"""Experiment runner for Jean Zay: one manifest, one rerunnable command.

    python scripts/xp.py status             # what is done / queued / stopped / to do
    python scripts/xp.py submit [names]     # (re)submit what is needed, all runs by default
    python scripts/xp.py submit --dry-run   # print the sbatch commands only
    python scripts/xp.py results            # table of every finished run's metrics
    python scripts/xp.py logs <name>        # tail of the run's latest SLURM log
    python scripts/xp.py cancel <name>      # scancel the run's jobs

`submit` can be run any number of times: it skips runs that are DONE or already queued,
resubmits STOPPED runs (they resume from their last checkpoint) and TODO runs whose
dependencies are done or queued (then chained with --dependency=afterok). Run it on the
login node, from the repo root. The manifest (experiments/runs.yaml by default, -m to change)
lists the runs; see that file for the fields.

Run states:
  DONE     finished (DONE marker / all shards / report written)
  RUNNING  a job of this run is running;  PENDING: queued, none running
  STOPPED  has progress but no job and not done: timeout, crash or cancel (see `logs`)
  TODO     never started;  BLOCKED: a dependency is not done and not queued
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
JOB_PREFIX = "xp_"

# Time requests stay close to what the jobs take (short jobs also start sooner: backfill).
# Measured on JZ: a 39k-clip study shard ~3 min (+ ~5 min for the file list in task 0), the
# strong-label stats ~3 min, the report a few minutes. Training runs have no default: each one
# sets `time` in the manifest, from the measured step rate (see `calibrate` in runs.yaml).
KIND_DEFAULTS = {
    "pretrain": {"gpus": 4, "cpus_per_task": 24, "script": "slurm/pretrain_bat.slurm"},
    "finetune": {"gpus": 1, "cpus_per_task": 24, "script": "slurm/finetune_bat.slurm"},
    "study": {"cpus_per_task": 40, "time": "00:20:00", "script": "slurm/study_array.slurm"},
    "strong": {"cpus_per_task": 4, "time": "00:15:00"},
    "report": {"cpus_per_task": 4, "time": "00:15:00"},
}


def expand(v):
    return os.path.expanduser(os.path.expandvars(v)) if isinstance(v, str) else v


def load_env_script(path: Path) -> None:
    """Variables exported by the jobs' env script (e.g. AUDIOSET), so that the manifest's $VARS
    expand here exactly as in the jobs. Sourced in a throwaway subshell; nothing else changes."""
    res = subprocess.run(["bash", "-c", f"source {shlex.quote(str(path))} >/dev/null 2>&1; env -0"],
                         capture_output=True, cwd=REPO)
    for item in res.stdout.split(b"\0"):
        k, sep, v = item.decode(errors="replace").partition("=")
        if sep and not k.startswith(("BASH_FUNC_", "_")):
            os.environ[k] = v


class Manifest:
    def __init__(self, path: Path):
        data = yaml.safe_load(path.read_text())
        raw = data.get("settings") or {}
        env_script = expand(raw.get("env_script"))
        if env_script:
            env_script = Path(env_script) if Path(env_script).is_absolute() else REPO / env_script
            if env_script.exists():
                load_env_script(env_script)
        self.settings = {k: expand(v) for k, v in raw.items()}
        if env_script:
            self.settings["env_script"] = str(env_script)
        self.root = Path(self.settings.get("root", "runs"))
        if not self.root.is_absolute():
            self.root = REPO / self.root
        self.runs = {}
        entries = []
        for r in data.get("runs") or []:  # seeds: [a, b] -> one run <name>_s<seed> per seed
            seeds = r.pop("seeds", None)
            entries += [{**r, "name": f"{r['name']}_s{sd}", "seed": sd} for sd in seeds] if seeds else [r]
        for r in entries:
            if r["name"] in self.runs:
                sys.exit(f"duplicate run name in manifest: {r['name']}")
            if r.get("kind") not in KIND_DEFAULTS:
                sys.exit(f"run {r['name']}: unknown kind {r.get('kind')!r} (one of {', '.join(KIND_DEFAULTS)})")
            run = {**KIND_DEFAULTS[r["kind"]], **{k: expand(v) for k, v in r.items()}}
            run["deps"] = list(run.get("depends_on") or [])
            for key in ("init", "per_class_ap", "study", "strong_run"):
                v = run.get(key)
                if isinstance(v, str) and v.startswith("run:") and v[4:] not in run["deps"]:
                    run["deps"].append(v[4:])
            self.runs[r["name"]] = run
        for name, run in self.runs.items():
            for d in run["deps"]:
                if d not in self.runs:
                    sys.exit(f"run {name}: unknown dependency {d!r}")

    def run_dir(self, name: str) -> Path:
        return self.root / name

    def ordered(self, names):
        """names plus nothing else, dependencies first."""
        out, seen = [], set()

        def visit(n, stack=()):
            if n in seen:
                return
            if n in stack:
                sys.exit(f"dependency cycle: {' -> '.join(stack + (n,))}")
            for d in self.runs[n]["deps"]:
                if d in names:
                    visit(d, stack + (n,))
            seen.add(n)
            out.append(n)

        for n in names:
            visit(n)
        return out


# ---------------------------------------------------------------- state

def queue() -> dict:
    """{run name: [(job id, state)]} for this user's xp_ jobs; {} if squeue is unavailable."""
    if not shutil.which("squeue"):
        return {}
    res = subprocess.run(["squeue", "--me", "--noheader", "--format=%i|%j|%T"], capture_output=True, text=True)
    jobs = {}
    for line in res.stdout.splitlines():
        jid, name, state = line.strip().split("|")
        if name.startswith(JOB_PREFIX):
            jobs.setdefault(name[len(JOB_PREFIX):], []).append((jid.split("_")[0], state))
    return jobs


def pretrain_step(d: Path):
    steps = [int(m.group(1)) for f in d.glob("step(*)_BAT_state_*.pt") if (m := re.match(r"step\((\d+)\)", f.name))]
    return max(steps) if steps else None


def progress(m: Manifest, name: str):
    """(done, progress text or None if nothing started)."""
    run, d = m.runs[name], m.run_dir(name)
    kind = run["kind"]
    if kind in ("pretrain", "finetune") and (d / "DONE").exists():
        return True, "done"
    if kind == "pretrain":
        s = pretrain_step(d)
        return False, (f"step {s}" if s is not None else None)
    if kind == "finetune":
        p = d / "progress.json"
        if p.exists():
            j = json.loads(p.read_text())
            return False, f"step {j['step']}/{j['total']}"
        return False, ("started" if any(d.glob("slurm-*.out")) else None)
    if kind == "study":
        n = int(run["num_shards"])
        done = sum(1 for i in range(n) if any((d / "shards" / f"{i:05d}{s}").exists() for s in (".parquet", ".csv.gz")))
        return done == n, (f"shards {done}/{n}" if done or (d / "files.csv").exists() else None)
    if kind == "strong":
        return (d / "strong_class_stats.csv").exists(), None
    if kind == "report":
        return (d / "report.md").exists(), None
    raise ValueError(kind)


def last_log_line(d: Path) -> str:
    logs = sorted(d.glob("slurm-*.out"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return ""
    text = logs[-1].read_text(errors="replace").splitlines()
    for line in reversed(text[-200:]):
        if re.search(r"DUE TO TIME LIMIT|CANCELLED|Error|Traceback|Killed|OOM|out of memory", line):
            return line.strip()[:120]
    return ""


def states(m: Manifest) -> dict:
    q = queue()
    out = {}
    for name in m.ordered(list(m.runs)):
        done, prog = progress(m, name)
        jobs = q.get(name, [])
        if done:
            st = "DONE"
        elif any(s == "RUNNING" for _, s in jobs):
            st = "RUNNING"
        elif jobs:
            st = "PENDING"
        elif prog is not None:
            st = "STOPPED"
        elif any(out[dep]["state"] not in ("DONE", "RUNNING", "PENDING") for dep in m.runs[name]["deps"]):
            st = "BLOCKED"
        else:
            st = "TODO"
        out[name] = {"state": st, "progress": prog or "", "jobs": [j for j, _ in jobs],
                     "note": last_log_line(m.run_dir(name)) if st == "STOPPED" else ""}
    return out


# ---------------------------------------------------------------- submission

def resolve_init(m: Manifest, run: dict):
    init = run.get("init", "bat_base")
    if init in (None, "none"):
        return None
    if init == "bat_base":
        return m.settings.get("bat_base", str(Path.home() / ".cache/bat_ssl/BAT_base.pt"))
    if init.startswith("run:"):
        src = m.runs[init[4:]]
        f = "final_ema_BAT_format.pt" if run.get("init_ema") else "final_BAT_format.pt"
        if src["kind"] != "pretrain":
            sys.exit(f"init {init}: only a pretrain run can be used as init")
        return str(m.run_dir(init[4:]) / f)
    return init


def overrides(run: dict) -> str:
    ov = run.get("overrides") or {}
    parts = []
    for k, v in ov.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            parts.append(flag)
        elif v is not False and v is not None:
            parts += [flag, str(v)]
    return " ".join(shlex.quote(p) for p in parts)


def sbatch_jobs(m: Manifest, name: str, dep_jobs: list) -> list:
    """[(sbatch argv, extra environment)], one item per job (several for a chain)."""
    run, d = m.runs[name], m.run_dir(name)
    s = m.settings
    kind = run["kind"]
    gpu = kind in ("pretrain", "finetune")
    base = ["sbatch", "--parsable", f"--job-name={JOB_PREFIX}{name}",
            f"--account={run.get('account') or s.get('account' if gpu else 'cpu_account')}",
            f"--cpus-per-task={run['cpus_per_task']}"]
    env = {"ENV_SCRIPT": s.get("env_script") or ""}
    if not env["ENV_SCRIPT"]:
        env.pop("ENV_SCRIPT")
    if run.get("seed") is not None:  # read by engine.utils.seed_everything (default 32)
        env["PYTHONHASHSEED"] = str(run["seed"])
    if gpu:
        base += [f"--gres=gpu:{run['gpus']}", f"--ntasks-per-node={run['gpus']}"]
    chain = int(run.get("chain") or 0)
    if chain:  # N x 2 h jobs on the dev QOS, each resuming the previous one
        base += ["--qos=qos_gpu_h100-dev" if gpu else "--qos=qos_cpu-dev", "--time=02:00:00"]
    else:
        base += [f"--time={run['time']}"] + ([f"--qos={run['qos']}"] if run.get("qos") else [])
    base += [f"--output={d}/slurm-%j.out" if kind != "study" else f"--output={d}/slurm-%A_%a.out"]

    audioset = run.get("audioset") or s.get("audioset")
    if kind == "pretrain":
        env.update(AUTO_RESUME="1", CONFIG=run["config"], LOG_DIR=str(d), INIT_CKPT=resolve_init(m, run) or "",
                   DATA_DIR=run.get("data_dir") or f"{audioset}/unbalanced_train_segments",
                   CACHE_DIR=str(m.root / "_cache"), SAVE_INTERVAL=str(run.get("save_interval", 1000)),
                   EXTRA=overrides(run))
        script = [run["script"]]
    elif kind == "finetune":
        env.update(CONFIG=run["config"], RUN_DIR=str(d), DATA_DIR=run.get("data_dir") or audioset,
                   INIT_CKPT=resolve_init(m, run) or "", MODEL_NAME=run.get("model_name", "BAT"),
                   EXTRA=overrides(run))
        script = [run["script"]]
    elif kind == "study":
        n = int(run["num_shards"])
        todo = [str(i) for i in range(n) if not any((d / "shards" / f"{i:05d}{x}").exists()
                                                    for x in (".parquet", ".csv.gz"))]
        base += [f"--array={','.join(todo)}", "--partition=cpu_p1", "--hint=nomultithread"]
        env.update(AUDIOSET=audioset, OUT_DIR=str(d), NUM_SHARDS=str(n))
        script = [run["script"]]
    elif kind == "strong":
        strong_dir = run.get("strong_dir") or f"{audioset}/strong"
        cmd = (f"python scripts/study/strong_label_stats.py stats --dir {shlex.quote(strong_dir)} "
               f"--ontology {shlex.quote(audioset + '/ontology.json')} --out-dir {shlex.quote(str(d))}")
        script = ["--partition=prepost", f"--wrap={env_prefix(s)}{cmd}"]
    else:  # report
        study = m.run_dir(run["study"][4:] if run["study"].startswith("run:") else run["study"])
        cmd = (f"python scripts/study/analyze.py --out-dir {shlex.quote(str(study))} "
               f"--dataset-dir {shlex.quote(audioset)} --report-dir {shlex.quote(str(d))}")
        if run.get("strong_run"):
            sr = run["strong_run"][4:] if run["strong_run"].startswith("run:") else run["strong_run"]
            cmd += f" --strong {shlex.quote(str(m.run_dir(sr) / 'strong_class_stats.csv'))}"
        ap = run.get("per_class_ap")
        if ap:
            ap_path = m.run_dir(ap[4:]) / "per_class_ap.csv" if ap.startswith("run:") else Path(ap)
            cmd += f" --per-class-ap {shlex.quote(str(ap_path))}"
        script = ["--partition=prepost", f"--wrap={env_prefix(s)}{cmd}"]

    jobs = []
    for i in range(max(chain, 1)):
        dep = ([f"--dependency=afterok:{':'.join(dep_jobs)}"] if dep_jobs else []) if i == 0 else ["CHAIN"]
        jobs.append((base + dep + script, env))
    return jobs


def env_prefix(s: dict) -> str:
    """Environment setup for --wrap jobs (the slurm scripts do the same themselves)."""
    if s.get("env_script"):
        return f"source {shlex.quote(s['env_script'])} && "
    return "module purge && module load pytorch-gpu/py3/2.8.0 && "


def check_inputs(m: Manifest, name: str):
    """Problems that would make the job fail right away (missing init, data, config)."""
    run, kind = m.runs[name], m.runs[name]["kind"]
    problems = []
    audioset = run.get("audioset") or m.settings.get("audioset")
    if not audioset or "$" in audioset:
        problems.append("settings.audioset (or $AUDIOSET) is not set")
    env_script = m.settings.get("env_script")
    if env_script and not Path(env_script).exists():
        problems.append(f"env_script not found: {env_script}")
    if kind in ("finetune", "report") and audioset and not run.get("data_dir"):
        missing = [f for f in ("ontology.json", "eval_segments.csv", "balanced_train_segments.csv",
                               "unbalanced_train_segments.csv") if not (Path(audioset) / f).exists()]
        if missing:
            problems.append(f"{audioset} lacks {', '.join(missing)} (label files: see settings.audioset)")
    if kind in ("pretrain", "finetune"):
        if not run.get("time") and not run.get("chain"):
            problems.append("no `time` (nor `chain`): set it from a measured step rate, e.g. with a short "
                            "calibration run (overrides: {optimization_steps: 300}, time: '00:20:00')")
        if not (REPO / run["config"]).exists():
            problems.append(f"config not found: {run['config']}")
        init = resolve_init(m, run)
        if init and not str(run.get("init") or "").startswith("run:") and not Path(init).exists():
            problems.append(f"init checkpoint not found: {init}")
    if kind == "strong":
        sd = Path(run.get("strong_dir") or f"{audioset}/strong")
        if not (sd / "audioset_train_strong.tsv").exists():
            problems.append(f"strong labels missing in {sd}: run "
                            f"`python scripts/study/strong_label_stats.py download --dir {sd}` (login node)")
    return problems


def submit(m: Manifest, names: list, dry_run: bool) -> None:
    st = states(m)
    submitted = {}  # run -> last job id of what we submit now
    for name in m.ordered(names):
        s = st[name]["state"]
        if s in ("DONE", "RUNNING", "PENDING"):
            print(f"[{s:>7}] {name}")
            continue
        dep_jobs, blocked = [], []
        for dep in m.runs[name]["deps"]:
            if dep in submitted:
                dep_jobs.append(submitted[dep])
            elif st[dep]["state"] in ("RUNNING", "PENDING"):
                dep_jobs += st[dep]["jobs"][-1:]  # the last job of a chain
            elif st[dep]["state"] != "DONE":
                blocked.append(dep)
        if blocked:
            print(f"[BLOCKED] {name}: waiting for {', '.join(blocked)} (not done, not queued; submit it too)")
            continue
        problems = check_inputs(m, name)
        if problems:
            print(f"[  ERROR] {name}: " + "; ".join(problems))
            continue
        run_dir = m.run_dir(name)
        prev = None
        for argv, env in sbatch_jobs(m, name, dep_jobs):
            argv = [a if a != "CHAIN" else f"--dependency=afterany:{prev}" for a in argv]
            shown = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " " + " ".join(shlex.quote(a) for a in argv)
            if dry_run:
                print(f"[dry-run] {name}: {shown}")
                prev = "<prev>"
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "xp_run.yaml").write_text(yaml.safe_dump({"run": m.runs[name], "settings": m.settings}))
            res = subprocess.run(argv, cwd=REPO, env={**os.environ, **env}, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[ FAILED] {name}: sbatch said: {res.stderr.strip()}")
                break
            prev = res.stdout.strip().split(";")[0]
            print(f"[SUBMIT ] {name}: job {prev} ({s.lower()})")
        if prev:
            submitted[name] = prev if not dry_run else f"<{name}>"


# ---------------------------------------------------------------- commands

def cmd_status(m: Manifest) -> None:
    st = states(m)
    w = max(len(n) for n in st) if st else 10
    print(f"{'run':<{w}}  {'kind':<8}  {'state':<8}  {'progress':<14}  jobs / note")
    for name, s in st.items():
        extra = ",".join(s["jobs"]) or s["note"]
        print(f"{name:<{w}}  {m.runs[name]['kind']:<8}  {s['state']:<8}  {s['progress']:<14}  {extra}")
    counts = {}
    for s in st.values():
        counts[s["state"]] = counts.get(s["state"], 0) + 1
    print("\n" + ", ".join(f"{k}: {v}" for k, v in counts.items()))


def cmd_results(m: Manifest, filt: str) -> None:
    rows = []
    for name, run in m.runs.items():
        if filt and not re.search(filt, name):
            continue
        p = m.run_dir(name) / "metrics.json"
        if p.exists():
            rows.append({"run": name, "init": str(run.get("init", "bat_base")), **json.loads(p.read_text())})
    if not rows:
        print("no finished run with metrics.json")
        return
    keys = ["run", "init"] + sorted({k for r in rows for k in r} - {"run", "init"})
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[k]:.2f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys) + " |")
    # Seed groups: <name>_s<seed> runs together with <name> itself (the default seed, 32).
    groups = {}
    for r in rows:
        groups.setdefault(re.sub(r"_s\d+$", "", r["run"]), []).append(r)
    groups = {g: rs for g, rs in groups.items() if len(rs) > 1}
    if groups:
        metric_keys = [k for k in keys[2:] if all(isinstance(r.get(k), float) for rs in groups.values() for r in rs)]
        lines += ["", "Over seeds (mean ± std, n):", "",
                  "| group | n | " + " | ".join(metric_keys) + " |", "|" + "---|" * (len(metric_keys) + 2)]
        for g, rs in groups.items():
            cells = []
            for k in metric_keys:
                v = [r[k] for r in rs]
                mu = sum(v) / len(v)
                sd = (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5
                cells.append(f"{mu:.2f} ± {sd:.2f}")
            lines.append(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")
    print("\n".join(lines))
    out = m.root / "results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwritten to {out}")


def cmd_logs(m: Manifest, name: str, n: int) -> None:
    logs = sorted(m.run_dir(name).glob("slurm-*.out"), key=lambda p: p.stat().st_mtime)
    if not logs:
        print(f"no log for {name} in {m.run_dir(name)}")
        return
    print(f"==> {logs[-1]}")
    print("\n".join(logs[-1].read_text(errors="replace").splitlines()[-n:]))


def cmd_cancel(m: Manifest, name: str) -> None:
    subprocess.run(["scancel", f"--name={JOB_PREFIX}{name}"], check=False)
    print(f"cancelled jobs named {JOB_PREFIX}{name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--manifest", type=Path, default=REPO / "experiments/runs.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    ps = sub.add_parser("submit")
    ps.add_argument("names", nargs="*", help="runs to submit (default: all); their dependencies must be "
                                             "done, queued or listed too")
    ps.add_argument("--dry-run", action="store_true")
    pr = sub.add_parser("results")
    pr.add_argument("--filter", default="", help="regex on run names")
    pl = sub.add_parser("logs")
    pl.add_argument("name")
    pl.add_argument("-n", type=int, default=40)
    pc = sub.add_parser("cancel")
    pc.add_argument("name")
    args = p.parse_args()

    m = Manifest(args.manifest)
    for n in getattr(args, "names", None) or []:
        if n not in m.runs:
            sys.exit(f"unknown run {n!r}; runs: {', '.join(m.runs)}")
    if args.cmd == "status":
        cmd_status(m)
    elif args.cmd == "submit":
        submit(m, args.names or list(m.runs), args.dry_run)
    elif args.cmd == "results":
        cmd_results(m, args.filter)
    elif args.cmd == "logs":
        cmd_logs(m, args.name, args.n)
    else:
        cmd_cancel(m, args.name)


if __name__ == "__main__":
    main()
