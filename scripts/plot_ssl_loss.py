"""Plot the SSL pretraining loss curve, stitching together the separate
BAT_history_<timestamp>.csv files a resumed run produces.

engine/ssl_trainer.py starts a fresh, empty history (and a fresh timestamp) every
time run_ssl_experiment() is called, and never writes a step number into the CSV --
just global_loss/local_loss, one row per completed optimizer step. So a training run
split across N SLURM submissions (via --resume-checkpoint) leaves N separate
BAT_history_*.csv files, each counting its own steps from 0. This script finds all of
them under --log-dir, orders them chronologically, and reconstructs the true absolute
step for every row by assuming what the normal --resume-checkpoint workflow guarantees:
each run resumed from the immediately-preceding run's latest checkpoint, so the
cumulative row count across earlier runs equals the step the next run started at. It
cross-checks that assumption against the checkpoints' own recorded resume chain
(args['resume_checkpoint'] -> that checkpoint's optimized_steps) and warns if a run
resumed from something other than the latest checkpoint of the previous run, since the
step axis would then be misaligned.

Usage:
    python scripts/plot_ssl_loss.py
    python scripts/plot_ssl_loss.py --log-dir logs/SSL --out logs/SSL/loss_curve.png --smooth 200
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

HISTORY_RE = re.compile(r"^BAT_history_(.+)\.csv$")
CHECKPOINT_RE = re.compile(r"^step\((\d+)\)_BAT_state_(.+)\.pt$")


def find_runs(log_dir: Path):
    """Return one (timestamp, csv_path, checkpoint_paths) tuple per training run
    (i.e. per SLURM job that called run_ssl_experiment), sorted chronologically --
    the timestamp format (%Y_%m_%d_%H_%M_%S) sorts correctly as a plain string."""
    checkpoints_by_ts = {}
    for p in log_dir.glob("step(*)_BAT_state_*.pt"):
        m = CHECKPOINT_RE.match(p.name)
        if m:
            checkpoints_by_ts.setdefault(m.group(2), []).append((int(m.group(1)), p))

    runs = []
    for csv_path in log_dir.glob("BAT_history_*.csv"):
        m = HISTORY_RE.match(csv_path.name)
        if not m:
            continue
        ts = m.group(1)
        ckpts = sorted(checkpoints_by_ts.get(ts, []))
        runs.append((ts, csv_path, ckpts))

    runs.sort(key=lambda r: r[0])
    return runs


def check_resume_alignment(csv_name: str, checkpoints, assumed_start_step: int) -> None:
    """Best-effort cross-check only -- never raises, just warns. Loads this run's
    latest checkpoint, follows its args['resume_checkpoint'] pointer back to the
    checkpoint it says it resumed from, and compares that checkpoint's own
    optimized_steps against what we assumed (sum of earlier runs' row counts)."""
    if not checkpoints:
        return
    _, latest_ckpt_path = checkpoints[-1]
    try:
        ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[Warning] Could not read {latest_ckpt_path} to verify resume alignment: {exc}")
        return

    resume_from = (ckpt.get("args") or {}).get("resume_checkpoint")
    if not resume_from:
        return  # this run wasn't resumed from anything (or args predates this field)

    resume_from = Path(resume_from)
    if not resume_from.exists():
        print(
            f"[Warning] {csv_name}: resumed from {resume_from}, which no longer exists -- "
            f"can't verify the step axis below lines up correctly for this run."
        )
        return

    try:
        prev_ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[Warning] Could not read {resume_from} to verify resume alignment: {exc}")
        return

    declared_start = prev_ckpt.get("optimized_steps")
    if declared_start is not None and declared_start != assumed_start_step:
        print(
            f"[Warning] {csv_name}: assumed it resumed at step {assumed_start_step:,} (sum of "
            f"earlier runs' rows), but it actually resumed from {resume_from.name} at step "
            f"{declared_start:,}. It looks like this run did not resume from the latest "
            f"checkpoint of the previous run -- the step axis may be misaligned from here on."
        )


def load_stitched(log_dir: Path) -> pd.DataFrame:
    runs = find_runs(log_dir)
    if not runs:
        raise SystemExit(f"No BAT_history_*.csv files found under {log_dir}")

    frames = []
    cumulative_steps = 0
    for ts, csv_path, ckpts in runs:
        check_resume_alignment(csv_path.name, ckpts, cumulative_steps)
        df = pd.read_csv(csv_path)
        df["step"] = range(cumulative_steps + 1, cumulative_steps + 1 + len(df))
        df["run"] = ts
        frames.append(df)
        cumulative_steps += len(df)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the stitched SSL pretraining loss curve")
    parser.add_argument(
        "--log-dir", default="logs/SSL", type=str,
        help="Directory with BAT_history_*.csv / step(N)_BAT_state_*.pt (default: logs/SSL)",
    )
    parser.add_argument(
        "--out", default=None, type=str, help="Output image path (default: <log-dir>/loss_curve.png)"
    )
    parser.add_argument(
        "--csv-out", default=None, type=str,
        help="Also save the stitched (step, global_loss, local_loss, run) table to this CSV",
    )
    parser.add_argument(
        "--smooth", default=200, type=int,
        help="Rolling-mean window in steps, overlaid on the raw curve; 0 disables it (default: 200)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_path = Path(args.out) if args.out else log_dir / "loss_curve.png"

    df = load_stitched(log_dir)
    print(f"Stitched {df['run'].nunique()} run(s), {len(df):,} total steps, from {log_dir}")

    if args.csv_out:
        df.to_csv(args.csv_out, index=False)
        print(f"Stitched table saved to {args.csv_out}")

    resume_boundaries = df.groupby("run")["step"].min().sort_values().tolist()[1:]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, col, title in zip(axes, ["global_loss", "local_loss"], ["Global loss", "Local loss"]):
        ax.plot(df["step"], df[col], color="tab:blue", alpha=0.3, linewidth=0.7, label="raw")
        if args.smooth > 1:
            ax.plot(
                df["step"], df[col].rolling(args.smooth, min_periods=1).mean(),
                color="tab:blue", linewidth=1.5, label=f"rolling mean ({args.smooth})",
            )
        for b in resume_boundaries:
            ax.axvline(b, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel(title)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Optimizer step")
    fig.suptitle("BAT SSL pretraining loss (dashed lines = resumed-job boundaries)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
