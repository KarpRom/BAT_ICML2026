"""Turn the AudioSet study outputs into a per-class report: <out-dir>/report.md + figures.

Inputs (all under --out-dir unless given):
  shards/*.parquet           per-clip stats from audio_stats.py
  strong_class_stats.csv     per-class event time scales from strong_label_stats.py (optional)
  --per-class-ap CSV         per-class AP of a finetuned model, columns name,ap (optional);
                             links the properties to what the model gets wrong

  python scripts/study/analyze.py --out-dir study --dataset-dir $AUDIOSET \
      [--per-class-ap runs/bat_as2m_ft/per_class_ap.csv]

Per-class AP mostly follows how many training clips a class has, so each property is tested
as AP ~ log(n_train) + property (standardized), not as a plain correlation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_CSVS = {"eval": "eval_segments.csv", "balanced": "balanced_train_segments.csv",
              "unbalanced": "unbalanced_train_segments.csv"}

# per-clip columns aggregated per class (median over the clips carrying the label)
CLIP_FEATURES = ["cutoff40_hz", "rolloff95_hz", "energy_above_8k", "onset_rate_hz", "ioi_median_s",
                 "ioi_below_token", "mod_above_token", "mod_above_half_token", "silence_frac"]
STRONG_FEATURES = ["dur_median_s", "short_160ms", "gap_below_160ms", "multi_160ms", "seq_160ms", "seq_80ms",
                   "seq_40ms"]


def read_shards(out_dir: Path) -> pd.DataFrame:
    files = sorted((out_dir / "shards").glob("*.parquet")) + sorted((out_dir / "shards").glob("*.csv.gz"))
    if not files:
        sys.exit(f"no shards in {out_dir / 'shards'}")
    df = pd.concat([pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files],
                   ignore_index=True)
    df["error"] = df["error"].fillna("")
    df["ytid"] = df.path.str.rsplit("/", n=1).str[-1].str.rsplit(".", n=1).str[0].str.replace(r"^Y", "", regex=True)
    return df


def read_labels(dataset_dir: Path) -> pd.DataFrame:
    parts = []
    for split, name in LABEL_CSVS.items():
        p = dataset_dir / name
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        d = pd.read_csv(p, comment="#", header=None, names=["ytid", "start", "end", "labels"],
                        skipinitialspace=True, quotechar='"')
        d["split"] = split
        parts.append(d[["ytid", "labels", "split"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ytid", "labels", "split"])


def ontology_info(dataset_dir: Path) -> tuple:
    p = dataset_dir / "ontology.json"
    if not p.exists():
        return {}, {}
    onto = json.loads(p.read_text())
    names = {o["id"]: o["name"] for o in onto}
    parent = {}
    for o in onto:
        for c in o.get("child_ids", []):
            parent.setdefault(c, o["id"])

    def root(mid: str) -> str:
        seen = set()
        while mid in parent and mid not in seen:
            seen.add(mid)
            mid = parent[mid]
        return names.get(mid, mid)

    return names, {m: root(m) for m in names}


def fit_vs_count(ap: np.ndarray, log_n: np.ndarray, x: np.ndarray) -> dict:
    """OLS of AP on [1, log n_train, z(x)]: the coefficient of z(x) is the AP change (points) per
    standard deviation of the property, at equal training-set size."""
    ok = np.isfinite(ap) & np.isfinite(log_n) & np.isfinite(x)
    if ok.sum() < 20:
        return {"n": int(ok.sum()), "coef": np.nan, "t": np.nan, "r2_gain": np.nan}
    ap, log_n, x = ap[ok], log_n[ok], x[ok]
    z = (x - x.mean()) / (x.std() + 1e-12)
    X0 = np.column_stack([np.ones_like(ap), log_n])
    X1 = np.column_stack([X0, z])
    b1, *_ = np.linalg.lstsq(X1, ap, rcond=None)
    b0, *_ = np.linalg.lstsq(X0, ap, rcond=None)
    rss1 = ((ap - X1 @ b1) ** 2).sum()
    rss0 = ((ap - X0 @ b0) ** 2).sum()
    tss = ((ap - ap.mean()) ** 2).sum()
    sigma2 = rss1 / (len(ap) - 3)
    se = np.sqrt(sigma2 * np.linalg.inv(X1.T @ X1)[2, 2])
    return {"n": int(len(ap)), "coef": float(b1[2]), "t": float(b1[2] / se), "r2_gain": float((rss0 - rss1) / tss)}


def quantiles(s: pd.Series, fmt: str = ",.0f", qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> str:
    s = s.dropna()
    return " · ".join(f"p{int(q * 100)} {s.quantile(q):{fmt}}" for q in qs) if len(s) else "n/a"


def table(d: pd.DataFrame, cols: list) -> str:
    cols = [c for c in cols if c in d]
    try:
        return d[cols].to_markdown(index=False, floatfmt=".3g")
    except ImportError:  # to_markdown needs tabulate
        return "```\n" + d[cols].to_string(index=False) + "\n```"


def figures(clips: pd.DataFrame, out_dir: Path) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made = []
    ok = clips[clips.error == ""]
    for col, title, fname in [("cutoff40_hz", "Effective bandwidth (cutoff, -40 dB)", "cutoff_hist.png"),
                              ("brickwall_hz", "Brickwall lowpass frequency (clips with one)", "brickwall_hist.png"),
                              ("onset_rate_hz", "Onset rate per clip", "onset_rate_hist.png")]:
        s = ok[col].dropna()
        if not len(s):
            continue
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.hist(s, bins=120, color="#3b6ea5")
        for f in (4000, 8000, 11025, 16000, 22050):
            if col != "onset_rate_hz":
                ax.axvline(f, color="#999", lw=0.6, ls="--")
        ax.set_title(title)
        ax.set_xlabel("Hz")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=120)
        plt.close(fig)
        made.append(fname)
    return made


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=Path("study"))
    p.add_argument("--dataset-dir", type=Path, required=True, help="AudioSet root with the label CSVs and ontology.json")
    p.add_argument("--strong", type=Path, default=None, help="strong_class_stats.csv (default: <out-dir>/...)")
    p.add_argument("--per-class-ap", type=Path, default=None, help="CSV with columns name,ap (AP in %% or 0-1)")
    p.add_argument("--report-dir", type=Path, default=None, help="where report.md + tables go (default: --out-dir)")
    args = p.parse_args()

    clips = read_shards(args.out_dir)
    out = args.report_dir or args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    summary_path = (args.strong.parent if args.strong else args.out_dir) / "strong_summary.json"
    labels = read_labels(args.dataset_dir)
    names, roots = ontology_info(args.dataset_dir)
    clips = clips.merge(labels[["ytid", "labels"]].drop_duplicates("ytid"), on="ytid", how="left")
    ok = clips[clips.error == ""]
    lines = ["# AudioSet sample-rate, bandwidth and time-scale study", ""]

    # ---- global
    lines += ["## Files", "",
              f"- clips analysed: {len(clips):,} (with errors: {(clips.error != '').sum():,}; "
              f"silent: {(clips.error == 'silent').sum():,}; without labels: {clips.labels.isna().sum():,})",
              "- by split: " + ", ".join(f"{k} {v:,}" for k, v in clips.split.value_counts().items()),
              "- header sample rate: " + ", ".join(f"{int(k)} Hz: {v:,}" for k, v in clips.sr.value_counts().items()
                                                    if pd.notna(k)),
              "- channels: " + ", ".join(f"{int(k)}: {v:,}" for k, v in clips.channels.value_counts().items()
                                         if pd.notna(k)),
              "- duration (s): " + quantiles(clips.duration_s, fmt=".2f"), ""]
    nyq = ok.sr / 2
    lines += ["## Effective bandwidth", "",
              f"- cutoff (-40 dB, Hz): {quantiles(ok.cutoff40_hz)}",
              f"- clips with a brickwall lowpass: {ok.brickwall_hz.notna().mean():.1%}; below 8 kHz "
              f"(narrower than BAT's 0-8 kHz input band):{(ok.brickwall_hz < 7800).mean():.1%}; below 4.2 kHz "
              f"(telephone-like): {(ok.brickwall_hz < 4200).mean():.1%}",
              f"- clips whose content reaches above 8 kHz (cutoff > 8.2 kHz, needs sr > 16 kHz to keep): "
              f"{(ok.cutoff40_hz > 8200).mean():.1%}",
              f"- energy above 8 kHz (share of clip energy, where sr > 16 kHz): {quantiles(ok.energy_above_8k, fmt='.2e')}",
              f"- clips whose cutoff is the header Nyquist (no lowpass below it): {(ok.cutoff40_hz >= 0.97 * nyq).mean():.1%}",
              ""]
    lines += ["## Time scales (per clip, BAT front end: 10 ms hop)", "",
              f"- onset rate (/s): {ok.onset_rate_hz.describe()[['25%', '50%', '75%']].round(2).to_dict()}",
              f"- share of inter-onset intervals shorter than one 160 ms token: "
              f"{ok.ioi_below_token.describe()[['25%', '50%', '75%']].round(3).to_dict()}",
              f"- share of envelope modulation power faster than one token (> 6.25 Hz): "
              f"{ok.mod_above_token.describe()[['25%', '50%', '75%']].round(3).to_dict()} "
              f"(stationary noise gives ~0.63: compare classes, not absolute values)", ""]

    # ---- per class
    lab = ok[["labels"] + CLIP_FEATURES + ["brickwall_hz"]].dropna(subset=["labels"]).copy()
    lab["mid"] = lab["labels"].str.split(",")
    lab = lab.explode("mid")
    lab["band_limited_8k"] = lab.brickwall_hz < 7800
    per = lab.groupby("mid").agg(n_clips=("labels", "size"), band_limited_8k=("band_limited_8k", "mean"),
                                 **{c: (c, "median") for c in CLIP_FEATURES}).reset_index()
    n_train = labels[labels.split != "eval"].assign(mid=lambda d: d["labels"].str.split(",")).explode("mid") \
        .groupby("mid").size().rename("n_train")
    per = per.merge(n_train, on="mid", how="left")
    per.insert(1, "name", per.mid.map(names).fillna(per.mid))
    per.insert(2, "branch", per.mid.map(roots).fillna(""))

    strong_path = args.strong or args.out_dir / "strong_class_stats.csv"
    if strong_path.exists():
        strong = pd.read_csv(strong_path)
        per = per.merge(strong[["mid"] + [c for c in STRONG_FEATURES if c in strong]], on="mid", how="left")
    per.to_csv(out / "class_table.csv", index=False)

    show = ["name", "branch", "n_clips", "cutoff40_hz", "band_limited_8k", "energy_above_8k", "onset_rate_hz",
            "mod_above_token", "dur_median_s", "seq_160ms", "seq_80ms"]
    branch = per.groupby("branch")[[c for c in show[2:] if c in per]].median().reset_index()
    lines += ["## Per ontology branch (median over classes)", "", table(branch, ["branch"] + show[2:]), ""]
    for col, title in [("energy_above_8k", "Most content above 8 kHz"),
                       ("band_limited_8k", "Most often band-limited below 8 kHz"),
                       ("mod_above_token", "Fastest envelope modulation"),
                       ("seq_160ms", "Most often sharing a 160 ms token with a sequential event"),
                       ("dur_median_s", "Shortest events")]:
        if col in per:
            top = per[per.n_clips >= 100].sort_values(col, ascending=col == "dur_median_s").head(20)
            lines += [f"## {title} (top 20 classes, >= 100 clips)", "", table(top, show), ""]

    if "seq_160ms" in per:
        summ = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if summ:
            lines += ["## Strong labels: what one token merges", "",
                      f"- events shorter than 40 / 80 / 160 ms: " + " / ".join(f"{v:.1%}" for v in summ["events_shorter_than"].values()),
                      "- active patches shared with another class (multi) or with a non-overlapping one (seq): " +
                      "; ".join(f"{w}: multi {v['multi']:.1%}, seq {v['seq']:.1%}" for w, v in summ["patches"].items()), ""]

    # ---- link to performance
    if args.per_class_ap:
        ap = pd.read_csv(args.per_class_ap)
        if ap.ap.max() <= 1.0:
            ap["ap"] *= 100
        perf = per.merge(ap[["name", "ap"]], on="name", how="inner")
        log_n = np.log10(perf.n_train.astype(float).clip(lower=1)).values
        rows = []
        for c in CLIP_FEATURES + ["band_limited_8k"] + [c for c in STRONG_FEATURES if c in perf]:
            r = fit_vs_count(perf.ap.values, log_n, perf[c].astype(float).values)
            rows.append({"property": c, **r})
        reg = pd.DataFrame(rows)
        reg.to_csv(out / "ap_vs_properties.csv", index=False)
        lines += [f"## Per-class AP vs properties ({args.per_class_ap}, {len(perf)} classes matched)", "",
                  "AP ~ log10(n_train) + z(property). coef = AP points per std of the property at equal training size; "
                  "|t| > 3 is a clear effect; r2_gain = variance explained beyond the class count.", "",
                  table(reg.round(3), list(reg.columns)), ""]

    figs = figures(clips, out)
    if figs:
        lines += ["## Figures", ""] + [f"![{f}]({f})" for f in figs] + [""]
    (out / "report.md").write_text("\n".join(lines))
    print(f"wrote {out / 'report.md'} and {out / 'class_table.csv'} ({len(per)} classes)")


if __name__ == "__main__":
    sys.exit(main())
