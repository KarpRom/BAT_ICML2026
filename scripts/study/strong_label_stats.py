"""Event time scales per class from the AudioSet strong labels (Hershey et al. 2021).

How long are events, and how often does one time patch of a given width hold two different
classes? This tells whether BAT's 160 ms time patch merges distinct events, and how much
80 ms or 40 ms patches would separate them.

  download  fetch the strong-label TSVs (front node: compute nodes have no internet)
  stats     per-class table -> <out-dir>/strong_class_stats.csv + strong_summary.json

  python scripts/study/strong_label_stats.py download --dir $AUDIOSET/strong
  python scripts/study/strong_label_stats.py stats --dir $AUDIOSET/strong \
      --ontology $AUDIOSET/ontology.json --out-dir study

Per class, for patch widths of 40 / 80 / 160 ms (on a 10 ms grid):
  multi_<w>  share of the class's active patches that also hold another class
  seq_<w>    share where that other class does not overlap it inside the patch: the two
             events only follow each other, so a finer patch would separate them
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/strong/"
FILES = ["audioset_train_strong.tsv", "audioset_eval_strong.tsv", "mid_to_display_name.tsv"]
FRAME_S = 0.010
CLIP_FRAMES = 1000  # 10 s
WIDTHS_MS = (40, 80, 160)
SHORT_S = (0.04, 0.08, 0.16)


def download(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        out = dest / name
        if out.exists() and out.stat().st_size > 0:
            print(f"[skip] {out}")
            continue
        print(f"downloading {BASE_URL + name}")
        tmp = out.with_suffix(".tmp")
        urllib.request.urlretrieve(BASE_URL + name, tmp)
        tmp.replace(out)


def read_strong(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    df = df.rename(columns={cols.get("segment_id", df.columns[0]): "segment_id",
                            cols.get("start_time_seconds", df.columns[1]): "start",
                            cols.get("end_time_seconds", df.columns[2]): "end",
                            cols.get("label", df.columns[3]): "mid"})
    return df[["segment_id", "start", "end", "mid"]]


def mid_names(strong_dir: Path, ontology: Path = None) -> dict:
    names = {}
    if ontology and ontology.exists():
        names.update({o["id"]: o["name"] for o in json.loads(ontology.read_text())})
    m2n = strong_dir / "mid_to_display_name.tsv"
    if m2n.exists():
        t = pd.read_csv(m2n, sep="\t", header=None, names=["mid", "name"])
        names.update({m: n for m, n in zip(t.mid, t.name) if m not in names})
    return names


def clip_activity(events: pd.DataFrame) -> tuple:
    """Bool activity (n_classes_in_clip, CLIP_FRAMES) on the 10 ms grid; same-class events are
    merged by the union."""
    mids = sorted(events.mid.unique())
    idx = {m: i for i, m in enumerate(mids)}
    act = np.zeros((len(mids), CLIP_FRAMES), dtype=bool)
    starts = np.clip(np.floor(events.start.values / FRAME_S).astype(int), 0, CLIP_FRAMES)
    ends = np.clip(np.ceil(events.end.values / FRAME_S).astype(int), 0, CLIP_FRAMES)
    for m, s, e in zip(events.mid.values, starts, ends):
        act[idx[m], s:max(e, s + 1)] = True  # an event shorter than a frame still covers one
    return mids, act


def patch_counts(act: np.ndarray, width: int) -> tuple:
    """Per class: active patches, patches shared with another class, and patches shared with a
    class that does not overlap it inside the patch."""
    n_cls = act.shape[0]
    n_p = CLIP_FRAMES // width
    a = act[:, :n_p * width].reshape(n_cls, n_p, width)
    present = a.any(axis=2)  # (C, P)
    if n_cls == 1:
        z = np.zeros(1, dtype=int)
        return present.sum(axis=1), z, z
    co = np.einsum("cpw,dpw->pcd", a.astype(np.uint8), a.astype(np.uint8)) > 0  # overlap in patch
    both = present.T[:, :, None] & present.T[:, None, :]  # (P, C, D)
    eye = np.eye(n_cls, dtype=bool)[None]
    multi = (both & ~eye).any(axis=2).T  # (C, P)
    seq = (both & ~co & ~eye).any(axis=2).T
    return present.sum(axis=1), (multi & present).sum(axis=1), (seq & present).sum(axis=1)


def stats(strong_dir: Path, ontology: Path, out_dir: Path) -> None:
    parts = []
    for split in ("train", "eval"):
        p = strong_dir / f"audioset_{split}_strong.tsv"
        if p.exists():
            d = read_strong(p)
            d["split"] = split
            parts.append(d)
            print(f"{split}: {len(d):,} events in {d.segment_id.nunique():,} clips")
    ev = pd.concat(parts, ignore_index=True)
    ev = ev[ev.end > ev.start]
    ev["dur"] = ev.end - ev.start

    per_class = {}
    for w in WIDTHS_MS:
        per_class[w] = {}
    gaps = []  # same-class gaps: two events of one class closer than a token merge into it
    glob = {w: np.zeros(3, dtype=np.int64) for w in WIDTHS_MS}
    clips = ev.groupby("segment_id", sort=False)
    for seg, events in tqdm(clips, total=clips.ngroups, desc="clips", unit="clip",
                            mininterval=0.5 if sys.stderr.isatty() else 30):
        mids, act = clip_activity(events)
        for w in WIDTHS_MS:
            pres, multi, seq = patch_counts(act, int(round(w / 1000 / FRAME_S)))
            glob[w] += [pres.sum(), multi.sum(), seq.sum()]
            for m, a, b, c in zip(mids, pres, multi, seq):
                acc = per_class[w].setdefault(m, np.zeros(3, dtype=np.int64))
                acc += [a, b, c]
        for m, e in events.sort_values("start").groupby("mid"):
            if len(e) > 1:
                g = e.start.values[1:] - np.maximum.accumulate(e.end.values)[:-1]
                gaps += [(m, x) for x in g[g > 0]]

    names = mid_names(strong_dir, ontology)
    rows = []
    gap_df = pd.DataFrame(gaps, columns=["mid", "gap"])
    gap_by = dict(list(gap_df.groupby("mid").gap)) if len(gap_df) else {}
    for m, e in ev.groupby("mid"):
        r = {"mid": m, "name": names.get(m, m), "n_events": len(e), "n_clips": e.segment_id.nunique(),
             "dur_p10_s": e.dur.quantile(0.1), "dur_median_s": e.dur.median(), "dur_p90_s": e.dur.quantile(0.9)}
        for s in SHORT_S:
            r[f"short_{int(s * 1000)}ms"] = float((e.dur < s).mean())
        g = gap_by.get(m)
        r["gap_median_s"] = float(g.median()) if g is not None else np.nan
        r["gap_below_160ms"] = float((g < 0.16).mean()) if g is not None else np.nan
        for w in WIDTHS_MS:
            a, b, c = per_class[w].get(m, (0, 0, 0))
            r[f"multi_{w}ms"] = b / a if a else np.nan
            r[f"seq_{w}ms"] = c / a if a else np.nan
        rows.append(r)
    table = pd.DataFrame(rows).sort_values("n_events", ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "strong_class_stats.csv", index=False)
    summary = {"n_events": int(len(ev)), "n_clips": int(ev.segment_id.nunique()),
               "n_classes": int(ev.mid.nunique()),
               "events_shorter_than": {f"{int(s * 1000)}ms": float((ev.dur < s).mean()) for s in SHORT_S},
               "patches": {f"{w}ms": {"multi": float(glob[w][1] / glob[w][0]), "seq": float(glob[w][2] / glob[w][0])}
                           for w in WIDTHS_MS}}
    (out_dir / "strong_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'strong_class_stats.csv'} ({len(table)} classes)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pd_ = sub.add_parser("download")
    pd_.add_argument("--dir", type=Path, required=True)
    ps = sub.add_parser("stats")
    ps.add_argument("--dir", type=Path, required=True, help="folder with the strong-label TSVs")
    ps.add_argument("--ontology", type=Path, default=None, help="AudioSet ontology.json (class names)")
    ps.add_argument("--out-dir", type=Path, default=Path("study"))
    args = p.parse_args()
    if args.cmd == "download":
        download(args.dir)
    else:
        stats(args.dir, args.ontology, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
