"""Per-clip sample-rate, bandwidth and time-scale statistics over the full AudioSet.

Answers "would another sample rate or token size help?" with numbers per clip, later
aggregated per class by analyze.py. Two subcommands:

  list   one sorted file list for eval + balanced + unbalanced (run once, on a CPU node)
  run    one shard of that list -> <out_dir>/shards/<shard>.parquet; a shard whose output
         exists is skipped, so resubmitting the array job only redoes the missing ones
  summary  headline numbers over the shards finished so far (per class: analyze.py)

  python scripts/study/audio_stats.py list --dataset-dir $AUDIOSET --out-dir study
  python scripts/study/audio_stats.py run --out-dir study --shard 0 --num-shards 100

slurm/study_array.slurm runs `run` as an array job (shard = SLURM_ARRAY_TASK_ID).
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# one thread per worker process: the pool already uses every core of the task
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np
import pandas as pd
from tqdm import tqdm

SPLITS = {"eval": "eval_segments", "balanced": "balanced_train_segments", "unbalanced": "unbalanced_train_segments"}

# BAT front end: 10 ms hop, 64 ms window (n_fft 1024 at 16 kHz), 128 mels up to 8 kHz
HOP_S = 0.010
WIN_S = 0.064
N_MELS = 128
TOKEN_S = 0.160  # 16 frames x 10 ms: one BAT time patch

ENERGY_ABOVE_HZ = (4000, 8000, 11000, 16000)


def list_files(dataset_dir: Path, out_dir: Path) -> None:
    rows = []
    for split, folder in SPLITS.items():
        d = dataset_dir / folder
        if not d.is_dir():
            print(f"[skip] {d} not found")
            continue
        names = sorted(e.name for e in os.scandir(d) if e.name.endswith((".wav", ".flac")))
        rows += [(split, (d / n).as_posix()) for n in names]
        print(f"{split}: {len(names):,} files")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["split", "path"])
    tmp = out_dir / "files.csv.tmp"
    df.to_csv(tmp, index=False)
    tmp.replace(out_dir / "files.csv")
    print(f"wrote {out_dir / 'files.csv'} ({len(df):,} files)")


def smooth_db(psd_db: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return psd_db
    kernel = np.ones(width) / width
    return np.convolve(np.pad(psd_db, (width // 2, width - 1 - width // 2), mode="edge"), kernel, mode="valid")


def bandwidth_stats(x: np.ndarray, sr: int) -> dict:
    """Effective bandwidth from the long-term spectrum: where the content actually stops,
    whatever the header sample rate says (upsampled phone audio, codec lowpass, ...)."""
    from scipy.signal import welch

    nperseg = int(2 ** np.ceil(np.log2(sr * 0.128)))  # ~128 ms -> ~8 Hz bins
    f, psd = welch(x, fs=sr, nperseg=min(nperseg, len(x)), noverlap=None, scaling="density")
    psd = np.maximum(psd, 1e-20)
    psd_db = 10 * np.log10(psd)
    df_hz = f[1] - f[0]
    sm = smooth_db(psd_db, max(1, int(round(100 / df_hz))))  # 100 Hz moving average

    out = {}
    total = psd.sum()
    cum = np.cumsum(psd) / total
    out["rolloff85_hz"] = float(f[np.searchsorted(cum, 0.85)])
    out["rolloff95_hz"] = float(f[np.searchsorted(cum, 0.95)])
    out["rolloff99_hz"] = float(f[min(np.searchsorted(cum, 0.99), len(f) - 1)])
    out["centroid_hz"] = float((f * psd).sum() / total)
    out["flatness"] = float(np.exp(np.log(psd).mean()) / psd.mean())
    for hz in ENERGY_ABOVE_HZ:
        out[f"energy_above_{hz // 1000}k"] = float(psd[f >= hz].sum() / total) if hz < sr / 2 else np.nan

    # Threshold cutoffs: last frequency still within 40 / 60 dB of the 0.2-4 kHz reference level.
    ref_band = (f >= 200) & (f <= 4000)
    ref = np.percentile(sm[ref_band], 90) if ref_band.any() else sm.max()
    for db in (40, 60):
        above = np.nonzero(sm > ref - db)[0]
        out[f"cutoff{db}_hz"] = float(f[above[-1]]) if len(above) else 0.0

    # Brickwall cutoff: steepest drop above 1 kHz over 300 Hz (resampling / codec lowpass).
    span = max(1, int(round(300 / df_hz)))
    drop = sm[:-span] - sm[span:]
    valid = f[:-span] >= 1000
    if valid.any():
        i = np.argmax(np.where(valid, drop, -np.inf))
        out["brickwall_drop_db"] = float(drop[i])
        out["brickwall_hz"] = float(f[i + span // 2]) if drop[i] >= 25 else np.nan
    else:
        out["brickwall_drop_db"], out["brickwall_hz"] = np.nan, np.nan
    return out


def mel_filterbank(sr: int, n_fft: int, n_mels: int, f_max: float) -> np.ndarray:
    """HTK-scale triangular filters, no normalization: torchaudio's MelSpectrogram defaults,
    i.e. BAT's front end. (numpy only: librosa's import chain needs extra packages.)"""
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    f_pts = mel2hz(np.linspace(hz2mel(0.0), hz2mel(f_max), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    lower = (freqs[None, :] - f_pts[:-2, None]) / (f_pts[1:-1, None] - f_pts[:-2, None])
    upper = (f_pts[2:, None] - freqs[None, :]) / (f_pts[2:, None] - f_pts[1:-1, None])
    return np.maximum(0.0, np.minimum(lower, upper))  # (n_mels, n_fft // 2 + 1)


def temporal_stats(x: np.ndarray, sr: int) -> dict:
    """Time scales on BAT's own front end (10 ms hop, 64 ms window, mels up to 8 kHz)."""
    from scipy.signal import stft, find_peaks

    hop = int(round(HOP_S * sr))
    n_fft = int(2 ** np.ceil(np.log2(WIN_S * sr)))
    _, _, z = stft(x, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary="even", padded=True)
    mel = mel_filterbank(sr, n_fft, N_MELS, min(8000, sr / 2)) @ (np.abs(z) ** 2)
    logmel = 10 * np.log10(np.maximum(mel, 1e-10))
    logmel = np.maximum(logmel, logmel.max() - 80)  # top_db=80, as AmplitudeToDB in BAT
    duration = len(x) / sr
    out = {}

    # Onsets: peaks of the spectral flux (positive log-mel increase, averaged over bands),
    # at least 30 ms apart, above median + max(3 robust std, 2 dB) of the clip's own flux and
    # 2 dB prominent (stationary noise then gives ~no onsets).
    env = np.maximum(0.0, np.diff(logmel, axis=1)).mean(axis=0)
    med = np.median(env)
    mad = np.median(np.abs(env - med)) + 1e-6
    peaks, _ = find_peaks(env, height=med + max(3 * 1.4826 * mad, 2.0), prominence=2.0, distance=3)
    onsets = peaks * HOP_S
    out["onset_rate_hz"] = len(onsets) / duration
    ioi = np.diff(onsets)
    out["ioi_median_s"] = float(np.median(ioi)) if len(ioi) else np.nan
    out["ioi_below_token"] = float((ioi < TOKEN_S).mean()) if len(ioi) else np.nan

    # Temporal modulation spectrum of each mel band's envelope (frame rate 100 Hz): the share
    # of modulation power faster than one token (6.25 Hz) or one half-token (12.5 Hz).
    e = logmel - logmel.mean(axis=1, keepdims=True)
    mod = np.abs(np.fft.rfft(e, axis=1)) ** 2
    mod_f = np.fft.rfftfreq(e.shape[1], d=HOP_S)
    band = (mod_f >= 0.5) & (mod_f <= 50)
    p = mod[:, band].sum(axis=0)
    p_total = p.sum() + 1e-12
    mf = mod_f[band]
    out["mod_above_token"] = float(p[mf > 1 / TOKEN_S].sum() / p_total)
    out["mod_above_half_token"] = float(p[mf > 2 / TOKEN_S].sum() / p_total)
    out["mod_centroid_hz"] = float((mf * p).sum() / p_total)
    return out


def analyze_file(path: str) -> dict:
    import soundfile as sf

    row = {"path": path, "error": ""}
    try:
        info = sf.info(path)
        row.update(sr=info.samplerate, channels=info.channels, frames=info.frames,
                   duration_s=info.frames / info.samplerate if info.samplerate else 0.0, subtype=info.subtype)
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if len(x) < int(0.5 * sr):
            row["error"] = "too_short"
            return row
        row["peak"] = float(np.abs(x).max())
        row["clip_frac"] = float((np.abs(x) >= 0.999).mean())
        rms = np.sqrt(np.mean(x ** 2))
        row["rms_dbfs"] = float(20 * np.log10(rms + 1e-12))
        frame = int(0.025 * sr)
        n = len(x) // frame
        frms = np.sqrt((x[:n * frame].reshape(n, frame) ** 2).mean(axis=1))
        row["silence_frac"] = float((20 * np.log10(frms + 1e-12) < -60).mean())
        if rms < 1e-5:
            row["error"] = "silent"
            return row
        row.update(bandwidth_stats(x, sr))
        row.update(temporal_stats(x, sr))
    except Exception as e:  # corrupt files exist in every AudioSet mirror: record, don't stop
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    return row


def write_table(df: pd.DataFrame, path: Path) -> Path:
    try:
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        final = path.with_suffix(".parquet")
    except ImportError:  # no pyarrow/fastparquet in this environment
        tmp = path.with_suffix(".csv.gz.tmp")
        df.to_csv(tmp, index=False, compression="gzip")
        final = path.with_suffix(".csv.gz")
    tmp.replace(final)
    return final


def summarize(df: pd.DataFrame) -> str:
    """The headline numbers of a set of clips: what analyze.py's report expands per class."""
    ok = df[df["error"].fillna("") == ""]
    q = lambda s, f=",.0f": " / ".join(f"{s.quantile(p):{f}}" for p in (0.1, 0.5, 0.9)) if s.notna().any() else "n/a"
    sr = ", ".join(f"{int(k)} Hz: {v / len(df):.1%}" for k, v in df["sr"].value_counts().items() if pd.notna(k))
    lines = [f"clips: {len(df):,} ({len(df) - len(ok):,} with errors, of which silent: {(df['error'] == 'silent').sum():,})",
             f"header sample rate: {sr}"]
    if len(ok):
        bw = ok["brickwall_hz"]
        lines += [
            f"cutoff -40 dB (Hz, p10 / p50 / p90): {q(ok['cutoff40_hz'])}",
            f"brickwall lowpass: {bw.notna().mean():.1%} of clips; below 7.8 kHz: {(bw < 7800).mean():.1%}; "
            f"below 4.2 kHz (telephone-like): {(bw < 4200).mean():.1%}",
            f"content above 8.2 kHz (lost by BAT's 16 kHz resampling): {(ok['cutoff40_hz'] > 8200).mean():.1%} of clips; "
            f"energy share above 8 kHz p50 / p90: {q(ok['energy_above_8k'], '.2e').split(' / ', 1)[1]}",
            f"onset rate (/s, p10 / p50 / p90): {q(ok['onset_rate_hz'], '.2f')}",
            f"inter-onset intervals < 160 ms (one token), p50 over clips: {ok['ioi_below_token'].median():.1%}",
            f"modulation power faster than one token, p50: {ok['mod_above_token'].median():.3f} "
            f"(stationary noise ~0.63: compare classes, not absolutes)"]
    return "\n".join("  " + l for l in lines)


def read_shards(out_dir: Path) -> pd.DataFrame:
    files = sorted((out_dir / "shards").glob("*.parquet")) + sorted((out_dir / "shards").glob("*.csv.gz"))
    if not files:
        sys.exit(f"no finished shard in {out_dir / 'shards'}")
    return pd.concat([pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files],
                     ignore_index=True)


def shard_done(out_dir: Path, shard: int) -> bool:
    stem = out_dir / "shards" / f"{shard:05d}"
    return any(stem.with_suffix(s).exists() for s in (".parquet", ".csv.gz"))


def run_shard(out_dir: Path, shard: int, num_shards: int, workers: int) -> None:
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = shard_dir / f"{shard:05d}"
    if shard_done(out_dir, shard):
        print(f"shard {shard} already done, skipping")
        return
    files = pd.read_csv(out_dir / "files.csv")
    part = files.iloc[shard::num_shards]
    print(f"shard {shard}/{num_shards}: {len(part):,} files, {workers} workers")
    t0 = time.time()
    with Pool(workers) as pool:
        # imap keeps the input order (rows line up with `part`); slow refresh in SLURM logs
        it = pool.imap(analyze_file, part["path"].tolist(), chunksize=16)
        rows = list(tqdm(it, total=len(part), desc=f"shard {shard}", unit="clip",
                         mininterval=0.5 if sys.stderr.isatty() else 30))
    df = pd.DataFrame(rows)
    df.insert(0, "split", part["split"].values)
    final = write_table(df, stem)
    print(f"wrote {final} in {time.time() - t0:.0f} s. Shard summary:\n{summarize(df)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list")
    pl.add_argument("--dataset-dir", type=Path, required=True, help="AudioSet root (eval_segments/, ...)")
    pl.add_argument("--out-dir", type=Path, default=Path("study"))
    pr = sub.add_parser("run")
    pr.add_argument("--out-dir", type=Path, default=Path("study"))
    pr.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    pr.add_argument("--num-shards", type=int, required=True)
    pr.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    pm = sub.add_parser("missing", help="print the shards not done yet, as a --array list (empty if none)")
    pm.add_argument("--out-dir", type=Path, default=Path("study"))
    pm.add_argument("--num-shards", type=int, required=True)
    ps = sub.add_parser("summary", help="headline numbers over the shards finished so far (any time)")
    ps.add_argument("--out-dir", type=Path, default=Path("study"))
    pt = sub.add_parser("file", help="print the stats of one file (debugging)")
    pt.add_argument("path")
    args = p.parse_args()

    if args.cmd == "list":
        list_files(args.dataset_dir, args.out_dir)
    elif args.cmd == "run":
        run_shard(args.out_dir, args.shard, args.num_shards, args.workers)
    elif args.cmd == "missing":
        print(",".join(str(i) for i in range(args.num_shards) if not shard_done(args.out_dir, i)))
    elif args.cmd == "summary":
        df = read_shards(args.out_dir)
        n = len(list((args.out_dir / "shards").glob("*")))
        print(f"{n} shard(s) in {args.out_dir / 'shards'}; by split: "
              + ", ".join(f"{k} {v:,}" for k, v in df["split"].value_counts().items()))
        print(summarize(df))
    else:
        for k, v in analyze_file(args.path).items():
            print(f"{k:>22}: {v}")


if __name__ == "__main__":
    sys.exit(main())
