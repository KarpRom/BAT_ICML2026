"""Pre-download the BAT ViT-Base checkpoint (BAT_base.pt, pretrained on AudioSet-2M).

Jean Zay compute nodes have no internet access, but continued pretraining
(`--init-checkpoint`, see engine/ssl_trainer.py) needs this file already sitting on
disk before a SLURM job can use it -- run this once on the front/login node (which
does have internet).

Usage:

    python scripts/download_bat_checkpoint.py

The checkpoint is Google Drive-hosted (not on the Hugging Face Hub like most other
checkpoints in this project), and large files there trigger Google's "can't scan for
viruses" interstitial that a plain HTTP GET can't get past -- this uses `gdown`
instead, which handles that. Install it once: `pip install gdown`.
"""

from pathlib import Path

import gdown

# BAT ViT-Base / AS2M, per the README's "Pretrained Weights" section.
CHECKPOINT_FILE_ID = "1fJ3FA4HC9fICQkpkk5AmZOd4O2sseg30"
CACHE_DIR = Path.home() / ".cache" / "bat_ssl"
CACHE_FILENAME = "BAT_base.pt"


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / CACHE_FILENAME

    if dest.exists():
        print(f"Already downloaded -> {dest} ({dest.stat().st_size:,} bytes)")
        return 0

    print(f"Downloading BAT_base.pt -> {dest} ...")
    gdown.download(id=CHECKPOINT_FILE_ID, output=str(dest))

    if not dest.exists() or dest.stat().st_size < 1_000_000:
        # Google's virus-scan interstitial silently returns a small HTML error page
        # instead of the real file for some large-file edge cases; catch that here
        # rather than leaving a corrupt/tiny "checkpoint" for a SLURM job to trip over.
        print(f"  FAILED: downloaded file is missing or suspiciously small "
              f"({dest.stat().st_size if dest.exists() else 0} bytes) -- check {dest} by hand.")
        return 1

    print(f"  OK -> {dest} ({dest.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
