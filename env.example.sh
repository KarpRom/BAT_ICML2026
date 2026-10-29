# Placeholder for env.sh: the environment every SLURM job sources (settings.env_script in
# experiments/runs.yaml). Copy it to env.sh next to it and adapt it; env.sh is not versioned
# (.gitignore), so each machine keeps its own.
#
#   cp env.example.sh env.sh
#
# It is sourced (not executed) by the slurm scripts, by GPU jobs (H100 nodes) and by CPU jobs
# (study / report). It needs a Python with torch, torchaudio, librosa, soundfile, scikit-learn,
# pandas, pyyaml, tqdm (+ pyarrow for the study).

# Machine-specific paths used by experiments/runs.yaml ($AUDIOSET). scripts/xp.py sources this file
# too (in a subshell) before expanding the manifest, so set them here and nowhere else.
export AUDIOSET=/path/to/audioset_root   # *_segments/ (audio), *_segments.csv, ontology.json, strong/

module purge
module load arch/h100                # H100 nodes: their own module tree (else torch fails on libmpi)
module load pytorch-gpu/py3/2.8.0
# e.g. extra packages on top of the module (librosa is not in it):
#   export PYTHONUSERBASE=$WORK/.local_pytorch_2.8.0
#   (once: pip install --user --no-cache-dir librosa soxr)
