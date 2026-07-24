#!/usr/bin/env bash
# Data-real: build DROID-100 + ALOHA eval clips with SAM2-derived controls.
set -eo pipefail

echo "=== DATA-REAL START $(date -u +%FT%TZ) ==="
nvidia-smi -L || true
df -h /shared || true

export HF_HOME=/shared/hf
export PIP_CACHE_DIR=/shared/pip
export SAM2_BUILD_CUDA=0

t0=$(date +%s)
pip install -q scikit-image opencv-python-headless pandas pyarrow \
    "imageio[ffmpeg]" av "huggingface_hub[hf_transfer]" pillow 2>&1 | tail -1
pip install -q "git+https://github.com/facebookresearch/sam2.git" 2>&1 | tail -1
echo "TIMING install_s=$(( $(date +%s) - t0 ))"

t0=$(date +%s)
python scripts/data_real.py
echo "TIMING data_real_s=$(( $(date +%s) - t0 ))"
echo "=== DATA-REAL DONE $(date -u +%FT%TZ) ==="
