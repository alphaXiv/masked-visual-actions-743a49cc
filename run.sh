#!/usr/bin/env bash
# Data-sim: robosuite URDF-rendered eval clips (Panda in-domain, UR5e/Sawyer held-out).
set -eo pipefail

echo "=== DATA-SIM START $(date -u +%FT%TZ) ==="
df -h /shared || true

export HF_HOME=/shared/hf
export PIP_CACHE_DIR=/shared/pip
export MUJOCO_GL=osmesa

t0=$(date +%s)
pip install -q "numpy==1.26.4" "mujoco==2.3.7" "robosuite==1.4.1" \
    "imageio[ffmpeg]" av "huggingface_hub[hf_transfer]" pillow scikit-image 2>&1 | tail -2
echo "TIMING install_s=$(( $(date +%s) - t0 ))"

t0=$(date +%s)
python scripts/data_sim.py
echo "TIMING data_sim_s=$(( $(date +%s) - t0 ))"
echo "=== DATA-SIM DONE $(date -u +%FT%TZ) ==="
