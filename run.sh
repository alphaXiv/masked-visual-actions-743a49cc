#!/usr/bin/env bash
# Root smoke: env install, weight downloads into shared cache, tiny inference timing.
set -eo pipefail

echo "=== SMOKE START $(date -u +%FT%TZ) ==="
nvidia-smi || true
df -h /hfcache || true

export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export DIFFSYNTH_MODEL_BASE_PATH=/hfcache/models
export HF_HOME=/hfcache/hf
export PIP_CACHE_DIR=/hfcache/pip

t0=$(date +%s)
git clone -q https://github.com/modelscope/DiffSynth-Studio.git /tmp/DiffSynth-Studio
cd /tmp/DiffSynth-Studio && git checkout -q 3743b1307caf2562af60d475b22d4b6be68e7cd0
pip install -q -e . 2>&1 | tail -2
pip install -q "huggingface_hub[hf_transfer]" imageio[ffmpeg] 2>&1 | tail -1
cd - >/dev/null
echo "TIMING install_s=$(( $(date +%s) - t0 ))"

t0=$(date +%s)
python inference/download_weights.py --out /hfcache/mva-loras
echo "TIMING lora_download_s=$(( $(date +%s) - t0 ))"
ls -la /hfcache/mva-loras

t0=$(date +%s)
python scripts/smoke.py
echo "TIMING smoke_total_s=$(( $(date +%s) - t0 ))"
echo "=== SMOKE DONE $(date -u +%FT%TZ) ==="
