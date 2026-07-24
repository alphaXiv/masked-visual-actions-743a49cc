#!/usr/bin/env bash
# Eval-node run.sh template: install env, fan gen_eval.py workers across GPUs,
# then aggregate. Eval nodes copy this over run.sh and commit eval_config.json.
set -eo pipefail

echo "=== EVAL START $(date -u +%FT%TZ) ==="
nvidia-smi -L || true
df -h /shared || true

export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export DIFFSYNTH_MODEL_BASE_PATH=/shared/models
export HF_HOME=/shared/hf
export PIP_CACHE_DIR=/shared/pip

t0=$(date +%s)
git clone -q https://github.com/modelscope/DiffSynth-Studio.git /tmp/DiffSynth-Studio
cd /tmp/DiffSynth-Studio && git checkout -q 3743b1307caf2562af60d475b22d4b6be68e7cd0
pip install -q -e . 2>&1 | tail -1
pip install -q "huggingface_hub[hf_transfer]" "imageio[ffmpeg]" av lpips scikit-image \
    opencv-python-headless 2>&1 | tail -1
cd - >/dev/null
echo "TIMING install_s=$(( $(date +%s) - t0 ))"

python inference/download_weights.py --out /shared/mva-loras >/dev/null
NGPU=$(python -c "import json;print(json.load(open('eval_config.json'))['gpus'])")
echo "fanning $NGPU workers"

pids=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$i python scripts/gen_eval.py --worker "$i" --workers "$NGPU" \
      2>&1 | sed "s/^/[w$i] /" &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "workers done rc=$rc"

python scripts/gen_eval.py --aggregate
echo "=== EVAL DONE $(date -u +%FT%TZ) ==="
