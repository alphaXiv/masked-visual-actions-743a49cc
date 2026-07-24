# Reproduction: Masked Visual Actions for Unified World Modeling (arXiv 2607.19343)

> This public fork reproduces the *released-checkpoint* claims of the paper on rebuilt public
> data. Detailed write-up: **[reports/mva-control/report.md](reports/mva-control/report.md)** ·
> Interactive notebook: **[notebooks/mva_reproduction.py](notebooks/mva_reproduction.py)**
> [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/masked-visual-actions-743a49cc/blob/main/notebooks/mva_reproduction.py)

**What was tested.** Whether the released MVA LoRAs (on `Wan2.2-Fun-A14B-Control`, 14B two-expert)
make the video model (1) more faithful and controllable than the matched base model on dense
masked-robot controls, (2) better with dense than sparse (end-effector / skeleton) visual controls,
and (3) robust on held-out robot embodiments, with failures quantified.

**Assessment: reproduced** (within the released-checkpoint scope). With dense masked-robot
controls and matched seeds, the MVA LoRAs roughly halve LPIPS on every subset (DROID 0.096 vs
0.206 for the base; ALOHA 0.113 vs 0.225) and add +3.5 to +12.2 dB robot-region PSNR. Our rebuilt
DROID row (LPIPS 0.096 / SSIM 0.895 / PSNR 24.3) lands within noise of the paper's Table 1 entry
(0.0945 / 0.887 / 23.74). Sparse end-effector/skeleton controls degrade the released checkpoint
monotonically and reproduce the paper's embodiment-hallucination failure on held-out robots
(skeleton control on UR5e yields a hallucinated Franka-like arm). Results are stable across extra
seeds and 50-step sampling.

**Substitutions** (authors' eval clips/rendering tools are unreleased): evaluation set rebuilt from
public data — lerobot `droid_100` (Franka, training domain), lerobot `aloha_static_coffee`
(held-out bimanual), and robosuite URDF renders (Panda in-domain; UR5e/Sawyer held-out) with exact
sim masks; SAM2 + hand-corrected boxes for real videos; 30–40 denoising steps instead of 50;
single matched seed per clip; the paper's separately-trained sparse checkpoints are unreleased, so
the sparse ablation feeds sparser controls to the dense-trained checkpoint.

**Compute.** Operator's Kubernetes cluster, 2×8 NVIDIA RTX PRO 6000 Blackwell (96 GB); peak 16
concurrent GPUs; 336 (main + robustness) 81-frame 480p videos; ~5 h wall time. Artifacts (eval set, generated
videos, per-clip metrics) are public at
[alphaXiv/mva-repro-artifacts](https://huggingface.co/datasets/alphaXiv/mva-repro-artifacts).

## Experiment log

Run command for every node is `bash run.sh` (the branch's committed `run.sh` + `.orx/k8s.yaml`
define behavior; jobs execute on Kubernetes via `orx exp run <exp> --backend k8s`).

| branch | purpose | exact run command | outcome | compute |
|---|---|---|---|---|
| [`orx/baseline-env-smoke-…`](https://github.com/alphaXiv/masked-visual-actions-743a49cc/tree/orx/baseline-env-smoke-weights-download-tiny-inferen) | env probe: pinned DiffSynth install, weight download, tiny inference | `bash run.sh` | pipeline fits one 96 GB GPU (69.1 GiB), 13.7 s/step | 1 GPU, ~3 min |
| [`orx/data-real-v3-…`](https://github.com/alphaXiv/masked-visual-actions-743a49cc/tree/orx/data-real-v3-human-annotated-sam2-box-prompts) | real eval set v3: DROID+ALOHA clips, SAM2 masks (hand boxes), dense/ee/skel controls | `bash run.sh` | 16+8 clips uploaded (v1/v2 mask failures fixed) | 1 GPU, ~8 min |
| [`orx/data-urdf-rendered-sim-…`](https://github.com/alphaXiv/masked-visual-actions-743a49cc/tree/orx/data-urdf-rendered-sim-eval-set-robosuite-panda) | sim eval set: robosuite URDF renders, exact masks, 3 robots | `bash run.sh` | 16 clips (Panda/UR5e/Sawyer) | CPU, ~4 min |
| [`orx/eval-sim-mva-…`](https://github.com/alphaXiv/masked-visual-actions-743a49cc/tree/orx/eval-sim-mva-loras-on-urdf-rendered-controls-den) | sim eval, MVA LoRAs, 48 gens | `bash run.sh` | 48/48 ok — dense ≫ sparse | 8 GPU, 49 min |
| [`orx/eval-sim-base-…`](https://github.com/alphaXiv/masked-visual-actions-743a49cc/tree/orx/eval-sim-base-wan2-2-fun-control-no-lora-on-same) | sim eval, base (no LoRA) control arm | `bash run.sh` | 48/48 ok — MVA wins every subset | 8 GPU, 57 min |
| [`orx/eval-real-mva-…`](https://github.com/alphaXiv/masked-visual-actions-743a49cc/tree/orx/eval-real-mva-loras-dense-ee-skeleton-controls-d) | real eval (DROID+ALOHA), MVA LoRAs, 72 gens | `bash run.sh` | 72/72 ok — DROID LPIPS matches paper table | 8 GPU, 1h25m |
| [`orx/eval-real-base-…`](https://github.com/alphaXiv/masked-visual-actions-743a49cc/tree/orx/eval-real-base-wan2-2-fun-control-no-lora-matche) | real eval, base control arm | `bash run.sh` | 72/72 ok | 8 GPU, 1h39m |

`main` — not run as an experiment (publication surface). Superseded iterations kept for lineage:
`orx/data-real-robot-eval-set-…` (v1: episode-picker bug), `orx/data-real-v2-…` (v2: motion-guided
masks, still leaked), plus first-attempt eval branches cancelled for infra fixes.

---

(original upstream README follows)

# Reproduction: Masked Visual Actions for Unified World Modeling (arXiv 2607.19343)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/masked-visual-actions-743a49cc/blob/main/notebooks/mva_reproduction.py)

**Verdict: reproduced** (released-checkpoint scope). We tested whether the released MVA LoRAs
make `Wan2.2-Fun-A14B-Control` more controllable and embodiment-tolerant than the same base
model without them, on a rebuilt public evaluation set (real DROID + held-out bimanual ALOHA +
URDF-rendered robosuite clips with in-domain and held-out arms), with matched controls, prompts,
and seeds.

- **Claim 1 (fidelity/adherence).** Reproduced. Dense-control robot-region PSNR: **+7.8 dB**
  (DROID), **+12.2 dB** (ALOHA) over base; MVA wins paired LPIPS on 34/36 clips. Our DROID
  absolutes (LPIPS 0.096 / SSIM 0.895 / PSNR 24.3) land near the paper's Table 1
  (0.0945 / 0.887 / 23.74) despite a different, rebuilt eval set.
- **Claim 2 (dense > sparse controls).** Reproduced as a proxy (the paper's sparse-trained
  checkpoints were never released; we feed sparser controls to the released dense-trained
  checkpoint). Sparse controls degrade every metric and freeze motion (static-video flags
  1/36 dense → 9/36 skeleton).
- **Claim 3 (held-out embodiment).** Reproduced. The dense-control advantage *grows* out of
  domain, and the paper's signature failure appears on cue: skeleton control on a held-out UR5e
  hallucinates a white training-prior arm.

**Substitutions / downscaling:** authors' eval clips and rendering tools are unreleased →
public data (lerobot droid_100, aloha_static_coffee; robosuite renders); DROID sources are 180p
upscaled; 30–40 denoising steps in main runs (50-step robustness check included); SAM2 masks
from hand-corrected box prompts. Untested: planning, policy evaluation, inverse modeling,
finetuning.

**Compute:** Kubernetes (CoreWeave), 2×8 NVIDIA RTX PRO 6000 Blackwell 96 GB, peak **16
concurrent GPUs**, 416 generated videos (81f/480p), 4.7 h wall time.

📄 **[Detailed report](reports/mva-reproduction/report.md)** · 📓
**[Tutorial notebook](notebooks/mva_reproduction.py)** (`marimo edit notebooks/mva_reproduction.py`)
· 📦 **[Artifacts & eval set](https://huggingface.co/datasets/alphaXiv/mva-repro-artifacts)**

## Experiment log

Every node runs the same fixed command, `bash run.sh` (launched via `orx exp run <node> --backend k8s`);
the branch's committed `run.sh` + `.orx/k8s.yaml` define the work and resources.

| branch | purpose | run command | outcome | compute |
|---|---|---|---|---|
| [baseline smoke](../../tree/orx/baseline-env-smoke-weights-download-tiny-inferen) | env probe: pinned DiffSynth on Blackwell, weight cache, timing | `bash run.sh` | ok: 69.1 GiB peak, 13.7 s/step (after 4 infra failures: ramdisk hostPath, HF redirect 404) | 1 GPU, ~3 min |
| [data-real v3](../../tree/orx/data-real-v3-human-annotated-sam2-box-prompts) | 16 DROID + 8 ALOHA clips, SAM2 masks from hand-annotated boxes, dense/ee/skel controls (v1: episode-picker bug; v2: motion-guided prompts still failed) | `bash run.sh` | 24 clips uploaded; preregistered filter keeps 20 | 1 GPU, ~8 min |
| [data-sim](../../tree/orx/data-urdf-rendered-sim-eval-set-robosuite-panda) | 16 robosuite URDF-rendered episodes (Panda/UR5e/Sawyer), exact masks | `bash run.sh` | 16 clips uploaded | CPU (osmesa), ~4 min |
| [eval real MVA](../../tree/orx/eval-real-mva-loras-dense-ee-skeleton-controls-d) | 72 gens: MVA, DROID+ALOHA × dense/ee/skel, 40 steps | `bash run.sh` | 72/72 ok → `gen/real_mva` | 8 GPUs, ~75 min |
| [eval real base](../../tree/orx/eval-real-base-wan2-2-fun-control-no-lora-matche) | 72 gens: base, matched seeds | `bash run.sh` | 72/72 ok → `gen/real_base` | 8 GPUs, ~75 min |
| [eval sim MVA](../../tree/orx/eval-sim-mva-loras-on-urdf-rendered-controls-den) | 48 gens: MVA, 3 sim robots × 3 controls, 30 steps | `bash run.sh` | 48/48 ok → `gen/sim_mva` | 8 GPUs, ~45 min |
| [eval sim base](../../tree/orx/eval-sim-base-wan2-2-fun-control-no-lora-on-same) | 48 gens: base, matched seeds | `bash run.sh` | 48/48 ok → `gen/sim_base` | 8 GPUs, ~45 min |
| [50-step MVA](../../tree/orx/eval-real-50-steps-mva-dense-droid) / [base](../../tree/orx/eval-real-50-steps-base-dense-droid) | robustness: DROID dense at library-default 50 steps | `bash run.sh` | gap unchanged (LPIPS 0.106 vs 0.197; robot PSNR +7.7 dB) | 4+4 GPUs |
| [extra-seed MVA](../../tree/orx/eval-real-seeds-mva-dense-droid-2-extra-seeds) / [base](../../tree/orx/eval-real-seeds-base-dense-droid-2-extra-seeds) | robustness: DROID dense, 2 extra seeds per clip | `bash run.sh` | gaps +8.3 / +7.5 dB — stable across seeds | 4+4 GPUs |
| [second-seed all subsets, MVA](../../tree/orx/robustness-second-seed-dense-controls-all-subset) / [base](../../tree/orx/robustness-second-seed-dense-controls-all-subset-2) | robustness: dense controls, every subset, new seed (80 gens) | `bash run.sh` | every ordering replicated (ALOHA 0.094 vs 0.217 LPIPS) | 8+8 GPUs |

`main` was not run as an experiment (publication surface). Earlier iterations (data-real v1/v2,
first sim-eval attempts) are preserved on their `orx/*` branches; run ids and notes live in the
experiment tree (`orx exp desc`).

---

# masked-visual-actions

Code for finetuning and running our robot-video **control model**: a LoRA on top
of [`PAI/Wan2.2-Fun-A14B-Control`](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control).
Given a **control video** (a rendered URDF robot), a **reference image** (the
first real frame), and a **text prompt**, it generates the corresponding RGB video.

We did not modify the video model or its trainer — we used
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) to train a LoRA
on our data and to run inference. This repo is the thin layer on top: the
inference script, the training recipe, and our weights.

```
inference/   infer.py, download_weights.py   — run the model with our LoRAs
training/    train_control.sh                — finetune a control LoRA on your CSV
```

Weights are on the Hugging Face Hub at
[`HadiZayer/masked-visual-actions`](https://huggingface.co/HadiZayer/masked-visual-actions).

## Setup

Install DiffSynth-Studio at the pinned commit, then this repo's light deps:

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
git checkout 3743b1307caf2562af60d475b22d4b6be68e7cd0
pip install -e .
pip install huggingface_hub
```

A CUDA GPU is required (the base model is 14B; `infer.py --low-vram` offloads to
disk if you are memory constrained).

## Inference

```bash
python inference/download_weights.py --out ./checkpoints

python inference/infer.py \
    --lora-high checkpoints/masked_world_lora_high.safetensors \
    --lora-low  checkpoints/masked_world_lora_low.safetensors \
    --control-video robot_render.mp4 \
    --reference-image first_frame.png \
    --prompt "a robot arm picks up a mug" \
    --output out.mp4
```

`Wan2.2-Fun-A14B-Control` is a two-expert MoE (a **high-noise** and a **low-noise**
DiT, split at timestep boundary 0.358), so there are two LoRAs — one loaded into
`pipe.dit`, one into `pipe.dit2`. `--reference-image` is optional (defaults to
frame 0 of the control video). See `infer.py --help` for resolution/seed/steps.

## Training

Provide a CSV with columns `prompt, reference_image, video, control_video` and run
the two-expert recipe (stock DiffSynth params, run from the DiffSynth-Studio root):

```bash
cd DiffSynth-Studio
DATASET_CSV=/path/to/train.csv OUTPUT_DIR=/path/to/out \
    bash /path/to/masked-visual-actions/training/train_control.sh
```

This writes `<OUTPUT_DIR>_high_noise/` and `<OUTPUT_DIR>_low_noise/`; point
`infer.py` at the `step-*.safetensors` checkpoints you want.

## Rendering control videos

Tools for rendering the URDF robot control videos from DROID episodes are coming
soon.

## License

Apache-2.0 (inherited from DiffSynth-Studio). See `LICENSE`.
