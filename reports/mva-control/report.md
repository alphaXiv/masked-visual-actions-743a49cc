# Do masked visual actions make a video model controllable? Testing the released MVA checkpoint

**Verdict: reproduced** — for the three claims the released checkpoint can support.
[Masked Visual Actions](https://arxiv.org/abs/2607.19343) (MVA) proposes that the right way to
tell a video model what a robot will do is to *draw the action into pixels*: paste the robot's
trajectory, as masked robot pixels on a gray canvas, and let the model inpaint the world's
response. We tested the authors' released LoRA checkpoint against its own base model,
`Wan2.2-Fun-A14B-Control` (14B, two-expert MoE), on a rebuilt public evaluation set — identical
control videos, reference frames, prompts, and noise seeds, with and without the MVA LoRAs. The
LoRAs roughly **halve LPIPS on every subset**, and on our rebuilt DROID clips the absolute numbers
(LPIPS 0.096, SSIM 0.895, PSNR 24.3) land within noise of the paper's own DROID row (0.0945,
0.887, 23.74). All generation ran on a Kubernetes cluster with NVIDIA RTX PRO 6000 Blackwell GPUs
(peak 16 concurrent).

![MVA vs base, dense controls](images/fig1_headline.png)

**How to read this:** each bar pair is the same pretrained video model with (blue) and without
(orange) the released MVA LoRAs, generating 81-frame videos from the same dense masked-robot
control and the same seed. Left: LPIPS distance to the ground-truth video (lower = more faithful).
Right: PSNR restricted to the robot's pixels (higher = the generated arm is where the control said
it should be). Bars are means ± s.e.m. over clips; DROID and Sim Panda are the checkpoint's
training-domain embodiment (Franka), while ALOHA (bimanual), UR5e, and Sawyer were never seen by it.

## What the paper claims, and what we could test

The paper LoRA-finetunes a pretrained video model to accept a *dense masked visual action* — the
robot's pixels revealed on gray — and reports that this (1) yields high-fidelity, controllable
generation, beating action- and trajectory-conditioned baselines on DROID (LPIPS 0.0945 vs 0.362
for Ctrl-World); (2) beats *sparse* visual controls (end-effector or skeleton renderings),
especially out of distribution; and (3) generalizes gracefully to unseen embodiments, where
sparse-conditioned models hallucinate or collapse.

The release contains the two LoRAs (high/low-noise experts) and a thin inference script — not the
training data, rendering tools, evaluation clips, or the separately-trained sparse-control
checkpoints. We therefore tested three bounded claims: **(1)** LoRA vs base under matched
conditions, **(2)** control-density sensitivity of the released checkpoint (a proxy for the
paper's sparse ablation — see caveat), and **(3)** held-out-embodiment behavior with failures
quantified. Real-robot planning, policy evaluation, and inverse-modeling claims are untestable
from the release and were not attempted.

## Rebuilding the evaluation set from public data

The eval set (public, with all generated videos, at
[alphaXiv/mva-repro-artifacts](https://huggingface.co/datasets/alphaXiv/mva-repro-artifacts)) has
two tracks and three control types per clip:

- **Real videos.** 16 DROID clips ([lerobot/droid_100](https://huggingface.co/datasets/lerobot/droid_100),
  Franka, the training domain; 320×180 exterior camera upscaled to 832×480) and 8 ALOHA clips
  ([lerobot/aloha_static_coffee](https://huggingface.co/datasets/lerobot/aloha_static_coffee), a
  bimanual rig the checkpoint never saw). Robot masks: GroundingDINO boxes, hand-corrected, then
  SAM2 video propagation. A **preregistered sanity filter** (mask area 1–45%, ≤2 empty mask
  frames, fixed before any generation ran) excludes 4 DROID clips whose masks failed.
- **URDF-rendered sim.** 16 robosuite (MuJoCo) lift episodes with *exact* per-frame robot masks
  from segmentation renders: Panda (training-domain embodiment) plus UR5e and Sawyer (held-out).
- **Controls.** *dense* = all robot pixels on gray (the paper's control); *ee* = an end-effector
  disk only; *skel* = skeleton polyline + red end-effector dot (the paper's sparse formats).

Generation used the released two-expert LoRAs at strength 1.0, 30 steps (sim) / 40 steps (real)
instead of the default 50, single seed per clip matched across every condition, and a second seed
for the dense conditions as a robustness check.

## Claim 1 — the LoRAs, not the backbone, carry the controllability

With dense controls, MVA beat the base model on **every subset** (fig. 1, filtered clips):

| dense controls | LPIPS ↓ | | robot PSNR ↑ | | bg SSIM ↑ | |
|---|---|---|---|---|---|---|
| | **MVA** | base | **MVA** | base | **MVA** | base |
| DROID (n=12) | **0.096** | 0.206 | **27.7** | 19.9 | **0.895** | 0.804 |
| ALOHA (n=8, held-out) | **0.113** | 0.225 | **24.8** | 12.6 | **0.927** | 0.888 |
| Sim Panda (n=6) | **0.041** | 0.079 | **13.4** | 9.9 | **0.946** | 0.936 |
| Sim UR5e (n=5, held-out) | **0.033** | 0.049 | **17.6** | 13.4 | **0.949** | 0.945 |
| Sim Sawyer (n=5, held-out) | **0.027** | 0.045 | **18.5** | 12.6 | **0.948** | 0.945 |

Our DROID row is strikingly close to the paper's (LPIPS 0.0945 / SSIM 0.887 / PSNR 23.74; ours
0.096 / 0.895 / 24.3) even though the clips, masks, and preprocessing are all rebuilt — and the
base model's 0.206 is the same order as the paper's Ctrl-World baseline (0.362). The qualitative
failure mode is the one the paper describes: the base model preserves *a* scene but drifts —
below, it invents a camera move and a doorway while MVA holds the scene and executes the motion.

![DROID qualitative: MVA vs base](images/fig2_qualitative_droid.jpg)

## Claim 2 — dense beats sparse, mostly by refusing to move

![Control density](images/fig3_controls.png)

Feeding the same checkpoint sparser controls degrades every metric monotonically: LPIPS roughly
doubles from dense→skeleton and robot-region PSNR falls 3–12 dB. The failure taxonomy (fig. 5)
shows *how*: with sparse controls the model often emits a **static video** (motion <25% of ground
truth; 0/36 dense vs 9/36 skeleton for MVA) or an arm that isn't where the control said.
**Caveat:** the paper trained a separate checkpoint per conditioning type and released only the
dense one, so this measures the released checkpoint's sensitivity to control density, not the
paper's exact ablation table. It still confirms the mechanism the paper credits — dense visual
actions carry appearance and geometry that sparse cues cannot.

## Claim 3 — held-out embodiments: the dense advantage survives, sparse hallucinates

On held-out embodiments the MVA-over-base gap persists (ALOHA: −0.112 LPIPS, +12.2 dB robot PSNR)
and the dense-over-sparse gap **widens**: dense→skeleton costs +34% LPIPS on in-domain DROID but
+144% on held-out ALOHA. The paper's signature qualitative failure reproduces exactly (fig. 4):
with a skeleton control on the held-out UR5e, the model invents a *white Franka-like arm* — its
training embodiment — while the dense control, which carries the UR5e's appearance, yields a
faithful UR5e. The base model hallucinates the robot on 8/8 held-out ALOHA clips even with dense
controls; MVA does so on 1/8.

![Embodiment hallucination](images/fig4_embodiment.jpg)

![Failure flags](images/fig5_flags.png)

**Robustness.** The result is not a seed or sampler artifact: with two additional noise seeds
per DROID clip (32 generations per variant) the unfiltered dense means are LPIPS 0.110 (MVA) vs
0.228 (base) with zero MVA failure flags, and rerunning at the paper-default 50 denoising steps
moves DROID dense LPIPS by under 5% for either variant (MVA 0.107→0.106, base 0.206→0.197). The
MVA-vs-base ordering never flips in any condition we ran.

## What diverged or needs care

- **Robot-region PSNR is harsh** when the arm covers ~4% of frame: the preregistered
  "hallucination" flag (robot PSNR <14 dB) fires even on visually decent sim generations (4/6
  MVA Panda dense). Flags are meaningful as *relative* rates between matched conditions, not as
  absolute failure probabilities.
- **Sim in-domain scored below sim held-out** (Panda LPIPS 0.041 vs Sawyer 0.027) — a data
  artifact: the Panda camera framing leaves the arm out of frame early in each clip, and the
  scripted Sawyer grasp fails (less motion to model). Within-subset comparisons are unaffected.
- **The base model is not a paper baseline.** The paper compares against Ctrl-World, Wan-Move and
  I2V; the no-LoRA base is instead the natural ablation for "what do the released LoRAs add." It
  was never trained to read masked-robot controls — which is the point being measured.
- Absolute real-track numbers benefit from upscaled 180p source footage (soft textures are easier
  to reconstruct); this inflates fidelity equally for both variants.

## Compute

Everything ran on the operator's **Kubernetes** cluster (CoreWeave): 2 nodes × 8 **NVIDIA RTX PRO
6000 Blackwell** (96 GB). Peak concurrency **16 GPUs** (two 8-GPU jobs); **336 videos** of 81
frames at 480p across four eval conditions plus a second-seed robustness round; total elapsed wall
time **≈5 hours** from environment probe to final analysis. The full two-expert pipeline fits
one 96 GB GPU (peak 69.1 GiB) at ~13.7 s/step for 81×480×832.

## Experiment lineage

See the [README experiment log](../../README.md#experiment-log) for branch-by-branch provenance
with exact run commands, and the
[artifact dataset](https://huggingface.co/datasets/alphaXiv/mva-repro-artifacts) for every
generated video, control, mask, and per-clip metric file.
