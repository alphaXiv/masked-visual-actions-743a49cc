# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "matplotlib",
#     "requests",
# ]
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
# Masked Visual Actions, reproduced from the released checkpoint

**Paper:** [Masked Visual Actions for Unified World Modeling](https://arxiv.org/abs/2607.19343)
(arXiv 2607.19343) proposes controlling a pretrained video model with **masked visual actions**:
instead of conditioning on joint angles or sparse keypoints, the robot's future trajectory is
*drawn into pixel space* (the robot's pixels on a gray canvas) and the model inpaints the rest of
the world around it.

**What this notebook shows.** We took the authors' released LoRA checkpoint (built on
`Wan2.2-Fun-A14B-Control`, a 14B two-expert video model) and asked three bounded questions:

1. Do the MVA LoRAs beat the *same* base model without them, on identical controls and seeds?
2. How does control density (dense mask vs end-effector-only vs skeleton) matter?
3. Does the advantage survive on robot embodiments the checkpoint never saw?

Everything below loads **precomputed results** from the public artifact dataset
[`alphaXiv/mva-repro-artifacts`](https://huggingface.co/datasets/alphaXiv/mva-repro-artifacts) —
no GPU needed. The generation itself ran on a Kubernetes cluster with NVIDIA RTX PRO 6000
Blackwell GPUs (up to 16 concurrent).
"""
    )
    return


@app.cell
def _():
    import json
    import urllib.request

    import pandas as pd

    HF = "https://huggingface.co/datasets/alphaXiv/mva-repro-artifacts/resolve/main"
    CONDS = ["real_mva", "real_base", "sim_mva", "sim_base"]

    def fetch_json(url):
        with urllib.request.urlopen(url) as r:
            return json.load(r)

    frames = []
    for cond_name in CONDS:
        try:
            res = fetch_json(f"{HF}/gen/{cond_name}/results.json")
            df_c = pd.DataFrame(res["items"])
            df_c["cond"] = cond_name
            frames.append(df_c)
        except Exception as exc:
            print(f"{cond_name}: {exc}")
    items = pd.concat(frames, ignore_index=True)
    items["track"] = items.subset.str.startswith("sim").map({True: "sim", False: "real"})
    items.head(3)
    return HF, fetch_json, items, pd


@app.cell
def _(mo):
    mo.md(
        r"""
## The evaluation set (rebuilt from public data)

The authors did not release their evaluation clips or rendering tools, so we rebuilt a matched
evaluation set from public sources:

| subset | source | embodiment | role |
|---|---|---|---|
| `droid` | [lerobot/droid_100](https://huggingface.co/datasets/lerobot/droid_100) | Franka | training domain |
| `aloha` | [lerobot/aloha_static_coffee](https://huggingface.co/datasets/lerobot/aloha_static_coffee) | bimanual ALOHA | **held-out embodiment** |
| `sim_panda` | robosuite render (URDF) | Panda | training-domain embodiment |
| `sim_ur5e`, `sim_sawyer` | robosuite render (URDF) | UR5e / Sawyer | **held-out embodiments** |

Real-video robot masks come from GroundingDINO + SAM2 (hand-corrected box prompts); sim masks are
exact segmentation renders. Three control types per clip: **dense** (all robot pixels on gray),
**ee** (end-effector disk only), **skel** (skeleton polyline + end-effector dot).

A preregistered sanity filter (mask area in [1%, 45%], ≤2 empty mask frames) excludes clips whose
automatic masks failed; the filter was fixed before any generation ran.
"""
    )
    return


@app.cell
def _(fetch_json, HF, items, pd):
    # clip sanity filter from the eval-set metadata (preregistered)
    metas = []
    for (subset_name, clip_name), _ in items.groupby(["subset", "clip"]):
        try:
            m = fetch_json(f"{HF}/evalset/{subset_name}/{clip_name}/meta.json")
            area = m.get("mask_area_mean", m.get("mask_area_frac"))
            metas.append(dict(subset=subset_name, clip=clip_name, mask_area=area,
                              empty=m.get("empty_frames", 0)))
        except Exception:
            metas.append(dict(subset=subset_name, clip=clip_name, mask_area=None, empty=None))
    meta = pd.DataFrame(metas)
    meta["clip_ok"] = meta.mask_area.between(0.01, 0.45) & (meta.empty <= 2)
    ok = items.merge(meta[["subset", "clip", "clip_ok"]], on=["subset", "clip"])
    ok = ok[ok.clip_ok]
    f"{len(ok)}/{len(items)} generations pass the preregistered clip filter"
    return meta, ok


@app.cell
def _(mo):
    mo.md(r"""## Headline: MVA LoRA vs. matched base model (dense controls, matched seeds)""")
    return


@app.cell
def _(ok):
    import matplotlib.pyplot as plt
    import numpy as np

    C_MVA, C_BASE = "#2a78d6", "#eb6834"
    d = ok[ok.control == "dense"]
    subsets = [s for s in ["droid", "aloha", "sim_panda", "sim_ur5e", "sim_sawyer"]
               if s in set(d.subset)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, metric, title in [(axes[0], "lpips", "LPIPS ↓ (reconstruction)"),
                              (axes[1], "psnr_robot", "robot-region PSNR ↑ (adherence)")]:
        x = np.arange(len(subsets))
        for off, (var, col) in enumerate([("mva", C_MVA), ("base", C_BASE)]):
            sel = d[d.variant == var].groupby("subset")[metric]
            ax.bar(x + (off - .5) * .36, [sel.mean().get(s, np.nan) for s in subsets],
                   yerr=[sel.sem().get(s, np.nan) for s in subsets],
                   width=.34, color=col, label={"mva": "MVA LoRA", "base": "base"}[var],
                   error_kw=dict(lw=1, capsize=2))
        ax.set_xticks(x, subsets, rotation=15)
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig
    return C_BASE, C_MVA, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
**What fig. 1 shows.** With dense masked-robot controls and matched seeds, the MVA LoRAs beat the
matched base model on every subset. On DROID the absolute numbers (LPIPS 0.096 / SSIM 0.895 /
PSNR 24.3) land near the paper's Table 1 (0.0945 / 0.887 / 23.74) even though the clips differ —
read the *paired* gaps, not the absolutes: robot-region PSNR improves by +7.8 dB (DROID),
+12.2 dB (ALOHA, held-out), +3.5 dB (Panda), +5.1 dB (held-out sim arms).
"""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Claim 2 — control density (dense vs end-effector vs skeleton)""")
    return


@app.cell
def _(np, ok, plt, C_BASE, C_MVA):
    d2 = ok[ok.variant == "mva"].copy()
    d2["domain"] = np.where(d2.subset.isin(["aloha", "sim_ur5e", "sim_sawyer"]),
                            "held-out embodiment", "in-domain")
    ctls = ["dense", "ee", "skel"]
    fig2, axes2 = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax2, (metric2, lab2) in zip(axes2, [("lpips", "LPIPS ↓"), ("psnr_robot", "robot PSNR ↑"),
                                            ("motion_ratio", "motion vs GT (1 = matched)")]):
        x2 = np.arange(len(ctls))
        for off2, (dom2, col2) in enumerate([("in-domain", C_MVA), ("held-out embodiment", C_BASE)]):
            g2 = d2[d2.domain == dom2].groupby("control")[metric2]
            ax2.bar(x2 + (off2 - .5) * .36, [g2.mean().get(c, np.nan) for c in ctls],
                    yerr=[g2.sem().get(c, np.nan) for c in ctls], width=.34, color=col2,
                    label=dom2, error_kw=dict(lw=1, capsize=2))
        ax2.set_xticks(x2, ctls)
        ax2.set_title(lab2)
    axes2[0].legend(frameon=False, fontsize=8)
    fig2.suptitle("Sparser controls degrade fidelity and adherence — and freeze motion", y=1.04)
    fig2.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md(
        r"""
The rightmost panel is the diagnostic: with sparse controls the model often simply *does not
move* (motion ratio collapses; static-video flags fire on 9/36 skeleton generations vs 1/36
dense). Caveat: the paper trained a separate checkpoint per control type (not released) — this
probes the released dense-trained checkpoint's sensitivity to control density.

## Claim 3 — held-out embodiments and the hallucination failure mode

The advantage *grows* out of domain (paired robot-PSNR gap: +7.8 dB in-domain DROID →
+12.2 dB held-out ALOHA). And the paper's signature failure reproduces: below, the same
checkpoint, clip, and seed on the held-out UR5e — the dense control yields a faithful UR5e,
the skeleton control yields a hallucinated white training-prior arm that then vanishes.
"""
    )
    return


@app.cell
def _(HF, mo):
    mo.vstack([
        mo.md("**Dense control (MVA):** faithful UR5e"),
        mo.image(f"{HF}/gen/sim_mva/sim_ur5e/clip000/dense/sheet.jpg"),
        mo.md("**Skeleton control (MVA):** hallucinated white arm"),
        mo.image(f"{HF}/gen/sim_mva/sim_ur5e/clip000/skel/sheet.jpg"),
        mo.md("**Held-out bimanual ALOHA, dense control — MVA vs base** (rows: GT / control / generation)"),
        mo.image(f"{HF}/gen/real_mva/aloha/clip002/dense/sheet.jpg"),
        mo.image(f"{HF}/gen/real_base/aloha/clip002/dense/sheet.jpg"),
    ])
    return


@app.cell
def _(ok, pd):
    flags = (ok.groupby(["variant", "control"])
             [["flag_static", "flag_scene_transform", "flag_robot_halluc"]].sum())
    flags["n"] = ok.groupby(["variant", "control"]).size()
    flags
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## Verdict and caveats

**Reproduced** (released-checkpoint scope): the MVA LoRAs make the base video model markedly
more faithful and more controllable under dense masked-robot controls; dense beats sparse; the
advantage persists — indeed grows — on held-out embodiments, and failures concentrate exactly
where the paper says (sparse controls, base model, out-of-domain).

Caveats: rebuilt 40-clip public eval set, 36 clips after the preregistered mask filter (not the authors' clips); DROID sources are 180p upscaled;
sparse-control comparison probes the dense-trained checkpoint (the paper's sparse checkpoints
were never released); 30–40 denoising steps for main runs (50-step robustness check unchanged);
robot-region PSNR is harsh in absolute terms — compare across conditions, not against 100%.

Untested (needs unreleased tooling / robot): planning, policy evaluation, inverse-dynamics
action extraction, and the finetuning itself.

**Compute:** Kubernetes, 2×8 NVIDIA RTX PRO 6000 Blackwell (96 GB), peak 16 concurrent GPUs;
~300 videos at 81 frames / 480p; the two-expert 14B pipeline fits one GPU (69 GiB peak).
"""
    )
    return


if __name__ == "__main__":
    app.run()
