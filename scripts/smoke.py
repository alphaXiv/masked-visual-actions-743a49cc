#!/usr/bin/env python
"""Smoke test: load full pipeline + MVA LoRAs on one GPU, run a short inference
on a synthetic control video, print timing/VRAM evidence, verify HF write access."""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.infer import build_pipeline

HF_BASE = "alibaba-pai/Wan2.2-Fun-A14B-Control"
ARTIFACT_REPO = "alphaXiv/mva-repro-artifacts"
H, W, F = 480, 832, 81


def synthetic_control(h, w, f):
    """Gray canvas with a moving red 'gripper' blob — stand-in control video."""
    frames = []
    for i in range(f):
        img = Image.new("RGB", (w, h), (128, 128, 128))
        d = ImageDraw.Draw(img)
        x = int(w * 0.2 + (w * 0.5) * i / (f - 1))
        y = int(h * 0.6 - (h * 0.2) * i / (f - 1))
        d.rectangle([x - 25, y - 60, x + 25, y + 20], fill=(200, 200, 210))
        d.rectangle([x - 25, y + 20, x - 8, y + 55], fill=(255, 40, 40))
        d.rectangle([x + 8, y + 20, x + 25, y + 55], fill=(255, 40, 40))
        frames.append(img)
    return frames


def main():
    # HF write check first (cheap, fails fast)
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    print(f"HF whoami: {who['name']}")
    api.create_repo(ARTIFACT_REPO, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=b"smoke ok\n", path_in_repo="smoke.txt",
                    repo_id=ARTIFACT_REPO, repo_type="dataset")
    print(f"HF write ok: datasets/{ARTIFACT_REPO}")

    t0 = time.time()
    pipe = build_pipeline(HF_BASE)
    print(f"TIMING pipeline_load_s={time.time()-t0:.0f}")

    for name, expert in [("high", "dit"), ("low", "dit2")]:
        t0 = time.time()
        pipe.load_lora(getattr(pipe, expert), f"/hfcache/mva-loras/masked_world_lora_{name}.safetensors", alpha=1.0)
        print(f"TIMING lora_{name}_load_s={time.time()-t0:.0f}")

    control = synthetic_control(H, W, F)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    video = pipe(
        prompt="a robot arm moves over a table",
        negative_prompt="static, blurry",
        control_video=control,
        reference_image=control[0],
        height=H, width=W, num_frames=F,
        num_inference_steps=6, seed=0, tiled=True,
    )
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(json.dumps({
        "smoke": "ok", "frames": F, "h": H, "w": W, "steps": 6,
        "gen_s": round(dt, 1), "s_per_step_est": round(dt / 6, 1),
        "peak_vram_gib": round(peak, 1),
        "gpu": torch.cuda.get_device_name(0),
    }))

    from diffsynth.utils.data import save_video
    os.makedirs("/tmp/out", exist_ok=True)
    save_video(video, "/tmp/out/smoke.mp4", fps=15, quality=5)
    api.upload_file(path_or_fileobj="/tmp/out/smoke.mp4", path_in_repo="smoke/smoke.mp4",
                    repo_id=ARTIFACT_REPO, repo_type="dataset")
    print("uploaded smoke/smoke.mp4")


if __name__ == "__main__":
    main()
