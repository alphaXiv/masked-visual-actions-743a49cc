#!/usr/bin/env python
"""Generate videos for one model variant across eval clips/controls and score them.

Driven by eval_config.json on the experiment branch:
  {"name": "...", "variant": "mva"|"base", "subsets": [...],
   "controls": ["dense","ee","skel"], "steps": 40, "gpus": 4}

Modes:
  --plan                   download the eval set, print the task list size
  --worker I --workers N   generate+score shard I (pin one GPU via CUDA_VISIBLE_DEVICES)
  --aggregate              collect item JSONs, print aggregate table, upload to HF

Matched seeds: seed = crc32(clip_id) so every variant/control sees the same noise.
Outputs land in /shared/gen/<name>/ (resumable) and HF dataset gen/<name>/.
"""
import argparse
import glob
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mva_common import read_video, save_video, contact_sheet, ARTIFACT_REPO

CFG = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_config.json")))
GEN_ROOT = f"/shared/gen/{CFG['name']}"
LORA_DIR = "/shared/mva-loras"
HF_BASE = "alibaba-pai/Wan2.2-Fun-A14B-Control"


def eval_set_dir():
    from huggingface_hub import snapshot_download
    pats = [f"evalset/{s}/*" for s in CFG["subsets"]]
    d = snapshot_download(ARTIFACT_REPO, repo_type="dataset", allow_patterns=pats,
                          local_dir="/scratch/evalset_dl")
    return os.path.join(d, "evalset")


def task_list(root):
    tasks = []
    for s in CFG["subsets"]:
        for cd in sorted(glob.glob(f"{root}/{s}/clip*")):
            for ctl in CFG["controls"]:
                tasks.append((s, cd, ctl))
    return tasks


def load_pipeline():
    from inference.infer import build_pipeline
    pipe = build_pipeline(HF_BASE)
    if CFG["variant"] == "mva":
        pipe.load_lora(pipe.dit, f"{LORA_DIR}/masked_world_lora_high.safetensors", alpha=1.0)
        pipe.load_lora(pipe.dit2, f"{LORA_DIR}/masked_world_lora_low.safetensors", alpha=1.0)
    return pipe


def gen_one(pipe, clip_dir, ctl, out_dir):
    from PIL import Image
    from inference.infer import DEFAULT_NEGATIVE_PROMPT
    meta = json.load(open(f"{clip_dir}/meta.json"))
    h, w, nf = meta["h"], meta["w"], meta["n_frames"]
    control = [Image.fromarray(np.asarray(f)) for f in read_video(f"{clip_dir}/control_{ctl}.mp4")][:nf]
    ref = Image.open(f"{clip_dir}/ref.png").convert("RGB").resize((w, h))
    seed = zlib.crc32(f"{os.path.basename(os.path.dirname(clip_dir))}/{os.path.basename(clip_dir)}".encode()) % 2 ** 31
    video = pipe(
        prompt=meta["prompt"], negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        control_video=control, reference_image=ref,
        height=h, width=w, num_frames=nf,
        num_inference_steps=CFG.get("steps", 40), seed=seed, tiled=True,
    )
    frames = [np.asarray(f.convert("RGB")) for f in video]
    save_video(frames, f"{out_dir}/gen.mp4", meta["fps"])
    return frames, meta, seed


def worker(idx, nworkers):
    import time
    import torch
    root = eval_set_dir()
    tasks = task_list(root)
    mine = tasks[idx::nworkers]
    print(f"worker {idx}: {len(mine)}/{len(tasks)} tasks on {torch.cuda.get_device_name(0)}")
    import lpips
    lp = lpips.LPIPS(net="alex").to("cuda").eval()
    from eval_metrics import video_metrics
    pipe = None
    for s, cd, ctl in mine:
        clip = os.path.basename(cd)
        od = f"{GEN_ROOT}/{s}/{clip}/{ctl}"
        mfile = f"{od}/metrics.json"
        if os.path.exists(mfile):
            print(f"skip done {s}/{clip}/{ctl}")
            continue
        if pipe is None:
            t0 = time.time()
            pipe = load_pipeline()
            print(f"worker {idx}: pipeline loaded in {time.time()-t0:.0f}s")
        os.makedirs(od, exist_ok=True)
        t0 = time.time()
        try:
            frames, meta, seed = gen_one(pipe, cd, ctl, od)
        except Exception as e:
            import traceback
            traceback.print_exc()
            json.dump({"error": repr(e)}, open(f"{od}/error.json", "w"))
            continue
        gt = [np.asarray(f) for f in read_video(f"{cd}/gt.mp4")][:len(frames)]
        masks = np.load(f"{cd}/masks.npz")["mask"].astype(bool)
        m = video_metrics(gt, frames, masks, lpips_model=lp)
        m.update(subset=s, clip=clip, control=ctl, variant=CFG["variant"],
                 seed=seed, gen_s=round(time.time() - t0, 1), steps=CFG.get("steps", 40))
        json.dump(m, open(mfile, "w"), indent=1)
        print("ITEM " + json.dumps(m))
        try:
            ctl_frames = [np.asarray(x) for x in read_video(f"{cd}/control_{ctl}.mp4")]
            contact_sheet([("gt", gt), (f"ctl:{ctl}", ctl_frames),
                           (f"gen:{CFG['variant']}", frames)], f"{od}/sheet.jpg")
        except Exception:
            pass


def aggregate():
    rows = []
    for mf in glob.glob(f"{GEN_ROOT}/*/*/*/metrics.json"):
        rows.append(json.load(open(mf)))
    print(f"aggregating {len(rows)} items")
    keys = ["lpips", "ssim", "psnr", "psnr_robot", "ssim_robot", "ssim_bg", "psnr_bg",
            "epe", "epe_robot", "motion_ratio"]
    agg = {}
    for s in CFG["subsets"]:
        for ctl in CFG["controls"]:
            sel = [r for r in rows if r.get("subset") == s and r.get("control") == ctl and "lpips" in r]
            if not sel:
                continue
            e = {k: [round(float(np.mean([r[k] for r in sel if r.get(k) is not None])), 4),
                     round(float(np.std([r[k] for r in sel if r.get(k) is not None])), 4)] for k in keys}
            e["n"] = len(sel)
            for fl in ["flag_static", "flag_scene_transform", "flag_robot_halluc"]:
                e[fl] = int(sum(bool(r.get(fl)) for r in sel))
            agg[f"{s}/{ctl}"] = e
    print("AGG " + json.dumps({"name": CFG["name"], "variant": CFG["variant"], "agg": agg}))
    json.dump({"config": CFG, "items": rows, "agg": agg}, open(f"{GEN_ROOT}/results.json", "w"), indent=1)
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(folder_path=GEN_ROOT, path_in_repo=f"gen/{CFG['name']}",
                      repo_id=ARTIFACT_REPO, repo_type="dataset")
    print(f"uploaded gen/{CFG['name']}")
    n_err = len(glob.glob(f"{GEN_ROOT}/*/*/*/error.json"))
    print("SUMMARY " + json.dumps({"items": len(rows), "errors": n_err}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    if a.plan:
        root = eval_set_dir()
        print(f"PLAN tasks={len(task_list(root))} root={root}")
    elif a.aggregate:
        aggregate()
    else:
        worker(a.worker or 0, a.workers)
