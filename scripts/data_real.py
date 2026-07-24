#!/usr/bin/env python
"""Build the real-video eval set (v2): DROID-100 (Franka, training domain) and
ALOHA static-coffee (bimanual, held-out embodiment).

Robot masks are motion-guided: accumulate optical flow, pick the max-motion
anchor frame, sample point prompts from the moving region, and propagate with
SAM2 in both directions. This avoids text-grounding failures (v1 latched onto
a screwdriver / camera mount instead of the robot)."""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mva_common import ffmpeg_extract, save_clip, upload_dir

N_FRAMES = 81
DROID_N, ALOHA_N = 16, 8
OUT = "/tmp/evalset"


def motion_prompts(frames, n_obj):
    """Point prompts per object from the moving region at the max-motion frame."""
    gs = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    step = 2
    mags = []
    for i in range(step, len(gs), step):
        fl = cv2.calcOpticalFlowFarneback(gs[i - step], gs[i], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        mags.append((i - step // 2, np.linalg.norm(fl, axis=2)))
    means = [m.mean() for _, m in mags]
    ai = int(np.argmax(means))
    anchor, amag = mags[ai]
    hot = (amag >= max(np.percentile(amag, 99.0), 0.5)).astype(np.uint8)
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    hot = cv2.morphologyEx(hot, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(hot)
    comps = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    comps = [c for c in comps if stats[c, cv2.CC_STAT_AREA] > 200]
    if not comps:
        return anchor, []
    W = frames[0].shape[1]
    if n_obj == 2:
        left = [c for c in comps if cents[c][0] < W / 2][:1]
        right = [c for c in comps if cents[c][0] >= W / 2][:1]
        groups = [g for g in (left, right) if g]
    else:
        groups = [comps[:2]]
    out = []
    for g in groups:
        m = np.isin(lab, g)
        ys, xs = np.nonzero(m)
        pts = [(int(xs.mean()), int(ys.mean()))]
        for idx in (np.argmin(ys), np.argmax(ys), np.argmin(xs), np.argmax(xs)):
            pts.append((int(xs[idx]), int(ys[idx])))
        pts = [p for p in pts if m[p[1], p[0]]][:5]
        if pts:
            out.append(pts)
    return anchor, out


def sam2_masks(frames, anchor, obj_points):
    """Propagate per-object point prompts from `anchor` in both directions."""
    import tempfile
    from PIL import Image
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    pred = sam2_masks.pred
    if pred is None:
        pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large", device="cuda")
        sam2_masks.pred = pred
    F = len(frames)
    h, w = frames[0].shape[:2]
    obj_masks = [[np.zeros((h, w), bool)] * F for _ in obj_points]
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(f"{td}/{i:05d}.jpg", quality=92)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = pred.init_state(video_path=td)
            for oi, pts in enumerate(obj_points):
                pred.add_new_points_or_box(
                    state, frame_idx=anchor, obj_id=oi,
                    points=np.array(pts, np.float32),
                    labels=np.ones(len(pts), np.int32))
            for reverse in (False, True):
                for fi, obj_ids, logits in pred.propagate_in_video(state, start_frame_idx=anchor, reverse=reverse):
                    for k, oid in enumerate(obj_ids):
                        obj_masks[oid][fi] = (logits[k, 0] > 0).cpu().numpy()
            pred.reset_state(state)
    return [list(m) for m in obj_masks]


def process_clip(frames, n_obj):
    anchor, obj_points = motion_prompts(frames, n_obj)
    if not obj_points:
        return None, None
    obj_masks = sam2_masks(frames, anchor, obj_points)
    union = [np.logical_or.reduce([om[i] for om in obj_masks]) for i in range(len(frames))]
    return union, obj_masks


def clip_stats(union):
    a = np.array([m.mean() for m in union])
    return {"mask_area_mean": round(float(a.mean()), 4),
            "mask_area_min": round(float(a.min()), 4),
            "empty_frames": int((a < 0.001).sum())}


def build_droid():
    ep = pd.read_parquet(hf_hub_download("lerobot/droid_100", "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset"))
    key = "observation.images.exterior_image_1_left"
    vid = hf_hub_download("lerobot/droid_100", f"videos/{key}/chunk-000/file-000.mp4", repo_type="dataset")
    fcol = f"videos/{key}/from_timestamp"

    picked, seen = [], set()
    for _, row in ep.iterrows():
        if int(row["length"]) < N_FRAMES + 4:
            continue
        tl = list(row["tasks"]) if row["tasks"] is not None else []
        prompt = str(tl[0]).strip() if tl else "a robot arm manipulates objects on a table"
        if prompt.lower() in seen:
            continue
        seen.add(prompt.lower())
        picked.append((row, prompt))
        if len(picked) >= DROID_N:
            break
    print(f"droid picked {len(picked)} episodes")

    stats = []
    for ci, (row, prompt) in enumerate(picked):
        frames = ffmpeg_extract(vid, float(row[fcol]), N_FRAMES, 1, 854, 480)
        frames = [f[:, 11:843].copy() for f in frames]
        if len(frames) < N_FRAMES:
            print(f"droid clip{ci}: only {len(frames)} frames, skip"); continue
        union, obj_masks = process_clip(frames, n_obj=1)
        if union is None:
            print(f"droid clip{ci}: no motion found, skip"); continue
        meta = save_clip(f"{OUT}/droid/clip{ci:03d}", frames, union, prompt, 15,
                         {"episode_index": int(row["episode_index"]), **clip_stats(union)},
                         obj_masks=obj_masks)
        print("CLIP", json.dumps({"set": "droid", "clip": ci, "prompt": prompt[:60], **clip_stats(union)}))
        stats.append(meta)
    return stats


def build_aloha():
    repo = "lerobot/aloha_static_coffee"
    ep = pd.read_parquet(hf_hub_download(repo, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset"))
    key = "observation.images.cam_high"
    vid = hf_hub_download(repo, f"videos/{key}/chunk-000/file-000.mp4", repo_type="dataset")
    fcol = f"videos/{key}/from_timestamp"
    stats = []
    for ci in range(ALOHA_N):
        row = ep.iloc[ci]
        frames = ffmpeg_extract(vid, float(row[fcol]), N_FRAMES, 3, 640, 480)
        if len(frames) < N_FRAMES:
            print(f"aloha clip{ci}: only {len(frames)} frames, skip"); continue
        union, obj_masks = process_clip(frames, n_obj=2)
        if union is None:
            print(f"aloha clip{ci}: no motion found, skip"); continue
        meta = save_clip(f"{OUT}/aloha/clip{ci:03d}", frames, union,
                         "two robot arms prepare coffee with a coffee machine", 16,
                         {"episode_index": int(row["episode_index"]), **clip_stats(union)},
                         obj_masks=obj_masks)
        print("CLIP", json.dumps({"set": "aloha", "clip": ci, **clip_stats(union)}))
        stats.append(meta)
    return stats


def main():
    sam2_masks.pred = None
    d = build_droid()
    a = build_aloha()
    print("SUMMARY", json.dumps({"droid_clips": len(d), "aloha_clips": len(a)}))
    upload_dir(OUT, "evalset")


if __name__ == "__main__":
    main()
