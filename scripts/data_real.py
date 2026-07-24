#!/usr/bin/env python
"""Build the real-video eval set: DROID-100 (Franka, training domain) and
ALOHA static-coffee (bimanual, held-out embodiment). Robot masks come from a
GroundingDINO box prompt on frame 0 propagated with SAM2."""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mva_common import ffmpeg_extract, save_clip, upload_dir

N_FRAMES = 81
DROID_N, ALOHA_N = 16, 8
OUT = "/tmp/evalset"


def load_detector():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    mid = "IDEA-Research/grounding-dino-base"
    proc = AutoProcessor.from_pretrained(mid)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to("cuda")
    return proc, det


def detect_boxes(proc, det, frame, text="a robot arm.", topk=1):
    from PIL import Image
    img = Image.fromarray(frame)
    inputs = proc(images=img, text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = det(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=0.25, text_threshold=0.25,
        target_sizes=[img.size[::-1]])[0]
    order = torch.argsort(res["scores"], descending=True)
    boxes = [res["boxes"][i].tolist() for i in order[:topk]]
    scores = [float(res["scores"][i]) for i in order[:topk]]
    return boxes, scores


def sam2_masks(frames, boxes):
    """Propagate box prompts on frame 0 through the clip; return union masks."""
    import tempfile
    from PIL import Image
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    pred = sam2_masks.pred
    if pred is None:
        pred = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large", device="cuda")
        sam2_masks.pred = pred
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(f"{td}/{i:05d}.jpg", quality=92)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = pred.init_state(video_path=td)
            for oi, box in enumerate(boxes):
                pred.add_new_points_or_box(state, frame_idx=0, obj_id=oi,
                                           box=np.array(box, dtype=np.float32))
            masks = [None] * len(frames)
            for fi, obj_ids, logits in pred.propagate_in_video(state):
                m = (logits > 0).any(dim=0)[0].cpu().numpy()
                masks[fi] = m
    return [m if m is not None else np.zeros(frames[0].shape[:2], bool) for m in masks]


def build_droid(proc, det):
    ep = pd.read_parquet(hf_hub_download("lerobot/droid_100", "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset"))
    tasks = pd.read_parquet(hf_hub_download("lerobot/droid_100", "meta/tasks.parquet", repo_type="dataset"))
    print("episode meta columns:", list(ep.columns))
    task_by_idx = {int(r["task_index"]): str(t) for t, r in tasks.iterrows()} if "task_index" in tasks.columns else {i: str(t) for i, t in enumerate(tasks.index)}
    key = "observation.images.exterior_image_1_left"
    vid = hf_hub_download("lerobot/droid_100", f"videos/{key}/chunk-000/file-000.mp4", repo_type="dataset")
    fcol, tcol = f"videos/{key}/from_timestamp", f"videos/{key}/to_timestamp"

    picked, seen_tasks = [], set()
    for _, row in ep.iterrows():
        if int(row["length"]) < N_FRAMES + 4:
            continue
        ti = int(np.atleast_1d(row.get("tasks/task_index", row.get("task_index", -1)))[0]) if "tasks/task_index" in ep.columns or "task_index" in ep.columns else -1
        if ti in seen_tasks and len(picked) < DROID_N * 2:
            continue
        seen_tasks.add(ti)
        picked.append(row)
        if len(picked) >= DROID_N:
            break
    print(f"droid picked {len(picked)} episodes")

    stats = []
    for ci, row in enumerate(picked):
        frames = ffmpeg_extract(vid, float(row[fcol]), N_FRAMES, 1, 854, 480)
        frames = [f[:, 11:843].copy() for f in frames]  # center-crop 854->832, keeps 16:9
        if len(frames) < N_FRAMES:
            print(f"droid clip{ci}: only {len(frames)} frames, skip"); continue
        boxes, scores = detect_boxes(proc, det, frames[0])
        if not boxes:
            print(f"droid clip{ci}: no robot detected, skip"); continue
        masks = sam2_masks(frames, boxes)
        prompt = "a robot arm manipulates objects on a table"
        try:
            ti = int(np.atleast_1d(row.get("tasks/task_index", -1))[0])
            prompt = task_by_idx.get(ti, prompt)
        except Exception:
            pass
        meta = save_clip(f"{OUT}/droid/clip{ci:03d}", frames, masks, prompt, 15,
                         {"det_score": scores[0], "episode_index": int(row["episode_index"])})
        print("CLIP", json.dumps({"set": "droid", "clip": ci, **{k: meta[k] for k in ('mask_area_frac',)}, "det": round(scores[0], 3)}))
        stats.append(meta)
    return stats


def build_aloha(proc, det):
    repo = "lerobot/aloha_static_coffee"
    ep = pd.read_parquet(hf_hub_download(repo, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset"))
    key = "observation.images.cam_high"
    vid = hf_hub_download(repo, f"videos/{key}/chunk-000/file-000.mp4", repo_type="dataset")
    fcol = f"videos/{key}/from_timestamp"
    stats = []
    for ci in range(ALOHA_N):
        row = ep.iloc[ci]
        frames = ffmpeg_extract(vid, float(row[fcol]), N_FRAMES, 3, 640, 480)  # 50fps -> ~16.7fps
        if len(frames) < N_FRAMES:
            print(f"aloha clip{ci}: only {len(frames)} frames, skip"); continue
        boxes, scores = detect_boxes(proc, det, frames[0], topk=2)
        if not boxes:
            print(f"aloha clip{ci}: no robot detected, skip"); continue
        masks = sam2_masks(frames, boxes)
        meta = save_clip(f"{OUT}/aloha/clip{ci:03d}", frames, masks,
                         "two robot arms prepare coffee with a coffee machine", 16,
                         {"det_score": scores[0], "episode_index": int(row["episode_index"])})
        print("CLIP", json.dumps({"set": "aloha", "clip": ci, "mask_area_frac": meta["mask_area_frac"], "det": round(scores[0], 3)}))
        stats.append(meta)
    return stats


def main():
    sam2_masks.pred = None
    proc, det = load_detector()
    d = build_droid(proc, det)
    a = build_aloha(proc, det)
    print("SUMMARY", json.dumps({"droid_clips": len(d), "aloha_clips": len(a)}))
    upload_dir(OUT, "evalset")


if __name__ == "__main__":
    main()
