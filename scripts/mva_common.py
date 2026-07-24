"""Shared utilities: video IO, mask-derived control videos, contact sheets, HF upload."""
import io
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

ARTIFACT_REPO = "alphaXiv/mva-repro-artifacts"
GRAY = 128


def read_video(path):
    """Decode a video into a list of RGB uint8 arrays (ffmpeg backend, AV1-safe)."""
    import imageio.v3 as iio
    return [f for f in iio.imiter(path, plugin="pyav")]


def save_video(frames, path, fps=15):
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=7,
                           pixelformat="yuv420p", macro_block_size=1)
    for f in frames:
        w.append_data(np.asarray(f, dtype=np.uint8))
    w.close()


def ffmpeg_extract(src, start_s, n_frames, step, out_w, out_h):
    """Extract n_frames frames (every `step`-th) from src starting at start_s, resized."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-i", src,
           "-vf", f"select=not(mod(n\\,{step})),scale={out_w}:{out_h}:flags=lanczos",
           "-vsync", "vfr", "-frames:v", str(n_frames),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    n = len(raw) // (out_w * out_h * 3)
    arr = np.frombuffer(raw[: n * out_w * out_h * 3], np.uint8).reshape(n, out_h, out_w, 3)
    return [arr[i].copy() for i in range(n)]


def ee_point_from_mask(mask):
    """Approximate end-effector pixel: skeleton point farthest from the arm's
    border attachment (where the mask touches the image edge)."""
    from skimage.morphology import skeletonize
    m = mask.astype(bool)
    if m.sum() < 50:
        return None
    h, w = m.shape
    border = np.zeros_like(m)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    attach = np.argwhere(m & border)
    if len(attach) == 0:
        # arm fully inside frame: use mask centroid as anchor
        attach = np.argwhere(m).mean(0, keepdims=True)
    base = attach.mean(0)
    skel = np.argwhere(skeletonize(m))
    if len(skel) == 0:
        skel = np.argwhere(m)
    d = np.linalg.norm(skel - base, axis=1)
    y, x = skel[int(np.argmax(d))]
    return int(x), int(y)


def build_controls(frames, masks, ee_radius_frac=0.10):
    """From GT frames + robot masks, build dense / ee / skeleton control videos."""
    from skimage.morphology import skeletonize
    dense, ee, skel = [], [], []
    h, w = masks[0].shape
    r = int(ee_radius_frac * max(h, w))
    for f, m in zip(frames, masks):
        m = m.astype(bool)
        g = np.full_like(f, GRAY)
        d = g.copy()
        d[m] = f[m]
        dense.append(d)

        pt = ee_point_from_mask(m)
        e_img = g.copy()
        if pt is not None:
            yy, xx = np.ogrid[:h, :w]
            disc = (yy - pt[1]) ** 2 + (xx - pt[0]) ** 2 <= r * r
            keep = disc & m
            e_img[keep] = f[keep]
        ee.append(e_img)

        s_pil = Image.fromarray(g)
        dr = ImageDraw.Draw(s_pil)
        sk = np.argwhere(skeletonize(m))
        for y, x in sk:
            dr.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 255, 255))
        if pt is not None:
            dr.ellipse([pt[0] - 6, pt[1] - 6, pt[0] + 6, pt[1] + 6], fill=(255, 40, 40))
        skel.append(np.asarray(s_pil))
    return dense, ee, skel


def contact_sheet(rows, path, idxs=(0, 20, 40, 60, 80), cell_w=320):
    """rows: list of (label, frames). Grid image: one row per video, one col per idx."""
    from PIL import ImageFont
    pads, labw = 4, 90
    n_r, n_c = len(rows), len(idxs)
    h0, w0 = rows[0][1][0].shape[:2]
    cell_h = int(cell_w * h0 / w0)
    W = labw + n_c * (cell_w + pads)
    H = n_r * (cell_h + pads)
    sheet = Image.new("RGB", (W, H), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for ri, (label, frames) in enumerate(rows):
        d.text((4, ri * (cell_h + pads) + cell_h // 2), label, fill=(255, 255, 255))
        for ci, fi in enumerate(idxs):
            fi = min(fi, len(frames) - 1)
            im = Image.fromarray(np.asarray(frames[fi], np.uint8)).resize((cell_w, cell_h))
            sheet.paste(im, (labw + ci * (cell_w + pads), ri * (cell_h + pads)))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sheet.save(path)


def upload_dir(local_dir, repo_path):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(ARTIFACT_REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=local_dir, path_in_repo=repo_path,
                      repo_id=ARTIFACT_REPO, repo_type="dataset")
    print(f"uploaded {local_dir} -> {ARTIFACT_REPO}/{repo_path}")


def save_clip(out_dir, gt, masks, prompt, fps, extra=None):
    """Write one eval clip folder: gt/ref/controls/masks/meta + preview sheet."""
    os.makedirs(out_dir, exist_ok=True)
    dense, ee, skel = build_controls(gt, masks)
    save_video(gt, f"{out_dir}/gt.mp4", fps)
    Image.fromarray(gt[0]).save(f"{out_dir}/ref.png")
    save_video(dense, f"{out_dir}/control_dense.mp4", fps)
    save_video(ee, f"{out_dir}/control_ee.mp4", fps)
    save_video(skel, f"{out_dir}/control_skel.mp4", fps)
    np.savez_compressed(f"{out_dir}/masks.npz", mask=np.stack(masks).astype(np.uint8))
    meta = {"prompt": prompt, "fps": fps, "h": gt[0].shape[0], "w": gt[0].shape[1],
            "n_frames": len(gt), "mask_area_frac": float(np.stack(masks).mean())}
    meta.update(extra or {})
    with open(f"{out_dir}/meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    contact_sheet([("gt", gt), ("dense", dense), ("ee", ee), ("skel", skel)],
                  f"{out_dir}/preview.png")
    return meta
