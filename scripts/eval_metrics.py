"""Reference-based metrics for generated robot videos.

All metrics compare a generated video against the ground-truth clip, using the
per-frame robot masks saved with the eval set:
  full-frame  : LPIPS / SSIM / PSNR (reconstruction fidelity)
  robot region: PSNR / SSIM / flow-EPE inside the dilated robot mask
                (trajectory adherence: did the arm go where the control said?)
  background  : PSNR / SSIM outside the mask (scene preservation)
  motion      : Farneback flow magnitude ratio + flow EPE (temporal dynamics)

Preregistered failure flags (thresholds fixed before any eval ran):
  static_video       motion_ratio < 0.25  (model ignored the control, froze)
  scene_transform    ssim_bg < 0.50       (background replaced/hallucinated)
  robot_hallucination psnr_robot < 14.0   (arm missing or morphed)
"""
import numpy as np

THRESH = {"static_video": 0.25, "scene_transform": 0.50, "robot_hallucination": 14.0}


def _gray(f):
    return (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])


def _psnr_masked(a, b, m):
    if m.sum() < 10:
        return None
    mse = ((a[m].astype(np.float64) - b[m].astype(np.float64)) ** 2).mean()
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-9)))


def _dilate(m, k):
    import cv2
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def _flow(a, b):
    import cv2
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.1, 0)


def video_metrics(gt, gen, masks, lpips_model=None, device="cuda"):
    """gt, gen: lists of HxWx3 uint8 (same length); masks: FxHxW bool."""
    import torch
    from skimage.metrics import structural_similarity
    F = min(len(gt), len(gen), len(masks))
    gt, gen = gt[:F], gen[:F]
    out = {k: [] for k in ["ssim", "psnr", "psnr_robot", "psnr_bg", "ssim_robot", "ssim_bg"]}
    epe_all, epe_robot, mag_gt, mag_gen = [], [], [], []

    for t in range(F):
        a, b = gt[t], gen[t]
        m = _dilate(masks[t], 9)
        bg = ~_dilate(masks[t], 21)
        out["psnr"].append(_psnr_masked(a, b, np.ones(m.shape, bool)))
        out["psnr_robot"].append(_psnr_masked(a, b, m))
        out["psnr_bg"].append(_psnr_masked(a, b, bg))
        ga, gb = _gray(a), _gray(b)
        s, smap = structural_similarity(ga, gb, data_range=255.0, full=True)
        out["ssim"].append(float(s))
        out["ssim_robot"].append(float(smap[m].mean()) if m.sum() > 10 else None)
        out["ssim_bg"].append(float(smap[bg].mean()) if bg.sum() > 10 else None)
        if t > 0:
            fa = _flow(_gray(gt[t - 1]).astype(np.uint8), ga.astype(np.uint8))
            fb = _flow(_gray(gen[t - 1]).astype(np.uint8), gb.astype(np.uint8))
            d = np.linalg.norm(fa - fb, axis=-1)
            epe_all.append(float(d.mean()))
            if m.sum() > 10:
                epe_robot.append(float(d[m].mean()))
            mag_gt.append(float(np.linalg.norm(fa, axis=-1).mean()))
            mag_gen.append(float(np.linalg.norm(fb, axis=-1).mean()))

    res = {k: float(np.mean([v for v in vs if v is not None])) for k, vs in out.items()}
    res["epe"] = float(np.mean(epe_all))
    res["epe_robot"] = float(np.mean(epe_robot)) if epe_robot else None
    res["motion_ratio"] = float(np.sum(mag_gen) / max(np.sum(mag_gt), 1e-6))

    if lpips_model is not None:
        vals = []
        for t in range(0, F, 4):
            ta = torch.from_numpy(gt[t]).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
            tb = torch.from_numpy(gen[t]).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
            with torch.no_grad():
                vals.append(float(lpips_model(ta, tb).item()))
        res["lpips"] = float(np.mean(vals))

    res["flag_static"] = bool(res["motion_ratio"] < THRESH["static_video"])
    res["flag_scene_transform"] = bool(res["ssim_bg"] < THRESH["scene_transform"])
    res["flag_robot_halluc"] = bool(res["psnr_robot"] < THRESH["robot_hallucination"])
    return res
