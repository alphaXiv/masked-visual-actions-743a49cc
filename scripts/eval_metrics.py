"""Per-clip video metrics: full-frame + robot-region + background-region
fidelity, and optical-flow trajectory adherence."""
import cv2
import numpy as np


def _dilate(m, r):
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.dilate(m.astype(np.uint8), k).astype(bool)


def region_psnr(a, b, mask):
    if mask.sum() < 10:
        return None
    mse = ((a[mask].astype(np.float64) - b[mask]) ** 2).mean()
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-8)))


def region_ssim(a, b, mask):
    from skimage.metrics import structural_similarity
    if mask.sum() < 10:
        return None
    _, smap = structural_similarity(a, b, channel_axis=2, full=True, data_range=255)
    return float(smap.mean(axis=2)[mask].mean())


def flow_seq(frames):
    gs = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    return [cv2.calcOpticalFlowFarneback(gs[i - 1], gs[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            for i in range(1, len(gs))]


def video_metrics(gt, gen, masks, lpips_model=None, device="cuda"):
    """gt/gen: lists of HxWx3 uint8 (same length); masks: per-frame robot bool."""
    import torch
    F = min(len(gt), len(gen))
    gt, gen, masks = gt[:F], gen[:F], masks[:F]
    robot = [_dilate(m, 4) for m in masks]
    bg = [~_dilate(m, 12) for m in masks]

    out = {k: [] for k in ("psnr", "ssim", "psnr_robot", "ssim_robot", "psnr_bg", "ssim_bg")}
    for a, b, rm, bm in zip(gt, gen, robot, bg):
        full = np.ones(rm.shape, bool)
        out["psnr"].append(region_psnr(a, b, full))
        out["ssim"].append(region_ssim(a, b, full))
        out["psnr_robot"].append(region_psnr(a, b, rm))
        out["ssim_robot"].append(region_ssim(a, b, rm))
        out["psnr_bg"].append(region_psnr(a, b, bm))
        out["ssim_bg"].append(region_ssim(a, b, bm))
    res = {k: round(float(np.mean([v for v in vs if v is not None])), 4) if any(v is not None for v in vs) else None
           for k, vs in out.items()}

    if lpips_model is not None:
        with torch.no_grad():
            vals = []
            for i in range(0, F, 8):
                a = torch.from_numpy(np.stack(gt[i:i + 8])).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1
                b = torch.from_numpy(np.stack(gen[i:i + 8])).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1
                vals += lpips_model(a, b).flatten().cpu().tolist()
        res["lpips"] = round(float(np.mean(vals)), 4)
        # background LPIPS: gray out the (dilated) robot region in both videos
        with torch.no_grad():
            vals = []
            for i in range(0, F, 8):
                ga = np.stack(gt[i:i + 8]).copy()
                gb = np.stack(gen[i:i + 8]).copy()
                for j, rm in enumerate(robot[i:i + 8]):
                    ga[j][rm] = 128
                    gb[j][rm] = 128
                a = torch.from_numpy(ga).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1
                b = torch.from_numpy(gb).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1
                vals += lpips_model(a, b).flatten().cpu().tolist()
        res["lpips_bg"] = round(float(np.mean(vals)), 4)

    fg, fe = flow_seq(gt), flow_seq(gen)
    epe_all, epe_robot, mag_gt, mag_gen = [], [], [], []
    for i, (fa, fb) in enumerate(zip(fg, fe)):
        d = np.linalg.norm(fa - fb, axis=2)
        epe_all.append(d.mean())
        rm = robot[i + 1]
        if rm.sum() > 10:
            epe_robot.append(d[rm].mean())
        mag_gt.append(np.linalg.norm(fa, axis=2).mean())
        mag_gen.append(np.linalg.norm(fb, axis=2).mean())
    res["epe"] = round(float(np.mean(epe_all)), 4)
    res["epe_robot"] = round(float(np.mean(epe_robot)), 4) if epe_robot else None
    res["motion_ratio"] = round(float(np.sum(mag_gen) / max(np.sum(mag_gt), 1e-6)), 4)

    # preregistered failure flags
    res["flag_scene_transform"] = bool(res["ssim_bg"] is not None and res["ssim_bg"] < 0.5)
    res["flag_static"] = bool(res["motion_ratio"] < 0.3)
    res["flag_robot_halluc"] = bool(res["psnr_robot"] is not None and res["psnr_robot"] < 14.0)
    return res
