#!/usr/bin/env python
"""URDF-rendered sim eval set via robosuite/MuJoCo (osmesa, CPU).

Per episode we render, at every control step:
  gt      - normal agentview render (ground-truth video)
  mask    - exact robot mask from segmentation render (robot+gripper bodies)
  dense   - robot-only render (gripper recolored red) composited on gray
  ee      - dense restricted to a disc around the projected end-effector site
  skel    - projected joint-chain polyline + red EE dot on gray

Panda = in-domain embodiment (paper trains on DROID/RoboCasa Franka);
UR5e and Sawyer = held-out embodiments.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mva_common import save_video, contact_sheet, upload_dir, GRAY

H, W, F = 480, 832, 81
OUT = "/tmp/evalset"
EPS = [("Panda", 6, "sim_panda"), ("UR5e", 5, "sim_ur5e"), ("Sawyer", 5, "sim_sawyer")]


def make_env(robot, seed):
    import robosuite as suite
    from robosuite import load_controller_config
    np.random.seed(seed)
    env = suite.make(
        env_name="Lift", robots=robot,
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, use_object_obs=True,
        control_freq=15, horizon=F + 5, ignore_done=True,
        placement_initializer=None,
    )
    env.reset()
    return env


def robot_geom_ids(sim):
    ids = []
    for gid in range(sim.model.ngeom):
        bname = sim.model.body_id2name(sim.model.geom_bodyid[gid]) or ""
        if bname.startswith(("robot0", "gripper0")):
            ids.append(gid)
    return np.array(ids)


def gripper_geom_ids(sim):
    return np.array([gid for gid in range(sim.model.ngeom)
                     if (sim.model.body_id2name(sim.model.geom_bodyid[gid]) or "").startswith("gripper0")])


def chain_points(sim, robot_prefix="robot0"):
    pts = []
    for bid in range(sim.model.nbody):
        name = sim.model.body_id2name(bid) or ""
        if name.startswith(robot_prefix) and ("link" in name or "base" in name):
            pts.append(sim.data.body_xpos[bid].copy())
    pts.append(sim.data.site_xpos[sim.model.site_name2id("gripper0_grip_site")].copy())
    return np.array(pts)


def render_all(env, rob_ids, grip_ids):
    from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
    sim = env.sim
    cam = "agentview"
    gt = sim.render(width=W, height=H, camera_name=cam)[::-1].copy()

    seg = sim.render(width=W, height=H, camera_name=cam, segmentation=True)[::-1]
    mask = np.isin(seg[..., 1], rob_ids)

    rgba = sim.model.geom_rgba.copy()
    new = rgba.copy()
    for gid in grip_ids:
        new[gid, :3] = (1.0, 0.12, 0.12)
    sim.model.geom_rgba[:] = new
    robot_render = sim.render(width=W, height=H, camera_name=cam)[::-1].copy()
    sim.model.geom_rgba[:] = rgba

    dense = np.full_like(gt, GRAY)
    dense[mask] = robot_render[mask]

    w2c = get_camera_transform_matrix(sim, cam, H, W)
    ee3d = sim.data.site_xpos[sim.model.site_name2id("gripper0_grip_site")]
    ee_px = project_points_from_world_to_camera(ee3d[None], w2c, H, W)[0]  # (row, col)
    r = int(0.10 * W)
    yy, xx = np.ogrid[:H, :W]
    disc = (yy - ee_px[0]) ** 2 + (xx - ee_px[1]) ** 2 <= r * r
    ee_img = np.full_like(gt, GRAY)
    keep = disc & mask
    ee_img[keep] = robot_render[keep]

    from PIL import Image, ImageDraw
    sk = Image.new("RGB", (W, H), (GRAY, GRAY, GRAY))
    d = ImageDraw.Draw(sk)
    ch = chain_points(sim)
    px = project_points_from_world_to_camera(ch, w2c, H, W)
    for i in range(len(px) - 1):
        d.line([(int(px[i][1]), int(px[i][0])), (int(px[i + 1][1]), int(px[i + 1][0]))],
               fill=(255, 255, 255), width=4)
    d.ellipse([int(ee_px[1]) - 8, int(ee_px[0]) - 8, int(ee_px[1]) + 8, int(ee_px[0]) + 8],
              fill=(255, 40, 40))
    return gt, mask, dense, ee_img, np.asarray(sk)


def scripted_action(obs, phase):
    eef = obs["robot0_eef_pos"]
    cube = obs["cube_pos"]
    if phase == 0:
        tgt, grip = cube + [0, 0, 0.10], -1
    elif phase == 1:
        tgt, grip = cube + [0, 0, -0.005], -1
    elif phase == 2:
        tgt, grip = eef, 1
    else:
        tgt, grip = eef + [0, 0, 0.20], 1
    delta = np.clip((tgt - eef) * 6.0, -1, 1)
    return np.concatenate([delta, [0, 0, 0], [grip]])


def run_episode(robot, seed):
    env = make_env(robot, seed)
    sim = env.sim
    rob_ids, grip_ids = robot_geom_ids(sim), gripper_geom_ids(sim)
    obs = env._get_observations(force_update=True)
    frames = {"gt": [], "mask": [], "dense": [], "ee": [], "skel": []}
    phase, phase_t = 0, 0
    for t in range(F):
        gt, mask, dense, ee_img, skel = render_all(env, rob_ids, grip_ids)
        for k, v in zip(("gt", "mask", "dense", "ee", "skel"), (gt, mask, dense, ee_img, skel)):
            frames[k].append(v)
        a = scripted_action(obs, phase)
        obs, _, _, _ = env.step(a)
        eef, cube = obs["robot0_eef_pos"], obs["cube_pos"]
        phase_t += 1
        if phase == 0 and np.linalg.norm(eef - (cube + [0, 0, 0.10])) < 0.02:
            phase, phase_t = 1, 0
        elif phase == 1 and (np.linalg.norm(eef[:2] - cube[:2]) < 0.012 and abs(eef[2] - cube[2]) < 0.015 or phase_t > 25):
            phase, phase_t = 2, 0
        elif phase == 2 and phase_t >= 5:
            phase, phase_t = 3, 0
    lifted = bool(obs["cube_pos"][2] > 0.88)
    env.close()
    return frames, lifted


def main():
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from PIL import Image
    n_ok = 0
    for robot, n_eps, setname in EPS:
        for e in range(n_eps):
            try:
                fr, lifted = run_episode(robot, seed=1000 + e)
            except Exception as ex:
                print(f"EPISODE FAIL {robot} ep{e}: {type(ex).__name__}: {ex}")
                continue
            cd = f"{OUT}/{setname}/clip{e:03d}"
            os.makedirs(cd, exist_ok=True)
            save_video(fr["gt"], f"{cd}/gt.mp4", 15)
            Image.fromarray(fr["gt"][0]).save(f"{cd}/ref.png")
            save_video(fr["dense"], f"{cd}/control_dense.mp4", 15)
            save_video(fr["ee"], f"{cd}/control_ee.mp4", 15)
            save_video(fr["skel"], f"{cd}/control_skel.mp4", 15)
            np.savez_compressed(f"{cd}/masks.npz", mask=np.stack(fr["mask"]).astype(np.uint8))
            meta = {"prompt": f"a {robot} robot arm reaches for and lifts a red cube on a table",
                    "fps": 15, "h": H, "w": W, "n_frames": F, "robot": robot, "lifted": lifted,
                    "mask_area_frac": float(np.stack(fr["mask"]).mean())}
            with open(f"{cd}/meta.json", "w") as fh:
                json.dump(meta, fh, indent=1)
            contact_sheet([("gt", fr["gt"]), ("dense", fr["dense"]), ("ee", fr["ee"]), ("skel", fr["skel"])],
                          f"{cd}/preview.png")
            print("CLIP", json.dumps({"set": setname, "clip": e, "lifted": lifted,
                                      "mask_area_frac": meta["mask_area_frac"]}))
            n_ok += 1
    print("SUMMARY", json.dumps({"sim_clips": n_ok}))
    upload_dir(OUT, "evalset")


if __name__ == "__main__":
    main()
