"""Synthetic fixtures mimicking RoboTwin raw data, LeRobot metadata and eval logs."""
from __future__ import annotations

import io
import json
from pathlib import Path

import h5py
import numpy as np

SETTING = "aloha-agilex_clean_50"


def jpeg_bytes(h: int = 240, w: int = 320) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 30, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def smooth_traj(T: int, phase: float = 0.0, noise: float = 0.0):
    t = np.arange(T)
    arm_l = np.stack([0.3 * np.sin(2 * np.pi * t / T + phase + k) for k in range(6)], axis=1)
    arm_r = np.stack([0.3 * np.cos(2 * np.pi * t / T + phase + k) for k in range(6)], axis=1)
    grip = np.ones(T)
    grip[T // 3: 2 * T // 3] = 0.0
    if noise:
        arm_l = arm_l + noise * ((-1.0) ** t)[:, None]
    return arm_l, grip.copy(), arm_r, grip.copy()


def write_episode(path: Path, T: int = 60, *, noise: float = 0.0, phase: float = 0.0, nan: bool = False,
                  cam_len_delta: int = 0, bad_jpeg: bool = False, static: bool = False,
                  drop_key: str | None = None) -> None:
    arm_l, gl, arm_r, gr = smooth_traj(T, phase, noise)
    if static:
        arm_l[:] = 0.1
        arm_r[:] = 0.2
    if nan:
        arm_l[5, 2] = np.nan
    jpg = jpeg_bytes()
    with h5py.File(path, "w") as f:
        joints = {"joint_action/left_arm": arm_l, "joint_action/left_gripper": gl,
                  "joint_action/right_arm": arm_r, "joint_action/right_gripper": gr}
        for key, value in joints.items():
            if key != drop_key:
                f.create_dataset(key, data=value)
        for cam in ("head_camera", "left_camera", "right_camera"):
            n = T + (cam_len_delta if cam == "head_camera" else 0)
            blob = b"garbage!" if bad_jpeg and cam == "left_camera" else jpg
            f.create_dataset(f"observation/{cam}/rgb", data=np.array([blob] * n, dtype=f"S{len(blob)}"))


def make_task(root: Path, task: str, n_eps: int = 5, *, seen=None, unseen=None, overrides=None) -> Path:
    sd = root / task / SETTING
    for sub in ("data", "instructions", "video"):
        (sd / sub).mkdir(parents=True, exist_ok=True)
    seen = seen or [f"{task} seen instruction {k}" for k in range(3)]
    unseen = unseen or [f"{task} unseen instruction {k}" for k in range(2)]
    overrides = overrides or {}
    for i in range(n_eps):
        kw = dict(T=60 + i, phase=0.01 * i)
        kw.update(overrides.get(i, {}))
        write_episode(sd / "data" / f"episode{i}.hdf5", **kw)
        (sd / "instructions" / f"episode{i}.json").write_text(json.dumps({"seen": seen, "unseen": unseen}))
        (sd / "video" / f"episode{i}.mp4").write_bytes(b"")
    (sd / "seed.txt").write_text(" ".join(str(100 + i) for i in range(n_eps)))
    return sd


def make_lerobot(root: Path, task: str, *, name_pattern: str = "{task}-aloha-agilex_clean_50-50",
                 episodes: int = 50, frames: int = 1000, task_strings=None, version: str = "v2.1") -> Path:
    ds = root / name_pattern.format(task=task)
    (ds / "meta").mkdir(parents=True, exist_ok=True)
    (ds / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": version, "total_episodes": episodes, "total_frames": frames, "fps": 50,
        "robot_type": "aloha"}))
    task_strings = task_strings if task_strings is not None else [f"{task} seen instruction 0"]
    if version == "v2.1":
        (ds / "meta" / "tasks.jsonl").write_text(
            "".join(json.dumps({"task_index": i, "task": s}) + "\n" for i, s in enumerate(task_strings)))
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.table({"task_index": list(range(len(task_strings))), "task": task_strings}),
                       ds / "meta" / "tasks.parquet")
    return ds


def episode_print(task: str, config: str, suc: int, num: int, seed: int, policy: str = "lingbot") -> str:
    pct = round(suc / num * 100, 1)
    return (f"\x1b[93m{task}\x1b[0m | \x1b[94m{policy}\x1b[0m | \x1b[92m{config}\x1b[0m | \x1b[91mckpt\x1b[0m\n"
            f"Success rate: \x1b[96m{suc}/{num}\x1b[0m => \x1b[95m{pct}%\x1b[0m, current seed: \x1b[90m{seed}\x1b[0m\n\n")


def episodes_log(task: str, config: str, outcomes: list[int], seed0: int = 100000) -> str:
    out, suc = [], 0
    for i, ok in enumerate(outcomes):
        suc += ok
        out.append("infer time 0.12s\n\x1b[92mSuccess!\x1b[0m\n" if ok else "\x1b[91mFail!\x1b[0m\n")
        out.append(episode_print(task, config, suc, i + 1, seed0 + i))
    return "".join(out)
