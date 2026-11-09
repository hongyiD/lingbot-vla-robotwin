"""Readers for RoboTwin raw episodes as unpacked from aloha-agilex_clean_50.zip.

Layout per task (unzipped into <raw_root>/<task>/):
  aloha-agilex_clean_50/data/episode{i}.hdf5
  aloha-agilex_clean_50/instructions/episode{i}.json   ({"seen": [...], "unseen": [...]})
  aloha-agilex_clean_50/video/episode{i}.mp4
  aloha-agilex_clean_50/seed.txt, scene_info.json, _traj_data/

HDF5 keys follow RoboTwin policy/pi0/scripts/process_data.py (RoboTwin @13c3c47).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

EPISODE_RE = re.compile(r"^episode_?(\d+)\.hdf5$")

JOINT_KEYS = {
    "left_arm": "joint_action/left_arm",
    "left_gripper": "joint_action/left_gripper",
    "right_arm": "joint_action/right_arm",
    "right_gripper": "joint_action/right_gripper",
}

CAMERA_KEYS = {
    "head_camera": "observation/head_camera/rgb",
    "left_camera": "observation/left_camera/rgb",
    "right_camera": "observation/right_camera/rgb",
}

ARM_SLICES = (slice(0, 6), slice(7, 13))
GRIPPER_COLUMNS = (6, 13)


def list_episodes(data_dir: Path) -> dict[int, Path]:
    episodes: dict[int, Path] = {}
    if not data_dir.is_dir():
        return episodes
    for p in data_dir.iterdir():
        m = EPISODE_RE.match(p.name)
        if m:
            episodes[int(m.group(1))] = p
    return dict(sorted(episodes.items()))


def instruction_file(setting_dir: Path, idx: int) -> Path | None:
    for candidate in (
        setting_dir / "instructions" / f"episode{idx}.json",
        setting_dir / "instructions" / f"episode_{idx:07d}.json",
        setting_dir / "instruction" / f"episode_{idx:07d}.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_instruction_pools(path: Path) -> tuple[list[str], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"instruction file is not a JSON object: {path}")
    seen = data.get("seen")
    unseen = data.get("unseen")
    for name, value in (("seen", seen), ("unseen", unseen)):
        if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
            raise ValueError(f"'{name}' must be a list of strings in {path}")
    return list(seen), list(unseen)


def task_instruction_pools(setting_dir: Path) -> tuple[set[str], set[str]]:
    """Union of seen / unseen instructions over all episodes of one task."""
    seen_all: set[str] = set()
    unseen_all: set[str] = set()
    for idx in list_episodes(setting_dir / "data"):
        f = instruction_file(setting_dir, idx)
        if f is None:
            continue
        seen, unseen = load_instruction_pools(f)
        seen_all.update(seen)
        unseen_all.update(unseen)
    return seen_all, unseen_all


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return arr[:, None] if arr.ndim == 1 else arr


def load_joint_state(h5) -> np.ndarray:
    """(T, 14) state in pi0 order: left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)."""
    parts = [_as_2d(h5[JOINT_KEYS[k]][()]) for k in ("left_arm", "left_gripper", "right_arm", "right_gripper")]
    lengths = {p.shape[0] for p in parts}
    if len(lengths) != 1:
        raise ValueError(f"joint arrays have different lengths: {sorted(lengths)}")
    return np.concatenate(parts, axis=1)


def hdf5_tree(h5) -> list[str]:
    lines: list[str] = []

    def visit(name, obj):
        shape = getattr(obj, "shape", None)
        if shape is not None:
            lines.append(f"{name}  shape={shape} dtype={obj.dtype}")
        else:
            lines.append(f"{name}/")

    h5.visititems(visit)
    return lines
