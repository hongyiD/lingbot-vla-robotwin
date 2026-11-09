#!/usr/bin/env python3
"""Per-episode motion-quality statistics with within-task Z-scores (flag only, never delete).

Features (arm joints only for derivatives; units are per frame):
  length, path_len, vel_rms, acc_rms, jerk_rms, jerk_max, max_step, idle_ratio,
  grip_toggles_l / grip_toggles_r (open/close transitions).
Flag rule: classic |z| > --z (default 3) on any continuous feature, or robust |z| > --robust_z
(median/MAD based, default 5) to catch outliers that mask themselves in the classic score.
Flagged episodes are candidates for manual video review; the decision to drop anything is
human-only and must be recorded in the ledger.

Example:
  python data_stats.py --raw_root ../data/raw --out_dir ../reports/data_stats
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lingbot_tools.io_utils import md_table, now_iso, write_csv, write_text  # noqa: E402
from lingbot_tools.raw_data import ARM_SLICES, GRIPPER_COLUMNS, list_episodes, load_joint_state  # noqa: E402
from lingbot_tools.tasks import CLEAN_SETTING_DIR, resolve_tasks  # noqa: E402

CONTINUOUS = ["length", "path_len", "vel_rms", "acc_rms", "jerk_rms", "jerk_max", "max_step", "idle_ratio"]
IDLE_EPS = 1e-4  # rad per frame


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _toggles(g: np.ndarray) -> int:
    lo, hi = float(np.min(g)), float(np.max(g))
    if hi - lo < 0.1:
        return 0
    state = g > (lo + hi) / 2.0
    return int(np.count_nonzero(state[1:] != state[:-1]))


def episode_features(q: np.ndarray) -> dict:
    arm = np.concatenate([q[:, s] for s in ARM_SLICES], axis=1)
    v = np.diff(arm, axis=0)
    a = np.diff(v, axis=0)
    j = np.diff(a, axis=0)
    speed = np.max(np.abs(v), axis=1) if v.size else np.zeros(0)
    return {
        "length": int(q.shape[0]),
        "path_len": float(np.sum(np.abs(v))),
        "vel_rms": _rms(v),
        "acc_rms": _rms(a),
        "jerk_rms": _rms(j),
        "jerk_max": float(np.max(np.abs(j))) if j.size else 0.0,
        "max_step": float(np.max(np.abs(v))) if v.size else 0.0,
        "idle_ratio": float(np.mean(speed < IDLE_EPS)) if speed.size else 1.0,
        "grip_toggles_l": _toggles(q[:, GRIPPER_COLUMNS[0]]),
        "grip_toggles_r": _toggles(q[:, GRIPPER_COLUMNS[1]]),
    }


def task_features(task: str, raw_root: str, setting_dir_name: str) -> dict:
    setting_dir = Path(raw_root) / task / setting_dir_name
    rows, errors, states = [], [], []
    for idx, path in list_episodes(setting_dir / "data").items():
        try:
            with h5py.File(path, "r") as h5:
                q = load_joint_state(h5)
        except Exception as exc:
            errors.append({"task": task, "episode": idx, "error": str(exc)[:300]})
            continue
        if q.shape[1] != 14 or not np.all(np.isfinite(q)):
            errors.append({"task": task, "episode": idx, "error": "non-finite or wrong dims; run check_raw_data.py"})
            continue
        feats = episode_features(q)
        feats.update(task=task, episode=idx, video=str(setting_dir / "video" / f"episode{idx}.mp4"))
        rows.append(feats)
        states.append(q.astype(np.float32))
    return {"rows": rows, "errors": errors, "states": states}


def add_zscores(rows: list[dict], z_thresh: float, robust_thresh: float) -> None:
    """Annotate rows of ONE task in place with z_/rz_ columns and flag reasons."""
    for feat in CONTINUOUS:
        x = np.array([r[feat] for r in rows], dtype=float)
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med))) * 1.4826
        for r, xi in zip(rows, x):
            r[f"z_{feat}"] = (xi - mean) / std if std > 0 else 0.0
            r[f"rz_{feat}"] = (xi - med) / mad if mad > 0 else 0.0
    for feat in ("grip_toggles_l", "grip_toggles_r"):
        vals = [r[feat] for r in rows]
        mode = max(set(vals), key=vals.count) if vals else 0
        for r in rows:
            r[f"{feat}_mode"] = mode
    for r in rows:
        reasons = [f"{f}:z={r[f'z_{f}']:+.2f}" for f in CONTINUOUS if abs(r[f"z_{f}"]) > z_thresh]
        reasons += [f"{f}:rz={r[f'rz_{f}']:+.1f}" for f in CONTINUOUS
                    if abs(r[f"rz_{f}"]) > robust_thresh and abs(r[f"z_{f}"]) <= z_thresh]
        r["flagged"] = int(bool(reasons))
        r["flag_reasons"] = ";".join(reasons)


def joint_quantiles(states: list[np.ndarray]) -> dict:
    allq = np.concatenate(states, axis=0)
    return {
        "frames": int(allq.shape[0]),
        "order": "left_arm(6),left_gripper(1),right_arm(6),right_gripper(1)",
        "mean": allq.mean(axis=0).round(6).tolist(),
        "std": allq.std(axis=0).round(6).tolist(),
        "q01": np.quantile(allq, 0.01, axis=0).round(6).tolist(),
        "q99": np.quantile(allq, 0.99, axis=0).round(6).tolist(),
        "min": allq.min(axis=0).round(6).tolist(),
        "max": allq.max(axis=0).round(6).tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--setting_dir", default=CLEAN_SETTING_DIR)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--robust_z", type=float, default=5.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args(argv)

    tasks = resolve_tasks(args.tasks)
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(task_features, tasks, [args.raw_root] * len(tasks),
                                    [args.setting_dir] * len(tasks)))
    else:
        results = [task_features(t, args.raw_root, args.setting_dir) for t in tasks]

    all_rows, errors, states, task_rows = [], [], [], []
    for task, res in zip(tasks, results):
        rows = res["rows"]
        errors.extend(res["errors"])
        states.extend(res["states"])
        if rows:
            add_zscores(rows, args.z, args.robust_z)
        all_rows.extend(rows)
        lengths = [r["length"] for r in rows]
        task_rows.append({
            "task": task, "episodes": len(rows), "flagged": sum(r["flagged"] for r in rows),
            "length_mean": round(float(np.mean(lengths)), 1) if lengths else "",
            "length_min": min(lengths) if lengths else "", "length_max": max(lengths) if lengths else "",
            "frames": sum(lengths), "lerobot_frames_est": sum(max(n - 1, 0) for n in lengths),
            "jerk_rms_median": round(float(np.median([r["jerk_rms"] for r in rows])), 6) if rows else "",
            "idle_ratio_median": round(float(np.median([r["idle_ratio"] for r in rows])), 4) if rows else "",
        })

    out = Path(args.out_dir)
    ep_fields = (["task", "episode", "flagged", "flag_reasons"] + CONTINUOUS + ["grip_toggles_l", "grip_toggles_r",
                 "grip_toggles_l_mode", "grip_toggles_r_mode"] + [f"z_{f}" for f in CONTINUOUS]
                 + [f"rz_{f}" for f in CONTINUOUS] + ["video"])
    write_csv(out / "episode_stats.csv", all_rows, ep_fields)
    flagged = [r for r in all_rows if r["flagged"]]
    write_csv(out / "flagged.csv", flagged, ["task", "episode", "flag_reasons", "length", "jerk_rms",
                                             "max_step", "idle_ratio", "video"])
    task_fields = ["task", "episodes", "flagged", "length_mean", "length_min", "length_max", "frames",
                   "lerobot_frames_est", "jerk_rms_median", "idle_ratio_median"]
    write_csv(out / "task_summary.csv", task_rows, task_fields)
    if errors:
        write_csv(out / "errors.csv", errors, ["task", "episode", "error"])
    quant = joint_quantiles(states) if states else {}
    write_text(out / "joint_quantiles.json", json.dumps(quant, indent=2) + "\n")

    total_frames = sum(t["frames"] for t in task_rows)
    lerobot_frames = sum(t["lerobot_frames_est"] for t in task_rows)
    summary = {"time": now_iso(), "raw_root": str(Path(args.raw_root).resolve()), "tasks": len(tasks),
               "episodes": len(all_rows), "flagged": len(flagged), "errors": len(errors),
               "frames": total_frames, "lerobot_frames_est": lerobot_frames,
               "z": args.z, "robust_z": args.robust_z}
    write_text(out / "summary.json", json.dumps(summary, indent=2) + "\n")
    report = [
        "# Data quality statistics (flag only)\n",
        f"- time: {summary['time']}", f"- raw_root: `{summary['raw_root']}`",
        f"- episodes: {len(all_rows)} over {len(tasks)} tasks, frames: {total_frames} "
        f"(LeRobot frames after pi0 processing ~ {lerobot_frames})",
        f"- flagged: {len(flagged)} (rule: |z|>{args.z} or robust |z|>{args.robust_z}); read errors: {len(errors)}",
        "- Flagged episodes are review candidates only. Watch the video, then record keep/drop in the ledger.\n",
        "## Per task\n", md_table(task_fields, [[t[k] for k in task_fields] for t in task_rows]), "",
        "## Flagged episodes\n",
        md_table(["task", "episode", "reasons", "video"],
                 [[r["task"], r["episode"], r["flag_reasons"], r["video"]] for r in flagged]) if flagged else "None.",
        "",
    ]
    write_text(out / "report.md", "\n".join(report))
    print(f"[data-stats] episodes={len(all_rows)} flagged={len(flagged)} errors={len(errors)} "
          f"frames={total_frames} lerobot_frames_est={lerobot_frames}")
    print(f"[data-stats] reports written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
