#!/usr/bin/env python3
"""Integrity check for unpacked RoboTwin clean data (aloha-agilex_clean_50) before conversion.

Per task:    episode count/indices, instructions (seen/unseen pools), seed.txt, videos.
Per episode: HDF5 readable, required keys, equal lengths, shapes, finite values, joint/gripper
             ranges, non-static motion, sampled camera frames are decodable JPEG.

Read-only: never modifies or deletes data. ERROR -> exit code 1; WARN is informational.

Examples:
  python check_raw_data.py --raw_root ../data/raw --out_dir ../reports/raw_check
  python check_raw_data.py --print_tree ../data/raw/click_bell/aloha-agilex_clean_50/data/episode0.hdf5
"""
from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lingbot_tools.io_utils import md_table, now_iso, write_csv, write_text  # noqa: E402
from lingbot_tools.raw_data import (  # noqa: E402
    ARM_SLICES, CAMERA_KEYS, GRIPPER_COLUMNS, JOINT_KEYS, hdf5_tree, instruction_file,
    list_episodes, load_instruction_pools, load_joint_state,
)
from lingbot_tools.tasks import CLEAN_SETTING_DIR, EPISODES_PER_TASK, resolve_tasks  # noqa: E402

JPEG_SOI = b"\xff\xd8\xff"
ARM_LIMIT = 2 * np.pi + 0.1
GRIPPER_RANGE = (-0.1, 1.1)
MAX_STEP_WARN = 0.5  # rad per frame


def _issue(task, episode, level, code, detail=""):
    return {"task": task, "episode": episode, "level": level, "code": code, "detail": str(detail)[:500]}


def _frame_bytes(ds, i):
    value = ds[i]
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value)
    if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 1:
        return value.tobytes()
    return None


def _decode_hw(blob: bytes):
    try:
        from PIL import Image
    except ImportError:
        return None
    with Image.open(io.BytesIO(blob)) as img:
        img.load()
        return (img.height, img.width)


def check_episode(path: Path, task: str, idx: int, sample_frames: int, decode: bool):
    issues, info = [], {"T": None, "image_hw": set()}
    try:
        h5 = h5py.File(path, "r")
    except Exception as exc:  # unreadable / truncated file
        return info, [_issue(task, idx, "ERROR", "unreadable_hdf5", exc)]
    with h5:
        missing = [k for k in list(JOINT_KEYS.values()) + list(CAMERA_KEYS.values()) if k not in h5]
        if missing:
            issues.append(_issue(task, idx, "ERROR", "missing_keys", ",".join(missing)))
            return info, issues
        try:
            q = load_joint_state(h5)
        except Exception as exc:
            issues.append(_issue(task, idx, "ERROR", "joint_shape", exc))
            return info, issues
        T = q.shape[0]
        info["T"] = T
        if q.shape[1] != 14:
            issues.append(_issue(task, idx, "ERROR", "joint_dims", f"expected 14 dims, got {q.shape[1]}"))
            return info, issues
        if T < 10:
            issues.append(_issue(task, idx, "ERROR", "too_short", f"T={T}"))
        if not np.all(np.isfinite(q)):
            issues.append(_issue(task, idx, "ERROR", "non_finite_joint", f"{int((~np.isfinite(q)).sum())} values"))
        arm = np.concatenate([q[:, s] for s in ARM_SLICES], axis=1)
        grip = q[:, list(GRIPPER_COLUMNS)]
        if np.isfinite(arm).all() and np.abs(arm).max() > ARM_LIMIT:
            issues.append(_issue(task, idx, "WARN", "arm_out_of_range", f"max|q|={np.abs(arm).max():.3f}"))
        if np.isfinite(grip).all() and (grip.min() < GRIPPER_RANGE[0] or grip.max() > GRIPPER_RANGE[1]):
            issues.append(_issue(task, idx, "WARN", "gripper_out_of_range",
                                 f"min={grip.min():.3f} max={grip.max():.3f}"))
        if T > 1 and np.isfinite(arm).all():
            step = np.abs(np.diff(arm, axis=0))
            if step.max() == 0.0:
                issues.append(_issue(task, idx, "ERROR", "static_episode", "arm never moves"))
            elif step.max() > MAX_STEP_WARN:
                issues.append(_issue(task, idx, "WARN", "joint_jump", f"max step {step.max():.3f} rad"))

        frames = sorted(set(np.linspace(0, max(T - 1, 0), num=max(1, sample_frames)).astype(int).tolist()))
        for cam, key in CAMERA_KEYS.items():
            ds = h5[key]
            if ds.shape[0] != T:
                issues.append(_issue(task, idx, "ERROR", "camera_length_mismatch", f"{cam}: {ds.shape[0]} vs T={T}"))
                continue
            if ds.ndim == 4:  # raw pixels instead of encoded bytes
                info["image_hw"].add(f"{ds.shape[1]}x{ds.shape[2]}")
                continue
            for i in frames:
                blob = _frame_bytes(ds, i)
                if blob is None or not blob.startswith(JPEG_SOI):
                    issues.append(_issue(task, idx, "ERROR", "bad_jpeg", f"{cam} frame {i}"))
                    break
                if decode:
                    try:
                        hw = _decode_hw(blob)
                    except Exception as exc:
                        issues.append(_issue(task, idx, "ERROR", "jpeg_decode_failed", f"{cam} frame {i}: {exc}"))
                        break
                    if hw:
                        info["image_hw"].add(f"{hw[0]}x{hw[1]}")
    return info, issues


def check_task(task: str, raw_root: str, setting_dir_name: str, expected: int, sample_frames: int,
               decode: bool) -> dict:
    setting_dir = Path(raw_root) / task / setting_dir_name
    issues: list[dict] = []
    row = {"task": task, "episodes_found": 0, "episodes_ok": 0, "errors": 0, "warnings": 0,
           "frames_total": 0, "T_min": "", "T_median": "", "T_max": "", "seen_min": "", "unseen_min": "",
           "image_hw": "", "videos": 0, "seeds": ""}
    if not setting_dir.is_dir():
        issues.append(_issue(task, "", "ERROR", "missing_setting_dir", setting_dir))
        row["errors"] = 1
        return {"row": row, "issues": issues}

    episodes = list_episodes(setting_dir / "data")
    row["episodes_found"] = len(episodes)
    if len(episodes) != expected:
        issues.append(_issue(task, "", "ERROR", "episode_count", f"found {len(episodes)}, expected {expected}"))
    missing_idx = sorted(set(range(expected)) - set(episodes))
    if missing_idx:
        issues.append(_issue(task, "", "ERROR", "missing_episode_indices", missing_idx[:20]))

    lengths, seen_counts, unseen_counts, hw_all = [], [], [], set()
    for idx, path in episodes.items():
        info, ep_issues = check_episode(path, task, idx, sample_frames, decode)
        instr = instruction_file(setting_dir, idx)
        if instr is None:
            ep_issues.append(_issue(task, idx, "ERROR", "missing_instruction_file"))
        else:
            try:
                seen, unseen = load_instruction_pools(instr)
                if not seen:
                    ep_issues.append(_issue(task, idx, "ERROR", "empty_seen_instructions"))
                if not unseen:
                    ep_issues.append(_issue(task, idx, "WARN", "empty_unseen_instructions"))
                seen_counts.append(len(seen))
                unseen_counts.append(len(unseen))
            except Exception as exc:
                ep_issues.append(_issue(task, idx, "ERROR", "bad_instruction_file", exc))
        if info["T"] is not None:
            lengths.append(info["T"])
        hw_all |= info["image_hw"]
        if not any(i["level"] == "ERROR" for i in ep_issues):
            row["episodes_ok"] += 1
        issues.extend(ep_issues)

    seed_file = setting_dir / "seed.txt"
    if seed_file.is_file():
        seeds = re.findall(r"-?\d+", seed_file.read_text(encoding="utf-8", errors="ignore"))
        row["seeds"] = len(seeds)
        if len(seeds) != len(episodes):
            issues.append(_issue(task, "", "WARN", "seed_count", f"{len(seeds)} seeds vs {len(episodes)} episodes"))
    else:
        issues.append(_issue(task, "", "WARN", "missing_seed_txt"))
    video_dir = setting_dir / "video"
    row["videos"] = len(list(video_dir.glob("*.mp4"))) if video_dir.is_dir() else 0
    if len(hw_all) > 1:
        issues.append(_issue(task, "", "WARN", "mixed_image_sizes", sorted(hw_all)))

    if lengths:
        row.update(frames_total=sum(lengths), T_min=min(lengths), T_median=int(statistics.median(lengths)),
                   T_max=max(lengths))
    if seen_counts:
        row["seen_min"] = min(seen_counts)
        row["unseen_min"] = min(unseen_counts)
    row["image_hw"] = ",".join(sorted(hw_all))
    row["errors"] = sum(1 for i in issues if i["level"] == "ERROR")
    row["warnings"] = sum(1 for i in issues if i["level"] == "WARN")
    return {"row": row, "issues": issues}


SUMMARY_FIELDS = ["task", "episodes_found", "episodes_ok", "errors", "warnings", "frames_total", "T_min",
                  "T_median", "T_max", "seen_min", "unseen_min", "image_hw", "videos", "seeds"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root", help="directory with one sub-directory per task")
    ap.add_argument("--setting_dir", default=CLEAN_SETTING_DIR)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--expect_episodes", type=int, default=EPISODES_PER_TASK)
    ap.add_argument("--sample_frames", type=int, default=3, help="frames per camera per episode to check")
    ap.add_argument("--no_decode", action="store_true", help="only check JPEG markers, skip decoding")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--print_tree", default=None, help="print the key tree of one HDF5 file and exit")
    args = ap.parse_args(argv)

    if args.print_tree:
        with h5py.File(args.print_tree, "r") as h5:
            print("\n".join(hdf5_tree(h5)))
        return 0
    if not args.raw_root:
        ap.error("--raw_root is required unless --print_tree is used")

    tasks = resolve_tasks(args.tasks)
    worker_args = (args.raw_root, args.setting_dir, args.expect_episodes, args.sample_frames, not args.no_decode)
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(check_task, tasks, *[[a] * len(tasks) for a in worker_args]))
    else:
        results = [check_task(t, *worker_args) for t in tasks]

    rows = [r["row"] for r in results]
    issues = [i for r in results for i in r["issues"]]
    n_err = sum(1 for i in issues if i["level"] == "ERROR")
    n_warn = sum(1 for i in issues if i["level"] == "WARN")
    total_eps = sum(r["episodes_found"] for r in rows)
    total_frames = sum(int(r["frames_total"] or 0) for r in rows)
    for r in rows:
        flag = "OK " if r["errors"] == 0 else "ERR"
        print(f"  {flag} {r['task']:<28} eps={r['episodes_found']:>3} ok={r['episodes_ok']:>3} "
              f"T={r['T_min']}-{r['T_max']} err={r['errors']} warn={r['warnings']}")
    print(f"[raw-check] tasks={len(rows)} episodes={total_eps} frames={total_frames} errors={n_err} warnings={n_warn}")

    if args.out_dir:
        out = Path(args.out_dir)
        write_csv(out / "summary.csv", rows, SUMMARY_FIELDS)
        write_csv(out / "issues.csv", issues, ["task", "episode", "level", "code", "detail"])
        summary = {"time": now_iso(), "raw_root": str(Path(args.raw_root).resolve()), "setting_dir": args.setting_dir,
                   "tasks": len(rows), "episodes": total_eps, "frames": total_frames, "errors": n_err,
                   "warnings": n_warn, "tasks_with_errors": [r["task"] for r in rows if r["errors"]]}
        write_text(out / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        report = [f"# Raw data integrity check\n", f"- time: {summary['time']}",
                  f"- raw_root: `{summary['raw_root']}` / `{args.setting_dir}`",
                  f"- tasks: {len(rows)}, episodes: {total_eps}, frames: {total_frames}",
                  f"- errors: {n_err}, warnings: {n_warn}\n", "## Per task\n",
                  md_table(SUMMARY_FIELDS, [[r[k] for k in SUMMARY_FIELDS] for r in rows]), "",
                  "## Issues (first 300)\n",
                  md_table(["task", "episode", "level", "code", "detail"],
                           [[i["task"], i["episode"], i["level"], i["code"], i["detail"]] for i in issues[:300]])
                  if issues else "None.", ""]
        write_text(out / "report.md", "\n".join(report))
        print(f"[raw-check] reports written to {out}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
