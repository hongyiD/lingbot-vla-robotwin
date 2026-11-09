#!/usr/bin/env python3
"""Write the LingBot multi-dataset training list ("<robot_config_name> <abs_dataset_path>" per line).

Only the 50 aloha-agilex clean datasets are allowed. Every dataset must exist, have
meta/info.json and the expected episode count; dataset names must not contain forbidden
tokens (randomized / other embodiments).

Examples:
  # official pi0 route (LeRobot v2.1, repo ids <task>-aloha-agilex_clean_50-50)
  python make_train_list.py --lerobot_root ~/.cache/huggingface/lerobot --out ../lists/train_c50.txt
  # AMD XPolicyLab route (LeRobot v3.0, repo ids <task>_joint_v30)
  python make_train_list.py --lerobot_root /RoboTwin/data/lerobot --name_pattern '{task}_joint_v30' --out ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lingbot_tools.io_utils import now_iso, write_text  # noqa: E402
from lingbot_tools.lerobot_meta import read_info  # noqa: E402
from lingbot_tools.tasks import EPISODES_PER_TASK, FORBIDDEN_DATA_TOKENS, resolve_tasks  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lerobot_root", required=True)
    ap.add_argument("--name_pattern", default="{task}-aloha-agilex_clean_50-50")
    ap.add_argument("--robot_name", default="robotwin_c50", help="robot config name (configs/robot_configs/<name>.yaml)")
    ap.add_argument("--tasks", default="all", help="subset only for smoke runs; submissions need all 50")
    ap.add_argument("--expect_episodes", type=int, default=EPISODES_PER_TASK)
    ap.add_argument("--global_batch", type=int, default=256, help="only used to print steps per epoch")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    tasks = resolve_tasks(args.tasks)
    errors, lines, per_task = [], [], []
    for task in tasks:
        ds = (Path(args.lerobot_root).expanduser() / args.name_pattern.format(task=task)).resolve()
        name = ds.name.lower()
        bad_tokens = [t for t in FORBIDDEN_DATA_TOKENS if t in name]
        if bad_tokens:
            errors.append(f"{task}: forbidden token(s) {bad_tokens} in dataset name {ds.name}")
            continue
        info = read_info(ds)
        if info is None:
            errors.append(f"{task}: missing {ds}/meta/info.json")
            continue
        episodes = int(info.get("total_episodes", -1))
        frames = int(info.get("total_frames", 0))
        if episodes != args.expect_episodes:
            errors.append(f"{task}: total_episodes={episodes}, expected {args.expect_episodes}")
        per_task.append({"task": task, "path": str(ds), "episodes": episodes, "frames": frames,
                         "codebase_version": info.get("codebase_version"), "fps": info.get("fps"),
                         "robot_type": info.get("robot_type")})
        lines.append(f"{args.robot_name} {ds}")

    if errors:
        print("[train-list] FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    out = Path(args.out)
    write_text(out, "\n".join(lines) + "\n")
    total_frames = sum(t["frames"] for t in per_task)
    versions = sorted({str(t["codebase_version"]) for t in per_task})
    summary = {"time": now_iso(), "list": str(out.resolve()), "robot_name": args.robot_name,
               "tasks": len(per_task), "total_frames": total_frames, "codebase_versions": versions,
               "steps_per_epoch_at_global_batch": {str(args.global_batch): round(total_frames / args.global_batch, 1)},
               "datasets": per_task}
    write_text(out.with_suffix(out.suffix + ".summary.json"), json.dumps(summary, indent=2) + "\n")
    print(f"[train-list] {len(lines)} datasets -> {out}")
    print(f"[train-list] total_frames={total_frames} versions={versions} "
          f"steps/epoch@{args.global_batch}={total_frames / args.global_batch:.1f}")
    if len(per_task) != 50:
        print(f"[train-list] NOTE: only {len(per_task)} tasks listed (smoke subset); submissions require all 50")
    return 0


if __name__ == "__main__":
    sys.exit(main())
