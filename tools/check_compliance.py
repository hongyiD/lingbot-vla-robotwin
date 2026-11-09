#!/usr/bin/env python3
"""Pre-training / pre-submission compliance checks for the preliminary round.

C1 data scope   : training list only references the 50 aloha-agilex clean datasets.
C2 instructions : every LeRobot task string is an original *seen* instruction of that task
                  (no unseen leakage, no generated/modified text). Needs --raw_root.
C3 norm stats   : robot config norm_stats is clean-only (not the official mixed robotwin.json)
                  and identical to data.norm_stats_file in the training config (inference reads that).
C4 base model   : model paths point to the base lingbot-vla-v2-6b, not a RoboTwin post-trained one.
C5 eval script  : one model path for all tasks; demo_clean never evaluated with unseen instructions.

Exit code 1 if any check FAILs. WARN needs a human look; SKIP means inputs were not provided.

Example:
  python check_compliance.py --train_list ../lists/train_c50.txt \
      --robot_config_root /path/lingbot-vla-v2/configs/robot_configs \
      --train_config /path/train_c50.yaml --raw_root ../data/raw --eval_script /path/eval.sh \
      --report ../reports/compliance.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lingbot_tools.io_utils import now_iso, sha256_file, write_text  # noqa: E402
from lingbot_tools.lerobot_meta import infer_task_from_name, read_info, read_tasks  # noqa: E402
from lingbot_tools.raw_data import task_instruction_pools  # noqa: E402
from lingbot_tools.tasks import (  # noqa: E402
    CLEAN_SETTING_DIR, EPISODES_PER_TASK, FORBIDDEN_DATA_TOKENS, OFFICIAL_MIXED_NORM_STATS, TASKS,
)


def _result(cid: str, name: str) -> dict:
    return {"id": cid, "name": name, "status": "PASS", "details": []}


def _fail(res: dict, msg: str) -> None:
    res["status"] = "FAIL"
    res["details"].append(f"FAIL: {msg}")


def _warn(res: dict, msg: str) -> None:
    if res["status"] == "PASS":
        res["status"] = "WARN"
    res["details"].append(f"WARN: {msg}")


def _info(res: dict, msg: str) -> None:
    res["details"].append(msg)


def parse_train_list(path: Path) -> list[tuple[str, Path]]:
    entries = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{n}: expected '<robot_config_name> <dataset_path>', got {raw!r}")
        entries.append((parts[0], Path(parts[1])))
    return entries


def check_c1(entries, expect_episodes: int) -> tuple[dict, dict[Path, str]]:
    res = _result("C1", "training data scope")
    ds_task: dict[Path, str] = {}
    seen_tasks: dict[str, Path] = {}
    for robot, ds in entries:
        name = ds.name.lower()
        bad = [t for t in FORBIDDEN_DATA_TOKENS if t in name]
        if bad:
            _fail(res, f"{ds}: forbidden token(s) {bad}")
        task = infer_task_from_name(ds.name)
        if task is None:
            _fail(res, f"{ds}: cannot map dataset name to one of the 50 tasks")
            continue
        if task in seen_tasks:
            _fail(res, f"task {task} listed twice ({seen_tasks[task]} and {ds})")
        seen_tasks[task] = ds
        ds_task[ds] = task
        info = read_info(ds)
        if info is None:
            _fail(res, f"{ds}: missing meta/info.json")
            continue
        if int(info.get("total_episodes", -1)) != expect_episodes:
            _fail(res, f"{ds}: total_episodes={info.get('total_episodes')}, expected {expect_episodes}")
    missing = [t for t in TASKS if t not in seen_tasks]
    if missing:
        _warn(res, f"{len(missing)} task(s) not in list (ok only for smoke runs): {', '.join(missing[:10])}")
    _info(res, f"{len(seen_tasks)} task datasets, robot names: {sorted({r for r, _ in entries})}")
    return res, ds_task


def check_c2(ds_task: dict[Path, str], raw_root: Path | None, setting_dir: str) -> dict:
    res = _result("C2", "instruction pool (seen only)")
    if raw_root is None:
        res["status"] = "SKIP"
        _info(res, "no --raw_root given")
        return res
    for ds, task in sorted(ds_task.items(), key=lambda kv: kv[1]):
        strings = read_tasks(ds)
        if strings is None:
            _warn(res, f"{task}: cannot read meta/tasks.jsonl or meta/tasks.parquet in {ds}")
            continue
        seen, unseen = task_instruction_pools(raw_root / task / setting_dir)
        if not seen:
            _warn(res, f"{task}: no raw instruction files under {raw_root / task / setting_dir}")
            continue
        unseen_only = sorted(set(strings) & (unseen - seen))
        unknown = sorted(set(strings) - seen - unseen)
        both = sorted(set(strings) & seen & unseen)
        if unseen_only:
            _fail(res, f"{task}: {len(unseen_only)} unseen instruction(s) used, e.g. {unseen_only[0]!r}")
        if unknown:
            _fail(res, f"{task}: {len(unknown)} instruction(s) not in raw pools, e.g. {unknown[0][:120]!r}")
        if both:
            _warn(res, f"{task}: {len(both)} string(s) appear in both seen and unseen pools")
    _info(res, f"checked {len(ds_task)} datasets against raw pools in {raw_root}")
    return res


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _resolve(path_str: str, root: Path) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (root / p)


def check_c3(entries, robot_config_root: Path | None, lingbot_root: Path | None, train_cfg: dict | None,
             allow_official: bool) -> dict:
    res = _result("C3", "normalization stats (clean-only, train == inference)")
    if robot_config_root is None:
        res["status"] = "SKIP"
        _info(res, "no --robot_config_root given")
        return res
    root = lingbot_root or robot_config_root.resolve().parents[1]
    official = root / OFFICIAL_MIXED_NORM_STATS
    official_sha = sha256_file(official) if official.is_file() else None
    robot_files = {}
    for robot in sorted({r for r, _ in entries}):
        cfg_path = robot_config_root / f"{robot}.yaml"
        if not cfg_path.is_file():
            _fail(res, f"robot config {cfg_path} not found")
            continue
        norm = _load_yaml(cfg_path).get("norm_stats")
        if not norm:
            _fail(res, f"{cfg_path}: no 'norm_stats' field")
            continue
        norm_path = _resolve(str(norm), root)
        if not norm_path.is_file():
            _fail(res, f"{robot}: norm_stats file {norm_path} does not exist")
            continue
        sha = sha256_file(norm_path)
        robot_files[robot] = (norm_path, sha)
        is_official = str(norm).replace("\\", "/").endswith(OFFICIAL_MIXED_NORM_STATS) or sha == official_sha
        if is_official:
            msg = f"{robot}: norm_stats is the official mixed {OFFICIAL_MIXED_NORM_STATS} (computed beyond clean data)"
            (_warn if allow_official else _fail)(res, msg)
        _info(res, f"{robot}: norm_stats={norm_path} sha256={sha[:16]}")
    if train_cfg is not None:
        nsf = (train_cfg.get("data") or {}).get("norm_stats_file")
        if not nsf:
            _warn(res, "training config has no data.norm_stats_file; inference falls back to the robot config path")
        else:
            nsf_path = _resolve(str(nsf), root)
            if not nsf_path.is_file():
                _fail(res, f"data.norm_stats_file {nsf_path} does not exist")
            else:
                nsf_sha = sha256_file(nsf_path)
                mismatched = [r for r, (_, s) in robot_files.items() if s != nsf_sha]
                if mismatched:
                    _fail(res, f"data.norm_stats_file differs from robot config norm_stats for {mismatched} "
                               "(training normalizes with one file, inference de-normalizes with the other)")
                _info(res, f"data.norm_stats_file={nsf_path} sha256={nsf_sha[:16]}")
    return res


def check_c4(train_cfg: dict | None, train_list: Path) -> dict:
    res = _result("C4", "base model and config wiring")
    if train_cfg is None:
        res["status"] = "SKIP"
        _info(res, "no --train_config given")
        return res
    model = train_cfg.get("model") or {}
    data = train_cfg.get("data") or {}
    train = train_cfg.get("train") or {}
    for key in ("model_path", "config_path"):
        value = model.get(key)
        if not value and key == "config_path" and model.get("model_path"):
            # lingbotvla/utils/arguments.py ModelArguments.__post_init__: config_path defaults to model_path
            _info(res, "model.config_path not set, defaults to model.model_path")
            continue
        if not value:
            _warn(res, f"model.{key} not set")
            continue
        leaf = Path(str(value)).name.lower()
        if "robotwin" in leaf:
            _fail(res, f"model.{key}={value} looks like a RoboTwin post-trained checkpoint (prohibited)")
        elif "lingbot-vla-v2" not in str(value).lower():
            _warn(res, f"model.{key}={value} is not recognisable as lingbot-vla-v2 base weights")
        _info(res, f"model.{key}={value}")
    if train.get("load_checkpoint_path"):
        _warn(res, f"train.load_checkpoint_path={train['load_checkpoint_path']} (must be our own run)")
    tp = data.get("train_path")
    if tp and Path(str(tp)).expanduser().resolve() != train_list.resolve():
        _warn(res, f"data.train_path={tp} differs from checked list {train_list}")
    if tp and str(tp).endswith(".txt") and data.get("data_name") != "multi":
        _fail(res, f"data.train_path is a list file but data.data_name={data.get('data_name')!r} (expected 'multi')")
    return res


_MODEL_PATH_RE = re.compile(r"--model_path[ =]+[\"']?([^\s\"']+)")


def check_c5(eval_script: Path | None) -> dict:
    res = _result("C5", "evaluation script")
    if eval_script is None:
        res["status"] = "SKIP"
        _info(res, "no --eval_script given")
        return res
    text = eval_script.read_text(encoding="utf-8", errors="ignore")
    joined = text.replace("\\\n", " ")
    paths = sorted(set(_MODEL_PATH_RE.findall(joined)))
    if len(paths) > 1:
        _fail(res, f"multiple --model_path values: {paths}")
    _info(res, f"--model_path values: {paths or 'none found'}")
    for n, line in enumerate(joined.splitlines(), 1):
        if "demo_clean" in line and re.search(r"instruction_type[ =]+[\"']?unseen", line):
            _fail(res, f"line {n}: demo_clean evaluated with unseen instructions")
    uses_launcher = "start_robotwin_infer_and_eval" in text or "eval_policy_client_lingbotvla" in text
    if uses_launcher and "instruction_type" not in text:
        _warn(res, "LingBot launcher without instruction_type override: its default for demo_clean is unseen")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_list", required=True)
    ap.add_argument("--robot_config_root", default=None)
    ap.add_argument("--lingbot_root", default=None, help="default: <robot_config_root>/../..")
    ap.add_argument("--train_config", default=None)
    ap.add_argument("--raw_root", default=None)
    ap.add_argument("--setting_dir", default=CLEAN_SETTING_DIR)
    ap.add_argument("--eval_script", default=None)
    ap.add_argument("--expect_episodes", type=int, default=EPISODES_PER_TASK)
    ap.add_argument("--allow_official_norm", action="store_true", help="downgrade official mixed norm stats to WARN")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    train_list = Path(args.train_list)
    entries = parse_train_list(train_list)
    train_cfg = _load_yaml(Path(args.train_config)) if args.train_config else None
    c1, ds_task = check_c1(entries, args.expect_episodes)
    results = [
        c1,
        check_c2(ds_task, Path(args.raw_root) if args.raw_root else None, args.setting_dir),
        check_c3(entries, Path(args.robot_config_root) if args.robot_config_root else None,
                 Path(args.lingbot_root) if args.lingbot_root else None, train_cfg, args.allow_official_norm),
        check_c4(train_cfg, train_list),
        check_c5(Path(args.eval_script) if args.eval_script else None),
    ]
    lines = [f"# Compliance check\n", f"- time: {now_iso()}", f"- train_list: `{train_list.resolve()}`", ""]
    for r in results:
        print(f"  [{r['status']:<4}] {r['id']} {r['name']}")
        for d in r["details"]:
            print(f"         {d}")
        lines.append(f"## {r['id']} {r['name']}: **{r['status']}**\n")
        lines.extend(f"- {d}" for d in r["details"])
        lines.append("")
    failed = [r["id"] for r in results if r["status"] == "FAIL"]
    verdict = "FAIL (" + ", ".join(failed) + ")" if failed else "PASS"
    print(f"[compliance] {verdict}")
    lines.insert(3, f"- verdict: **{verdict}**")
    if args.report:
        write_text(args.report, "\n".join(lines) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
