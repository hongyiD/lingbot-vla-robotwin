#!/usr/bin/env python3
"""Parse a RoboTwin evaluation run into per-episode rows (CSV) plus a summary.

Supported run layouts (auto-detected):
  launcher  : lingbot-vla-v2 experiment/robotwin/start_robotwin_infer_and_eval.sh
              <run_dir>/eval_logs/<task>.log, <run_dir>/eval_results/<task>/_result.txt
              Retries append "[attempt N/M]" banners; only the last attempt block is used.
  benchmark : Robotwin-radeon-cloud run_clean_benchmark.py (XPolicyLab client)
              <run_dir>/logs/eval_<task_config>__<task>.log

Both clients print, per episode:
  <task> | <policy> | <task_config> | <ckpt>
  Success rate: S/N => P%, current seed: SEED
The XPolicyLab batch mode line "<task> | episode=E | worker=W | seed=S | success=B | batch success: S/N"
is also accepted.

Example:
  python parse_eval_run.py --run_dir /path/to/run --out_dir ../evals/<run_id>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lingbot_tools.io_utils import md_table, now_iso, strip_ansi, write_csv, write_text  # noqa: E402
from lingbot_tools.tasks import EXPECTED_INSTRUCTION_TYPE, TASKS, setting_from_task_config  # noqa: E402

HEADER_RE = re.compile(r"^(?P<task>[a-z0-9_]+) \| (?P<policy>[^|]+?) \| (?P<config>[^|]+?) \| (?P<ckpt>.*?)\s*$")
RATE_RE = re.compile(r"Success rate:\s*(?P<suc>\d+)/(?P<num>\d+)\s*=>\s*[0-9.]+%,\s*current seed:\s*(?P<seed>-?\d+)")
BATCH_RE = re.compile(r"^(?P<task>[a-z0-9_]+) \| episode=(?P<ep>[^|]*?) \| worker=[^|]*? \| seed=(?P<seed>-?\d+) "
                      r"\| success=(?P<success>True|False|true|false|1|0) \| batch success:\s*(?P<suc>\d+)/(?P<num>\d+)")
ATTEMPT_RE = re.compile(r"\[attempt\s+\d+/\d+\]")
INSTR_RE = re.compile(r"(?:Eval Instruction|Instruction Type):\s*(seen|unseen)", re.IGNORECASE)
FINAL_RE = re.compile(r"Final (?:batch )?success rate:\s*(\d+)/(\d+)")
BENCH_LOG_RE = re.compile(r"^eval_(?P<config>[^_].*?)__(?P<task>[a-z0-9_]+)\.log$")

ROW_FIELDS = ["run_id", "setting", "task_config", "task", "episode_idx", "seed", "success",
              "instruction_type", "policy", "ckpt", "source"]


def last_attempt_block(text: str) -> str:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if ATTEMPT_RE.search(line)]
    return "\n".join(lines[starts[-1]:]) if starts else text


def parse_log_text(text: str, *, task_hint: str | None = None, config_hint: str | None = None) -> dict:
    """Return {"episodes": [...], "instruction_type": str|None, "anomalies": [...], "final": (s, n)|None}."""
    text = strip_ansi(text.replace("\r", "\n"))
    block = last_attempt_block(text)
    episodes, anomalies = [], []
    prev_suc, prev_num = 0, 0
    last_header = None
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m_batch = BATCH_RE.match(stripped)
        if m_batch:
            suc, num = int(m_batch["suc"]), int(m_batch["num"])
            success = m_batch["success"].lower() in ("true", "1")
            episodes.append({"task": m_batch["task"], "task_config": config_hint, "episode_idx": num - 1,
                             "seed": int(m_batch["seed"]), "success": int(success), "policy": "", "ckpt": ""})
            if num != prev_num + 1:
                anomalies.append(f"batch episode counter jumped {prev_num} -> {num}")
            prev_suc, prev_num = suc, num
            continue
        m_head = HEADER_RE.match(stripped)
        if m_head and "Success rate" not in stripped:
            last_header = m_head
            continue
        m_rate = RATE_RE.search(stripped)
        if m_rate:
            suc, num, seed = int(m_rate["suc"]), int(m_rate["num"]), int(m_rate["seed"])
            delta = suc - prev_suc
            if num != prev_num + 1:
                anomalies.append(f"episode counter jumped {prev_num} -> {num} (seed {seed})")
            if delta not in (0, 1):
                anomalies.append(f"success counter jumped {prev_suc} -> {suc} at episode {num}")
            head = last_header
            episodes.append({
                "task": head["task"] if head else task_hint,
                "task_config": head["config"].strip() if head else config_hint,
                "episode_idx": num - 1, "seed": seed, "success": int(delta == 1),
                "policy": head["policy"].strip() if head else "", "ckpt": head["ckpt"].strip() if head else "",
            })
            prev_suc, prev_num = suc, num
            last_header = None
    instr = INSTR_RE.findall(block) or INSTR_RE.findall(text)
    finals = FINAL_RE.findall(block)
    final = (int(finals[-1][0]), int(finals[-1][1])) if finals else None
    if final and episodes and final != (sum(e["success"] for e in episodes), len(episodes)):
        anomalies.append(f"final line {final[0]}/{final[1]} != parsed {sum(e['success'] for e in episodes)}/{len(episodes)}")
    return {"episodes": episodes, "instruction_type": instr[-1].lower() if instr else None,
            "anomalies": anomalies, "final": final}


def discover_logs(run_dir: Path) -> list[dict]:
    """List log files with task / config hints and optional _result.txt path."""
    items = []
    if (run_dir / "eval_logs").is_dir():
        for log in sorted((run_dir / "eval_logs").glob("*.log")):
            task = log.stem
            items.append({"log": log, "task": task, "config": None,
                          "result_txt": run_dir / "eval_results" / task / "_result.txt"})
    elif (run_dir / "logs").is_dir():
        for log in sorted((run_dir / "logs").glob("eval_*__*.log")):
            m = BENCH_LOG_RE.match(log.name)
            if m:
                items.append({"log": log, "task": m["task"], "config": m["config"], "result_txt": None})
    return items


def parse_run(run_dir: Path, run_id: str, config_override: str | None = None) -> dict:
    rows, problems = [], []
    for item in discover_logs(run_dir):
        parsed = parse_log_text(item["log"].read_text(encoding="utf-8", errors="ignore"),
                                task_hint=item["task"], config_hint=config_override or item["config"])
        instr = parsed["instruction_type"]
        rt = item["result_txt"]
        if rt is not None and rt.is_file():
            m = INSTR_RE.search(rt.read_text(encoding="utf-8", errors="ignore"))
            if m:
                instr = m.group(1).lower()
        for a in parsed["anomalies"]:
            problems.append({"task": item["task"], "level": "WARN", "detail": a})
        if not parsed["episodes"]:
            problems.append({"task": item["task"], "level": "WARN", "detail": f"no episodes parsed in {item['log']}"})
        for ep in parsed["episodes"]:
            task_config = config_override or ep["task_config"] or item["config"] or ""
            setting = setting_from_task_config(task_config) or ""
            if ep["task"] != item["task"]:
                problems.append({"task": item["task"], "level": "WARN",
                                 "detail": f"log contains episodes of {ep['task']}"})
            rows.append({"run_id": run_id, "setting": setting, "task_config": task_config, "task": ep["task"],
                         "episode_idx": ep["episode_idx"], "seed": ep["seed"], "success": ep["success"],
                         "instruction_type": instr or "", "policy": ep["policy"], "ckpt": ep["ckpt"],
                         "source": str(item["log"])})
    return {"rows": rows, "problems": problems}


def summarize(rows: list[dict], expect_episodes: int | None) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["setting"], r["task"])].append(r)
    table, problems = [], []
    for (setting, task), eps in sorted(groups.items(), key=lambda kv: (kv[0][0], TASKS.index(kv[0][1]) if kv[0][1] in TASKS else 999)):
        instr = sorted({e["instruction_type"] for e in eps})
        succ = sum(int(e["success"]) for e in eps)
        table.append({"setting": setting, "task": task, "attempts": len(eps), "successes": succ,
                      "success_rate": round(100.0 * succ / len(eps), 1), "instruction_type": ",".join(instr)})
        expected = EXPECTED_INSTRUCTION_TYPE.get(setting)
        if not setting:
            problems.append({"task": task, "level": "WARN", "detail": "unknown task_config -> setting"})
        elif expected and instr != [expected]:
            problems.append({"task": task, "level": "ERROR",
                             "detail": f"{setting} evaluated with instruction_type={instr}, expected {expected}"})
        if expect_episodes and len(eps) != expect_episodes:
            problems.append({"task": task, "level": "WARN", "detail": f"{setting}: {len(eps)} episodes, expected {expect_episodes}"})
        seeds = [e["seed"] for e in eps]
        if len(seeds) != len(set(seeds)):
            problems.append({"task": task, "level": "WARN", "detail": f"{setting}: duplicate seeds"})
    return table, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--run_id", default=None, help="default: run_dir name")
    ap.add_argument("--task_config", default=None, help="override task_config for every log")
    ap.add_argument("--expect_episodes", type=int, default=None)
    ap.add_argument("--out_dir", default=None, help="default: <run_dir>/parsed")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    run_id = args.run_id or run_dir.name
    parsed = parse_run(run_dir, run_id, args.task_config)
    rows = parsed["rows"]
    table, sum_problems = summarize(rows, args.expect_episodes)
    problems = parsed["problems"] + sum_problems
    out = Path(args.out_dir) if args.out_dir else run_dir / "parsed"
    write_csv(out / "episodes.csv", rows, ROW_FIELDS)

    per_setting = {}
    for setting in sorted({t["setting"] for t in table}):
        ts = [t for t in table if t["setting"] == setting]
        att = sum(t["attempts"] for t in ts)
        suc = sum(t["successes"] for t in ts)
        per_setting[setting] = {"tasks": len(ts), "attempts": att, "successes": suc,
                                "episode_rate": round(100.0 * suc / att, 2) if att else 0.0,
                                "task_mean_rate": round(sum(t["success_rate"] for t in ts) / len(ts), 2) if ts else 0.0}
    summary = {"time": now_iso(), "run_id": run_id, "run_dir": str(run_dir.resolve()), "episodes": len(rows),
               "per_setting": per_setting, "errors": sum(p["level"] == "ERROR" for p in problems),
               "warnings": sum(p["level"] == "WARN" for p in problems), "problems": problems}
    write_text(out / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    md = [f"# Eval run {run_id}\n", f"- parsed: {summary['time']}", f"- run_dir: `{summary['run_dir']}`",
          f"- episodes: {len(rows)}, errors: {summary['errors']}, warnings: {summary['warnings']}", ""]
    for setting, s in per_setting.items():
        md.append(f"- **{setting}**: {s['successes']}/{s['attempts']} = {s['episode_rate']}% "
                  f"(task mean {s['task_mean_rate']}%, {s['tasks']} tasks)")
    md += ["", md_table(["setting", "task", "attempts", "successes", "success_rate", "instruction_type"],
                        [[t[k] for k in ("setting", "task", "attempts", "successes", "success_rate", "instruction_type")]
                         for t in table]), ""]
    if problems:
        md += ["## Problems\n", md_table(["task", "level", "detail"], [[p["task"], p["level"], p["detail"]] for p in problems]), ""]
    write_text(out / "summary.md", "\n".join(md))

    for setting, s in per_setting.items():
        print(f"[eval-parse] {setting}: {s['successes']}/{s['attempts']} = {s['episode_rate']}% "
              f"(task mean {s['task_mean_rate']}%, {s['tasks']} tasks)")
    for p in problems:
        print(f"  {p['level']}: {p['task']}: {p['detail']}")
    print(f"[eval-parse] {len(rows)} episodes -> {out / 'episodes.csv'}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
