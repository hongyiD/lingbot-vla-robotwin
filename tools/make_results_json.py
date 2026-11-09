#!/usr/bin/env python3
"""Fill the official result template JSON from parsed per-episode CSVs.

Inputs are episodes.csv files from parse_eval_run.py (one or more runs; each (setting, task)
must come from exactly one run). Key order, schema_version and task keys follow
templates/prelim_results_template.json. Refuses to write when a setting was evaluated with the
wrong instruction pool (clean -> seen, randomized -> unseen) unless --allow_instruction_mismatch.

This tool only writes a local file; uploading to Tianchi is a manual step.

Example:
  python make_results_json.py --episodes ../evals/runA_clean/episodes.csv ../evals/runA_rand/episodes.csv \
      --team_id 123456 --out ../submissions/2026-10-25_runA.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lingbot_tools.io_utils import EMBODIEDX_ROOT, read_csv, write_text  # noqa: E402
from lingbot_tools.tasks import EXPECTED_INSTRUCTION_TYPE  # noqa: E402

DEFAULT_TEMPLATE = EMBODIEDX_ROOT / "templates" / "prelim_results_template.json"


def build_results(template: dict, rows: list[dict], *, allow_instruction_mismatch: bool) -> tuple[dict, list[str], list[str]]:
    errors, warnings = [], []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["setting"], r["task"])].append(r)

    for (setting, task), eps in groups.items():
        if setting not in template["results"]:
            errors.append(f"unknown setting {setting!r} (task {task})")
            continue
        if task not in template["results"][setting]:
            errors.append(f"unknown task {task!r} in setting {setting}")
            continue
        runs = sorted({e["run_id"] for e in eps})
        if len(runs) > 1:
            errors.append(f"{setting}/{task}: episodes from multiple runs {runs}; pass one run per task")
        instr = sorted({e.get("instruction_type", "") for e in eps})
        expected = EXPECTED_INSTRUCTION_TYPE[setting]
        if instr != [expected]:
            msg = f"{setting}/{task}: instruction_type={instr}, expected {expected}"
            (warnings if allow_instruction_mismatch else errors).append(msg)
        seeds = [e["seed"] for e in eps]
        if len(seeds) != len(set(seeds)):
            errors.append(f"{setting}/{task}: duplicate seeds")

    out = {"schema_version": template["schema_version"], "team_id": template["team_id"], "results": {}}
    for setting, tasks in template["results"].items():
        out["results"][setting] = {}
        for task in tasks:
            eps = groups.get((setting, task), [])
            if not eps:
                warnings.append(f"{setting}/{task}: no episodes (attempts=0)")
            out["results"][setting][task] = {"attempts": len(eps),
                                             "successes": sum(int(e["success"]) for e in eps)}
    return out, errors, warnings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument("--team_id", required=True)
    ap.add_argument("--min_attempts", type=int, default=50, help="warn below this many attempts per task")
    ap.add_argument("--allow_instruction_mismatch", action="store_true")
    ap.add_argument("--allow_missing", action="store_true", help="allow tasks with zero attempts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    rows = [r for path in args.episodes for r in read_csv(path)]
    result, errors, warnings = build_results(template, rows, allow_instruction_mismatch=args.allow_instruction_mismatch)
    result["team_id"] = args.team_id

    missing = [w for w in warnings if "attempts=0" in w]
    if missing and not args.allow_missing:
        errors.append(f"{len(missing)} (setting, task) pairs have no episodes; use --allow_missing for drafts")
    for setting, tasks in result["results"].items():
        low = [t for t, v in tasks.items() if 0 < v["attempts"] < args.min_attempts]
        if low:
            warnings.append(f"{setting}: {len(low)} task(s) below {args.min_attempts} attempts, e.g. {low[0]}")

    for w in warnings:
        print(f"  WARN: {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    if errors:
        print("[results-json] not written")
        return 1

    write_text(args.out, json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for setting, tasks in result["results"].items():
        att = sum(v["attempts"] for v in tasks.values())
        suc = sum(v["successes"] for v in tasks.values())
        rates = [v["successes"] / v["attempts"] for v in tasks.values() if v["attempts"]]
        task_mean = 100.0 * sum(rates) / len(rates) if rates else 0.0
        print(f"[results-json] {setting}: {suc}/{att} episodes, task-mean {task_mean:.2f}%")
    print(f"[results-json] written to {args.out} (upload manually)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
