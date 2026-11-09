#!/usr/bin/env python3
"""Paired A/B decision between two evaluation runs (baseline vs candidate).

Pairing: episodes are matched on (setting, task, seed); both runs must use the same eval code
so that seeds coincide (fallback: episode_idx with --pair_on episode_idx).

Statistics:
  - McNemar exact test on paired binary outcomes (primary; pooled and per setting).
  - Wilcoxon signed-rank gate on per-(setting, task) success rates, same semantics as
    agenticx.rl.acceptance.wilcoxon_gate (zero diffs dropped, <2 non-zero pairs -> reject).
  - Bootstrap 95% CI of the mean per-task difference (percentage points).

Adoption rule (plan section 7):
  ADOPT        pooled McNemar p < --alpha_mcnemar and diff > 0, no setting drops more than
               --max_setting_drop points, and the Wilcoxon gate accepts at --alpha_wilcoxon.
  REJECT       pooled diff <= 0 or a setting drops more than --max_setting_drop points.
  INCONCLUSIVE otherwise (collect more episodes or keep baseline).

Example:
  python ab_test.py --baseline ../evals/runA/episodes.csv --candidate ../evals/runA_k8/episodes.csv \
      --out_md ../evals/ab_runA_vs_k8.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lingbot_tools.io_utils import md_table, now_iso, read_csv, write_text  # noqa: E402
from lingbot_tools.stats import bootstrap_mean_ci, mcnemar_exact, wilcoxon_gate  # noqa: E402


def pair_episodes(base_rows: list[dict], cand_rows: list[dict], pair_on: str) -> tuple[list[dict], dict]:
    def index(rows):
        idx = {}
        dup = 0
        for r in rows:
            key = (r["setting"], r["task"], str(r[pair_on]))
            dup += key in idx
            idx[key] = int(r["success"])
        return idx, dup

    bi, bdup = index(base_rows)
    ci, cdup = index(cand_rows)
    keys = sorted(set(bi) & set(ci))
    pairs = [{"setting": k[0], "task": k[1], "key": k[2], "base": bi[k], "cand": ci[k]} for k in keys]
    info = {"baseline_episodes": len(base_rows), "candidate_episodes": len(cand_rows), "paired": len(pairs),
            "baseline_unpaired": len(set(bi) - set(ci)), "candidate_unpaired": len(set(ci) - set(bi)),
            "baseline_duplicate_keys": bdup, "candidate_duplicate_keys": cdup}
    return pairs, info


def mcnemar_block(pairs: list[dict]) -> dict:
    b = sum(1 for p in pairs if p["base"] == 0 and p["cand"] == 1)
    c = sum(1 for p in pairs if p["base"] == 1 and p["cand"] == 0)
    n = len(pairs)
    base_rate = 100.0 * sum(p["base"] for p in pairs) / n if n else 0.0
    cand_rate = 100.0 * sum(p["cand"] for p in pairs) / n if n else 0.0
    return {"n": n, "baseline_rate": round(base_rate, 2), "candidate_rate": round(cand_rate, 2),
            "diff_pp": round(cand_rate - base_rate, 2), "b_fail_to_success": b, "c_success_to_fail": c,
            "p_value": mcnemar_exact(b, c)}


def task_rates(pairs: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for p in pairs:
        groups[(p["setting"], p["task"])].append(p)
    out = []
    for (setting, task), ps in sorted(groups.items()):
        n = len(ps)
        br = 100.0 * sum(p["base"] for p in ps) / n
        cr = 100.0 * sum(p["cand"] for p in ps) / n
        out.append({"setting": setting, "task": task, "n": n, "base": br, "cand": cr, "diff": cr - br})
    return out


def analyse(pairs: list[dict], *, alpha_mcnemar: float, alpha_wilcoxon: float, max_setting_drop: float,
            n_boot: int) -> dict:
    pooled = mcnemar_block(pairs)
    settings = {s: mcnemar_block([p for p in pairs if p["setting"] == s]) for s in sorted({p["setting"] for p in pairs})}
    units = task_rates(pairs)
    gate = wilcoxon_gate([u["base"] for u in units], [u["cand"] for u in units], alpha=alpha_wilcoxon) if units else None
    ci = bootstrap_mean_ci([u["diff"] for u in units], n_boot=n_boot) if units else (float("nan"), float("nan"))

    drops = {s: v["diff_pp"] for s, v in settings.items() if v["diff_pp"] < -max_setting_drop}
    if pooled["diff_pp"] <= 0 or drops:
        decision = "REJECT"
    elif pooled["p_value"] < alpha_mcnemar and gate is not None and gate.accepted:
        decision = "ADOPT"
    else:
        decision = "INCONCLUSIVE"
    reasons = []
    if drops:
        reasons.append(f"setting drop beyond {max_setting_drop}pp: {drops}")
    if pooled["diff_pp"] <= 0:
        reasons.append("pooled difference is not positive")
    if pooled["p_value"] >= alpha_mcnemar:
        reasons.append(f"McNemar p={pooled['p_value']:.4f} >= {alpha_mcnemar}")
    if gate is not None and not gate.accepted:
        reasons.append(f"Wilcoxon gate: {gate.reason} (p={gate.p_value:.4f})")
    return {"pooled": pooled, "per_setting": settings, "wilcoxon_gate": gate.to_json() if gate else None,
            "task_diff_mean_pp": round(sum(u["diff"] for u in units) / len(units), 2) if units else None,
            "task_diff_ci95_pp": [round(ci[0], 2), round(ci[1], 2)], "decision": decision, "reasons": reasons,
            "units": units}


def render_md(res: dict, info: dict, args) -> str:
    p = res["pooled"]
    lines = [f"# A/B: {Path(args.candidate).parent.name} vs {Path(args.baseline).parent.name}\n",
             f"- time: {now_iso()}", f"- baseline: `{args.baseline}`", f"- candidate: `{args.candidate}`",
             f"- pairing on ({args.pair_on}): {info['paired']} paired, unpaired base/cand = "
             f"{info['baseline_unpaired']}/{info['candidate_unpaired']}",
             f"- **decision: {res['decision']}**" + (f" ({'; '.join(res['reasons'])})" if res["reasons"] else ""), "",
             "## McNemar (paired episodes)\n",
             md_table(["scope", "n", "baseline %", "candidate %", "diff pp", "fail->success", "success->fail", "p"],
                      [["pooled", p["n"], p["baseline_rate"], p["candidate_rate"], p["diff_pp"], p["b_fail_to_success"],
                        p["c_success_to_fail"], f"{p['p_value']:.4g}"]]
                      + [[s, v["n"], v["baseline_rate"], v["candidate_rate"], v["diff_pp"], v["b_fail_to_success"],
                          v["c_success_to_fail"], f"{v['p_value']:.4g}"] for s, v in res["per_setting"].items()]), ""]
    g = res["wilcoxon_gate"]
    if g:
        lines += ["## Per-task Wilcoxon gate\n",
                  f"- units: {len(res['units'])} (setting x task), non-zero pairs: {g['n_pairs']}",
                  f"- statistic={g['statistic']:.3f}, p={g['p_value']:.4g}, accepted={g['accepted']} ({g['reason']})",
                  f"- mean task diff = {res['task_diff_mean_pp']} pp, bootstrap 95% CI = {res['task_diff_ci95_pp']}", ""]
    movers = sorted(res["units"], key=lambda u: u["diff"])
    worst, best = movers[:8], movers[-8:][::-1]
    fmt = lambda us: [[u["setting"], u["task"], u["n"], round(u["base"], 1), round(u["cand"], 1), round(u["diff"], 1)] for u in us]
    lines += ["## Largest gains\n", md_table(["setting", "task", "n", "base %", "cand %", "diff"], fmt(best)), "",
              "## Largest regressions\n", md_table(["setting", "task", "n", "base %", "cand %", "diff"], fmt(worst)), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--pair_on", choices=["seed", "episode_idx"], default="seed")
    ap.add_argument("--alpha_mcnemar", type=float, default=0.05)
    ap.add_argument("--alpha_wilcoxon", type=float, default=0.10)
    ap.add_argument("--max_setting_drop", type=float, default=1.0, help="percentage points")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args(argv)

    pairs, info = pair_episodes(read_csv(args.baseline), read_csv(args.candidate), args.pair_on)
    if not pairs:
        print("[ab] no paired episodes; check that both runs used the same eval code / seeds")
        return 2
    res = analyse(pairs, alpha_mcnemar=args.alpha_mcnemar, alpha_wilcoxon=args.alpha_wilcoxon,
                  max_setting_drop=args.max_setting_drop, n_boot=args.n_boot)
    p = res["pooled"]
    print(f"[ab] paired={info['paired']} (unpaired base/cand {info['baseline_unpaired']}/{info['candidate_unpaired']})")
    print(f"[ab] pooled: {p['baseline_rate']}% -> {p['candidate_rate']}% ({p['diff_pp']:+.2f}pp), "
          f"McNemar b={p['b_fail_to_success']} c={p['c_success_to_fail']} p={p['p_value']:.4g}")
    for s, v in res["per_setting"].items():
        print(f"[ab] {s}: {v['baseline_rate']}% -> {v['candidate_rate']}% ({v['diff_pp']:+.2f}pp) p={v['p_value']:.4g}")
    if res["wilcoxon_gate"]:
        g = res["wilcoxon_gate"]
        print(f"[ab] wilcoxon gate: {g['reason']} p={g['p_value']:.4g}; task diff CI95 {res['task_diff_ci95_pp']}")
    print(f"[ab] DECISION: {res['decision']}" + (f" ({'; '.join(res['reasons'])})" if res["reasons"] else ""))
    if args.out_md:
        write_text(args.out_md, render_md(res, info, args))
    if args.out_json:
        payload = {k: v for k, v in res.items() if k != "units"} | {"pairing": info}
        write_text(args.out_json, json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
