#!/usr/bin/env python3
"""Summarise LingBot training logs for the ledger.

usage: train_metrics.py <logs dir | log files...> [--bin 500] [--ckpt_dir DIR] [--csv PATH]

Reads every train_*.log (not *.stop.log) in order, keeps the later log when a step appears twice (a resumed run
overrides the steps it re-did), and prints: per-log timing, resume point, saves and exception types; per-epoch and
per-bin means of every logged metric; checkpoint states; a tail-trend verdict. Writes all per-step values to a CSV.
"""
import argparse
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

TS = r"(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d)"
STEP = re.compile(TS + r" - INFO - __main__ - Step (\d+)/(\d+), Epoch (\d+), (.*)$")
KV = re.compile(r"([A-Za-z_]+)\s+([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))s?(?=,|$)")
DCP = re.compile(TS + r" - INFO - __main__ - Distributed checkpoint saved at \S*global_step_(\d+) successfully")
HF = re.compile(TS + r" - INFO - __main__ - \[async_hf\] HF checkpoint finished for \S*global_step_(\d+) in ([0-9.]+)s")
LOAD = re.compile(r"Load distributed checkpoint from \S*global_step_(\d+) successfully")
SCRATCH = re.compile(r"Starting training from scratch")
REACHED = re.compile(TS + r" - INFO - __main__ - Reached max_steps=(\d+)")
EXC = re.compile(r"^(?:\[rank\d+\]:\s*)?(?:[A-Za-z_][\w.]*\.)?([A-Za-z_]\w*(?:Error|Exception|Interrupt))(?::|\s*$)")

LOSSES = ["Loss", "VLA_Loss", "Depth_Loss", "Future_Depth_Loss", "FutureVideo_Loss", "SeqWise_Loss", "RouterZ_Loss"]
OTHERS = ["MaxVio", "AvgSigmoid", "GradNorm", "LR", "Expert_LR", "StepTime", "Depth_Forward_Time", "Ignore_Batch_Num"]
KEYS = LOSSES + OTHERS
SHORT = {"Loss": "Loss", "VLA_Loss": "VLA", "Depth_Loss": "Depth", "Future_Depth_Loss": "FutD",
         "FutureVideo_Loss": "FutV", "SeqWise_Loss": "SeqW", "RouterZ_Loss": "RtrZ", "MaxVio": "MaxVio",
         "AvgSigmoid": "AvgSig", "GradNorm": "Grad", "LR": "LR", "StepTime": "StepT"}


def ts(s):
    return datetime.strptime(s, "%m/%d/%Y %H:%M:%S")


def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.median(xs) if xs else None


def fmt(v, key):
    if v is None:
        return "-"
    if key in ("LR", "Expert_LR"):
        return f"{v:.2e}"
    if key in ("StepTime", "Depth_Forward_Time", "GradNorm", "MaxVio", "AvgSigmoid"):
        return f"{v:.3f}"
    return f"{v:.4f}"


def parse_log(path):
    info = {"path": path, "steps": {}, "dcp": {}, "hf": {}, "load": None, "scratch": False, "reached": None,
            "exc": Counter(), "tracebacks": 0, "first_ts": None, "last_ts": None}
    with open(path, errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if "Traceback (most recent call last)" in line:
                info["tracebacks"] += 1
            m = EXC.match(line.strip())
            if m:
                info["exc"][m.group(1)] += 1
            m = STEP.search(line)
            if m:
                t = ts(m.group(1))
                rec = {"step": int(m.group(2)), "epoch": int(m.group(4)), "ts": t}
                for k, v in KV.findall(m.group(5)):
                    if k in KEYS:
                        rec[k] = float(v)
                info["steps"][rec["step"]] = rec
                info["first_ts"] = info["first_ts"] or t
                info["last_ts"] = t
                continue
            m = DCP.search(line)
            if m:
                info["dcp"][int(m.group(2))] = ts(m.group(1))
                info["last_ts"] = ts(m.group(1))
                continue
            m = HF.search(line)
            if m:
                info["hf"].setdefault(int(m.group(2)), []).append((ts(m.group(1)), float(m.group(3))))
                info["last_ts"] = ts(m.group(1))
                continue
            m = REACHED.search(line)
            if m:
                info["reached"] = (int(m.group(2)), ts(m.group(1)))
                continue
            m = LOAD.search(line)
            if m:
                info["load"] = int(m.group(1))
                continue
            if SCRATCH.search(line):
                info["scratch"] = True
    return info


def table(title, groups, keys):
    print(f"\n# {title}")
    head = f"{'group':>11} {'n':>5} " + " ".join(f"{SHORT.get(k, k):>8}" for k in keys)
    print(head)
    for label, rows in groups:
        cells = []
        for k in keys:
            vals = [r.get(k) for r in rows if not (k == "StepTime" and r.get("first_of_log"))]
            v = median(vals) if k == "StepTime" else mean(vals)
            cells.append(f"{fmt(v, k):>8}")
        print(f"{label:>11} {len(rows):5d} " + " ".join(cells))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--bin", type=int, default=500)
    ap.add_argument("--ckpt_dir")
    ap.add_argument("--csv")
    a = ap.parse_args(argv)

    logs = []
    for p in map(Path, a.paths):
        if p.is_dir():
            logs += sorted(x for x in p.glob("train_*.log") if not x.name.endswith(".stop.log"))
        else:
            logs.append(p)
    if not logs:
        sys.exit("no train_*.log found")
    infos = [parse_log(p) for p in logs]

    print("# logs (later log wins for repeated steps)")
    merged = {}
    for info in infos:
        st = info["steps"]
        if not st:
            print(f"{info['path'].name}: no step lines; tracebacks {info['tracebacks']}; "
                  f"exceptions {dict(info['exc']) or '-'}")
            continue
        lo, hi = min(st), max(st)
        for i, s in enumerate(sorted(st)):
            st[s]["first_of_log"] = i < 5
        wall = (info["last_ts"] - info["first_ts"]).total_seconds()
        med = median([r.get("StepTime") for r in st.values() if not r["first_of_log"]])
        start = f"resumed from {info['load']}" if info["load"] else ("from scratch" if info["scratch"] else "start ?")
        print(f"{info['path'].name}: steps {lo}-{hi} ({len(st)}), {start}, "
              f"{info['first_ts']:%m-%d %H:%M:%S} -> {info['last_ts']:%m-%d %H:%M:%S} "
              f"({wall / 3600:.2f} h, {wall / max(len(st), 1):.2f} s/step incl. saves), "
              f"StepTime median {fmt(med, 'StepTime')}s")
        if info["reached"]:
            print(f"  reached max_steps={info['reached'][0]} at {info['reached'][1]:%m-%d %H:%M:%S}")
        for s in sorted(set(info["dcp"]) | set(info["hf"])):
            t_step = st.get(s, {}).get("ts")
            dcp_s = (info["dcp"][s] - t_step).total_seconds() if s in info["dcp"] and t_step else None
            hf = ", ".join(f"{sec:.0f}s" for _, sec in info["hf"].get(s, [])) or "-"
            print(f"  save global_step_{s}: dcp {f'{dcp_s:.0f}s' if dcp_s is not None else '-'}, hf {hf}")
        print(f"  tracebacks {info['tracebacks']}; exception lines {dict(info['exc']) or '-'}")
        merged.update(st)

    rows = [merged[s] for s in sorted(merged)]
    if not rows:
        sys.exit("no step lines in any log")
    steps = [r["step"] for r in rows]
    gaps = [(a + 1, b - 1) for a, b in zip(steps, steps[1:]) if b - a > 1]
    nonfinite = [r["step"] for r in rows if any(k in r and not math.isfinite(r[k]) for k in LOSSES + ["GradNorm"])]
    print(f"\n# merged: steps {steps[0]}-{steps[-1]} ({len(rows)}), gaps {gaps or 'none'}, "
          f"non-finite steps {len(nonfinite)}{' first ' + str(nonfinite[:5]) if nonfinite else ''}, "
          f"Ignore_Batch_Num total {int(sum(r.get('Ignore_Batch_Num', 0) for r in rows))}")
    print(f"LR {fmt(rows[0].get('LR'), 'LR')} -> {fmt(rows[-1].get('LR'), 'LR')}; "
          f"Expert_LR {fmt(rows[0].get('Expert_LR'), 'LR')} -> {fmt(rows[-1].get('Expert_LR'), 'LR')}")

    main_keys = LOSSES[:5] + ["GradNorm", "MaxVio", "LR", "StepTime"]
    epochs = sorted({r["epoch"] for r in rows})
    table("by epoch (mean; StepTime median)",
          [(f"ep{e}", [r for r in rows if r["epoch"] == e]) for e in epochs], main_keys)
    bins = []
    hi = a.bin
    while hi - a.bin < steps[-1]:
        g = [r for r in rows if hi - a.bin < r["step"] <= hi]
        if g:
            bins.append((f"<={g[-1]['step']}", g))
        hi += a.bin
    table(f"every {a.bin} steps (mean; StepTime median)", bins, main_keys)
    table("small terms by epoch", [(f"ep{e}", [r for r in rows if r["epoch"] == e]) for e in epochs],
          ["SeqWise_Loss", "RouterZ_Loss", "AvgSigmoid", "Depth_Forward_Time"])

    def seg(lo, hi, k="VLA_Loss"):
        return mean([r.get(k) for r in rows if lo < r["step"] <= hi])

    end = steps[-1]
    print("\n# trend")
    for k in ["Loss", "VLA_Loss", "Depth_Loss", "Future_Depth_Loss", "FutureVideo_Loss"]:
        first, last = seg(0, 100, k), seg(end - 100, end, k)
        tail = [seg(end - 1500, end - 1000, k), seg(end - 1000, end - 500, k), seg(end - 500, end, k)]
        if None in tail or first is None or last is None:
            continue
        rel = (tail[2] - tail[0]) / tail[0]
        verdict = "falling" if rel < -0.03 else ("rising" if rel > 0.03 else "flat (within 3%)")
        print(f"{SHORT[k]:>6}: first100 {first:.4f} -> last100 {last:.4f} ({(last - first) / first:+.1%}); "
              f"last 1500 in 3 blocks {tail[0]:.4f} {tail[1]:.4f} {tail[2]:.4f} ({rel:+.1%}) {verdict}")

    ckpt_dir = Path(a.ckpt_dir) if a.ckpt_dir else logs[0].parent.parent / "checkpoints"
    if ckpt_dir.is_dir():
        print(f"\n# checkpoints in {ckpt_dir}")
        for c in sorted(ckpt_dir.glob("global_step_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])):
            es = len(list((c / "extra_state").glob("*"))) if (c / "extra_state").is_dir() else 0
            hf = c / "hf_ckpt"
            size = sum(f.stat().st_size for f in hf.rglob("*") if f.is_file()) if hf.is_dir() else 0
            resume = "resume parts kept" if (c / "model").is_dir() and (c / "optimizer").is_dir() else "hf only"
            tmp = len(list(c.glob(".hf_ckpt.tmp.*")))
            print(f"  {c.name}: extra_state={es} hf_ckpt={'yes' if hf.is_dir() else 'no'} "
                  f"({size / 2**30:.1f} GiB) {resume}{f' tmp={tmp}' if tmp else ''}")

    out = Path(a.csv) if a.csv else logs[0].parent / "metrics_steps.csv"
    with open(out, "w") as f:
        f.write("step,epoch,time," + ",".join(KEYS) + "\n")
        for r in rows:
            f.write(f"{r['step']},{r['epoch']},{r['ts']:%Y-%m-%d %H:%M:%S}," +
                    ",".join("" if r.get(k) is None else repr(r[k]) for k in KEYS) + "\n")
    print(f"\ncsv {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
