#!/usr/bin/env python3
"""Append-only experiment ledger (ledger.jsonl) with a rendered ledger.md.

Every training run, evaluation, A/B decision, data decision and submission gets one entry,
optionally with sha256 of the files that define it (configs, lists, norm stats, results JSON).

Examples:
  python ledger.py add --kind data --title "clean50 downloaded" --field revision=3dc3b79 --hash ../data/manifests/clean50.json
  python ledger.py add --kind eval --title "runA step10k 50x10" --field clean=61.2 --field randomized=40.4
  python ledger.py add --kind submit --title "runA 100x50" --hash ../submissions/2026-10-25_runA.json
  python ledger.py show --last 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lingbot_tools.io_utils import EMBODIEDX_ROOT, append_jsonl, md_table, now_iso, read_jsonl, sha256_file, write_text  # noqa: E402

KINDS = ("data", "run", "eval", "ab", "submit", "decision", "note")
DEFAULT_DIR = Path(os.environ.get("LINGBOT_LEDGER_DIR", EMBODIEDX_ROOT / "ledger"))


def load(ledger_dir: Path) -> list[dict]:
    return list(read_jsonl(ledger_dir / "ledger.jsonl"))


def render(ledger_dir: Path) -> Path:
    entries = load(ledger_dir)
    rows = []
    for e in entries:
        fields = ", ".join(f"{k}={v}" for k, v in e.get("fields", {}).items())
        hashes = ", ".join(f"{Path(p).name}:{h[:10]}" for p, h in e.get("hashes", {}).items())
        rows.append([e["id"], e["time"], e["kind"], e["title"], fields, hashes])
    submits = [e for e in entries if e["kind"] == "submit"]
    text = (f"# Experiment ledger\n\n- entries: {len(entries)}, submissions: {len(submits)}\n"
            f"- rendered: {now_iso()} (source of truth: ledger.jsonl)\n\n"
            + md_table(["id", "time", "kind", "title", "fields", "hashes"], rows) + "\n")
    path = ledger_dir / "ledger.md"
    write_text(path, text)
    return path


def add(ledger_dir: Path, kind: str, title: str, fields: list[str], hashes: list[str]) -> dict:
    parsed = {}
    for item in fields:
        if "=" not in item:
            raise ValueError(f"--field expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        parsed[k.strip()] = v.strip()
    digests = {}
    for path in hashes:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"--hash file not found: {path}")
        digests[str(p.resolve())] = sha256_file(p)
    entries = load(ledger_dir)
    entry = {"id": (entries[-1]["id"] + 1) if entries else 1, "time": now_iso(), "kind": kind, "title": title,
             "fields": parsed, "hashes": digests}
    append_jsonl(ledger_dir / "ledger.jsonl", entry)
    render(ledger_dir)
    return entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger_dir", default=str(DEFAULT_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--kind", choices=KINDS, required=True)
    a.add_argument("--title", required=True)
    a.add_argument("--field", action="append", default=[], help="key=value, repeatable")
    a.add_argument("--hash", action="append", default=[], help="file to fingerprint, repeatable")
    s = sub.add_parser("show")
    s.add_argument("--last", type=int, default=20)
    s.add_argument("--kind", choices=KINDS, default=None)
    sub.add_parser("render")
    args = ap.parse_args(argv)

    ledger_dir = Path(args.ledger_dir)
    if args.cmd == "add":
        entry = add(ledger_dir, args.kind, args.title, args.field, args.hash)
        print(f"[ledger] #{entry['id']} {entry['kind']}: {entry['title']}")
    elif args.cmd == "show":
        entries = [e for e in load(ledger_dir) if args.kind is None or e["kind"] == args.kind]
        for e in entries[-args.last:]:
            fields = " ".join(f"{k}={v}" for k, v in e.get("fields", {}).items())
            print(f"#{e['id']:<4} {e['time']}  {e['kind']:<8} {e['title']}  {fields}")
    else:
        print(f"[ledger] rendered {render(ledger_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
