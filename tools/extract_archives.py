#!/usr/bin/env python3
"""Extract verified per-task archives into the RoboTwin raw-data layout.

<download_dir>/dataset/<task>/<archive>  ->  <raw_root>/<task>/<top-level dir inside zip>/...
For aloha-agilex_clean_50.zip this yields <raw_root>/<task>/aloha-agilex_clean_50/{data,instructions,...},
which is exactly what RoboTwin policy/pi0/scripts/process_data.py expects under data/<task>/<setting>/.

Already-extracted tasks (marker file present) are skipped. Member paths are validated
against path traversal before extraction.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lingbot_tools.io_utils import now_iso  # noqa: E402

MARKER = ".extracted.json"


def safe_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in zf.infolist():
        p = PurePosixPath(info.filename)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe path in archive: {info.filename}")
        members.append(info)
    return members


def top_level_dirs(members: list[zipfile.ZipInfo]) -> set[str]:
    return {PurePosixPath(m.filename).parts[0] for m in members if PurePosixPath(m.filename).parts}


def extract_one(archive: Path, dest: Path, expect_top: str | None, force: bool) -> dict:
    marker = dest / MARKER
    if marker.exists() and not force:
        return {"archive": str(archive), "status": "SKIPPED", "dest": str(dest)}
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = safe_members(zf)
        tops = top_level_dirs(members)
        if expect_top and tops != {expect_top}:
            return {"archive": str(archive), "status": "UNEXPECTED_LAYOUT", "dest": str(dest),
                    "top_level": sorted(tops)}
        zf.extractall(dest, members=members)
    record = {"archive": str(archive), "status": "EXTRACTED", "dest": str(dest),
              "top_level": sorted(tops), "members": len(members), "time": now_iso()}
    marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--download_dir", required=True)
    ap.add_argument("--raw_root", required=True, help="destination root, one sub-directory per task")
    ap.add_argument("--expect_top", default=None,
                    help="required single top-level dir inside each zip (default: archive stem)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    expect_top = args.expect_top or Path(manifest["archive"]).stem
    failures = 0
    for entry in manifest["files"]:
        archive = Path(args.download_dir) / entry["path"]
        if not archive.is_file():
            print(f"  MISSING            {entry['task']}")
            failures += 1
            continue
        try:
            rec = extract_one(archive, Path(args.raw_root) / entry["task"], expect_top, args.force)
        except (zipfile.BadZipFile, ValueError, OSError) as exc:
            rec = {"status": "FAILED", "error": str(exc)}
        detail = f"  top={rec.get('top_level')}" if rec["status"] == "UNEXPECTED_LAYOUT" else ""
        detail += f"  {rec['error']}" if rec.get("error") else ""
        print(f"  {rec['status']:<18} {entry['task']}{detail}")
        failures += rec["status"] not in ("EXTRACTED", "SKIPPED")
    print(f"[extract] done, failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
