#!/usr/bin/env python3
"""Verify downloaded archives against a manifest produced by hf_manifest.py.

Checks existence, byte size and sha256 (LFS oid). Exit code 0 only if every file is OK.

Example:
  python verify_downloads.py --manifest ../data/manifests/clean50.json --download_dir ../data/hf
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lingbot_tools.io_utils import md_table, now_iso, sha256_file, write_text  # noqa: E402


def check_file(entry: dict, download_dir: Path, size_only: bool) -> dict:
    path = download_dir / entry["path"]
    result = {"task": entry["task"], "path": str(path), "rel_path": entry["path"], "expected_size": entry["size"],
              "expected_sha256": entry.get("sha256"), "status": "OK", "local_size": None}
    if not path.is_file():
        result["status"] = "MISSING"
        return result
    result["local_size"] = path.stat().st_size
    if result["local_size"] != entry["size"]:
        result["status"] = "SIZE_MISMATCH"
        return result
    if not size_only and entry.get("sha256"):
        local_sha = sha256_file(path)
        result["local_sha256"] = local_sha
        if local_sha != entry["sha256"]:
            result["status"] = "SHA_MISMATCH"
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--download_dir", required=True, help="--local-dir used for hf download")
    ap.add_argument("--size_only", action="store_true", help="skip sha256 (fast pre-check)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--report", default=None, help="optional markdown report path")
    args = ap.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    download_dir = Path(args.download_dir)
    entries = manifest["files"]
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(lambda e: check_file(e, download_dir, args.size_only), entries))

    bad = [r for r in results if r["status"] != "OK"]
    for r in results:
        print(f"  {r['status']:<14} {r['task']}")
    print(f"[verify] {len(results) - len(bad)}/{len(results)} OK "
          f"(revision {manifest.get('revision', '?')[:12]}, sha256 {'skipped' if args.size_only else 'checked'})")
    if bad:
        include = " ".join(f"--include '{r['rel_path']}'" for r in bad)
        print("[verify] re-download failed files:")
        print(f"  hf download {manifest['repo']} --repo-type {manifest.get('repo_type', 'dataset')} "
              f"--revision {manifest['revision']} {include} --local-dir {download_dir}")

    if args.report:
        rows = [(r["task"], r["status"], r["expected_size"], r["local_size"]) for r in results]
        text = (f"# Download verification\n\n- time: {now_iso()}\n- manifest: {args.manifest}\n"
                f"- revision: {manifest.get('revision')}\n- archive: {manifest.get('archive')}\n"
                f"- sha256: {'skipped' if args.size_only else 'checked'}\n"
                f"- result: {len(results) - len(bad)}/{len(results)} OK\n\n"
                + md_table(["task", "status", "expected_size", "local_size"], rows) + "\n")
        write_text(args.report, text)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
