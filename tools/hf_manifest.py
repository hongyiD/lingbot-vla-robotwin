#!/usr/bin/env python3
"""Build a pinned download manifest (path, size, LFS sha256) for RoboTwin 2.0 archives,
or for every file of a model repo (to check weights that someone else downloaded).

Reads metadata only from the Hugging Face API; nothing is downloaded.

Example:
  python hf_manifest.py --archive aloha-agilex_clean_50.zip --out ../data/manifests/clean50.json
  python hf_manifest.py --archive demo_clean.zip --revision a967b852... --tasks click_bell --out m.json
  python hf_manifest.py --repo_type model --repo Ruicheng/moge-2-vitb-normal --revision ca5f0e07... \\
      --exclude .gitattributes --exclude '*.md' --out ../data/manifests/moge-2-vitb-normal@ca5f0e07.json
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lingbot_tools.io_utils import now_iso  # noqa: E402
from lingbot_tools.tasks import CLEAN_ARCHIVE, resolve_tasks  # noqa: E402

DEFAULT_REPO = "TianxingChen/RoboTwin2.0"
DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def _get_json(url: str, token: str | None, timeout: float) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "lingbot-tools/1.0"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_revision(endpoint: str, repo: str, revision: str, token: str | None, timeout: float) -> str:
    url = f"{endpoint}/api/datasets/{repo}/revision/{urllib.parse.quote(revision, safe='')}"
    info = _get_json(url, token, timeout)
    sha = info.get("sha") if isinstance(info, dict) else None
    if not sha:
        raise RuntimeError(f"cannot resolve revision {revision!r} via {url}")
    return sha


def list_task_dir(endpoint: str, repo: str, revision: str, task: str, token: str | None,
                  timeout: float) -> list[dict]:
    url = f"{endpoint}/api/datasets/{repo}/tree/{revision}/dataset/{task}"
    entries = _get_json(url, token, timeout)
    if not isinstance(entries, list):
        raise RuntimeError(f"unexpected tree response for {task}: {entries!r}")
    return entries


def build_manifest(repo: str, revision: str, archive: str, tasks: list[str], endpoint: str,
                   token: str | None, timeout: float) -> dict:
    sha = resolve_revision(endpoint, repo, revision, token, timeout)
    files, missing = [], []
    for task in tasks:
        path = f"dataset/{task}/{archive}"
        entry = next((e for e in list_task_dir(endpoint, repo, sha, task, token, timeout)
                      if e.get("path") == path), None)
        if entry is None:
            missing.append(task)
            continue
        lfs = entry.get("lfs") or {}
        files.append({
            "task": task,
            "path": path,
            "size": int(lfs.get("size") or entry.get("size") or 0),
            "sha256": lfs.get("oid"),
        })
        print(f"  {task:<28} {files[-1]['size'] / 1e9:7.2f} GB  {str(files[-1]['sha256'])[:12]}")
    return {
        "repo": repo,
        "repo_type": "dataset",
        "endpoint": endpoint,
        "revision_requested": revision,
        "revision": sha,
        "archive": archive,
        "created_at": now_iso(),
        "total_bytes": sum(f["size"] for f in files),
        "missing_tasks": missing,
        "files": files,
    }


def build_model_manifest(repo: str, revision: str, endpoint: str, token: str | None, timeout: float,
                         exclude: list[str]) -> dict:
    url = f"{endpoint}/api/models/{repo}/revision/{urllib.parse.quote(revision, safe='')}?blobs=true"
    info = _get_json(url, token, timeout)
    sha = info.get("sha") if isinstance(info, dict) else None
    if not sha:
        raise RuntimeError(f"cannot resolve revision {revision!r} via {url}")
    files = []
    for sibling in sorted(info.get("siblings") or [], key=lambda s: s["rfilename"]):
        path = sibling["rfilename"]
        if any(fnmatch.fnmatch(path, pattern) for pattern in exclude):
            continue
        lfs = sibling.get("lfs") or {}
        files.append({
            "task": path,
            "path": path,
            "size": int(lfs.get("size") or sibling.get("size") or 0),
            "sha256": lfs.get("sha256"),  # small non-LFS files: size check only
        })
        print(f"  {path:<55} {files[-1]['size'] / 1e9:7.2f} GB  {str(files[-1]['sha256'] or '-')[:12]}")
    return {
        "repo": repo,
        "repo_type": "model",
        "endpoint": endpoint,
        "revision_requested": revision,
        "revision": sha,
        "created_at": now_iso(),
        "total_bytes": sum(f["size"] for f in files),
        "missing_tasks": [],
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--repo_type", choices=("dataset", "model"), default="dataset",
                    help="model: list every file of --repo; --archive and --tasks are ignored")
    ap.add_argument("--revision", default="main", help="branch, tag or commit; resolved to a commit sha")
    ap.add_argument("--archive", default=CLEAN_ARCHIVE)
    ap.add_argument("--tasks", default="all", help='"all" or comma-separated task names')
    ap.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                    help="model only: skip paths matching this glob, e.g. '*.md' (repeatable)")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="e.g. https://hf-mirror.com")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.repo_type == "model" and args.repo == DEFAULT_REPO:
        ap.error("--repo_type model needs --repo <org/name>")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if args.repo_type == "model":
        print(f"[manifest] {args.repo}@{args.revision} (model repo, all files)")
        manifest = build_model_manifest(args.repo, args.revision, args.endpoint.rstrip("/"), token,
                                        args.timeout, args.exclude)
    else:
        tasks = resolve_tasks(args.tasks)
        print(f"[manifest] {args.repo}@{args.revision} archive={args.archive} tasks={len(tasks)}")
        manifest = build_manifest(args.repo, args.revision, args.archive, tasks,
                                  args.endpoint.rstrip("/"), token, args.timeout)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[manifest] revision sha = {manifest['revision']}")
    print(f"[manifest] files = {len(manifest['files'])}, total = {manifest['total_bytes'] / 1e9:.2f} GB")
    if args.repo_type == "model":
        print(f"[manifest] written to {out}")
        print("[manifest] download (pinned):")
        print(f"  hf download {args.repo} --revision {manifest['revision']} --local-dir <DOWNLOAD_DIR>")
        return 0
    if manifest["missing_tasks"]:
        print(f"[manifest] MISSING archive for: {', '.join(manifest['missing_tasks'])}")
    print(f"[manifest] written to {out}")
    print("[manifest] download (pinned):")
    print(f"  hf download {args.repo} --repo-type dataset --revision {manifest['revision']} \\")
    print(f"    --include 'dataset/*/{args.archive}' --local-dir <DOWNLOAD_DIR>")
    return 1 if manifest["missing_tasks"] else 0


if __name__ == "__main__":
    sys.exit(main())
