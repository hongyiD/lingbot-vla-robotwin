#!/usr/bin/env python3
"""Daily progress report from the ledger, optionally pushed to a DingTalk / Feishu custom bot.

Content: ledger entries of the last N hours, latest eval numbers, submission count, and the
tail of a training log. Report text only; it never triggers training, evaluation or submission.

Webhook settings come from flags or env: LINGBOT_WEBHOOK_KIND (dingtalk|feishu),
LINGBOT_WEBHOOK_URL, LINGBOT_WEBHOOK_SECRET (signing secret, optional).

Examples:
  python daily_report.py --out ../reports/daily/$(date +%F).md
  python daily_report.py --training_log /path/train.log --webhook_kind feishu --dry_run
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import DEFAULT_DIR, load  # noqa: E402
from lingbot_tools.io_utils import strip_ansi, write_text  # noqa: E402


def build_report(entries: list[dict], since_hours: float, training_log: Path | None, tail: int,
                 now: dt.datetime | None = None) -> tuple[str, str]:
    now = now or dt.datetime.now().astimezone()
    cutoff = now - dt.timedelta(hours=since_hours)
    recent = [e for e in entries if dt.datetime.fromisoformat(e["time"]) >= cutoff]
    evals = [e for e in entries if e["kind"] == "eval"]
    submits = [e for e in entries if e["kind"] == "submit"]
    title = f"灵波初赛日报 {now:%Y-%m-%d}"
    lines = [f"### {title}", "", f"**最近 {since_hours:g} 小时记录 ({len(recent)})**"]
    lines += [f"- #{e['id']} [{e['kind']}] {e['title']}" for e in recent] or ["- 无"]
    lines += ["", "**最近一次评测**"]
    if evals:
        e = evals[-1]
        fields = ", ".join(f"{k}={v}" for k, v in e.get("fields", {}).items())
        lines.append(f"- #{e['id']} {e['title']} ({fields})")
    else:
        lines.append("- 暂无")
    lines += ["", f"**已提交次数**: {len(submits)}"]
    if training_log is not None:
        lines += ["", f"**训练日志尾部** (`{training_log.name}`)"]
        if training_log.is_file():
            tail_lines = [strip_ansi(x).rstrip() for x in training_log.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
            lines += ["```", *tail_lines[-tail:], "```"]
        else:
            lines.append("- 日志不存在")
    return title, "\n".join(lines) + "\n"


def dingtalk_signed_url(url: str, secret: str, ts_ms: int) -> str:
    string_to_sign = f"{ts_ms}\n{secret}".encode("utf-8")
    sign = base64.b64encode(hmac.new(secret.encode("utf-8"), string_to_sign, hashlib.sha256).digest()).decode()
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}timestamp={ts_ms}&sign={urllib.parse.quote_plus(sign)}"


def feishu_sign(secret: str, ts_s: int) -> str:
    key = f"{ts_s}\n{secret}".encode("utf-8")
    return base64.b64encode(hmac.new(key, b"", hashlib.sha256).digest()).decode()


def build_request(kind: str, url: str, secret: str | None, title: str, text: str, now_s: float) -> tuple[str, dict]:
    if kind == "dingtalk":
        payload = {"msgtype": "markdown", "markdown": {"title": title, "text": text}}
        if secret:
            url = dingtalk_signed_url(url, secret, int(now_s * 1000))
        return url, payload
    if kind == "feishu":
        payload = {"msg_type": "text", "content": {"text": text}}
        if secret:
            ts = int(now_s)
            payload.update(timestamp=str(ts), sign=feishu_sign(secret, ts))
        return url, payload
    raise ValueError(f"unsupported webhook kind {kind!r}")


def post_json(url: str, payload: dict, timeout: float = 15.0) -> str:
    req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger_dir", default=str(DEFAULT_DIR))
    ap.add_argument("--since_hours", type=float, default=24.0)
    ap.add_argument("--training_log", default=None)
    ap.add_argument("--tail", type=int, default=8)
    ap.add_argument("--out", default=None, help="write the markdown report here")
    ap.add_argument("--webhook_kind", default=os.environ.get("LINGBOT_WEBHOOK_KIND", "none"),
                    choices=["none", "dingtalk", "feishu"])
    ap.add_argument("--webhook_url", default=os.environ.get("LINGBOT_WEBHOOK_URL"))
    ap.add_argument("--secret", default=os.environ.get("LINGBOT_WEBHOOK_SECRET"))
    ap.add_argument("--dry_run", action="store_true", help="print the webhook payload instead of sending")
    args = ap.parse_args(argv)

    title, text = build_report(load(Path(args.ledger_dir)), args.since_hours,
                               Path(args.training_log) if args.training_log else None, args.tail)
    print(text)
    if args.out:
        write_text(args.out, text)
    if args.webhook_kind == "none":
        return 0
    if not args.webhook_url:
        print("[daily-report] no webhook url (flag or LINGBOT_WEBHOOK_URL)")
        return 1
    url, payload = build_request(args.webhook_kind, args.webhook_url, args.secret, title, text, time.time())
    if args.dry_run:
        print(json.dumps({"url": url.split("?")[0], "payload": payload}, ensure_ascii=False, indent=2))
        return 0
    print(f"[daily-report] webhook response: {post_json(url, payload)[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
