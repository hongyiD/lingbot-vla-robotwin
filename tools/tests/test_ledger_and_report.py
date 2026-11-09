import base64
import datetime as dt
import hashlib
import hmac
import json
import urllib.parse

import daily_report
import ledger


def test_ledger_add_show_render(tmp_path, capsys):
    cfg = tmp_path / "train.yaml"
    cfg.write_text("lr: 1e-4\n")
    d = str(tmp_path / "ledger")
    assert ledger.main(["--ledger_dir", d, "add", "--kind", "run", "--title", "runA", "--field", "steps=10000",
                        "--hash", str(cfg)]) == 0
    assert ledger.main(["--ledger_dir", d, "add", "--kind", "submit", "--title", "runA 100x50"]) == 0
    entries = ledger.load(tmp_path / "ledger")
    assert [e["id"] for e in entries] == [1, 2]
    assert entries[0]["fields"] == {"steps": "10000"}
    assert list(entries[0]["hashes"].values())[0] == hashlib.sha256(cfg.read_bytes()).hexdigest()
    md = (tmp_path / "ledger" / "ledger.md").read_text()
    assert "submissions: 1" in md and "runA 100x50" in md
    assert ledger.main(["--ledger_dir", d, "show", "--kind", "run"]) == 0
    assert "runA" in capsys.readouterr().out


def test_daily_report_content(tmp_path):
    now = dt.datetime(2026, 10, 20, 22, 0, tzinfo=dt.timezone(dt.timedelta(hours=8)))
    entries = [
        {"id": 1, "time": "2026-10-18T10:00:00+08:00", "kind": "data", "title": "old", "fields": {}},
        {"id": 2, "time": "2026-10-20T09:00:00+08:00", "kind": "eval", "title": "runA 50x10",
         "fields": {"clean": "61.2", "randomized": "40.4"}},
        {"id": 3, "time": "2026-10-20T21:00:00+08:00", "kind": "submit", "title": "first submit", "fields": {}},
    ]
    log = tmp_path / "train.log"
    log.write_text("\n".join(f"\x1b[32mstep {i}\x1b[0m loss 0.{i}" for i in range(20)))
    title, text = daily_report.build_report(entries, 24, log, 3, now=now)
    assert title == "灵波初赛日报 2026-10-20"
    assert "#2 [eval] runA 50x10" in text and "#3 [submit]" in text and "old" not in text
    assert "clean=61.2" in text and "**已提交次数**: 1" in text
    assert "step 19 loss 0.19" in text and "step 16" not in text and "\x1b" not in text


def test_webhook_signing():
    secret, ts_ms = "SECabc", 1700000000000
    url = daily_report.dingtalk_signed_url("https://oapi.dingtalk.com/robot/send?access_token=t", secret, ts_ms)
    expected = base64.b64encode(hmac.new(secret.encode(), f"{ts_ms}\n{secret}".encode(), hashlib.sha256).digest()).decode()
    assert url.endswith(f"&timestamp={ts_ms}&sign={urllib.parse.quote_plus(expected)}")
    fs = daily_report.feishu_sign(secret, 1700000000)
    assert fs == base64.b64encode(hmac.new(f"1700000000\n{secret}".encode(), b"", hashlib.sha256).digest()).decode()
    url, payload = daily_report.build_request("feishu", "https://open.feishu.cn/hook/x", secret, "t", "body", 1700000000.5)
    assert payload["timestamp"] == "1700000000" and payload["content"]["text"] == "body"
    url, payload = daily_report.build_request("dingtalk", "https://oapi.dingtalk.com/robot/send?access_token=t", None,
                                              "t", "body", 0)
    assert payload == {"msgtype": "markdown", "markdown": {"title": "t", "text": "body"}}
    assert json.dumps(payload)
