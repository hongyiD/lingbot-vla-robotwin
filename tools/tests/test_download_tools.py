import json
import zipfile

import pytest

import extract_archives
import hf_manifest
import verify_downloads
from lingbot_tools.io_utils import sha256_file

ARCHIVE = "aloha-agilex_clean_50.zip"


def _make_zip(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def _fake_hf(monkeypatch, download_dir, tasks):
    meta = {}
    for task in tasks:
        p = download_dir / "dataset" / task / ARCHIVE
        meta[task] = (p.stat().st_size, sha256_file(p))

    def fake_get_json(url, token, timeout):
        if "/revision/" in url:
            return {"sha": "abc123def456"}
        task = url.rsplit("/", 1)[-1]
        size, sha = meta[task]
        return [{"type": "file", "path": f"dataset/{task}/demo_clean.zip", "size": 1, "lfs": {"oid": "x", "size": 1}},
                {"type": "file", "path": f"dataset/{task}/{ARCHIVE}", "size": size, "lfs": {"oid": sha, "size": size}}]

    monkeypatch.setattr(hf_manifest, "_get_json", fake_get_json)


def test_manifest_verify_extract(tmp_path, monkeypatch):
    dl, raw, manifest = tmp_path / "hf", tmp_path / "raw", tmp_path / "manifest.json"
    for task in ("click_bell", "lift_pot"):
        _make_zip(dl / "dataset" / task / ARCHIVE, {
            "aloha-agilex_clean_50/data/episode0.hdf5": b"h5",
            "aloha-agilex_clean_50/instructions/episode0.json": b"{}",
            "aloha-agilex_clean_50/seed.txt": b"1"})
    _fake_hf(monkeypatch, dl, ["click_bell", "lift_pot"])

    assert hf_manifest.main(["--tasks", "click_bell,lift_pot", "--out", str(manifest)]) == 0
    m = json.loads(manifest.read_text())
    assert m["revision"] == "abc123def456"
    assert [f["task"] for f in m["files"]] == ["click_bell", "lift_pot"]
    assert m["total_bytes"] == sum(f["size"] for f in m["files"])

    assert verify_downloads.main(["--manifest", str(manifest), "--download_dir", str(dl)]) == 0

    assert extract_archives.main(["--manifest", str(manifest), "--download_dir", str(dl), "--raw_root", str(raw)]) == 0
    assert (raw / "click_bell" / "aloha-agilex_clean_50" / "seed.txt").read_text() == "1"
    assert (raw / "lift_pot" / ".extracted.json").exists()
    # second run is idempotent (skipped)
    assert extract_archives.main(["--manifest", str(manifest), "--download_dir", str(dl), "--raw_root", str(raw)]) == 0

    target = dl / "dataset" / "lift_pot" / ARCHIVE
    data = bytearray(target.read_bytes())
    data[12] ^= 0xFF
    target.write_bytes(bytes(data))
    assert verify_downloads.main(["--manifest", str(manifest), "--download_dir", str(dl)]) == 1
    target.unlink()
    assert verify_downloads.main(["--manifest", str(manifest), "--download_dir", str(dl), "--size_only"]) == 1


def test_model_manifest_and_verify(tmp_path, monkeypatch):
    weights, manifest = tmp_path / "weights", tmp_path / "manifest.json"
    blob = weights / "depth" / "model.pt"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"weights" * 100)
    (weights / "config.json").write_text("{}")
    siblings = [
        {"rfilename": ".gitattributes", "size": 10},
        {"rfilename": "README.md", "size": 5},
        {"rfilename": "depth/model.pt", "size": 700, "lfs": {"sha256": sha256_file(blob), "size": 700}},
        {"rfilename": "config.json", "size": 2},
    ]
    urls = []

    def fake_get_json(url, token, timeout):
        urls.append(url)
        return {"sha": "fedcba987654", "siblings": siblings}

    monkeypatch.setattr(hf_manifest, "_get_json", fake_get_json)
    assert hf_manifest.main(["--repo_type", "model", "--repo", "org/model", "--exclude", ".gitattributes",
                             "--exclude", "*.md", "--out", str(manifest)]) == 0
    assert urls == [f"{hf_manifest.DEFAULT_ENDPOINT.rstrip('/')}/api/models/org/model/revision/main?blobs=true"]
    m = json.loads(manifest.read_text())
    assert (m["repo_type"], m["revision"], m["total_bytes"]) == ("model", "fedcba987654", 702)
    assert [(f["path"], f["sha256"] is None) for f in m["files"]] == [("config.json", True), ("depth/model.pt", False)]

    assert verify_downloads.main(["--manifest", str(manifest), "--download_dir", str(weights)]) == 0
    blob.write_bytes(b"x" * 700)  # same size, other content
    assert verify_downloads.main(["--manifest", str(manifest), "--download_dir", str(weights)]) == 1

    with pytest.raises(SystemExit):
        hf_manifest.main(["--repo_type", "model", "--out", str(manifest)])


def test_extract_rejects_unsafe_and_unexpected_layout(tmp_path, monkeypatch):
    dl, raw, manifest = tmp_path / "hf", tmp_path / "raw", tmp_path / "manifest.json"
    _make_zip(dl / "dataset" / "click_bell" / ARCHIVE, {"../evil.txt": b"x"})
    _make_zip(dl / "dataset" / "lift_pot" / ARCHIVE, {"other_top/data/episode0.hdf5": b"x"})
    _fake_hf(monkeypatch, dl, ["click_bell", "lift_pot"])
    assert hf_manifest.main(["--tasks", "click_bell,lift_pot", "--out", str(manifest)]) == 0
    assert extract_archives.main(["--manifest", str(manifest), "--download_dir", str(dl), "--raw_root", str(raw)]) == 1
    assert not (tmp_path / "evil.txt").exists()
    assert not (raw / "lift_pot" / "other_top").exists()
