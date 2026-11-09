"""gpu_env_probe.py: output parsers, error hints and the report paths that run without a GPU."""
from __future__ import annotations

import argparse
import importlib.util
import json

import pytest

import gpu_env_probe as probe

TOPO = (
    "\tGPU0\tGPU1\tGPU2\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\n"
    "GPU0\t X \tNV18\tNV18\tSYS\t0-55\t0\t\tN/A\n"
    "GPU1\tNV18\t X \t{link}\tSYS\t0-55\t0\t\tN/A\n"
    "GPU2\tNV18\t{link}\t X \tSYS\t56-111\t1\t\tN/A\n"
    "NIC0\tSYS\tSYS\tSYS\t X \n"
    "\n"
    "Legend:\n"
    "\n"
    "  X    = Self\n"
    "  NV#  = Connection traversing a bonded set of # NVLinks\n"
)

FABRIC = (
    "GPU 00000000:18:00.0\n"
    "    Fabric\n"
    "        State                             : Completed\n"
    "        Status                            : Success\n"
    "        Health\n"
    "            Status                        : Healthy\n"
    "    Processes                             : None\n"
    "\n"
    "GPU 00000000:2A:00.0\n"
    "    Fabric\n"
    "        State                             : In Progress\n"
    "        Status                            : N/A\n"
)


def test_topology_all_nvlink():
    assert probe.parse_topology(TOPO.format(link="NV18")) == {"n_gpu": 3, "links": {"NV18": 6}, "all_nvlink": True}


def test_topology_flags_pcie_pair():
    topo = probe.parse_topology(TOPO.format(link="PIX"))
    assert topo["links"] == {"NV18": 4, "PIX": 2}
    assert topo["all_nvlink"] is False


def test_topology_without_matrix():
    assert probe.parse_topology("")["n_gpu"] == 0


def test_fabric_reads_first_level_fields_only():
    assert probe.parse_fabric(FABRIC) == [("Completed", "Success"), ("In Progress", "N/A")]


def test_version_parsers():
    banner = "| NVIDIA-SMI 580.00      Driver Version: 580.00      CUDA Version: 13.0     |"
    assert probe.parse_cuda_driver_version(banner) == (13, 0)
    assert probe.parse_cuda_driver_version("no banner") is None
    ptxas = "ptxas: NVIDIA (R) Ptx optimizing assembler\nCuda compilation tools, release 12.9, V12.9.0\n"
    assert probe.parse_ptxas_version(ptxas) == (12, 9)
    assert probe.parse_ptxas_version("") is None
    assert probe.parse_compute_cap("10.3") == (10, 3)
    assert probe.parse_compute_cap("12.0") == (12, 0)
    for bad in ("sm_103", "103", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            probe.parse_compute_cap(bad)


@pytest.mark.parametrize("cap, expected", [
    ((10, 3), "same_major"),
    ((10, 0), "exact"),
    ((8, 9), "same_major"),
    ((11, 0), "unsupported"),
    ((13, 0), "ptx_jit"),
])
def test_classify_arch(cap, expected):
    arch_list = ["sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]
    assert probe.classify_arch(cap, arch_list) == expected


def test_hints():
    assert "TRITON_PTXAS_PATH" in probe.hint_for_error(
        "PTXASError: ptxas fatal   : Value 'sm_103a' is not defined for option 'gpu-name'")
    assert "12.9" in probe.hint_for_error(
        "RuntimeError: Triton only support CUDA 10.0 or higher, but got CUDA version: 13.0")
    assert "cu129" in probe.hint_for_error("CUDA error: no kernel image is available for execution on the device")
    assert "libgl1" in probe.hint_for_error("ImportError: libGL.so.1: cannot open shared object file")
    assert probe.hint_for_error("ValueError: something else") == ""


def test_guard_turns_exception_into_fail_with_hint():
    rep = probe.Report("test")

    def broken():
        raise RuntimeError("ptxas fatal   : Value 'sm_103a' is not defined for option 'gpu-name'")

    rep.guard("triton.kernel", broken)
    check = rep.checks[-1]
    assert check["status"] == "FAIL"
    assert "TRITON_PTXAS_PATH" in check["hint"]
    assert check["data"]["traceback"]


def test_judge_flags_nan_and_over_limit():
    res = probe._judge({"fwd": 1e-3, "grad": float("nan")}, {"fwd": 3e-2, "grad": 5e-2})
    assert res["status"] == "FAIL" and "grad" in res["detail"]
    assert probe._judge({"fwd": 1e-3}, {"fwd": 3e-2})["status"] == "PASS"


def test_missing_c_compiler_is_fail(monkeypatch, tmp_path):
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    rep = probe.Report("test")
    probe.check_build_tools(rep, 0)
    assert [c["name"] for c in rep.checks] == ["platform.cc"]
    assert rep.checks[0]["status"] == "FAIL" and "build-essential" in rep.checks[0]["hint"]


def test_libcuda_lookup_follows_triton(monkeypatch, tmp_path):
    (tmp_path / "libcuda.so.1").write_bytes(b"")
    monkeypatch.delenv("TRITON_LIBCUDA_PATH", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    def libcuda_status(ldconfig_output):
        monkeypatch.setattr(probe, "_run", lambda cmd, timeout=60: (0, ldconfig_output, ""))
        rep = probe.Report("test")
        probe.check_build_tools(rep, 8)
        return rep.checks[-1]

    assert libcuda_status(f"\tlibcuda.so.1 (libc6,x86-64) => {tmp_path}/libcuda.so.1\n")["status"] == "PASS"
    missing = libcuda_status("\tlibc.so.6 (libc6,x86-64) => /lib/x86_64-linux-gnu/libc.so.6\n")
    assert missing["status"] == "FAIL" and "TRITON_LIBCUDA_PATH" in missing["hint"]
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
    assert libcuda_status("")["status"] == "PASS"


def _run_probe(tmp_path, *args, data_dir=None):
    out = tmp_path / "report.json"
    rc = probe.main([*args, "--data_dir", str(data_dir or tmp_path), "--skip_net", "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    return rc, report, {c["name"]: c for c in report["checks"]}


def test_platform_only_report(tmp_path):
    rc, report, checks = _run_probe(tmp_path, "--platform_only")
    assert report["mode"] == "platform"
    assert {"platform.cpu_arch", "platform.cpu_ram", "platform.disk", "gpu.inventory"} <= set(checks)
    assert "net.reachability" not in checks
    assert all(c["status"] in probe.STATUSES for c in checks.values())
    assert rc == (1 if any(c["status"] == "FAIL" for c in checks.values()) else 0)
    assert sum(report["counts"].values()) == len(report["checks"])


def test_missing_data_dir_is_fail(tmp_path):
    rc, _, checks = _run_probe(tmp_path, "--platform_only", data_dir=tmp_path / "missing")
    assert checks["platform.disk"]["status"] == "FAIL"
    assert rc == 1


@pytest.mark.skipif(importlib.util.find_spec("triton") is not None, reason="triton is installed")
def test_triton_target_without_triton_fails(tmp_path):
    out = tmp_path / "report.json"
    assert probe.main(["--triton_target", "10.3", "--out", str(out)]) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["mode"] == "triton_cross"
    assert [(c["name"], c["status"]) for c in report["checks"]] == [("triton.cross_compile", "FAIL")]


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None, reason="torch is installed")
def test_full_mode_without_torch_fails(tmp_path):
    rc, report, checks = _run_probe(tmp_path)
    assert report["mode"] == "full"
    assert checks["torch.import"]["status"] == "FAIL"
    assert checks["py.versions"]["status"] == "WARN"
    assert rc == 1
