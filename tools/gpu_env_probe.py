#!/usr/bin/env python3
"""Probe a rented NVIDIA machine (B300 first) before LingBot-VLA 2.0 data processing and training.

Copy this single file to the machine; it does not depend on lingbot_tools.

  # G0, right after login: system python3 is enough (stdlib only)
  python3 gpu_env_probe.py --platform_only --data_dir /data --out ~/reports/g0_platform.json

  # G1, inside the LingBot training env (torch / triton / flash-attn / lingbotvla installed)
  python gpu_env_probe.py --data_dir /data --out ~/reports/g1_probe.json

  # G1, collectives across all GPUs (one process per GPU)
  torchrun --nproc-per-node 8 gpu_env_probe.py --nccl --out ~/reports/g1_nccl.json

  # Before shipping a training image: compile the Triton probe kernels for B300 (sm_103a)
  # on any x86_64 Linux box, no B300 (or any GPU) needed
  python gpu_env_probe.py --triton_target 10.3 --out ~/reports/triton_sm103.json

Every check reports PASS / WARN / FAIL / SKIP; the exit code is 1 if any check FAILs.
Kernel checks run each GPU code path LingBot depends on (cuBLAS, cuDNN conv3d, SDPA, Triton,
torch.compile, FlexAttention, flash-attn 2, LingBot's fused and inference MoE) and compare it with
an fp32/fp64 reference, so a kernel that runs but computes wrong numbers is caught as well.
`--device cpu` runs the non-GPU parts, as a self-test of this script.
"""
import argparse
import importlib
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# No `from __future__ import annotations`: Triton reads the tl.constexpr annotations below at
# decoration time, and --platform_only has to run on whatever python3 the image ships.
try:
    import triton
    import triton.language as tl

    TRITON_IMPORT_ERROR = ""
except Exception as _exc:  # noqa: BLE001
    triton = None
    tl = None
    TRITON_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

STATUSES = ("PASS", "WARN", "FAIL", "SKIP")

LINGBOT_PINS = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "torchaudio": "2.8.0",
    "triton": "3.4.0",
    "flash_attn": "2.8.3",
    "transformers": "4.57.3",
    "datasets": "3.6.0",
    "numpy": "1.26.4",
    "av": "15.0.0",
    "torchcodec": "0.6.0",
    "opencv-python-headless": "4.11.0.86",
}

NET_TARGETS = (
    ("huggingface.co", "https://huggingface.co/api/models/robbyant/lingbot-vla-v2-6b"),
    ("hf-mirror.com", "https://hf-mirror.com/"),
    ("pypi.org", "https://pypi.org/simple/pip/"),
    ("download.pytorch.org cu129", "https://download.pytorch.org/whl/cu129/torch/"),
    ("github lerobot tarball", "https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"),
    ("github flash-attn release", "https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3"),
)

GPU_FIELDS = ("index", "name", "memory.total", "driver_version", "compute_cap", "mig.mode.current")

MISMATCH_HINT = ("Runs but disagrees with the fp32/fp64 reference: treat this code path as broken on this "
                 "GPU and use the fallback from the runbook table.")
TRITON_HINT = "Triton is required by LingBot's fused MoE, FlexAttention and torch.compile."
LINGBOT_HINT = "Run inside the env built by tools/create_train_env.sh, or pass --lingbot_root <lingbot-vla-v2>."
FA_HINT = ("LingBot hard-codes flash_attention_2 for the Qwen3-VL text model and the action expert, so "
           "transformers will not build the model without flash-attn; install flash-attn 2.8.3.")

_HINTS = (
    ("triton only support cuda",
     "TRITON_PTXAS_PATH points to a ptxas Triton 3.4 cannot parse (13.x): use a CUDA 12.9 ptxas."),
    ("ptxas",
     "Triton's bundled ptxas (CUDA 12.8) cannot target B300 (sm_103a): install a CUDA 12.9 ptxas and "
     "export TRITON_PTXAS_PATH=<dir>/ptxas (runbook 3.2). Do not use a 13.x ptxas with Triton 3.4."),
    ("no kernel image is available",
     "A prebuilt binary has no kernels for this GPU: use the cu129 torch wheels; rebuild flash-attn "
     "with FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=100 (runbook 3.5)."),
    ("system not yet initialized",
     "NVSwitch fabric is not ready: check `systemctl status nvidia-fabricmanager` and the Fabric "
     "section of `nvidia-smi -q`."),
    ("driver version is insufficient",
     "The driver is older than this env's CUDA runtime: B300 needs a driver that supports CUDA >= 12.9."),
    ("out of resource",
     "A Triton kernel config exceeds this GPU's limits: note the kernel name and use the runbook fallback."),
    ("libgl.so",
     "cv2 is the GUI build: apt-get install -y libgl1 libglib2.0-0, or keep only opencv-python-headless."),
    ("failed to find c compiler",
     "Triton compiles a C launcher for its kernels with the host compiler: apt-get install -y build-essential, "
     "or export CC."),
    ("libcuda.so cannot found",
     "Triton finds libcuda.so.1 through `/sbin/ldconfig -p`: run ldconfig, or export TRITON_LIBCUDA_PATH=<dir "
     "containing libcuda.so.1> (common inside containers)."),
    ("python.h",
     "Triton's launcher build needs Python headers: use the conda python from create_train_env.sh, or install "
     "python3.12-dev."),
    ("nccl",
     "Rerun with NCCL_DEBUG=INFO; check Fabric Manager and `nvidia-smi topo -m`."),
)


def hint_for_error(message: str) -> str:
    """First matching remediation for an error message, or ''."""
    low = message.lower()
    for key, hint in _HINTS:
        if key in low:
            return hint
    return ""


def parse_cuda_driver_version(text: str) -> Optional[Tuple[int, int]]:
    """CUDA version supported by the driver, from the `nvidia-smi` banner."""
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_ptxas_version(text: str) -> Optional[Tuple[int, int]]:
    """Release from `ptxas --version` ("Cuda compilation tools, release 12.9, ...")."""
    m = re.search(r"release (\d+)\.(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_compute_cap(text: str) -> Tuple[int, int]:
    """Compute capability as nvidia-smi prints it ("10.3" for B300)."""
    m = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"expected a compute capability like 10.3, got {text!r}")
    return int(m.group(1)), int(m.group(2))


def parse_fabric(text: str) -> List[Tuple[str, str]]:
    """(State, Status) of every `Fabric` block in `nvidia-smi -q`, one per GPU; nested blocks ignored."""
    lines = text.splitlines()
    result = []
    for i, line in enumerate(lines):
        if line.strip() != "Fabric":
            continue
        parent = len(line) - len(line.lstrip())
        child = None
        fields = {}
        for nxt in lines[i + 1:]:
            if not nxt.strip():
                break
            indent = len(nxt) - len(nxt.lstrip())
            if indent <= parent:
                break
            if child is None:
                child = indent
            if indent == child and ":" in nxt:
                key, _, value = nxt.partition(":")
                fields[key.strip()] = value.strip()
        result.append((fields.get("State", "?"), fields.get("Status", "?")))
    return result


def parse_topology(text: str) -> Dict:
    """Pairwise GPU link types from `nvidia-smi topo -m` (X = self, NV# = NVLink)."""
    lines = [line for line in text.splitlines() if line.strip()]
    gpu_token = re.compile(r"GPU\d+$")
    header = next((line.split() for line in lines if line[:1].isspace() and line.split()[0] == "GPU0"), None)
    if header is None:
        return {"n_gpu": 0, "links": {}, "all_nvlink": False}
    n_gpu = 0
    for token in header:
        if not gpu_token.match(token):
            break
        n_gpu += 1
    links = Counter()
    for line in lines:
        tokens = line.split()
        if line[:1].isspace() or not gpu_token.match(tokens[0]):
            continue
        row = int(tokens[0][3:])
        for col, link in enumerate(tokens[1:1 + n_gpu]):
            if col != row:
                links[link] += 1
    all_nvlink = n_gpu > 1 and bool(links) and all(k.startswith("NV") for k in links)
    return {"n_gpu": n_gpu, "links": dict(links), "all_nvlink": all_nvlink}


def classify_arch(cap: Tuple[int, int], arch_list: List[str]) -> str:
    """How a torch build covers a GPU: exact SASS, same-major SASS, PTX JIT, or unsupported."""
    sass, ptx = [], []
    for arch in arch_list:
        m = re.fullmatch(r"(sm|compute)_(\d+)(\d)", arch)
        if m:
            (sass if m.group(1) == "sm" else ptx).append((int(m.group(2)), int(m.group(3))))
    cap = tuple(cap)
    if cap in sass:
        return "exact"
    if any(major == cap[0] and minor <= cap[1] for major, minor in sass):
        return "same_major"
    if any(version <= cap for version in ptx):
        return "ptx_jit"
    return "unsupported"


def _short(text, limit: int = 400) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _run(cmd: List[str], timeout: int = 60) -> Tuple[Optional[int], str, str]:
    """(returncode, stdout, stderr); returncode is None if the command could not run."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None, "", f"{cmd[0]}: command not found"
    except subprocess.TimeoutExpired:
        return None, "", f"{' '.join(cmd)}: timed out after {timeout}s"
    except OSError as exc:
        return None, "", f"{cmd[0]}: {exc}"
    return proc.returncode, proc.stdout, proc.stderr


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _result(status: str, detail: str = "", hint: str = "", **data) -> Dict:
    return {"status": status, "detail": detail, "hint": hint, "data": data}


def _judge(errors: Dict[str, float], limits: Dict[str, float], prefix: str = "", **data) -> Dict:
    bad = [k for k, v in errors.items() if not v <= limits[k]]
    detail = prefix + ", ".join(f"{k} {v:.1e}" for k, v in errors.items())
    if bad:
        detail += " (over limit: " + ", ".join(bad) + ")"
    return _result("FAIL" if bad else "PASS", detail, MISMATCH_HINT if bad else "",
                   errors=errors, limits=limits, **data)


class Report:
    def __init__(self, mode: str):
        self.mode = mode
        self.checks: List[Dict] = []
        self.started = time.time()

    def add(self, name: str, status: str, detail: str = "", hint: str = "", data: Optional[Dict] = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status}")
        self.checks.append({"name": name, "status": status, "detail": detail, "hint": hint, "data": data or {}})
        print(f"[probe] {status:<4} {name}: {detail}", flush=True)

    def guard(self, name: str, fn, *args) -> None:
        """Run a check function returning _result(); an exception becomes a FAIL with a hint."""
        try:
            res = fn(*args)
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            tail = traceback.format_exc().strip().splitlines()[-12:]
            self.add(name, "FAIL", _short(msg), hint_for_error(msg + " " + " ".join(tail)), {"traceback": tail})
            return
        self.add(name, res["status"], res["detail"], res["hint"], res["data"])

    def finish(self, out: str) -> int:
        counts = Counter(c["status"] for c in self.checks)
        print("\n[probe] summary: " + ", ".join(f"{s} {counts.get(s, 0)}" for s in STATUSES))
        for c in self.checks:
            if c["status"] in ("FAIL", "WARN"):
                print(f"  {c['status']} {c['name']}: {c['detail']}")
                if c["hint"]:
                    print(f"       -> {c['hint']}")
        if out:
            path = os.path.abspath(os.path.expanduser(out))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            report = {
                "tool": "gpu_env_probe",
                "mode": self.mode,
                "host": socket.gethostname(),
                "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                "duration_s": round(time.time() - self.started, 1),
                "counts": dict(counts),
                "checks": self.checks,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            print(f"[probe] wrote {path}")
        return 1 if counts.get("FAIL") else 0


# ---------------------------------------------------------------- platform (stdlib only)

def _cgroup_limits() -> Tuple[Optional[float], Optional[int]]:
    """(CPU quota in cores, memory limit in bytes) of this container; None when unlimited or unknown."""
    cpus = None
    quota = _read_text("/sys/fs/cgroup/cpu.max").split()
    if len(quota) == 2 and quota[0].isdigit() and quota[1].isdigit():
        cpus = int(quota[0]) / int(quota[1])
    else:
        q = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        p = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q.isdigit() and p.isdigit() and int(p) > 0:
            cpus = int(q) / int(p)
    mem = None
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        value = _read_text(path)
        if value.isdigit() and int(value) < 1 << 60:
            mem = int(value)
            break
    return cpus, mem


def check_os(rep: Report) -> None:
    machine = platform.machine()
    if machine in ("x86_64", "AMD64"):
        rep.add("platform.cpu_arch", "PASS", machine)
    else:
        rep.add("platform.cpu_arch", "WARN", machine,
                "Not x86_64 (GB300 Grace is aarch64): flash-attn release wheels are x86_64 only and other "
                "dependencies may need source builds; prefer an x86 HGX B300 host.")
    libc, version = platform.libc_ver()
    nums = [int(x) for x in re.findall(r"\d+", version)[:2]]
    if libc != "glibc" or len(nums) < 2:
        rep.add("platform.glibc", "SKIP", f"libc '{libc or 'unknown'}' {version}".strip())
    elif tuple(nums) >= (2, 28):
        rep.add("platform.glibc", "PASS", f"glibc {version}")
    else:
        rep.add("platform.glibc", "FAIL", f"glibc {version}",
                "torch 2.8 wheels need glibc >= 2.28: use an Ubuntu 22.04+ image.")


def check_build_tools(rep: Report, n_gpu: int) -> None:
    """Host pieces Triton 3.4 needs when it compiles a kernel, looked up the way Triton does."""
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    if cc:
        rep.add("platform.cc", "PASS", cc)
    else:
        rep.add("platform.cc", "FAIL", "no $CC, gcc or clang", hint_for_error("failed to find c compiler"))
    if not n_gpu:
        return
    hint = hint_for_error("libcuda.so cannot found")
    if os.environ.get("TRITON_LIBCUDA_PATH"):
        rep.add("platform.libcuda", "PASS", f"TRITON_LIBCUDA_PATH={os.environ['TRITON_LIBCUDA_PATH']}")
        return
    rc, out, _ = _run(["/sbin/ldconfig", "-p"], timeout=10)
    if rc is None:
        rep.add("platform.libcuda", "FAIL", "/sbin/ldconfig not found", hint)
        return
    locs = [line.split()[-1] for line in out.splitlines() if "libcuda.so.1" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    if not dirs:
        dirs = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") if d]
    found = [d for d in dirs if os.path.exists(os.path.join(d, "libcuda.so.1"))]
    if found:
        rep.add("platform.libcuda", "PASS", os.path.join(found[0], "libcuda.so.1"))
    else:
        rep.add("platform.libcuda", "FAIL", "libcuda.so.1 is not in `ldconfig -p` or LD_LIBRARY_PATH", hint)


def check_resources(rep: Report, data_dir: str, n_gpu: int) -> None:
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 0)
    quota, mem_limit = _cgroup_limits()
    if quota:
        cpus = min(cpus, int(quota))
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        ram = 0
    if mem_limit:
        ram = min(ram, mem_limit) if ram else mem_limit
    per = max(n_gpu, 1)
    low = []
    if cpus < 8 * per:
        low.append("< 8 CPUs per GPU")
    if ram < 32 * per * 2**30:
        low.append("< 32 GiB RAM per GPU")
    detail = f"{cpus} usable CPUs, {ram / 2**30:.0f} GiB RAM, {n_gpu} GPU(s)"
    rep.add("platform.cpu_ram", "WARN" if low else "PASS", detail + (f" ({'; '.join(low)})" if low else ""),
            "LingBot runs 8 dataloader workers per GPU plus depth/video teachers; a CPU-starved host shows "
            "up as low GPU utilization." if low else "",
            {"cpus": cpus, "ram_gib": round(ram / 2**30, 1), "cgroup_cpu_quota": quota, "cgroup_mem_limit": mem_limit})

    if os.path.isdir("/dev/shm"):
        shm = shutil.disk_usage("/dev/shm").total
        ok = shm >= 32 * 2**30
        rep.add("platform.shm", "PASS" if ok else "WARN", f"/dev/shm {shm / 2**30:.0f} GiB",
                "" if ok else "DataLoader workers pass batches through /dev/shm: in Docker use --ipc=host "
                              "or --shm-size=64g.")
    else:
        rep.add("platform.shm", "SKIP", "no /dev/shm")

    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        ok = soft == resource.RLIM_INFINITY or soft >= 65536
        rep.add("platform.nofile", "PASS" if ok else "WARN", f"ulimit -n soft {soft}, hard {hard}",
                "" if ok else "Raise with `ulimit -n 1048576` (Docker: --ulimit nofile=1048576:1048576).")
    except ImportError:
        rep.add("platform.nofile", "SKIP", "resource module unavailable")

    path = os.path.abspath(os.path.expanduser(data_dir))
    if not os.path.isdir(path):
        rep.add("platform.disk", "FAIL", f"{path} does not exist", "Pass --data_dir on the large data disk.")
        return
    usage = shutil.disk_usage(path)
    rc, out, _ = _run(["df", "-PT", path], timeout=10)
    rows = out.splitlines() if rc == 0 else []
    fs_type = rows[1].split()[1] if len(rows) > 1 and len(rows[1].split()) > 1 else ""
    inodes = os.statvfs(path).f_favail
    ok = usage.free >= 2**40
    detail = f"{path}: {usage.free / 2**30:.0f} GiB free of {usage.total / 2**30:.0f} GiB"
    if fs_type:
        detail += f", {fs_type}"
    if inodes:
        detail += f", {inodes / 1e6:.1f}M free inodes"
    rep.add("platform.disk", "PASS" if ok else "WARN", detail,
            "" if ok else "Converted LeRobot image datasets, HF caches, weights and checkpoints (tens of GB each) "
                          "need >= 1 TiB on this disk.",
            {"free_gib": round(usage.free / 2**30), "fs_type": fs_type, "free_inodes": inodes})


def check_gpus(rep: Report) -> List[Dict[str, str]]:
    fields = GPU_FIELDS
    query = ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
    rc, out, err = _run(query)
    if rc not in (None, 0):
        fields = GPU_FIELDS[:4]
        rc, out, err = _run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
    if rc != 0:
        msg = _short(err or out or "nvidia-smi failed")
        rep.add("gpu.inventory", "FAIL", msg,
                hint_for_error(msg) or "No working NVIDIA driver visible (Docker needs --gpus all).")
        return []
    gpus = [dict(zip(fields, [p.strip() for p in line.split(",")])) for line in out.strip().splitlines() if line.strip()]
    if not gpus:
        rep.add("gpu.inventory", "FAIL", "nvidia-smi lists no GPU")
        return []
    first = gpus[0]
    mem = first.get("memory.total", "")
    mem_txt = f"{int(float(mem)) // 1024} GiB" if mem.replace(".", "", 1).isdigit() else mem
    detail = (f"{len(gpus)} x {first.get('name')} ({mem_txt}), compute_cap {first.get('compute_cap', '?')}, "
              f"driver {first.get('driver_version', '?')}")
    names = Counter(g.get("name") for g in gpus)
    mig = [g["index"] for g in gpus if g.get("mig.mode.current", "").lower() == "enabled"]
    if mig:
        rep.add("gpu.inventory", "FAIL", detail + f"; MIG enabled on GPU {','.join(mig)}",
                "MIG slices cannot run this training: ask the provider for whole GPUs.", {"gpus": gpus})
    elif len(names) > 1:
        rep.add("gpu.inventory", "WARN", detail + f"; mixed models {dict(names)}", "", {"gpus": gpus})
    else:
        rep.add("gpu.inventory", "PASS", detail, "", {"gpus": gpus})
    return gpus


def check_driver(rep: Report, gpus: List[Dict[str, str]]) -> None:
    rc, out, err = _run(["nvidia-smi"])
    version = parse_cuda_driver_version(out)
    caps = {g.get("compute_cap") for g in gpus}
    if version is None:
        rep.add("gpu.cuda_driver", "WARN", "could not read 'CUDA Version' from nvidia-smi")
    elif "10.3" in caps and version < (12, 9):
        rep.add("gpu.cuda_driver", "FAIL", f"driver supports CUDA {version[0]}.{version[1]}",
                "B300 (compute capability 10.3) needs a driver that supports CUDA >= 12.9: ask the provider.")
    else:
        rep.add("gpu.cuda_driver", "PASS", f"driver supports CUDA {version[0]}.{version[1]}")


def check_fabric(rep: Report) -> None:
    rc, out, err = _run(["nvidia-smi", "-q"], timeout=120)
    states = [s for s in parse_fabric(out) if s[0] not in ("N/A", "?")]
    if not states:
        rep.add("gpu.fabric", "SKIP", "no Fabric state reported (no NVSwitch, or old driver)")
    elif all(state == "Completed" and status == "Success" for state, status in states):
        rep.add("gpu.fabric", "PASS", f"{len(states)} GPUs Completed/Success")
    else:
        summary = dict(Counter(f"{state}/{status}" for state, status in states))
        rep.add("gpu.fabric", "FAIL", f"fabric states {summary}",
                "Until the fabric is Completed, CUDA init fails or NVLink is unused: check "
                "`systemctl status nvidia-fabricmanager` with the provider.")


def check_topology(rep: Report) -> None:
    rc, out, err = _run(["nvidia-smi", "topo", "-m"])
    topo = parse_topology(out if rc == 0 else "")
    if topo["n_gpu"] <= 1:
        rep.add("gpu.topology", "SKIP", f"{topo['n_gpu']} GPU in topology matrix")
    elif topo["all_nvlink"]:
        rep.add("gpu.topology", "PASS", f"all {topo['n_gpu']} GPUs pairwise NVLink {topo['links']}", "", topo)
    else:
        rep.add("gpu.topology", "WARN", f"non-NVLink pairs present {topo['links']}",
                "FSDP all-gather/reduce-scatter will be slower without NVLink between every GPU pair.", topo)


def _probe_url(url: str, timeout: float = 8.0) -> Tuple[bool, str]:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "gpu-env-probe"})
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            code = response.status
    except urllib.error.HTTPError as exc:
        code = exc.code
    except Exception as exc:  # noqa: BLE001
        return False, _short(f"{type(exc).__name__}: {exc}", 160)
    return code < 500, f"HTTP {code}, {(time.time() - start) * 1000:.0f} ms"


def check_network(rep: Report) -> None:
    targets = list(NET_TARGETS)
    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    if endpoint and endpoint not in ("https://huggingface.co", "https://hf-mirror.com"):
        targets.insert(0, ("HF_ENDPOINT", endpoint))
    results, down = {}, []
    for label, url in targets:
        ok, info = _probe_url(url)
        results[label] = {"url": url, "ok": ok, "info": info}
        if not ok:
            down.append(label)
    proxies = sorted(k for k in os.environ if k.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"))
    data = {"targets": results, "proxy_vars_set": proxies}
    if down:
        rep.add("net.reachability", "WARN", "unreachable: " + ", ".join(down),
                "Use HF_ENDPOINT=https://hf-mirror.com, a pip mirror or HTTPS_PROXY; anything still unreachable "
                "(flash-attn wheel, lerobot tarball, MoGe git deps) must be fetched elsewhere and copied in.", data)
    else:
        rep.add("net.reachability", "PASS", f"{len(targets)} endpoints reachable", "", data)


# ---------------------------------------------------------------- python stack

def _dist_version(name: str) -> Optional[str]:
    from importlib import metadata

    for candidate in (name, name.replace("_", "-"), name.replace("-", "_")):
        try:
            return metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None


def check_python_stack(rep: Report) -> None:
    found = {"python": platform.python_version()}
    issues = []
    if not found["python"].startswith("3.12."):
        issues.append(f"python {found['python']} (LingBot env uses 3.12)")
    for pkg, pin in LINGBOT_PINS.items():
        version = _dist_version(pkg)
        found[pkg] = version
        if version is None:
            issues.append(f"{pkg} missing")
        elif version.split("+")[0] != pin:
            issues.append(f"{pkg} {version} != {pin}")
    rep.add("py.versions", "WARN" if issues else "PASS",
            "; ".join(issues) if issues else f"python 3.12 and all {len(LINGBOT_PINS)} pins match",
            "Drift is recorded, not auto-fixed; the GPU checks and the G3 smoke runs decide whether it matters."
            if issues else "", found)

    extras = []
    if importlib.util.find_spec("flash_attn_interface") is not None:
        extras.append("flash_attn_interface (FlashAttention-3, Hopper only)")
    if importlib.util.find_spec("xformers") is not None:
        extras.append("xformers (MoGe / LingBot-Depth DINOv2 layers switch to it automatically)")
    cv2_dists = [d for d in ("opencv-python", "opencv-python-headless", "opencv-contrib-python") if _dist_version(d)]
    if len(cv2_dists) > 1:
        extras.append("several packages own cv2: " + ", ".join(cv2_dists))
    rep.add("py.conflicts", "WARN" if extras else "PASS",
            "; ".join(extras) if extras else "no FA3 / xformers / duplicate opencv",
            "Uninstall FA3 / xformers unless verified here (or export XFORMERS_DISABLED=1); keep only "
            "opencv-python-headless." if extras else "")

    failures = {}
    for module in ("lingbotvla", "lerobot", "cv2", "moge", "mdm", "utils3d"):
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            failures[module] = _short(f"{type(exc).__name__}: {exc}", 200)
    hint = ""
    if failures:
        hint = hint_for_error(" ".join(failures.values())) or LINGBOT_HINT
    rep.add("py.imports", "FAIL" if failures else "PASS",
            "; ".join(f"{k}: {v}" for k, v in failures.items()) if failures else "lingbotvla, lerobot, cv2, moge, mdm, utils3d",
            hint, {"failures": failures})


# ---------------------------------------------------------------- GPU kernels (torch)

if triton is not None:

    @triton.jit
    def _probe_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

    @triton.jit
    def _probe_dot_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                          stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc.to(tl.bfloat16),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _rel(out, ref) -> float:
    if out is None:  # e.g. a backward that returned no gradient for this input
        return float("inf")
    out = out.detach().cpu().double()
    ref = ref.detach().cpu().double()
    return float((out - ref).norm() / ref.norm().clamp_min(1e-12))


def _attn_reference(torch, q, k, v, gout, causal: bool):
    """fp64 CPU SDPA forward and input gradients for (B, H, S, D) tensors."""
    import torch.nn.functional as F

    qr, kr, vr = (t.double().requires_grad_() for t in (q, k, v))
    out = F.scaled_dot_product_attention(qr, kr, vr, is_causal=causal)
    (out * gout.double()).sum().backward()
    return out.detach(), (qr.grad, kr.grad, vr.grad)


def k_torch_cuda(torch) -> Dict:
    n = torch.cuda.device_count()
    caps = [torch.cuda.get_device_capability(i) for i in range(n)]
    arch_list = torch.cuda.get_arch_list()
    support = {f"{c[0]}.{c[1]}": classify_arch(c, arch_list) for c in sorted(set(caps))}
    nccl = ""
    try:
        version = torch.cuda.nccl.version()
        nccl = ".".join(map(str, version)) if isinstance(version, tuple) else str(version)
    except Exception:  # noqa: BLE001
        pass
    cap0 = f"{caps[0][0]}.{caps[0][1]}"
    detail = (f"{n} x {torch.cuda.get_device_name(0)}, capability {cap0} ({support[cap0]}), torch {torch.__version__}, "
              f"CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, NCCL {nccl or '?'}")
    data = {"arch_list": arch_list, "support": support, "cuda": torch.version.cuda, "nccl": nccl}
    if "unsupported" in support.values():
        return _result("FAIL", detail, "This torch build has no kernels for the GPU: install the cu129 torch "
                                       "2.8.0 wheels (runbook 3.1).", **data)
    cuda = tuple(int(x) for x in re.findall(r"\d+", torch.version.cuda or "0.0")[:2])
    if (10, 3) in caps and cuda < (12, 9):
        return _result("WARN", detail, "Runs through sm_100 binary compatibility; the cu129 build is preferred on "
                                       "B300 (runbook 3.1).", **data)
    return _result("PASS", detail, **data)


def k_matmul(torch, dev) -> Dict:
    g = torch.Generator().manual_seed(0)
    a = torch.randn(1024, 1024, generator=g).bfloat16().float()
    b = torch.randn(1024, 1024, generator=g).bfloat16().float()
    ref = a.double() @ b.double()
    ad, bd = a.to(dev), b.to(dev)
    flag = torch.backends.cuda.matmul.allow_tf32
    try:
        errors = {"bf16": _rel(ad.bfloat16() @ bd.bfloat16(), ref)}
        torch.backends.cuda.matmul.allow_tf32 = True
        errors["tf32"] = _rel(ad @ bd, ref)
        torch.backends.cuda.matmul.allow_tf32 = False
        errors["fp32"] = _rel(ad @ bd, ref)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = flag
    return _judge(errors, {"bf16": 2e-2, "tf32": 5e-3, "fp32": 1e-4})


def k_conv3d(torch, dev) -> Dict:
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 3, 4, 64, 64, generator=g).bfloat16().float()
    w = (torch.randn(64, 3, 2, 16, 16, generator=g) * 0.05).bfloat16().float()
    stride = (2, 16, 16)
    ref = F.conv3d(x.double(), w.double(), stride=stride)
    errors = {
        "bf16": _rel(F.conv3d(x.to(dev).bfloat16(), w.to(dev).bfloat16(), stride=stride), ref),
        "fp32": _rel(F.conv3d(x.to(dev), w.to(dev), stride=stride), ref),
    }
    return _judge(errors, {"bf16": 3e-2, "fp32": 5e-3}, "patch-embed conv3d, ")


def k_sdpa(torch, dev) -> Dict:
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    g = torch.Generator().manual_seed(2)
    q, k, v, gout = (torch.randn(2, 8, 256, 64, generator=g).bfloat16().float() for _ in range(4))
    ref_out, ref_grads = _attn_reference(torch, q, k, v, gout, causal=False)
    results = {}
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.MATH):
        name = backend.name.lower()
        try:
            qd, kd, vd = (t.to(dev, torch.bfloat16, copy=True).requires_grad_() for t in (q, k, v))
            with sdpa_kernel([backend]):
                out = F.scaled_dot_product_attention(qd, kd, vd)
            (out.float() * gout.to(dev)).sum().backward()
            errs = [_rel(out, ref_out)] + [_rel(t.grad, r) for t, r in zip((qd, kd, vd), ref_grads)]
            results[name] = "ok" if errs[0] <= 3e-2 and max(errs[1:]) <= 5e-2 else f"mismatch {max(errs):.1e}"
        except Exception as exc:  # noqa: BLE001
            results[name] = "unavailable: " + _short(f"{type(exc).__name__}: {exc}", 160)
    ok = {n for n, r in results.items() if r == "ok"}
    wrong = sorted(n for n, r in results.items() if r.startswith("mismatch"))
    detail = ", ".join(f"{n} {r.split(':')[0]}" for n, r in results.items())
    if wrong or "math" not in ok or not ok & {"flash_attention", "efficient_attention"}:
        return _result("FAIL", detail, "A backend that computes wrong values may be picked automatically; disable it "
                                       "(e.g. torch.backends.cuda.enable_cudnn_sdp(False)) or treat as broken.",
                       backends=results)
    if len(ok) < len(results):
        return _result("WARN", detail, "Some SDPA backends are unavailable here; PyTorch falls back automatically.",
                       backends=results)
    return _result("PASS", detail, backends=results)


def k_triton_toolchain(caps: List[Tuple[int, int]]) -> Dict:
    if triton is None:
        return _result("FAIL", TRITON_IMPORT_ERROR, TRITON_HINT)
    path = ""
    try:
        from triton.backends.nvidia.compiler import get_ptxas

        path = get_ptxas().path
    except Exception:  # noqa: BLE001
        path = os.environ.get("TRITON_PTXAS_PATH", "")
    rc, out, err = _run([path, "--version"]) if path else (None, "", "ptxas path unknown")
    version = parse_ptxas_version(out + err)
    data = {"triton": triton.__version__, "ptxas_path": path, "ptxas_release": version,
            "TRITON_PTXAS_PATH": os.environ.get("TRITON_PTXAS_PATH", "")}
    detail = f"triton {triton.__version__}, ptxas {'%d.%d' % version if version else '?'} at {path or '?'}"
    if version is None:
        return _result("FAIL", detail + f" ({_short(err, 120)})",
                       "Cannot run ptxas: point TRITON_PTXAS_PATH at a working CUDA 12.9 ptxas.", **data)
    if triton.__version__.startswith("3.4") and version[0] >= 13:
        return _result("FAIL", detail, "Triton 3.4 rejects 13.x ptxas ('Triton only support CUDA ...'): point "
                                       "TRITON_PTXAS_PATH at a CUDA 12.9 ptxas.", **data)
    if (10, 3) in caps and version < (12, 9):
        return _result("FAIL", detail, "B300 (sm_103a) needs ptxas >= 12.9 and Triton's bundled one is older: "
                                       "install a CUDA 12.9 ptxas and export TRITON_PTXAS_PATH (runbook 3.2).", **data)
    return _result("PASS", detail, **data)


def k_triton_kernels(torch, dev) -> Dict:
    if triton is None:
        return _result("FAIL", TRITON_IMPORT_ERROR, TRITON_HINT)
    start = time.time()
    n = 100003
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    out = torch.empty_like(x)
    _probe_add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    m = 520
    a = torch.randn(m, m, device=dev, dtype=torch.bfloat16)
    b = torch.randn(m, m, device=dev, dtype=torch.bfloat16)
    c = torch.empty(m, m, device=dev, dtype=torch.bfloat16)
    _probe_dot_kernel[(triton.cdiv(m, 64), triton.cdiv(m, 64))](
        a, b, c, m, m, m, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64)
    torch.cuda.synchronize(dev)
    errors = {"add": float((out - (x + y)).abs().max()), "dot_bf16": _rel(c, a.cpu().double() @ b.cpu().double())}
    return _judge(errors, {"add": 1e-6, "dot_bf16": 2e-2}, f"compiled and ran in {time.time() - start:.0f}s, ")


def k_triton_cross(cap: Tuple[int, int]) -> Dict:
    """Compile the probe kernels for `cap` through Triton's LLVM and ptxas; nothing is launched."""
    if triton is None:
        return _result("FAIL", TRITON_IMPORT_ERROR, TRITON_HINT)
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas, sm_arch_from_capability
    from triton.compiler import ASTSource

    arch = cap[0] * 10 + cap[1]
    expected = sm_arch_from_capability(arch)
    ptxas = get_ptxas()
    dot_sig = {name: "*bf16" for name in ("a_ptr", "b_ptr", "c_ptr")}
    dot_sig.update({name: "i32" for name in ("M", "N", "K", "stride_am", "stride_ak", "stride_bk", "stride_bn",
                                             "stride_cm", "stride_cn")})
    dot_sig.update({name: "constexpr" for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K")})
    kernels = (
        ("add", _probe_add_kernel,
         {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"}, {"BLOCK": 1024}),
        ("dot_bf16", _probe_dot_kernel, dot_sig, {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}),
    )
    start = time.time()
    cubin_bytes, targets = {}, {}
    for name, fn, signature, constexprs in kernels:
        compiled = triton.compile(ASTSource(fn, signature, constexprs), target=GPUTarget("cuda", arch, 32))
        m = re.search(r"^\.target\s+(\S+)", compiled.asm["ptx"], re.M)
        targets[name] = m.group(1) if m else "?"
        cubin_bytes[name] = len(compiled.asm["cubin"])
    data = {"target": expected, "ptx_targets": targets, "cubin_bytes": cubin_bytes, "ptxas_path": ptxas.path,
            "ptxas_release": ptxas.version, "TRITON_PTXAS_PATH": os.environ.get("TRITON_PTXAS_PATH", "")}
    detail = (f"{expected} cubins: " + ", ".join(f"{k} {v} B" for k, v in cubin_bytes.items())
              + f"; triton {triton.__version__}, ptxas {ptxas.version} at {ptxas.path} ({time.time() - start:.0f}s)")
    wrong = {k: v for k, v in targets.items() if v != expected}
    if wrong or not all(cubin_bytes.values()):
        return _result("FAIL", detail + f"; PTX targets {targets}",
                       "Triton did not produce a cubin for the requested architecture.", **data)
    return _result("PASS", detail, **data)


def k_compile(torch, dev) -> Dict:
    import torch.nn as nn

    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Linear(1024, 256))
    model = model.to(dev, torch.bfloat16)
    x = torch.randn(64, 512, generator=torch.Generator().manual_seed(3)).to(dev, torch.bfloat16)

    def run(fn):
        xi = x.clone().requires_grad_()
        out = fn(xi)
        out.float().square().mean().backward()
        return out.detach(), xi.grad.detach()

    ref_out, ref_grad = run(model)
    start = time.time()
    out, grad = run(torch.compile(model))
    return _judge({"fwd": _rel(out, ref_out), "grad_x": _rel(grad, ref_grad)}, {"fwd": 3e-2, "grad_x": 5e-2},
                  f"MLP vs eager, compiled in {time.time() - start:.0f}s, ")


def k_flex(torch, dev) -> Dict:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    g = torch.Generator().manual_seed(4)
    B, H, S, D = 2, 8, 512, 64
    q, k, v, gout = (torch.randn(B, H, S, D, generator=g).bfloat16().float() for _ in range(4))
    ref_out, ref_grads = _attn_reference(torch, q, k, v, gout, causal=True)

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device=str(dev))
    compiled = torch.compile(flex_attention, dynamic=False)
    start = time.time()
    errors = {}
    for dtype, tag in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
        qd, kd, vd = (t.to(dev, dtype, copy=True).requires_grad_() for t in (q, k, v))
        out = compiled(qd, kd, vd, block_mask=block_mask)
        (out.float() * gout.to(dev)).sum().backward()
        errors[f"{tag}_fwd"] = _rel(out, ref_out)
        errors[f"{tag}_grad"] = max(_rel(t.grad, r) for t, r in zip((qd, kd, vd), ref_grads))
    return _judge(errors, {"bf16_fwd": 3e-2, "bf16_grad": 5e-2, "fp32_fwd": 1e-2, "fp32_grad": 2e-2},
                  f"causal block mask, compiled in {time.time() - start:.0f}s, ")


def k_fa_func(torch, dev, flash_attn_func) -> Dict:
    g = torch.Generator().manual_seed(5)
    B, S, H, D = 2, 512, 8, 64
    q, k, v, gout = (torch.randn(B, S, H, D, generator=g).bfloat16().float() for _ in range(4))
    ref_out, ref_grads = _attn_reference(torch, *(t.transpose(1, 2) for t in (q, k, v)), gout.transpose(1, 2),
                                         causal=False)
    qd, kd, vd = (t.to(dev, torch.bfloat16).requires_grad_() for t in (q, k, v))
    out = flash_attn_func(qd, kd, vd, causal=False)
    (out.float() * gout.to(dev)).sum().backward()
    errors = {"fwd": _rel(out.transpose(1, 2), ref_out)}
    for name, t, r in zip(("dq", "dk", "dv"), (qd, kd, vd), ref_grads):
        errors[name] = _rel(t.grad.transpose(1, 2), r)
    return _judge(errors, {"fwd": 3e-2, "dq": 5e-2, "dk": 5e-2, "dv": 5e-2})


def k_fa_varlen(torch, dev, flash_attn_varlen_func) -> Dict:
    g = torch.Generator().manual_seed(6)
    lens, H, D = (300, 212), 8, 64
    total = sum(lens)
    q, k, v, gout = (torch.randn(total, H, D, generator=g).bfloat16().float() for _ in range(4))
    outs, grads = [], ([], [], [])
    start = 0
    for length in lens:
        seg = slice(start, start + length)
        out, seg_grads = _attn_reference(torch, *(t[seg].transpose(0, 1).unsqueeze(0) for t in (q, k, v)),
                                         gout[seg].transpose(0, 1).unsqueeze(0), causal=True)
        outs.append(out[0].transpose(0, 1))
        for bucket, grad in zip(grads, seg_grads):
            bucket.append(grad[0].transpose(0, 1))
        start += length
    cu_seqlens = torch.tensor([0, lens[0], total], dtype=torch.int32, device=dev)
    qd, kd, vd = (t.to(dev, torch.bfloat16).requires_grad_() for t in (q, k, v))
    out = flash_attn_varlen_func(qd, kd, vd, cu_seqlens, cu_seqlens, max(lens), max(lens), causal=True)
    (out.float() * gout.to(dev)).sum().backward()
    errors = {"fwd": _rel(out, torch.cat(outs))}
    for name, t, bucket in zip(("dq", "dk", "dv"), (qd, kd, vd), grads):
        errors[name] = _rel(t.grad, torch.cat(bucket))
    return _judge(errors, {"fwd": 3e-2, "dq": 5e-2, "dk": 5e-2, "dv": 5e-2}, "2 packed causal sequences, ")


_MOE_CACHE: Dict[str, Dict] = {}
_MOE_INPUTS = ("route", "x", "w_gate", "w_up", "w_down")


def _moe_case(torch, dev) -> Dict:
    """Shared MoE inputs plus an fp32 per-expert reference (forward and gradients)."""
    if "case" in _MOE_CACHE:
        return _MOE_CACHE["case"]
    import torch.nn.functional as F

    E, H, I, T, K = 32, 1024, 512, 2048, 4
    g = torch.Generator().manual_seed(7)
    scores = torch.rand(T, E, generator=g)
    scores[:, 0] = -1.0  # expert 0 receives no tokens: exercises empty groups
    case = {
        "dims": (E, H, I, T, K),
        "x": torch.randn(T, H, generator=g),
        "w_gate": torch.randn(E, I, H, generator=g) * H ** -0.5,
        "w_up": torch.randn(E, I, H, generator=g) * H ** -0.5,
        "w_down": torch.randn(E, H, I, generator=g) * I ** -0.5,
        "route": torch.rand(T, K, generator=g),
        "sel": scores.topk(K, dim=-1).indices,
        "gout": torch.randn(T, H, generator=g),
    }
    for name in _MOE_INPUTS:
        case[name] = case[name].bfloat16().float()
    ref = {name: case[name].to(dev, copy=True).requires_grad_() for name in _MOE_INPUTS}
    sel = case["sel"].to(dev)
    flag = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = torch.zeros(T, H, device=dev)
        for slot in range(K):
            for e in range(E):
                idx = (sel[:, slot] == e).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                xe = ref["x"][idx]
                act = F.silu(xe @ ref["w_gate"][e].t()) * (xe @ ref["w_up"][e].t())
                out = out.index_add(0, idx, ref["route"][idx, slot:slot + 1] * (act @ ref["w_down"][e].t()))
        (out * case["gout"].to(dev)).sum().backward()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = flag
    case["ref_out"] = out.detach()
    case["ref_grads"] = {name: ref[name].grad for name in _MOE_INPUTS}
    _MOE_CACHE["case"] = case
    return case


def _import_lingbot(module: str, name: str):
    try:
        return getattr(importlib.import_module(module), name)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "lingbotvla":
            return None
        raise


def k_fused_moe(torch, dev) -> Dict:
    fused = _import_lingbot("lingbotvla.ops.fused_moe", "FusedMoeExpertFunction")
    available = _import_lingbot("lingbotvla.utils.import_utils", "is_fused_moe_available")
    if fused is None or available is None:
        return _result("FAIL", "lingbotvla is not importable", LINGBOT_HINT)
    if not available():
        return _result("FAIL", "is_fused_moe_available() is False (needs CUDA + triton)", TRITON_HINT)
    case = _moe_case(torch, dev)
    E, H, I, T, K = case["dims"]
    p = {name: case[name].to(dev, torch.bfloat16).requires_grad_() for name in _MOE_INPUTS}
    out = fused.apply(E, p["route"], case["sel"].to(dev), p["x"], p["w_gate"], p["w_up"], p["w_down"])
    (out.float() * case["gout"].to(dev)).sum().backward()
    errors = {"out": _rel(out, case["ref_out"])}
    errors.update({f"d_{name}": _rel(p[name].grad, case["ref_grads"][name]) for name in _MOE_INPUTS})
    limits = {key: 3e-2 if key == "out" else 5e-2 for key in errors}
    return _judge(errors, limits, f"E={E} H={H} I={I} T={T} top{K} (expert 0 empty), ")


def k_robby_moe(torch, dev) -> Dict:
    robby = _import_lingbot("lingbotvla.ops.robby_moe", "robby_moe_forward")
    if robby is None:
        return _result("FAIL", "lingbotvla is not importable", LINGBOT_HINT)
    case = _moe_case(torch, dev)
    with torch.no_grad():
        out = robby(*(case[name].to(dev, torch.bfloat16) for name in ("x", "route")), case["sel"].to(dev),
                    *(case[name].to(dev, torch.bfloat16) for name in ("w_gate", "w_up", "w_down")))
    res = _judge({"out": _rel(out, case["ref_out"])}, {"out": 3e-2}, "inference-only MoE, ")
    if res["status"] == "FAIL":
        res["hint"] = ("The model would silently use these wrong values at inference: disable it before loading "
                       "the model (runbook fallback table). Training is not affected.")
    return res


def k_memory(torch, dev, alloc_gib: float) -> Dict:
    free, total = torch.cuda.mem_get_info(dev)
    want = alloc_gib * 2**30 if alloc_gib > 0 else 0.8 * free
    buf = torch.empty(int(want // 2), dtype=torch.bfloat16, device=dev)
    buf.fill_(1.0)
    torch.cuda.synchronize(dev)
    del buf
    torch.cuda.empty_cache()
    return _result("PASS", f"allocated and filled {want / 2**30:.0f} GiB; total {total / 2**30:.0f} GiB, "
                           f"free before {free / 2**30:.0f} GiB",
                   total_gib=round(total / 2**30, 1), free_gib=round(free / 2**30, 1))


def k_video() -> Dict:
    try:
        import av
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return _result("FAIL", _short(f"{type(exc).__name__}: {exc}"),
                       "lingbotvla imports av at module level: pip install av==15.0.0.")
    with tempfile.TemporaryDirectory(prefix="gpu_env_probe_") as tmp:
        path = os.path.join(tmp, "probe.mp4")
        with av.open(path, "w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for i in range(10):
                frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 20, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(path) as container:
            decoded = sum(1 for _ in container.decode(video=0))
        try:
            from torchcodec.decoders import VideoDecoder

            torchcodec = f"torchcodec ok {tuple(VideoDecoder(path)[0].shape)}"
        except Exception as exc:  # noqa: BLE001
            torchcodec = "torchcodec failed: " + _short(f"{type(exc).__name__}: {exc}", 200)
    detail = f"av {av.__version__} decoded {decoded}/10 frames; {torchcodec}"
    if decoded != 10:
        return _result("FAIL", detail, "PyAV cannot round-trip a video here.")
    if torchcodec.startswith("torchcodec failed"):
        return _result("WARN", detail, "torchcodec needs system FFmpeg libraries (apt-get install -y ffmpeg); only "
                                       "video-mode datasets (LeRobot v3.0, AMD route) use it.")
    return _result("PASS", detail)


def check_torch_stack(rep: Report, device: str, alloc_gib: float) -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        rep.add("torch.import", "FAIL", _short(f"{type(exc).__name__}: {exc}"),
                "Run G1 inside the LingBot training env (torch 2.8.0).")
        return
    dev = torch.device(device)
    on_gpu = dev.type == "cuda"
    caps: List[Tuple[int, int]] = []
    if on_gpu:
        if not torch.cuda.is_available():
            rep.add("torch.cuda", "FAIL", f"torch {torch.__version__} (CUDA {torch.version.cuda}) sees no GPU",
                    "Check driver / container GPU passthrough (docker --gpus all) and that torch is a CUDA build.")
            return
        torch.cuda.set_device(dev)
        caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
        rep.guard("torch.cuda", k_torch_cuda, torch)
    else:
        rep.add("torch.cuda", "SKIP", f"device {dev}: self-test mode, GPU-only checks skipped")

    rep.guard("kernel.matmul", k_matmul, torch, dev)
    rep.guard("kernel.conv3d", k_conv3d, torch, dev)
    rep.guard("kernel.sdpa", k_sdpa, torch, dev)
    gpu_only = ("triton.toolchain", "triton.kernel", "flash_attn.func", "flash_attn.varlen",
                "lingbot.fused_moe", "lingbot.robby_moe", "gpu.memory")
    if on_gpu:
        rep.guard("triton.toolchain", k_triton_toolchain, caps)
        rep.guard("triton.kernel", k_triton_kernels, torch, dev)
    rep.guard("compile.inductor", k_compile, torch, dev)
    rep.guard("flex_attention", k_flex, torch, dev)
    if on_gpu:
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except Exception as exc:  # noqa: BLE001
            msg = _short(f"{type(exc).__name__}: {exc}")
            rep.add("flash_attn.func", "FAIL", msg, hint_for_error(msg) or FA_HINT)
            rep.add("flash_attn.varlen", "SKIP", "flash_attn import failed")
        else:
            rep.guard("flash_attn.func", k_fa_func, torch, dev, flash_attn_func)
            rep.guard("flash_attn.varlen", k_fa_varlen, torch, dev, flash_attn_varlen_func)
        rep.guard("lingbot.fused_moe", k_fused_moe, torch, dev)
        rep.guard("lingbot.robby_moe", k_robby_moe, torch, dev)
        _MOE_CACHE.clear()
        rep.guard("gpu.memory", k_memory, torch, dev, alloc_gib)
    else:
        for name in gpu_only:
            rep.add(name, "SKIP", "GPU only")
    rep.guard("video.decode", k_video)


# ---------------------------------------------------------------- NCCL (torchrun)

def _collectives(torch, dist, dev, rank: int, world: int) -> Dict[str, bool]:
    checks = {}
    x = torch.full((1 << 20,), float(rank + 1), device=dev)
    dist.all_reduce(x)
    checks["all_reduce"] = bool((x == world * (world + 1) / 2).all())
    gathered = torch.empty(world * 4, device=dev)
    dist.all_gather_into_tensor(gathered, torch.full((4,), float(rank), device=dev))
    checks["all_gather"] = bool((gathered == torch.arange(world, device=dev, dtype=torch.float32).repeat_interleave(4)).all())
    source = torch.arange(world * 4, device=dev, dtype=torch.float32)
    part = torch.empty(4, device=dev)
    dist.reduce_scatter_tensor(part, source)
    checks["reduce_scatter"] = bool((part == source[rank * 4:(rank + 1) * 4] * world).all())
    flags = torch.tensor([int(v) for v in checks.values()], device=dev)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    return {key: bool(value) for key, value in zip(checks, flags.tolist())}


def _bus_bandwidth(torch, dist, dev, world: int, nbytes: int, iters: int = 20) -> float:
    """all_reduce bus bandwidth in GB/s (nccl-tests convention), minimum across ranks."""
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    buf = torch.zeros(nbytes // torch.tensor([], dtype=dtype).element_size(), dtype=dtype, device=dev)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    for _ in range(5):
        dist.all_reduce(buf)
    sync()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(buf)
    sync()
    seconds = (time.perf_counter() - start) / iters
    busbw = nbytes / seconds / 1e9 * 2 * (world - 1) / world
    value = torch.tensor([busbw], dtype=torch.float64, device=dev)
    dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return float(value.item())


def nccl_main(out: str, device: str) -> int:
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:  # noqa: BLE001
        rep = Report("nccl")
        rep.add("nccl.collectives", "FAIL", _short(f"{type(exc).__name__}: {exc}"), "Run inside the LingBot env.")
        return rep.finish(out)
    for key, value in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29533"), ("RANK", "0"),
                       ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")):
        os.environ.setdefault(key, value)
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    use_cuda = not device.startswith("cpu")
    backend = "nccl" if use_cuda else "gloo"
    results: Dict = {}
    error = ""
    try:
        if use_cuda:
            torch.cuda.set_device(local)
            dev = torch.device("cuda", local)
            dist.init_process_group(backend, timeout=timedelta(minutes=5), device_id=dev)
        else:
            dev = torch.device("cpu")
            dist.init_process_group(backend, timeout=timedelta(minutes=5))
        results["collectives"] = _collectives(torch, dist, dev, rank, world)
        if world > 1:
            results["busbw"] = {label: round(_bus_bandwidth(torch, dist, dev, world, nbytes), 1)
                                for label, nbytes in (("64MiB", 64 << 20), ("1GiB", 1 << 30))}
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        print(f"[probe] rank {rank}: {error}", file=sys.stderr, flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    collectives = results.get("collectives", {})
    if rank != 0:
        return 1 if error or not all(collectives.values()) else 0

    nccl = ""
    if use_cuda:
        try:
            version = torch.cuda.nccl.version()
            nccl = ".".join(map(str, version)) if isinstance(version, tuple) else str(version)
        except Exception:  # noqa: BLE001
            pass
    rep = Report("nccl")
    env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
    head = f"world {world}, {backend}" + (f" {nccl}" if nccl else "")
    if collectives:
        bad = [k for k, v in collectives.items() if not v]
        rep.add("nccl.collectives", "FAIL" if bad else "PASS",
                f"{head}: " + ", ".join(f"{k} {'ok' if v else 'WRONG'}" for k, v in collectives.items()),
                "Collective results are wrong: do not train on this machine." if bad else "",
                {"env": env, "results": collectives})
    else:
        rep.add("nccl.collectives", "FAIL", f"{head}: {_short(error)}", hint_for_error(error), {"env": env})
    busbw = results.get("busbw")
    if world == 1:
        rep.add("nccl.busbw", "SKIP", "single process; launch with torchrun --nproc-per-node <gpus>")
    elif busbw is None:
        rep.add("nccl.busbw", "FAIL", _short(error) or "not measured", hint_for_error(error))
    else:
        slow = busbw["1GiB"] < 200
        rep.add("nccl.busbw", "WARN" if slow else "PASS",
                f"all_reduce busbw 64MiB {busbw['64MiB']} GB/s, 1GiB {busbw['1GiB']} GB/s (min across ranks)",
                "Far below NVLink speeds: rerun with NCCL_DEBUG=INFO and check whether P2P/NVLS is used "
                "(Fabric Manager, topology)." if slow else "", busbw)
    return rep.finish(out)


# ---------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform_only", action="store_true", help="G0: OS / driver / GPU / network checks (stdlib)")
    ap.add_argument("--nccl", action="store_true", help="collectives + bandwidth; launch with torchrun")
    ap.add_argument("--data_dir", default=".", help="disk for data, caches, weights and checkpoints (default: cwd)")
    ap.add_argument("--device", default="cuda:0", help="device for kernel checks; 'cpu' = self-test of this script")
    ap.add_argument("--alloc_gib", type=float, default=0, help="GPU memory check size (default: 80%% of free)")
    ap.add_argument("--lingbot_root", default="", help="lingbot-vla-v2 checkout to put on sys.path if not installed")
    ap.add_argument("--skip_net", action="store_true", help="skip the network reachability check")
    ap.add_argument("--triton_target", type=parse_compute_cap, default=None, metavar="X.Y",
                    help="only compile the Triton probe kernels for this compute capability (10.3 = B300); "
                         "needs triton, but no GPU")
    ap.add_argument("--out", default="", help="write the JSON report here")
    args = ap.parse_args(argv)

    if args.lingbot_root:
        sys.path.insert(0, os.path.abspath(os.path.expanduser(args.lingbot_root)))
    if args.nccl:
        return nccl_main(args.out, args.device)
    if args.triton_target:
        rep = Report("triton_cross")
        rep.guard("triton.cross_compile", k_triton_cross, args.triton_target)
        return rep.finish(args.out)

    rep = Report("platform" if args.platform_only else "full")
    check_os(rep)
    gpus = check_gpus(rep)
    check_resources(rep, args.data_dir, len(gpus))
    check_build_tools(rep, len(gpus))
    if gpus:
        check_driver(rep, gpus)
        check_fabric(rep)
        check_topology(rep)
    if args.platform_only:
        if not args.skip_net:
            check_network(rep)
    else:
        check_python_stack(rep)
        check_torch_stack(rep, args.device, args.alloc_gib)
    return rep.finish(args.out)


if __name__ == "__main__":
    sys.exit(main())
