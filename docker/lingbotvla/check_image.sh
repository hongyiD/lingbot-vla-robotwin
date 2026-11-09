#!/usr/bin/env bash
# Image self-check, no GPU needed: package versions, B300 kernel coverage, Triton sm_103a cross-compile.
# Runs as the last docker build step (any FAIL stops the build). Rerun anywhere:
#   docker run --rm <image> bash /opt/embodiedx/docker/check_image.sh /tmp/check
set -euo pipefail
OUT=${1:-/opt/embodiedx/reports/build}
mkdir -p "$OUT"

python - <<'PY' 2>&1 | tee "$OUT/check_image.txt"
import importlib
import importlib.util
import os
import re
import subprocess
import sys
from importlib.metadata import version

failed = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        failed.append(what)


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=600).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<{type(exc).__name__}: {exc}>"


import torch
import triton

check(torch.__version__ == "2.8.0+cu129" and torch.version.cuda == "12.9",
      f"torch {torch.__version__}, CUDA {torch.version.cuda} (want 2.8.0+cu129 / 12.9)")
# torch.cuda.get_arch_list() is empty without a GPU; the compiled-in flags are not
archs = (torch._C._cuda_getArchFlags() or "").split()
check(bool({"sm_100", "sm_103"} & set(archs)), f"torch kernels: {' '.join(archs)} (B300 needs sm_100 or sm_103)")
check(triton.__version__ == "3.4.0", f"triton {triton.__version__}")

ptxas = os.environ.get("TRITON_PTXAS_PATH", "")
banner = run([ptxas, "--version"]).splitlines() if os.access(ptxas, os.X_OK) else []
# The release line is not the last one: 12.9 ends the banner with "Build cuda_12.9.r12.9/compiler..."
release = next((line.strip() for line in banner if "release " in line), "")
check("release 12.9" in release, f"TRITON_PTXAS_PATH={ptxas or '(unset)'}: {release or 'missing or not executable'}")

import flash_attn

spec = importlib.util.find_spec("flash_attn_2_cuda")
cuobjdump = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
fa_archs = sorted(set(re.findall(r"sm_\d+[af]?", run([cuobjdump, "--list-elf", spec.origin])))) if spec else []
check(flash_attn.__version__ == "2.8.3", f"flash_attn {flash_attn.__version__}")
check(bool({"sm_100", "sm_100f", "sm_103", "sm_103f"} & set(fa_archs)),
      f"flash_attn_2_cuda cubins: {' '.join(fa_archs) or 'none'} (B300 needs a non-'a' sm_100/sm_103 cubin)")

for mod in ("lingbotvla", "lerobot", "cv2", "av", "moge", "mdm", "utils3d",
            "lerobot.datasets.v30.convert_dataset_v21_to_v30"):
    try:
        importlib.import_module(mod)
        check(True, f"import {mod}")
    except Exception as exc:  # report every broken import, not just the first
        check(False, f"import {mod}: {type(exc).__name__}: {exc}")


def ver(dist):
    try:
        return version(dist)
    except Exception:
        return "-"


print("versions: " + ", ".join(f"{d} {ver(d)}" for d in (
    "lerobot", "transformers", "datasets", "huggingface-hub", "numpy", "opencv-python", "opencv-python-headless")))
sys.exit(1 if failed else 0)
PY

TRITON_CACHE_DIR=$(mktemp -d) python /opt/embodiedx/tools/gpu_env_probe.py --triton_target 10.3 --out "$OUT/triton_sm103.json"
python /opt/embodiedx/k8s/make_train_configs.py --help > /dev/null
pip freeze > "$OUT/pip_freeze.txt"
echo "check_image: all PASS, reports in $OUT"
