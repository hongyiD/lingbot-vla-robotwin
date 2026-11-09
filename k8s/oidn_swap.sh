#!/usr/bin/env bash
# Make SAPIEN's ray-tracing denoiser work on Blackwell (B300, compute capability 10.3).
#
# sapien 3.0.0b1 bundles OIDN 2.0.1 in sapien/oidn_library/. Its CUDA module only carries kernels for sm_70/75/80/90
# and rejects devices above SM 9.9, so on B300 svulkan2 logs "OIDN Error: unsupported device type: CUDA" and renders
# without denoising (rt+oidn images come out identical to rt+none). OIDN 2.3.3 adds sm_100/sm_120 kernels and raises
# the limit to SM 12.0; its C API is backward compatible within major version 2.
#
# sapien/_oidn_tricks.py preloads oidn_library/libOpenImageDenoise_core.so.2.0.1 and libOpenImageDenoise.so.2.0.1 by
# path; libsvulkan2.so then binds libOpenImageDenoise.so.2 by soname. This script keeps those two file names as
# symlinks to the 2.3.3 libraries and puts libOpenImageDenoise_device_cuda.so.2.3.3 next to them, where the 2.3.3
# core looks for it. The originals are kept as *.orig.
#
# Usage: oidn_swap.sh <python of the env that has sapien> [--restore]
set -euo pipefail

PY=${1:?usage: oidn_swap.sh <python> [--restore]}
MODE=${2:-swap}
D=${D:-/mnt/work}
VER=2.3.3
TGZ=oidn-$VER.x86_64.linux.tar.gz
SHA256=3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f  # 64809031 bytes, GitHub release asset
URL=https://github.com/RenderKit/oidn/releases/download/v$VER/$TGZ
OLD=(libOpenImageDenoise.so.2.0.1 libOpenImageDenoise_core.so.2.0.1)

OL=$("$PY" -c 'import importlib.util, os; s = importlib.util.find_spec("sapien"); print(os.path.join(os.path.dirname(s.origin), "oidn_library"))')
[ -d "$OL" ] || { echo "oidn_swap: $OL not found" >&2; exit 1; }
cd "$OL"

if [ "$MODE" = "--restore" ]; then
  for f in "${OLD[@]}"; do
    if [ -f "$f.orig" ]; then rm -f "$f"; mv "$f.orig" "$f"; fi
  done
  rm -f libOpenImageDenoise*.so."$VER"
  ls -l "$OL"
  exit 0
fi

mkdir -p "$D/downloads"
T=$D/downloads/$TGZ
if [ ! -s "$T" ]; then
  export https_proxy=${https_proxy:-${HTTPS_PROXY:-${http_proxy:-}}}
  "$PY" -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$URL" "$T.part"
  mv "$T.part" "$T"
fi
echo "$SHA256  $T" | sha256sum -c -

X=$(mktemp -d)
trap 'rm -rf "$X"' EXIT
tar xzf "$T" -C "$X" --strip-components=1 \
  "oidn-$VER.x86_64.linux/lib/libOpenImageDenoise.so.$VER" \
  "oidn-$VER.x86_64.linux/lib/libOpenImageDenoise_core.so.$VER" \
  "oidn-$VER.x86_64.linux/lib/libOpenImageDenoise_device_cuda.so.$VER"
install -m 755 "$X"/lib/libOpenImageDenoise*.so."$VER" "$OL"/

for f in "${OLD[@]}"; do
  if [ ! -L "$f" ]; then mv "$f" "$f.orig"; fi
done
ln -sfn "libOpenImageDenoise.so.$VER" libOpenImageDenoise.so.2.0.1
ln -sfn "libOpenImageDenoise_core.so.$VER" libOpenImageDenoise_core.so.2.0.1
ls -l "$OL"
echo "oidn_swap: $OL now uses OIDN $VER"
