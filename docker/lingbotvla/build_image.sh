#!/usr/bin/env bash
# Build the LingBot training image from a small staged context (tools, manifests, k8s helpers; no data, no .venv).
# Usage: bash build_image.sh <tag> [extra docker build args...]
#   bash build_image.sh 20261018-be969b8-r1
#   bash build_image.sh 20261018-be969b8-r1 --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#        --build-arg HTTPS_PROXY=http://<proxy>:<port>
# A local flash-attn wheel in ./wheels/ (same file name as the release asset) skips the GitHub download.
set -euo pipefail
TAG=${1:?usage: build_image.sh <tag> [docker build args...]}
shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EX=$(cd "$HERE/../.." && pwd)
IMAGE=${IMAGE:-lingbotvla-train}

CTX=$(mktemp -d)
trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/embodiedx/docker" "$CTX/wheels"
tar -C "$EX" --exclude .venv --exclude tests --exclude __pycache__ --exclude .pytest_cache \
    -cf - tools data/manifests k8s | tar -C "$CTX/embodiedx" -xf -
cp "$HERE/check_image.sh" "$CTX/embodiedx/docker/"
cp "$HERE/Dockerfile" "$CTX/"
if compgen -G "$HERE/wheels/*.whl" > /dev/null; then cp "$HERE"/wheels/*.whl "$CTX/wheels/"; fi
printf 'image=%s:%s\nbuilt_at=%s\nbuilt_on=%s\n' "$IMAGE" "$TAG" "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$(hostname)" > "$CTX/embodiedx/IMAGE_INFO"

DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -t "$IMAGE:$TAG" "$@" "$CTX"
docker image inspect "$IMAGE:$TAG" --format 'built {{index .RepoTags 0}}  {{.Architecture}}  {{.Size}} bytes'
