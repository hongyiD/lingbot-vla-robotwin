# LingBot-VLA 2.0 training image for B300 (sm_103) on a Kubernetes GPU cluster.
# Build on any x86_64 Linux box with internet; no GPU needed. Use build_image.sh, which stages the context:
#   Dockerfile, embodiedx/{tools,data/manifests,k8s,docker/check_image.sh}, wheels/ (optional local flash-attn wheel)
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Asia/Shanghai \
    PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# APT_MIRROR (e.g. http://mirrors.aliyun.com/ubuntu) replaces archive/security.ubuntu.com; empty keeps the defaults
ARG APT_MIRROR=
# Compiler for Triton's launcher builds at runtime; libGL/glib for opencv-python; ffmpeg libs for torchcodec
RUN if [ -n "$APT_MIRROR" ]; then sed -i -e "s#http://archive.ubuntu.com/ubuntu#$APT_MIRROR#g" \
        -e "s#http://security.ubuntu.com/ubuntu#$APT_MIRROR#g" /etc/apt/sources.list; fi \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates libgl1 libglib2.0-0 ffmpeg \
        tmux vim less procps iproute2 rsync pigz tzdata \
    && rm -rf /var/lib/apt/lists/*

# The official environment script drives conda, so keep conda; one empty Python 3.12 env for it to resume into
ARG MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
RUN curl -fsSL -o /tmp/miniforge.sh "$MINIFORGE_URL" \
    && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda create -y -n lingbotvla python=3.12 pip \
    && /opt/conda/bin/conda clean -afy
ENV PATH=/opt/conda/envs/lingbotvla/bin:/opt/conda/bin:$PATH

# Editable installs point into this checkout: the path must not change after the install step
ARG LINGBOT_REPO=https://github.com/Robbyant/lingbot-vla-v2.git
ARG LINGBOT_SHA=be969b8fd117fb70550c5d4bf4bc328211b5b1b6
RUN git init -q /opt/lingbot-vla-v2 && cd /opt/lingbot-vla-v2 \
    && git remote add origin "$LINGBOT_REPO" \
    && git fetch -q --depth 1 origin "$LINGBOT_SHA" && git checkout -q --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$LINGBOT_SHA" \
    && chmod -R a+rwX /opt/lingbot-vla-v2

# PIP_INDEX_URL can point at a PyPI mirror; the torch line always uses the PyTorch index.
# torch==2.8.0 in the official script accepts +cu129 (PEP 440 local version), so --resume keeps this build.
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# Official script minus its CUDA-available assertion (no GPU inside docker build); a wheel in wheels/ wins over
# the download. The patched copy must sit in tools/ because the script derives REPO_ROOT from its own path.
ARG FA_WHEEL_BASE=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=wheels,target=/wheels \
    set -eux; cd /opt/lingbot-vla-v2; \
    abi=$(python -c "import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())"); \
    whl="flash_attn-2.8.3+cu12torch2.8cxx11abi${abi}-cp312-cp312-linux_x86_64.whl"; \
    if [ -f "/wheels/$whl" ]; then cp "/wheels/$whl" /tmp/; else curl -fL -o "/tmp/$whl" "$FA_WHEEL_BASE/$whl"; fi; \
    test "$(grep -c 'assert torch.cuda.is_available()' tools/create_train_env.sh)" = 1; \
    sed '/assert torch.cuda.is_available()/d' tools/create_train_env.sh > tools/.create_train_env.nogpu.sh; \
    test "$(grep -c 'assert torch.cuda.is_available()' tools/.create_train_env.nogpu.sh)" = 0; \
    bash tools/.create_train_env.nogpu.sh --resume --flash-attn-wheel "/tmp/$whl"; \
    rm -f tools/.create_train_env.nogpu.sh "/tmp/$whl"; \
    conda clean -afy

# Triton 3.4.0 bundles ptxas 12.8, which cannot target sm_103a. Triton silently falls back to the bundled
# ptxas if this path is not executable, hence the chmod and the version check.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --target /opt/cuda-ptxas-12.9 nvidia-cuda-nvcc-cu12==12.9.86 \
    && chmod +x /opt/cuda-ptxas-12.9/nvidia/cuda_nvcc/bin/ptxas \
    && /opt/cuda-ptxas-12.9/nvidia/cuda_nvcc/bin/ptxas --version | grep -q "release 12.9"
ENV TRITON_PTXAS_PATH=/opt/cuda-ptxas-12.9/nvidia/cuda_nvcc/bin/ptxas

COPY embodiedx/ /opt/embodiedx/
RUN bash /opt/embodiedx/docker/check_image.sh /opt/embodiedx/reports/build

LABEL org.opencontainers.image.title="lingbotvla-train" org.opencontainers.image.revision="$LINGBOT_SHA"
WORKDIR /opt/lingbot-vla-v2
CMD ["sleep", "infinity"]
