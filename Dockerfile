FROM python:3.12.7-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_WAIT_POLICY=PASSIVE

WORKDIR /app

# libopus0  — Opus codec required by opuslib-next (voice receive/send)
# libgomp1  — OpenMP runtime needed by onnxruntime / sherpa-onnx
# git       — pip git+ installs (discord.py-self)
# curl+ca-certificates — WARP sidecar bootstrap + proxy health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 \
    libgomp1 \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Cloudflare WARP sidecar tooling (wgcf registers a throwaway WireGuard
# device, wireproxy exposes it as a local HTTP CONNECT proxy). Discord
# blocks Render's datacenter egress for user tokens but cannot block
# Cloudflare's own AS — this is the DISCORD_PROXY backend. Enabled only
# when USE_WARP_PROXY=1 (see scripts/warp_entrypoint.sh).
ARG WGCF_SHA256=572d102720ee39a6a0a6a22a0b4689fdf92b8974ede927111e3daf8d2ed5dec1
ARG WIREPROXY_SHA256=e88c1d090740373fc606c1bafd81d9a5eadc642cce5667616e20e9d7a444f51c
RUN curl -fsSL -o /tmp/wgcf "https://github.com/ViRb3/wgcf/releases/download/v2.2.29/wgcf_2.2.29_linux_amd64" \
    && curl -fsSL -o /tmp/wireproxy.tgz "https://github.com/windtf/wireproxy/releases/download/v1.1.3/wireproxy_linux_amd64.tar.gz" \
    && echo "${WGCF_SHA256}  /tmp/wgcf" | sha256sum -c - \
    && echo "${WIREPROXY_SHA256}  /tmp/wireproxy.tgz" | sha256sum -c - \
    && mv /tmp/wgcf /usr/local/bin/wgcf \
    && tar -xzf /tmp/wireproxy.tgz -C /usr/local/bin \
    && rm /tmp/wireproxy.tgz \
    && chmod +x /usr/local/bin/wgcf /usr/local/bin/wireproxy

COPY requirements-render.txt .
# discord-native-voice installed with --no-deps: it declares
# discord.py-self>=2.2.0 but the pinned git build reports 2.2.0a0
RUN pip install -r requirements-render.txt \
    && pip install --no-deps "discord-native-voice>=0.1.1"

COPY . .

CMD ["bash", "scripts/warp_entrypoint.sh"]
