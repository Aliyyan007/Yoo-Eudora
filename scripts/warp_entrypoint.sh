#!/usr/bin/env bash
# Render entrypoint. When USE_WARP_PROXY=1 (and DISCORD_PROXY is unset),
# registers a throwaway Cloudflare WARP device, brings up a userspace
# WireGuard tunnel (wireproxy) exposing a local HTTP CONNECT proxy, verifies
# Discord is reachable through it, and exports DISCORD_PROXY so the bot's
# REST + gateway WS + CDN traffic all egress on a Cloudflare IP — which
# Discord does not block (Discord itself runs on Cloudflare).
#
# Failure mode is safe: if any step fails we continue WITHOUT a proxy rather
# than pointing the bot at a dead one.

set -uo pipefail

WARP_PORT="${WARP_PROXY_PORT:-40000}"
WARP_DIR="/tmp/warp"

start_warp() {
    echo "[warp] registering Cloudflare WARP device..."
    mkdir -p "$WARP_DIR" && cd "$WARP_DIR" || return 1
    wgcf register --accept-tos || { echo "[warp] wgcf register failed" >&2; return 1; }
    wgcf generate || { echo "[warp] wgcf generate failed" >&2; return 1; }
    python /app/scripts/build_wireproxy_conf.py \
        "$WARP_DIR/wgcf-profile.conf" "$WARP_DIR/wireproxy.conf" "$WARP_PORT" || return 1
    cd /app || return 1

    wireproxy -c "$WARP_DIR/wireproxy.conf" &
    echo "[warp] wireproxy started (pid $!), waiting for proxy on :$WARP_PORT"

    for _ in $(seq 1 45); do
        if curl -fsS -m 5 --proxy "http://127.0.0.1:$WARP_PORT" \
                -o /dev/null https://discord.com/api/v9/gateway; then
            echo "[warp] Discord reachable through WARP"
            return 0
        fi
        sleep 1
    done
    echo "[warp] proxy never came up" >&2
    return 1
}

if [ "${USE_WARP_PROXY:-0}" = "1" ] && [ -z "${DISCORD_PROXY:-}" ]; then
    if start_warp; then
        export DISCORD_PROXY="http://127.0.0.1:${WARP_PORT}"
        echo "[warp] DISCORD_PROXY=$DISCORD_PROXY"
    else
        echo "[warp] WARP setup failed — continuing on direct egress" >&2
    fi
elif [ -n "${DISCORD_PROXY:-}" ]; then
    echo "[warp] DISCORD_PROXY already set — WARP sidecar skipped"
fi

exec python -m src.main "$@"
