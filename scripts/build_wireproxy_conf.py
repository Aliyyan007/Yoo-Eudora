"""Convert a wgcf WireGuard profile into a wireproxy config.

Only whitelisted fields are emitted (a wgcf profile may contain keys wireproxy
doesn't understand), IPv6 lines are dropped (containers usually lack v6), and a
local HTTP CONNECT proxy section is appended. HTTP rather than SOCKS5 because
aiohttp's `proxy=` kwarg — used by discord.py-self's CDN leg — only accepts
http:// proxies; libcurl (REST + gateway WS) handles HTTP CONNECT fine too.

Usage: build_wireproxy_conf.py <wgcf-profile.conf> <wireproxy.conf> <port>
"""
import sys


def _parse(path: str) -> dict:
    sections: dict = {}
    current = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("#", ";")):
                continue
            if s.startswith("[") and s.endswith("]"):
                current = s[1:-1].strip()
                sections.setdefault(current, [])
            elif "=" in s and current:
                k, v = s.split("=", 1)
                sections[current].append((k.strip(), v.strip()))
    return sections


def _v4_only(value: str) -> str:
    return ", ".join(p.strip() for p in value.split(",") if ":" not in p)


def main() -> int:
    src, dst, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
    conf = _parse(src)
    iface = dict(conf.get("Interface", []))
    peer = dict(conf.get("Peer", []))

    private_key = iface.get("PrivateKey")
    public_key = peer.get("PublicKey")
    if not private_key or not public_key:
        print("[warp] wgcf profile missing keys", file=sys.stderr)
        return 1

    address = _v4_only(iface.get("Address", ""))
    dns = iface.get("DNS", "1.1.1.1")
    endpoint = peer.get("Endpoint", "engage.cloudflareclient.com:2408")
    allowed = _v4_only(peer.get("AllowedIPs", "0.0.0.0/0")) or "0.0.0.0/0"

    out = (
        "[Interface]\n"
        f"PrivateKey = {private_key}\n"
        f"Address = {address}\n"
        f"DNS = {dns}\n"
        "MTU = 1280\n"
        "\n[Peer]\n"
        f"PublicKey = {public_key}\n"
        f"AllowedIPs = {allowed}\n"
        f"Endpoint = {endpoint}\n"
        # Keep the NAT mapping warm — without a persistent keepalive the peer
        # goes quiet in ~15s behind NAT, and wireproxy re-handshakes every
        # ~30s ("stopped hearing back" spam + brief stalls on each rekey).
        "PersistentKeepalive = 25\n"
        "\n[http]\n"
        f"BindAddress = 127.0.0.1:{port}\n"
    )
    with open(dst, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"[warp] wireproxy config written ({dst})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
