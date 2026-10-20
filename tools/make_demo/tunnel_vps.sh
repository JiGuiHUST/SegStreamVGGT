#!/usr/bin/env bash
# Reverse SSH tunnel: expose this demo on the VPS Tailscale IP so a remote
# browser can open http://<TAILSCALE-IP>:8787 .
#
# This mirrors the existing DSH tunnel (3080) and does NOT touch it.
# Only the Tailscale interface + VPS loopback are bound, not the public IP.
#
# Start:  nohup setsid ./tunnel_vps.sh >/dev/null 2>&1 &
# Stop:   pkill -f 'tunnel_vps[.]sh'; pkill -f 'ssh -N -R 8787'
# Status: ps aux | grep '[s]sh -N -R 8787'; tail logs/tunnel.log
set -u
cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/tunnel.log"
VPS_HOST="${VPS_HOST:-<VPS-HOST>}"
VPS_PORT="${VPS_PORT:-<VPS-SSH-PORT>}"
VPS_KEY="${VPS_KEY:-~/.ssh/id_ed25519_vps}"
TAILSCALE_IP="${TAILSCALE_IP:-<TAILSCALE-IP>}"

echo "[tunnel] supervisor started $(date)" >> "$LOG"
while true; do
  echo "[tunnel] connecting $(date)" >> "$LOG"
  ssh -N \
    -R 8787:127.0.0.1:8787 \
    -R "${TAILSCALE_IP}:8787:127.0.0.1:8787" \
    -p "$VPS_PORT" -i "$VPS_KEY" \
    -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
    -o TCPKeepAlive=yes -o LogLevel=ERROR \
    "root@${VPS_HOST}" >> "$LOG" 2>&1
  echo "[tunnel] disconnected $(date), retry in 5s" >> "$LOG"
  sleep 5
done
