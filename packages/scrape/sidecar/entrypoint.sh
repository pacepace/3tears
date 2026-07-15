#!/usr/bin/env bash
set -euo pipefail

# Xvfb startup discipline mirrors Pace's 14-eng-playwright-headed reference
# repo: poll for the X11 socket file directly rather than depending on
# x11-utils, and clean up stale lock files from a previous run before and
# after.

DISPLAY_NUM="${DISPLAY_NUM:-99}"
SCREEN_GEOM="${SCREEN_GEOM:-1920x1080x24}"
PORT="${PORT:-8088}"

export DISPLAY=":${DISPLAY_NUM}"

cleanup() {
    echo "[entrypoint] cleaning up"
    pkill -P $$ || true
    rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}" || true
}
trap cleanup EXIT INT TERM

rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}" || true

echo "[entrypoint] starting Xvfb on ${DISPLAY} ${SCREEN_GEOM}"
Xvfb "${DISPLAY}" -screen 0 "${SCREEN_GEOM}" -nolisten tcp -ac &

for _ in $(seq 1 100); do
    if [ -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; then
        break
    fi
    sleep 0.1
done
[ -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ] || { echo "[entrypoint] Xvfb failed to start"; exit 1; }

echo "[entrypoint] starting nodriver sidecar on 0.0.0.0:${PORT}"
exec uvicorn main:app --host 0.0.0.0 --port "${PORT}"
