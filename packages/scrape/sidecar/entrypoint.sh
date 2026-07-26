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

# A window manager, without which this display cannot be operated by a human. Bare Xvfb maps
# windows in creation order with no title bars, no alt-tab and no click-to-focus, so a session
# holding several targets stacks them and only the last one opened is reachable. That makes the
# bounded working set undeliverable: three of four slots would hold targets nobody can get to.
#
# Started at boot rather than with the VNC, unlike x11vnc and websockify. Those are an exposure
# surface and stay down until someone arrives; a window manager is not reachable from outside
# and costs a few megabytes of RSS, and starting it on demand would mean restacking windows
# that already exist -- openbox adopts what is mapped when it starts, but Chromium has already
# chosen its geometry by then.
# `openbox-session` rather than `openbox`, because it runs the autostart file and plain openbox
# does not. Everything the desktop needs -- the DPI that UI_SCALE turns into Xft.dpi, and the
# taskbar -- lives there instead of here.
#
# That placement is the fix, not a preference. Doing it in this script means racing Xvfb: the
# socket path appears before the server accepts connections, and an `xrdb` that loses that race
# reports success having set nothing, so the log claimed a DPI that `xrdb -query` could not
# find. Autostart runs after the window manager has connected, which is proof the server is
# ready rather than a guess about when it will be.
#
# `cleanup` reaps it: `pkill -P $$` takes every direct child.
echo "[entrypoint] starting openbox-session on ${DISPLAY}"
openbox-session &

echo "[entrypoint] starting nodriver sidecar on 0.0.0.0:${PORT}"
exec uvicorn main:app --host 0.0.0.0 --port "${PORT}"
