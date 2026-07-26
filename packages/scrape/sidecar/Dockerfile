# Debian (not Ubuntu) base: Debian's `chromium` apt package is a real binary;
# Ubuntu's is a snap-wrapper stub that fails with no snapd in a container.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# x11vnc/websockify/novnc are the human-in-the-loop path: they put the Xvfb
# display Chromium already draws to in front of a person. Installed here but
# started on demand by hitl.py, never at boot -- an idle VNC surface would exist
# for the whole life of a container that is usually unattended. Debian ships all
# three, and novnc's static client lands in /usr/share/novnc (verified in this
# base image, not recalled: that tree has vnc.html / vnc_lite.html and no
# index.html, which is why hitl.py names a page explicitly).
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        xvfb \
        curl \
        x11vnc \
        websockify \
        novnc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir \
        "nodriver>=0.50,<0.51" \
        "fastapi>=0.115,<0.116" \
        "uvicorn[standard]>=0.32,<0.33"

COPY main.py /app/main.py
COPY hitl.py /app/hitl.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# EGRESS_PROXY / EGRESS_NAME are unset by default: the default route, named
# "direct". A deployment that wants this container to leave by TOR or a VPN sets
# EGRESS_PROXY to a --proxy-server value and EGRESS_NAME to the identifier it
# records results against. Chromium takes --proxy-server process-wide, so one
# container is one exit; two exits are two containers.
ENV DISPLAY_NUM=99 \
    NOVNC_ROOT=/usr/share/novnc \
    SCREEN_GEOM=1920x1080x24 \
    CHROMIUM_PATH=/usr/bin/chromium \
    PORT=8088

# 6080 is websockify: the only route to the display, and only while a session
# is open. x11vnc binds loopback and is never published.
EXPOSE 8088 6080

ENTRYPOINT ["/app/entrypoint.sh"]
