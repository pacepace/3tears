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
#
# openbox is a window manager, and without one this display is not operable by a human.
# Xvfb draws windows in map order with no title bars, no alt-tab and no click-to-focus, so a
# session holding several targets stacks them and only the last one opened can be reached --
# which makes the bounded working set undeliverable, since three of four slots would be
# occupied by targets nobody can get to. Found by an operator on the real screen: "I see three
# targets open, I cannot switch between them."
#
# openbox alone was not enough. It restored focus and stacking, and the operator's next report
# was that switching between four windows was still awkward and a minimised one could not be
# recovered at all -- there was nowhere for it to go. tint2 is the taskbar: one click per
# target, a visible list of what the session currently holds, and somewhere for a minimised
# window to come back from.
#
# The alternative considered and rejected was removing the minimise button, which would have
# closed the trap without making switching any easier. That fixes the symptom the operator hit
# second and ignores the one they hit first.
#
# x11-xserver-utils carries xrdb, which is how UI_SCALE reaches openbox and tint2: both size
# their furniture from Xft.dpi and read it once at startup. Without it the page would scale and
# the titlebars and taskbar would not, and a mismatch like that reads as a rendering fault
# rather than a setting. It also brings xrandr, which is what established that Xvfb has a single
# fixed mode and therefore cannot follow a client's viewport.
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        xvfb \
        curl \
        x11vnc \
        websockify \
        novnc \
        openbox \
        tint2 \
        x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/*

# One virtual desktop, and no way to leave it.
#
# openbox ships four, with the mousewheel bound to GoToDesktop on the desktop background. On a
# display whose whole job is holding a bounded set of targets in front of one person, that is a
# trap rather than a feature: an idle scroll moves the operator to an empty desktop, their
# targets appear to have vanished, and tint2's default `single_desktop` taskbar shows nothing
# there either -- which is exactly how a working taskbar reads as a missing one.
#
# Both edits are needed. Dropping to one desktop removes the destination; unbinding the wheel
# removes the gesture, so a stray scroll over the background does nothing at all instead of
# silently succeeding at something pointless.
RUN sed -i \
        -e '/<desktops>/,/<\/desktops>/ s|<number>4</number>|<number>1</number>|' \
        -e '/<context name="Desktop">/,/<\/context>/ s|<action name="GoToDesktop"><to>previous</to></action>||' \
        -e '/<context name="Desktop">/,/<\/context>/ s|<action name="GoToDesktop"><to>next</to></action>||' \
        /etc/xdg/openbox/rc.xml \
    && grep -q '<number>1</number>' /etc/xdg/openbox/rc.xml


WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir \
        "nodriver>=0.50,<0.51" \
        "fastapi>=0.115,<0.116" \
        "uvicorn[standard]>=0.32,<0.33"

COPY openbox-autostart /etc/xdg/openbox/autostart
COPY main.py /app/main.py
COPY hitl.py /app/hitl.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# EGRESS_PROXY / EGRESS_NAME are unset by default: the default route, named
# "direct". A deployment that wants this container to leave by TOR or a VPN sets
# EGRESS_PROXY to a --proxy-server value and EGRESS_NAME to the identifier it
# records results against.
#
# That sets the container's DEFAULT exit, not its only one: a render can ask for a
# different one per request, because a browser context carries its own proxy even
# though the --proxy-server flag is process-wide. One container therefore serves
# several exits, and a second copy is not how you get a second exit.
ENV DISPLAY_NUM=99 \
    NOVNC_ROOT=/usr/share/novnc \
    SCREEN_GEOM=1920x1080x24 \
    CHROMIUM_PATH=/usr/bin/chromium \
    PORT=8088

# 6080 is websockify: the only route to the display, and only while a session
# is open. x11vnc binds loopback and is never published.
EXPOSE 8088 6080

ENTRYPOINT ["/app/entrypoint.sh"]
