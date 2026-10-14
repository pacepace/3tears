# The remote test lane's image (scripts/remote-setup.sh builds it on each host).
#
# It runs the unit + enforcement suite only. Integration tests spin NATS/Postgres
# through testcontainers, which needs a Docker daemon this container is not given,
# so no Postgres binaries and no docker CLI live here. The nodriver sidecar is
# AGPL and never enters the workspace venv; nothing here installs it.
#
# Python and uv match CI (.github/workflows/ci.yml installs 3.14; uv is pinned
# here to the release the workspace is developed with, since CI floats "latest").
FROM ghcr.io/astral-sh/uv:0.11.32 AS uv
FROM python:3.14-slim

# procps (pgrep, ps): the scrape suite's orphaned-regex-worker test finds and watches its
# worker with them, as CI's ubuntu runner can; python:slim ships neither.
RUN apt-get update \
    && apt-get install -y --no-install-recommends rsync git ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/

# The host's uid/gid, so files the suite writes under the /testruns bind mount
# stay owned by the host account that rsyncs into it.
ARG UID=1000
ARG GID=1000
RUN groupadd -o -g "$GID" lane && useradd -o -m -u "$UID" -g "$GID" -s /bin/bash lane

ENV UV_LINK_MODE=copy \
    UV_CACHE_DIR=/testruns/.uv-cache \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1
USER lane
WORKDIR /testruns
CMD ["sleep", "infinity"]
