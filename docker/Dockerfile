# threetears-base: shared Python 3.14 base image for every 3tears-consuming app.
#
# Build context: this repo (3tears) root.
#
#     docker build -f docker/Dockerfile -t "$REGISTRY/threetears-base:$VERSION" .
#
# In the platform layout this image is built via Docker Bake; see the
# top-level docker-bake.hcl one directory up.
#
# Image surface:
#   * /opt/threetears/.venv  -- venv with every 3tears wheel installed
#   * /threetears-wheels/    -- wheels themselves, available for downstream
#                              `COPY --from=...` if a consumer needs the raw
#                              wheels (e.g. multi-stage builds that install
#                              into a different venv)
#   * tini at /usr/bin/tini  -- PID-1 wrapper (signal forwarding, zombie reap)
#   * appuser (uid 1001)     -- non-root runtime user
#   * WORKDIR /app
#   * USER appuser
#
# Image is intentionally aibots-agnostic: it carries every 3tears wheel found
# under packages/ at build time and nothing else. Sub-images (e.g. aibots-base
# under 14-eng-ai-bot-agents/docker/Dockerfile) layer additional wheels.

# ---------------------------------------------------------------------------
# Stage 1: build every 3tears wheel
# ---------------------------------------------------------------------------
# Dynamic enumeration: every directory under packages/ that contains a
# pyproject.toml is built, regardless of nesting depth. New 3tears packages
# added to the workspace pick up automatically; no Dockerfile edit.
#
# packages/ has two layout shapes: top-level packages (e.g. ``packages/core``)
# and grouped packages (e.g. ``packages/agent/{acl,audit,memory,tools,workspace}``
# after the agent reorg). ``find -name pyproject.toml`` covers both.
FROM python:3.14-slim AS wheels

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir hatchling build

COPY packages /build/packages

RUN set -e; \
    find /build/packages -name pyproject.toml -type f | sort | while read -r toml; do \
      pkg_dir=$(dirname "$toml"); \
      pkg_rel=${pkg_dir#/build/packages/}; \
      echo "Building wheel for 3tears/${pkg_rel}"; \
      python -m build --wheel --outdir /wheels "$pkg_dir"; \
    done

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.14-slim AS runtime

# tini = PID-1 wrapper. Without it Python receives the docker stop SIGTERM but
# zombie children are not reaped and signal forwarding to subprocesses
# (uvicorn workers, honcho-managed processes, etc.) is unreliable; AgentBootstrap
# drain hooks would not run.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user. UID 1001 matches the platform convention (metallm,
# every other 3tears-consuming app) so volume mounts behave consistently
# across services.
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home --home-dir /home/appuser appuser

# Wheels live in two places: copied into the venv, and preserved on disk
# under /threetears-wheels for downstream `COPY --from=...` access.
COPY --from=wheels /wheels /threetears-wheels

# Canonical venv path. /opt/threetears/.venv is the shared convention for any
# 3tears-consuming app. Sub-images install into the same venv.
RUN python -m venv /opt/threetears/.venv
ENV VIRTUAL_ENV=/opt/threetears/.venv
ENV PATH="/opt/threetears/.venv/bin:${PATH}"

# Build tooling INTO the venv so downstream consumer Dockerfiles
# (aibots-base, hub, admin, schema, future 3tears apps) can `python -m build`
# and `pip install /app` from a source tree without spinning their own
# builder stage. Must be inside the venv because PATH puts venv-bin first;
# system-pip-installed `build` would be invisible to `python -m build`.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir hatchling build

RUN pip install --no-cache-dir /threetears-wheels/*.whl

# /app is the canonical working directory for every consumer. Owned by
# appuser so consumer COPY operations land with the right owner.
WORKDIR /app
RUN chown appuser:appuser /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

USER appuser

# tini is PID-1; the consumer's CMD runs as tini's child.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command for the bare base image: print Python version and exit.
# Consumer images override CMD with the actual service entry point.
CMD ["python", "--version"]
