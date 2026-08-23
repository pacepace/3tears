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
# Image is intentionally aibots-agnostic: it carries every 3tears WORKSPACE
# MEMBER wheel at build time and nothing else. Sub-images (e.g. aibots-base
# under 14-eng-ai-bot-agents/docker/Dockerfile) layer additional wheels.

# ---------------------------------------------------------------------------
# Stage 1: build every 3tears wheel
# ---------------------------------------------------------------------------
# Enumeration is delegated to uv, which resolves the member list from the root
# pyproject's [tool.uv.workspace] globs. New 3tears packages pick up
# automatically; no Dockerfile edit.
#
# It must be uv's list and not our own. This stage used to run
# `find /build/packages -name pyproject.toml` and build each hit, which is
# strictly BROADER than the workspace: it swept in packages/scrape/sidecar, a
# nodriver/Chromium companion app with a flat main.py/hitl.py, its own
# Dockerfile, its own AGPL-isolated dependency set, and `[tool.uv]
# package = false`. It is a deployable, never a distributable, and building a
# wheel from it fails with "Multiple top-level modules discovered in a
# flat-layout: ['main', 'hitl']". That killed every cold-cache base build from
# the moment the sidecar landed; a warm registry :cache tag hid it for eight
# releases, so it surfaced as an unrelated-looking break long after the change
# that caused it. Re-deriving the member list by hand is how that happened --
# so don't, ask uv.
FROM python:3.14-slim AS wheels

# uv, not pip. This platform is uv-only everywhere else (see the "uv-Only" rule
# in the consuming repos' CLAUDE.md) and the image builds were the last holdout
# -- which is exactly where it hurt: pip's resolver backtracks across the
# cross-product of every published version when a dependency graph is even
# slightly under-constrained, then reports ResolutionImpossible against whichever
# node it happened to be holding rather than the package actually in conflict.
# One such message named an innocent, correctly-installed package and cost most
# of a day. uv resolved the identical dependency set in seconds and names the
# real conflict.
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /usr/local/bin/uv

# The root pyproject.toml is the workspace definition -- it is the INPUT to the
# build below, not incidental metadata. Copied on its own so a packages/ change
# does not invalidate it, and vice versa.
COPY pyproject.toml /build/pyproject.toml
COPY packages /build/packages

WORKDIR /build
RUN uv build --all-packages --wheel --out-dir /wheels

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

# uv is carried into the RUNTIME image, not just the builder, because every
# downstream consumer (aibots-base, hub, admin, schema) installs into this venv
# and must do so with the same resolver. Leaving them on pip is what let the
# image chain drift away from the platform's uv-only rule in the first place.
# It is a single static binary -- no Python dependency, negligible layer cost.
#
# THIS VERSION IS A CONSUMER CONTRACT, not a detail of this image. Every
# consumer (aibots-base, hub, admin, schema) inherits this binary rather than
# installing its own -- aibots-base's Dockerfile says so outright -- and each of
# them runs `uv export` against ITS OWN uv.lock at build time. So this uv has to
# be able to read the lockfiles the platform currently produces, and it could
# not: 0.9.9 predates both the friendly-duration `exclude-newer = "14 days"` the
# supply-chain cooldown uses and the `[options]` table the newer uv writes into
# uv.lock. Every consumer Docker build failed identically the moment the
# cooldown landed:
#
#   failed to parse year in date "14 days"
#   Failed to parse `uv.lock` ... [options]: invalid type: map, expected a
#   timestamp string
#
# Bumping it here rather than overriding uv per consumer: nine repos carry that
# cooldown, so a per-repo override is the same workaround written nine times,
# and it would leave this image shipping a resolver older than the locks it is
# handed.
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /usr/local/bin/uv

# Build tooling INTO the venv so downstream consumer Dockerfiles
# (aibots-base, hub, admin, schema, future 3tears apps) can `python -m build`
# and install /app from a source tree without spinning their own builder stage.
# Must be inside the venv because PATH puts venv-bin first; a system-installed
# `build` would be invisible to `python -m build`.
#
# UV_PROJECT_ENVIRONMENT + the VIRTUAL_ENV set above make `uv pip install` target
# this venv without an explicit --python on every call.
RUN uv pip install --no-cache hatchling build

RUN uv pip install --no-cache /threetears-wheels/*.whl

# The family-from-base helper ships WITH the base, because every consumer that
# inherits this image runs the same check against the same family. Copied per
# repo it becomes N copies that drift apart silently -- the same second-source-
# of-truth failure the helper itself exists to catch one layer down. Consumers
# invoke it at this path; nothing copies it into a build context.
COPY docker/family_from_base.py /opt/threetears/family_from_base.py

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
