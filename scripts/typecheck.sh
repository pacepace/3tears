#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run mypy across the typechecked packages. The agent-namespace move
# (packages/agent-memory -> packages/agent/memory, etc.) nests src/
# two levels deep which mypy cannot resolve from file paths alone, so
# we address the modules by ``-p threetears.<x>``, resolving them through
# MYPYPATH + the installed-editable layout.
#
# This is the SINGLE SOURCE OF TRUTH for the mypy target set: CI calls this
# script (see .github/workflows/ci.yml) so the two can never drift apart.
#
# The target list is narrower than ``[tool.mypy] files`` in pyproject.toml, and the
# gap is a known, measured backlog item rather than an oversight. Every package that
# passes strict mypy today IS listed here, so none of them can regress. Five are
# absent because they do not pass yet, with the error count each would contribute
# when this was last measured (2026-07-25):
#
#   threetears.models              116    (by far the bulk of the remaining work)
#   threetears.conversations        10
#   threetears.agent.workspace       8    (mostly one L3Backend | None narrowing pattern)
#   threetears.langgraph             6
#   threetears.mcp                   4
#
# That is the whole remainder: every other package in `[tool.mypy] files` is listed
# below. Tracked as TYP-8H5R.
#
# Add each one HERE as it is fixed, in the same change that fixes it. Adding a
# package before it passes turns the gate red for everyone, which is how a
# not-yet-checked package becomes a permanently-skipped one.
MYPYPATH=packages/core/src:packages/nats/src:packages/observe/src:packages/agent/acl/src:packages/agent/audit/src:packages/agent/identity/src:packages/agent/intention/src:packages/agent/knowledge/src:packages/agent/memory/src:packages/agent/skills/src:packages/agent/tools/src:packages/agent/wake/src:packages/channels/src:packages/datasources/src:packages/enforcement/src:packages/epoch/src:packages/langgraph/src:packages/media-contracts/src:packages/models/src:packages/object-store/src:packages/registry/src:packages/scheduled-jobs/src:packages/scrape/src:packages/backup/src \
    uv run mypy \
        --explicit-package-bases \
        -p threetears.core \
        -p threetears.agent.identity \
        -p threetears.agent.intention \
        -p threetears.agent.knowledge \
        -p threetears.agent.memory \
        -p threetears.agent.skills \
        -p threetears.agent.tools \
        -p threetears.agent.wake \
        -p threetears.channels \
        -p threetears.datasources \
        -p threetears.media.contracts \
        -p threetears.object_store \
        -p threetears.backup \
        -p threetears.scrape \
        -p threetears.registry \
        -p threetears.epoch \
        -p threetears.scheduled_jobs \
        -p threetears.enforcement \
        -p threetears.agent.acl \
        -p threetears.agent.audit \
        -p threetears.nats \
        -p threetears.observe \
        "$@"
