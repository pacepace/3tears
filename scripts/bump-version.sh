#!/usr/bin/env bash
set -euo pipefail

# Bump (or verify) the workspace version across every place the version
# string lives, in lockstep.
#
# Usage:
#   ./scripts/bump-version.sh                 # patch bump (default)
#   ./scripts/bump-version.sh patch
#   ./scripts/bump-version.sh minor
#   ./scripts/bump-version.sh major
#   ./scripts/bump-version.sh 0.10.2          # bump to an explicit version
#   ./scripts/bump-version.sh --verify 0.10.2 # check everything matches X.Y.Z (no edits)
#   ./scripts/bump-version.sh --no-lock 0.10.2  # bump but skip the `uv lock` refresh
#
# What gets touched (this is the FULL lockstep list -- if a version
# string lives somewhere else, add it here OR the release workflow's
# pre-flight check will fail):
#
#   1. Every `packages/*/pyproject.toml` `^version = "X.Y.Z"` line.
#   2. Every `packages/*/tests/test_smoke.py` `assert __version__ ==
#      "X.Y.Z"` line (5 files today: core, langgraph, observe,
#      agent/memory, agent/tools).
#   3. `docker-bake.hcl` `variable "VERSION" { default = "vX.Y.Z" }`.
#   4. `docker-bake.hcl` hardcoded image refs `:vX.Y.Z` on the
#      threetears-base / aibots-base / aibots-hub / aibots-admin /
#      aibots-schema target lines (used by buildx for the contexts
#      and ARG defaults so the in-flight base target stays linked
#      to the consumer build).
#   5. `uv.lock` refresh via `uv lock` (skip with --no-lock).
#
# What this script DOES NOT touch (deliberately, because they need
# human-written prose, not a regex):
#
#   - Root `CHANGELOG.md`: a new `## vX.Y.Z -- date` section.
#   - Per-package `packages/*/CHANGELOG.md`: a new `## [X.Y.Z]` section.
#
# A reminder to write those is printed at the end. `--verify` does
# NOT enforce CHANGELOG presence (you may legitimately ship a release
# with no new CHANGELOG entries when there are no user-facing changes
# in that package).

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

_sed_i() {
    # Portable in-place sed (macOS BSD vs. GNU).
    if sed --version 2>/dev/null | grep -q GNU; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

_usage() {
    cat >&2 <<EOF
Usage: $0 [patch|minor|major|X.Y.Z|--verify X.Y.Z] [--no-lock]
See header comment for the full lockstep target list.
EOF
    exit 1
}

# Argument parsing.
VERIFY_ONLY=0
RUN_LOCK=1
ARG=""
for a in "$@"; do
    case "$a" in
        --verify) VERIFY_ONLY=1 ;;
        --no-lock) RUN_LOCK=0 ;;
        -h|--help) _usage ;;
        *)
            if [ -z "$ARG" ]; then ARG="$a"; else _usage; fi
            ;;
    esac
done
if [ -z "$ARG" ]; then ARG="patch"; fi

# Read the canonical version from packages/core/pyproject.toml.
CURRENT=$(grep '^version = ' "$REPO_ROOT/packages/core/pyproject.toml" \
    | head -1 \
    | sed 's/version = "\(.*\)"/\1/')

if [ -z "$CURRENT" ]; then
    echo "ERROR: could not read current version from packages/core/pyproject.toml" >&2
    exit 1
fi

# Resolve the target version. Accept bump keywords, an explicit
# X.Y.Z, or in --verify mode require an explicit X.Y.Z.
if [[ "$ARG" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    NEW="$ARG"
elif [[ "$ARG" == "major" || "$ARG" == "minor" || "$ARG" == "patch" ]]; then
    if [ "$VERIFY_ONLY" -eq 1 ]; then
        echo "ERROR: --verify requires an explicit X.Y.Z target, not a bump keyword" >&2
        exit 1
    fi
    IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"
    case "$ARG" in
        major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
        minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
        patch) PATCH=$((PATCH + 1)) ;;
    esac
    NEW="$MAJOR.$MINOR.$PATCH"
else
    _usage
fi

# --------------------------------------------------------------------
# Verify mode: scan every lockstep location, fail loudly if any does
# not match $NEW. Same target list as the bump path so the contract
# "bump-version.sh sets X" / "bump-version.sh --verify X" agree.
# --------------------------------------------------------------------

if [ "$VERIFY_ONLY" -eq 1 ]; then
    MISMATCH=0

    # 1. pyproject.toml across packages/.
    while IFS= read -r FILE; do
        REL="${FILE#$REPO_ROOT/}"
        V=$(grep '^version = ' "$FILE" | head -1 | sed 's/version = "\(.*\)"/\1/' || true)
        if [ "$V" != "$NEW" ]; then
            echo "  MISMATCH $REL: expected $NEW, got $V" >&2
            MISMATCH=1
        fi
    done < <(find "$REPO_ROOT/packages" -name pyproject.toml -type f)

    # 2. test_smoke.py __version__ assertions.
    #
    # Two shapes appear in the wild:
    #   - assert __version__ == "X.Y.Z"           (single-package import)
    #   - assert <alias>_version == "X.Y.Z"       (aliased cross-package
    #     import, e.g. ``from threetears.core import __version__ as
    #     core_version``)
    # The lockstep guarantee covers both -- a stale aliased assertion
    # is the exact regression that broke v0.10.2 CI (packages/core
    # asserted ``core_version == "0.10.1"`` while every other location
    # had bumped to 0.10.2).
    while IFS= read -r FILE; do
        REL="${FILE#$REPO_ROOT/}"
        BAD=$(grep -oE '(__version__|[a-zA-Z_]+_version) == "[0-9]+\.[0-9]+\.[0-9]+"' "$FILE" \
              | grep -v "\"$NEW\"" || true)
        if [ -n "$BAD" ]; then
            echo "  MISMATCH $REL: $BAD (expected $NEW)" >&2
            MISMATCH=1
        fi
    done < <(find "$REPO_ROOT/packages" -path '*/tests/test_smoke.py' -type f)

    # 3 + 4. docker-bake.hcl VERSION variable + every image tag ref.
    if [ -f "$REPO_ROOT/docker-bake.hcl" ]; then
        BAKE_VERSION=$(grep -E '^[[:space:]]*default = "v[0-9]+\.[0-9]+\.[0-9]+"$' \
                       "$REPO_ROOT/docker-bake.hcl" | head -1 \
                       | sed -E 's/.*"v([0-9]+\.[0-9]+\.[0-9]+)".*/\1/' || true)
        if [ "$BAKE_VERSION" != "$NEW" ]; then
            echo "  MISMATCH docker-bake.hcl VERSION: expected $NEW, got ${BAKE_VERSION:-<missing>}" >&2
            MISMATCH=1
        fi
        # Hardcoded `<image>:v<version>` references.
        BAD_REFS=$(grep -oE ':v[0-9]+\.[0-9]+\.[0-9]+' "$REPO_ROOT/docker-bake.hcl" \
                   | grep -v ":v$NEW" || true)
        if [ -n "$BAD_REFS" ]; then
            for REF in $BAD_REFS; do
                echo "  MISMATCH docker-bake.hcl image-tag $REF (expected :v$NEW)" >&2
            done
            MISMATCH=1
        fi
    fi

    if [ "$MISMATCH" -eq 0 ]; then
        echo "All version locations at $NEW."
        exit 0
    else
        echo "" >&2
        echo "ERROR: lockstep verification failed for version $NEW." >&2
        echo "Run \`./scripts/bump-version.sh $NEW\` to bring every location into sync." >&2
        exit 1
    fi
fi

# --------------------------------------------------------------------
# Bump mode.
# --------------------------------------------------------------------

if [[ "$CURRENT" == "$NEW" ]]; then
    echo "Canonical version (packages/core/pyproject.toml) already at $NEW."
    echo "Continuing through the lockstep targets to catch any drifted locations."
else
    echo "Bumping: $CURRENT -> $NEW"
fi

# 1. pyproject.toml across packages.
#
# Heal any drift, not just files at the canonical-current value. The
# verify mode is a hard lockstep check (expects every file at $NEW), so
# the bump mode has to be able to bring every file to $NEW too --
# otherwise a release could pass --verify only after the operator
# hand-edits the drifted file. The original script preserved
# "intentional pins" (non-canonical version held deliberately for
# compat); today's workspace has none, and any reintroduction would
# need both --verify and bump to know about it (e.g. a future
# ``.bump-pinned`` allow-list file consulted by both modes). When that
# need actually shows up we add the allow-list explicitly; in the
# meantime the simpler "always heal" behaviour wins.
PYPROJECT_UPDATED=0
while IFS= read -r FILE; do
    REL="${FILE#$REPO_ROOT/}"
    OTHER=$(grep '^version = ' "$FILE" | head -1 | sed 's/version = "\(.*\)"/\1/' || true)
    if [ -z "$OTHER" ] || [ "$OTHER" = "$NEW" ]; then
        continue  # no version line, or already at target
    fi
    _sed_i -E "s/^version = \"$OTHER\"\$/version = \"$NEW\"/" "$FILE"
    echo "  updated $REL ($OTHER -> $NEW)"
    PYPROJECT_UPDATED=$((PYPROJECT_UPDATED + 1))
done < <(find "$REPO_ROOT/packages" -name pyproject.toml -type f)

# 2. test_smoke.py __version__ assertions. Replace any value (the
# files might have been at a different prior version than the
# canonical) so the lockstep is fully restored.
#
# Handle both shapes:
#   - assert __version__ == "X.Y.Z"           (single-package import)
#   - assert <alias>_version == "X.Y.Z"       (aliased cross-package
#     import; the capture group preserves the LHS name on rewrite)
# The aliased shape is what slipped past v0.10.1 -> 0.10.2's first
# bump attempt: packages/core/tests/test_smoke.py's
# test_cross_package_imports() asserts against ``core_version``,
# ``memory_version``, ``tools_version`` aliases, none of which match
# the literal ``__version__`` token.
SMOKE_UPDATED=0
while IFS= read -r FILE; do
    REL="${FILE#$REPO_ROOT/}"
    # only rewrite if at least one assertion is NOT already at $NEW
    BAD=$(grep -oE '(__version__|[a-zA-Z_]+_version) == "[0-9]+\.[0-9]+\.[0-9]+"' "$FILE" \
          | grep -v "\"$NEW\"" || true)
    if [ -n "$BAD" ]; then
        _sed_i -E "s/(__version__|[a-zA-Z_]+_version) == \"[0-9]+\\.[0-9]+\\.[0-9]+\"/\\1 == \"$NEW\"/g" "$FILE"
        echo "  updated $REL"
        SMOKE_UPDATED=$((SMOKE_UPDATED + 1))
    fi
done < <(find "$REPO_ROOT/packages" -path '*/tests/test_smoke.py' -type f)

# 3 + 4. docker-bake.hcl: the VERSION variable + every hardcoded
# `:v<old>` image tag reference. Replace based on the current bake
# VERSION value (which may have drifted from the canonical -- that's
# the bug this script's lockstep guarantees prevent in the future).
BAKE_UPDATED=0
if [ -f "$REPO_ROOT/docker-bake.hcl" ]; then
    BAKE_VERSION=$(grep -E '^[[:space:]]*default = "v[0-9]+\.[0-9]+\.[0-9]+"$' \
                   "$REPO_ROOT/docker-bake.hcl" | head -1 \
                   | sed -E 's/.*"v([0-9]+\.[0-9]+\.[0-9]+)".*/\1/' || true)
    if [ -n "$BAKE_VERSION" ] && [ "$BAKE_VERSION" != "$NEW" ]; then
        _sed_i -E "s/^([[:space:]]*default = )\"v$BAKE_VERSION\"$/\\1\"v$NEW\"/" \
            "$REPO_ROOT/docker-bake.hcl"
        echo "  updated docker-bake.hcl VERSION default"
        BAKE_UPDATED=$((BAKE_UPDATED + 1))
    fi
    # Hardcoded image tag references like ``:v0.9.1`` on context /
    # ARG default lines. Match anything that looks like ``:vX.Y.Z`` not
    # already at the target.
    BAD_TAGS=$(grep -oE ':v[0-9]+\.[0-9]+\.[0-9]+' "$REPO_ROOT/docker-bake.hcl" \
               | grep -v ":v$NEW" || true)
    if [ -n "$BAD_TAGS" ]; then
        _sed_i -E "s/:v[0-9]+\\.[0-9]+\\.[0-9]+/:v$NEW/g" "$REPO_ROOT/docker-bake.hcl"
        echo "  updated docker-bake.hcl image-tag references"
        BAKE_UPDATED=$((BAKE_UPDATED + 1))
    fi
fi

# 5. uv.lock refresh.
LOCK_UPDATED=0
if [ "$RUN_LOCK" -eq 1 ]; then
    if command -v uv >/dev/null 2>&1; then
        echo "  running 'uv lock' to refresh editable package pins ..."
        (cd "$REPO_ROOT" && uv lock >/dev/null 2>&1) || {
            echo "" >&2
            echo "WARNING: 'uv lock' failed. Run it manually and inspect; the" >&2
            echo "version bump did succeed in pyproject + smoke + docker-bake," >&2
            echo "the lockfile just hasn't refreshed." >&2
        }
        LOCK_UPDATED=1
    else
        echo "  skipped uv.lock refresh: 'uv' not on PATH"
    fi
fi

echo ""
echo "Done. New version: $NEW"
echo "  $PYPROJECT_UPDATED pyproject.toml files updated"
echo "  $SMOKE_UPDATED test_smoke.py files updated"
echo "  $BAKE_UPDATED docker-bake.hcl edits"
if [ "$LOCK_UPDATED" -eq 1 ]; then
    echo "  uv.lock refreshed"
fi

echo ""
echo "Reminder: CHANGELOGs are NOT auto-edited (they need release notes,"
echo "not regex). Add a new section in each of these if there are"
echo "user-facing changes worth recording:"
echo "  - CHANGELOG.md                              ## v$NEW -- $(date +%Y-%m-%d)"
for CH in "$REPO_ROOT"/packages/*/CHANGELOG.md; do
    [ -f "$CH" ] || continue
    REL="${CH#$REPO_ROOT/}"
    echo "  - $REL    ## [$NEW]"
done

echo ""
echo "Verify lockstep (the release workflow runs the same check):"
echo "  ./scripts/bump-version.sh --verify $NEW"
