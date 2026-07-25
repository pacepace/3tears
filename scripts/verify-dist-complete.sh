#!/usr/bin/env bash
set -euo pipefail

# Assert that a built dist/ contains an sdist AND a wheel for every workspace
# member, and fail loudly naming any that are missing.
#
# Why this exists, concretely: v0.18.0 published 26 of 27 packages. A step in
# release.yml deleted 3tears-scrape from dist/ before the upload, because at the
# time scrape had no PyPI project and an upload including it would have been
# rejected mid-publish. That step's own comment said to remove it for the release
# where scrape shipped -- but the instruction lived inside the step it was telling
# you to delete, so the only person who would ever read it was someone already
# editing release.yml, and cutting a release does not require editing release.yml.
# The build worked perfectly, the artifacts were correct, and one of them was
# dropped seconds before upload with nothing objecting.
#
# The lesson generalises past that one step: anything that removes, filters or
# skips a package between "build" and "publish" is invisible at release time. So
# the invariant is checked against the workspace itself rather than trusted to a
# comment -- if a package exists in the workspace and is not in dist/, the release
# stops.
#
# Reads the member globs out of the root pyproject's [tool.uv.workspace] rather
# than restating them, for the same reason the migration drift guard derives its
# columns: a list you have to remember to update is a list that silently rots,
# which is the failure this is here to prevent. An earlier version of this script
# claimed to do that while actually hardcoding a third copy of the globs -- the
# exact drift it was written to stop, in its own header.
#
# Usage:  ./scripts/verify-dist-complete.sh [dist-dir]

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DIST_DIR="${1:-dist}"

if [ ! -d "$DIST_DIR" ]; then
    echo "error: dist directory '$DIST_DIR' does not exist -- nothing was built." >&2
    exit 1
fi

missing=()
checked=0

# The member globs come from the root pyproject, so adding a workspace tier
# there is enough -- this script needs no edit and cannot fall behind it. A
# nested pyproject deeper than a member glob (packages/scrape/sidecar) is a
# separate deployable, is not a member, and is correctly not demanded here.
MEMBER_GLOBS="$(
    python3 - <<'PY'
import sys, tomllib, pathlib
data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
ws = data.get("tool", {}).get("uv", {}).get("workspace", {})
members = ws.get("members") or []
if not members:
    sys.exit("error: [tool.uv.workspace] members is empty or absent in pyproject.toml")
excluded = set(ws.get("exclude") or [])
print("\n".join(m for m in members if m not in excluded))
PY
)" || exit 1

# Fed by a herestring, NOT a pipe: a piped `while read` runs in a subshell and
# every `missing`/`checked` update would be discarded at the `done`, leaving a
# guard that always reports success.
while IFS= read -r glob; do
    [ -n "$glob" ] || continue
    for pyproject in $glob/pyproject.toml; do
        [ -f "$pyproject" ] || continue

        name="$(sed -n 's/^name = "\(.*\)"/\1/p' "$pyproject" | head -1)"
        if [ -z "$name" ]; then
            echo "error: could not read a package name from $pyproject" >&2
            exit 1
        fi

        # PyPI normalizes '-' to '_' in built artifact filenames.
        normalized="${name//-/_}"
        checked=$((checked + 1))

        wheel_count=$(find "$DIST_DIR" -maxdepth 1 -name "${normalized}-*.whl" | wc -l | tr -d ' ')
        sdist_count=$(find "$DIST_DIR" -maxdepth 1 -name "${normalized}-*.tar.gz" | wc -l | tr -d ' ')

        if [ "$wheel_count" -eq 0 ]; then
            missing+=("$name (no wheel)")
        fi
        if [ "$sdist_count" -eq 0 ]; then
            missing+=("$name (no sdist)")
        fi
    done
done <<< "$MEMBER_GLOBS"

if [ "$checked" -eq 0 ]; then
    echo "error: found no workspace members to check -- the globs are wrong, not the dist." >&2
    exit 1
fi

if [ "${#missing[@]}" -gt 0 ]; then
    echo "::error::$DIST_DIR is missing artifacts for ${#missing[@]} workspace package(s):" >&2
    for m in "${missing[@]}"; do
        echo "  - $m" >&2
    done
    echo "" >&2
    echo "Every workspace member must be built and published together -- the versions are" >&2
    echo "in lockstep, so a package left behind strands consumers on a version of it that" >&2
    echo "does not exist. If a package genuinely must not ship, remove it from the" >&2
    echo "workspace; do not filter it out between build and publish." >&2
    exit 1
fi

echo "OK: dist/ carries an sdist and a wheel for all $checked workspace packages."
echo "Publishing:"
ls -1 "$DIST_DIR"
