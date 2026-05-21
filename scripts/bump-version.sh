#!/usr/bin/env bash
set -euo pipefail

# Bump version across every workspace package.
# Usage:
#   ./scripts/bump-version.sh                 # patch bump (default)
#   ./scripts/bump-version.sh patch
#   ./scripts/bump-version.sh minor
#   ./scripts/bump-version.sh major
#   ./scripts/bump-version.sh 0.9.0           # bump to an explicit version
#
# Discovers every packages/**/pyproject.toml that currently carries the
# repo's canonical version and rewrites it to the new value. The
# canonical version is read from packages/core/pyproject.toml; every
# other package is expected to track it (16-package workspace, single
# version line). Packages whose version does not match the canonical
# value are skipped (they were intentionally pinned elsewhere; touching
# them would diverge the workspace).

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARG="${1:-patch}"

_sed_i() {
    # Portable in-place sed (macOS BSD vs. GNU).
    if sed --version 2>/dev/null | grep -q GNU; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

# Read the canonical version from packages/core/pyproject.toml.
CURRENT=$(grep '^version = ' "$REPO_ROOT/packages/core/pyproject.toml" \
    | head -1 \
    | sed 's/version = "\(.*\)"/\1/')

if [ -z "$CURRENT" ]; then
    echo "ERROR: could not read current version from packages/core/pyproject.toml"
    exit 1
fi

# Resolve the new version. Accept "patch"/"minor"/"major" bump keywords
# or a literal X.Y.Z value.
if [[ "$ARG" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    NEW="$ARG"
elif [[ "$ARG" == "major" || "$ARG" == "minor" || "$ARG" == "patch" ]]; then
    IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"
    case "$ARG" in
        major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
        minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
        patch) PATCH=$((PATCH + 1)) ;;
    esac
    NEW="$MAJOR.$MINOR.$PATCH"
else
    echo "Usage: $0 [major|minor|patch|<x.y.z>]"
    exit 1
fi

if [[ "$CURRENT" == "$NEW" ]]; then
    echo "Already at $NEW; nothing to do."
    exit 0
fi

echo "Bumping: $CURRENT -> $NEW"

UPDATED_COUNT=0
SKIPPED_COUNT=0

# Discover every workspace pyproject.toml and rewrite its version line
# if it currently matches the canonical version.
while IFS= read -r FILE; do
    if grep -q "^version = \"$CURRENT\"\$" "$FILE"; then
        _sed_i "s/^version = \"$CURRENT\"\$/version = \"$NEW\"/" "$FILE"
        REL="${FILE#$REPO_ROOT/}"
        echo "  updated $REL"
        UPDATED_COUNT=$((UPDATED_COUNT + 1))
    else
        REL="${FILE#$REPO_ROOT/}"
        OTHER=$(grep '^version = ' "$FILE" | head -1 | sed 's/version = "\(.*\)"/\1/' || true)
        if [ -n "$OTHER" ] && [[ "$OTHER" != "$NEW" ]]; then
            echo "  SKIPPED $REL (pinned to $OTHER, not the canonical $CURRENT)"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        fi
    fi
done < <(find "$REPO_ROOT/packages" -name pyproject.toml -type f)

echo ""
echo "Done. New version: $NEW"
echo "  $UPDATED_COUNT pyproject.toml files updated."
if [ "$SKIPPED_COUNT" -gt 0 ]; then
    echo "  $SKIPPED_COUNT files skipped (intentional pins; review with 'git diff packages')."
fi

echo ""
echo "Next steps:"
echo "  uv sync --extra vision                            # refresh uv.lock"
echo "  git add packages uv.lock                          # stage version files (no -A)"
echo "  git commit -m \"chore: bump all packages to $NEW\""
echo "  git tag v$NEW"
echo "  git push origin <branch> --tags                   # branch, not develop, if on a PR"
