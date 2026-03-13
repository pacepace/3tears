#!/usr/bin/env bash
set -euo pipefail

# Bump version across all packages.
# Usage: ./scripts/bump-version.sh [major|minor|patch]
# Defaults to patch if no argument given.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BUMP="${1:-patch}"

if [[ "$BUMP" != "major" && "$BUMP" != "minor" && "$BUMP" != "patch" ]]; then
    echo "Usage: $0 [major|minor|patch]"
    exit 1
fi

# Portable in-place sed (macOS requires -i '', GNU requires -i)
_sed_i() {
    if sed --version 2>/dev/null | grep -q GNU; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

# Read current version from core package
CURRENT=$(grep '^version = ' "$REPO_ROOT/packages/core/pyproject.toml" | head -1 | sed 's/version = "\(.*\)"/\1/')

if [ -z "$CURRENT" ]; then
    echo "Could not read current version from packages/core/pyproject.toml"
    exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

case "$BUMP" in
    major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
    minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
    patch) PATCH=$((PATCH + 1)) ;;
esac

NEW="$MAJOR.$MINOR.$PATCH"

echo "Bumping $BUMP: $CURRENT -> $NEW"

# Update all package pyproject.toml files
for PKG in core agent-memory agent-tools; do
    FILE="$REPO_ROOT/packages/$PKG/pyproject.toml"
    if [ -f "$FILE" ]; then
        _sed_i "s/^version = \"$CURRENT\"/version = \"$NEW\"/" "$FILE"
        echo "  Updated packages/$PKG/pyproject.toml"
    fi
done

# Update __version__ in __init__.py files
for INIT in \
    "$REPO_ROOT/packages/core/src/threetears/core/__init__.py" \
    "$REPO_ROOT/packages/agent-memory/src/threetears/agent/memory/__init__.py" \
    "$REPO_ROOT/packages/agent-tools/src/threetears/agent/tools/__init__.py"; do
    if [ -f "$INIT" ]; then
        _sed_i "s/__version__ = \"$CURRENT\"/__version__ = \"$NEW\"/" "$INIT"
        echo "  Updated $(basename "$(dirname "$INIT")")/__init__.py"
    fi
done

echo "Done. New version: $NEW"
echo ""
echo "Next steps:"
echo "  git add -A && git commit -m \"bump version to $NEW\""
echo "  git tag v$NEW"
echo "  git push origin develop --tags"
