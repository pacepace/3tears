#!/usr/bin/env bash
set -euo pipefail

# Install this repository's git hooks: a commit-msg hook that refuses a message crediting an agent
# (CLAUDE.md, "No agent attribution"), by running ./scripts/check-attribution.sh --message on it --
# the same check CI's `check` job runs over a pull request's commits, applied before the commit
# exists instead of after it is pushed.
#
# Usage:
#   ./scripts/install-hooks.sh
#
# Idempotent: running it again rewrites the same hook. It installs into the hooks directory git
# itself uses (`git rev-parse --git-path hooks`: honours core.hooksPath, and is the common
# directory's hooks for every worktree of the clone), and the hook runs the check script of the
# checkout the commit is made in, so each worktree checks with its own copy.
#
# A commit-msg hook already there that this script did not write (a git template's, a tool's) is
# kept, not replaced: it is moved to commit-msg.chained and the installed hook runs it first, then
# the check -- so the check judges the message the other hook leaves (one that strips trailers
# strips them before the check reads it), and a refusal from either stops the commit. If a
# different commit-msg.chained is already there, nothing is moved and the install fails, naming both.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# the line that marks a hook as this script's own, so a re-run may replace it and nothing else
MARKER="# installed by scripts/install-hooks.sh"

HOOKS_DIR="$(git rev-parse --path-format=absolute --git-path hooks)"
HOOK="$HOOKS_DIR/commit-msg"

CHAINED="$HOOKS_DIR/commit-msg.chained"

mkdir -p "$HOOKS_DIR"
if [ -e "$HOOK" ] && ! grep -qF "$MARKER" "$HOOK"; then
    if [ -e "$CHAINED" ] && ! cmp -s "$HOOK" "$CHAINED"; then
        echo "error: $HOOK was not installed by this script, and $CHAINED already holds a different hook;" >&2
        echo "       refusing to move either. merge them by hand, then run this again." >&2
        exit 1
    fi
    mv -f "$HOOK" "$CHAINED"
    echo "==> kept the existing commit-msg hook; it now runs first, as $CHAINED"
fi

tmp="$(mktemp "$HOOKS_DIR/commit-msg.XXXXXX")"
cat > "$tmp" <<HOOK_BODY
#!/bin/sh
$MARKER
# a commit-msg hook that was here first runs first, on the same message file
chained="\$(dirname "\$0")/commit-msg.chained"
if [ -x "\$chained" ]; then
    "\$chained" "\$@" || exit \$?
fi
# refuse a commit message that credits an agent (CLAUDE.md, "No agent attribution")
exec "\$(git rev-parse --show-toplevel)/scripts/check-attribution.sh" --message "\$1"
HOOK_BODY
chmod +x "$tmp"
mv -f "$tmp" "$HOOK"
echo "==> commit-msg hook installed: $HOOK"
