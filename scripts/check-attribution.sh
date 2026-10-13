#!/usr/bin/env bash
set -euo pipefail

# Refuse agent attribution in commit messages: CLAUDE.md, "No agent attribution. Anywhere. Ever."
#
# Usage:
#   ./scripts/check-attribution.sh <base>..<head>     every commit in the range (CI: a PR's own commits)
#   ./scripts/check-attribution.sh --message <file>   one message (a commit-msg hook passes its file)
#
# A commit reachable from <base> is never examined, so history already landed stays landed. The
# commits that landed with attribution before this check existed, and cannot be rewritten without
# a force-push, are listed by full hash in scripts/attribution-exemptions.txt.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXEMPTIONS="$REPO_ROOT/scripts/attribution-exemptions.txt"

# case-insensitive extended regexes; naming Claude Code as a tool the project uses is fine, so only
# authorship credit is matched
PATTERNS=(
    'co-authored-by:.*(claude|anthropic|copilot|chatgpt|openai|gemini|codex)'
    'noreply@anthropic\.com'
    'generated (with|by) \[?(claude|an ai|copilot|chatgpt)'
    'claude-session:'
    '🤖'
)

offending() {
    # print each line of the message on stdin that carries attribution
    local pattern
    local message
    message="$(cat)"
    for pattern in "${PATTERNS[@]}"; do
        printf '%s\n' "$message" | grep -inE -- "$pattern" || true
    done
}

if [ "${1:-}" = "--message" ]; then
    found="$(offending < "${2:?a message file}")"
    if [ -n "$found" ]; then
        echo "this commit message credits an agent (CLAUDE.md: No agent attribution). Remove:" >&2
        echo "$found" >&2
        exit 1
    fi
    exit 0
fi

range="${1:?usage: check-attribution.sh <base>..<head> | --message <file>}"
failed=0
for sha in $(git -C "$REPO_ROOT" rev-list "$range"); do
    if [ -f "$EXEMPTIONS" ] && grep -q "^$sha\b" "$EXEMPTIONS"; then
        continue
    fi
    found="$(git -C "$REPO_ROOT" log -1 --format=%B "$sha" | offending)"
    if [ -n "$found" ]; then
        echo "commit $sha credits an agent (CLAUDE.md: No agent attribution):" >&2
        echo "$found" | sed 's/^/    /' >&2
        failed=1
    fi
done
if [ "$failed" -ne 0 ]; then
    echo "reword these commits on the branch (an interactive rebase before the PR merges); never force-push a shared branch." >&2
    exit 1
fi
echo "no agent attribution in $range"
