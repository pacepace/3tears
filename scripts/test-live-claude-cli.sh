#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run the live Claude CLI batch -- structured output and the session pool -- and
# record that it ran.
#
# Usage: CLAUDE_CODE_OAUTH_TOKEN=<token> ./scripts/test-live-claude-cli.sh [extra pytest args...]
#
# Real calls through the real Claude CLI on a subscription
# (packages/models/tests/live/test_claude_cli_structured_output_live.py): a batch
# of structured calls that must all answer in their schema, and the pool's live
# proofs -- a reused pooled CLI shows the next call nothing of the last, and one
# pooled CLI switches between system prompts. Every other test of the subscription
# route and of claude_cli_pool fakes the CLI's messages, which is how 0.55.0
# shipped with a third of one consumer's structured calls failing. The release
# runs this before tagging anything that touches packages/models
# (docs/releasing.md, "Cutting a release").
#
# The token is a subscription token from `claude setup-token`. A logged-in
# `claude` does not stand in for it: 3tears runs the CLI in an isolated
# configuration directory where a stored login is never read.
#
# The script turns the test on itself, so it cannot skip: with no token or no
# Agent SDK the test fails. On every run -- pass or fail -- it writes one line to
# build/release-evidence/live-claude-cli.txt: the commit it ran on, whether the
# tree had changes on top of it, the version, and pytest's summary line. That
# line goes into the release PR body, which is what shows the batch ran against
# the commit being tagged.

if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    echo "CLAUDE_CODE_OAUTH_TOKEN is not set: give it a subscription token from \`claude setup-token\`" >&2
    exit 2
fi

EVIDENCE_DIR="build/release-evidence"
EVIDENCE="$EVIDENCE_DIR/live-claude-cli.txt"
OUTPUT="$(mktemp)"
trap 'rm -f "$OUTPUT"' EXIT
mkdir -p "$EVIDENCE_DIR"

STATUS=0
THREETEARS_LIVE_CLAUDE_CLI=1 uv run --frozen pytest \
    packages/models/tests/live/test_claude_cli_structured_output_live.py -rs \
    ${1+"$@"} 2>&1 | tee "$OUTPUT" || STATUS=$?

COMMIT="$(git rev-parse HEAD)"
TREE="clean"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    TREE="dirty"
fi
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' packages/models/pyproject.toml)"
SUMMARY="$(grep -E '(passed|failed|error|skipped)' "$OUTPUT" | tail -n 1 | sed 's/=//g; s/^ *//; s/ *$//')"
RECORD="$(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$COMMIT tree=$TREE version=$VERSION exit=$STATUS summary=\"$SUMMARY\""
echo "$RECORD" > "$EVIDENCE"
echo
echo "Recorded in $EVIDENCE (paste it into the release PR body):"
echo "$RECORD"
exit "$STATUS"
