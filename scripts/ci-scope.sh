#!/usr/bin/env bash
# Decide whether a CI run needs the suite; writes needed=true|false to $GITHUB_OUTPUT.
#
#   EVENT_NAME=pull_request BASE_SHA=<sha> ./scripts/ci-scope.sh
#
# One classifier for every job that skips its work on a documentation-only PR: ci.yml's `check`,
# each `unit` shard and `unit-pool` (the testament pool route) all run it as their `scope` step, so
# neither unit route can disagree with the gate about what a doc-only change is.
# tests/enforcement/test_ci_doc_only_skip.py runs THIS file against fixture changesets and fails if
# any of those jobs stops calling it.
#
# A documentation-only PR runs no suite. The skip is decided inside the jobs, rather than by a
# `paths-ignore:` on the trigger -- and that is not a style preference. `check` is a REQUIRED
# status context on both develop and main. A workflow that `paths-ignore` filters out never
# starts, so it never reports that context, so the PR waits on a check that can never arrive and
# blocks forever. Keeping the jobs unconditional and skipping their STEPS keeps the contract:
# `check` always reports, and a docs PR gets its green in seconds instead of minutes.
set -uo pipefail

if [ "${EVENT_NAME:-}" != "pull_request" ]; then
  # Every push to develop runs the full suite whatever it touched.
  # The develop -> main release PR is gated on THIS commit's stored
  # result (see ci.yml's push trigger comment), so a stamp that skipped
  # the suite would hand main a green light nothing verified.
  echo "needed=true" >> "$GITHUB_OUTPUT"
  echo "push to develop: running the full suite."
  exit 0
fi
changed=$(git diff --name-only "${BASE_SHA}...HEAD")
echo "changed files:"
echo "$changed"
# Classified with shell `case` rather than a grep expression, on
# purpose. `grep -qv` means "some line did not match", which is one
# negation away from "no line matched" and reads the same at a
# glance -- and the two differ exactly on the case that matters
# here, a PR mixing docs with code. `case` states the doc shapes
# positively and needs no negation to be read correctly.
#
# The default branch is `needed=true`: an unrecognised path is
# treated as code, so a new kind of file is tested rather than
# silently skipped. A changeset that is somehow empty also runs.
needed=false
while IFS= read -r file; do
  [ -z "$file" ] && continue
  case "$file" in
    *.md|docs/*|LICENSE|*/LICENSE) ;;
    *) needed=true; break ;;
  esac
done <<CHANGED
$changed
CHANGED
if [ -z "$changed" ]; then
  needed=true
fi
echo "needed=$needed" >> "$GITHUB_OUTPUT"
if [ "$needed" = "true" ]; then
  echo "-> code changed: running the full suite."
else
  echo "-> documentation only: skipping lint, typecheck and tests."
fi
