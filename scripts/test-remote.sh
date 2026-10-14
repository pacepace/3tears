#!/usr/bin/env bash
# Dispatch ./scripts/test.sh to a remote test host (see docs/testing.md, "The remote test lane").
#
#   scripts/test-remote.sh HOST [test.sh args...]          sync this tree, run the suite there
#   scripts/test-remote.sh --probe HOST [test.sh args...]  exit 0 iff the lane can take this run
#
# scripts/test.sh calls both when THREETEARS_TEST_REMOTE is set; running this by
# hand is the same thing without the local fallback. With
# THREETEARS_TEST_REMOTE_DRY_RUN=1 every ssh/rsync is printed instead of run and
# the probe answers yes.
#
# Only the lane key is ever offered: HOST is resolved through `ssh -G` for its
# hostname, user and port, and the connection itself runs with `-F none`, so no
# IdentityFile or ControlMaster socket from ~/.ssh/config (another lane's key, an
# admin master) can ride along.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER=threetears-test

PROBE=0
if [ "${1:-}" = --probe ]; then
    PROBE=1
    shift
fi
HOST="${1:?usage: scripts/test-remote.sh [--probe] HOST [test.sh args...]}"
shift
DRY="${THREETEARS_TEST_REMOTE_DRY_RUN:-}"

# The worktree's remote directory: sanitised basename plus a hash of the full
# path, so two checkouts with the same basename never share (and --delete) a tree.
base="$(printf '%s' "${REPO_ROOT##*/}" | tr -c 'A-Za-z0-9._-' '_')"
NAME="${base#.}-$(printf '%x' "$(printf '%s' "$REPO_ROOT" | cksum | cut -d' ' -f1)")"

# Every argument travels as one single-quoted word, and the host's shim admits
# only this alphabet, so refuse here, before any connection, with the reason.
REMOTE_ARGS=""
for arg in "$@"; do
    case "$arg" in
        "" | -- | ---* | *[!A-Za-z0-9._/=:\ -]*)
            echo "remote lane: argument '$arg' cannot cross the lane (allowed: A-Za-z0-9 . _ / = : - space)" >&2
            exit 64
            ;;
    esac
    REMOTE_ARGS="$REMOTE_ARGS '$arg'"
done

KEY="${THREETEARS_TEST_REMOTE_KEY:-$HOME/.ssh/threetears-testlane}"
# A public key alone is enough when the private half is held by a (forwarded) agent.
if [ ! -e "$KEY" ] && [ -e "$KEY.pub" ]; then KEY="$KEY.pub"; fi
if [ -z "$DRY" ] && [ ! -e "$KEY" ]; then
    echo "remote lane: no lane key at $KEY (set THREETEARS_TEST_REMOTE_KEY)" >&2
    exit 65
fi

hostname="$HOST" user="" port=22
if resolved="$(ssh -G "$HOST" 2>/dev/null)"; then
    hostname="$(printf '%s\n' "$resolved" | sed -n 's/^hostname //p')"
    user="$(printf '%s\n' "$resolved" | sed -n 's/^user //p')"
    port="$(printf '%s\n' "$resolved" | sed -n 's/^port //p')"
fi
SSH_OPTS="-F none -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=5 -o ControlPath=none"
SSH_OPTS="$SSH_OPTS -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p ${port:-22}${user:+ -l $user}"

run() {
    if [ -n "$DRY" ]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

if [ "$PROBE" = 1 ]; then
    [ -z "$DRY" ] || { run ssh $SSH_OPTS "$hostname" docker inspect -f '{{.State.Running}}' "$CONTAINER"; exit 0; }
    TIMEOUT=""
    if command -v timeout >/dev/null 2>&1; then TIMEOUT="timeout 5"; fi
    state="$($TIMEOUT ssh $SSH_OPTS "$hostname" docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)"
    [ "$state" = true ]
    exit
fi

run ssh $SSH_OPTS "$hostname" "mkdir -p 'testruns-3tears/$NAME'"
# No -l/-o/-g/-D: the shim refuses links, ownership and devices on the receiver.
run rsync -rtpz --delete \
    --exclude .git --exclude .venv --exclude .worktrees --exclude '**/__pycache__' \
    --exclude .pytest_cache --exclude .mypy_cache --exclude .ruff_cache --exclude '*.egg-info' \
    -e "ssh $SSH_OPTS" "$REPO_ROOT/" "$hostname:testruns-3tears/$NAME/"
# The release gates (tests/enforcement/test_api_growth_requires_a_minor_bump.py)
# read tags and old trees from git, so the objects and tags travel too, into a
# bare-bones .git the tree rsync above leaves alone. Incremental after the first.
if common="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
    run rsync -rtpz --include /HEAD --include /packed-refs --include '/objects/***' --include /refs/ \
        --include '/refs/tags/***' --exclude '*' \
        -e "ssh $SSH_OPTS" "$common/" "$hostname:testruns-3tears/$NAME/.git/"
fi
run ssh $SSH_OPTS "$hostname" "docker exec -i $CONTAINER bash /testruns/$NAME/scripts/remote-test.sh$REMOTE_ARGS"
