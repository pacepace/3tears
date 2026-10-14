#!/usr/bin/env bash
# Dispatch ./scripts/test.sh to a remote test host (see docs/testing.md, "The remote test lane").
#
#   scripts/test-remote.sh HOST [test.sh args...]          sync this tree, run the suite there
#   scripts/test-remote.sh --probe HOST [test.sh args...]  exit 0 iff the lane can take this run
#   scripts/test-remote.sh --tree DIR HOST [args...]       sync and test the checkout at DIR instead
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
#
# The runner travels with the dispatcher. The shim only ever executes
# /testruns/<name>/scripts/remote-test.sh from the SYNCED tree, so a tree that
# predates the lane (any branch cut before it, tested with --tree) would have no
# runner, or a stale one. So the tree rsync excludes /scripts/remote-test.sh and
# a second rsync ships THIS checkout's runner into its place. It goes through
# the shim's existing tree rule (destination testruns-3tears/<name>/, no
# --delete, a sender-side filter admitting that one file), so no wider shim rule
# is needed and an already-provisioned host takes it. Overwriting that one file
# on the far side is safe: the synced tree is a copy, never a git checkout, and
# the runner's contract (shim argv in, test.sh args out) belongs to the lane, not
# to the branch under test.
#
# One dispatch per <name> at a time. Two dispatches of one tree share
# testruns-3tears/<name>/, and the second's --delete rsync would rewrite the tree
# under the first's running pytest. So before the probe or the first connection
# the dispatcher takes ${XDG_STATE_HOME:-~/.local/state}/threetears-testlane/<name>.lock,
# a directory (mkdir is atomic everywhere; macOS has no flock) holding an `owner`
# file with the holder's pid and start time. It is held until the remote command
# returns and released by an EXIT trap on every path, Ctrl-C included. A second
# dispatch of the same tree says who holds it and waits, polling every 5 s, up to
# THREETEARS_LANE_WAIT_S seconds (default 1800; 0 means do not wait), then exits
# 75. A lock whose pid is no longer alive is stale: it is removed and taken. The
# dry run takes and releases the lock too. Different trees have different names
# and never wait on each other. --probe locks as well, so test.sh's probe waits
# out a run in flight instead of probing a host about to be busy with it, and a
# 75 from that probe stops test.sh rather than sending the suite to run locally.
set -euo pipefail

DISPATCHER_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$DISPATCHER_ROOT"
CONTAINER=threetears-test
USAGE="usage: scripts/test-remote.sh [--probe] [--tree DIR] HOST [test.sh args...]"

PROBE=0
while :; do
    case "${1:-}" in
        --probe)
            PROBE=1
            shift
            ;;
        --tree)
            [ -n "${2:-}" ] && [ -d "$2" ] || { echo "remote lane: --tree needs an existing directory" >&2; exit 64; }
            REPO_ROOT="$(cd "$2" && pwd)"
            shift 2
            ;;
        *) break ;;
    esac
done
HOST="${1:?$USAGE}"
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

WAIT_S="${THREETEARS_LANE_WAIT_S:-1800}"
case "$WAIT_S" in
    "" | *[!0-9]*)
        echo "remote lane: THREETEARS_LANE_WAIT_S must be a whole number of seconds, not '$WAIT_S'" >&2
        exit 64
        ;;
esac
LOCK_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/threetears-testlane"
LOCK="$LOCK_DIR/$NAME.lock"
HELD=0
holder_pid="" holder_started=""

# Only ever removes a lock this process took and still owns (an empty owner is
# this process interrupted between its mkdir and writing the file).
release() {
    if [ "$HELD" = 1 ]; then
        owner=""
        { read -r owner <"$LOCK/owner"; } 2>/dev/null || true
        if [ -z "$owner" ] || [ "$owner" = "$$" ]; then rm -rf "$LOCK"; fi
    fi
    HELD=0
}
trap release EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

claim_lock() {
    mkdir "$LOCK" 2>/dev/null || return 1
    HELD=1
    printf '%s\n%s\n' "$$" "$(date '+%Y-%m-%d %H:%M:%S')" >"$LOCK/owner"
}

# One attempt: 0 when this process now holds the lock. A holder whose pid is not
# alive left the lock behind (kill -9, a closed laptop): remove it and try once
# more. A lock with no owner file yet is being taken right now, so it is live.
take_lock() {
    if claim_lock; then return 0; fi
    holder_pid="" holder_started=""
    if [ -f "$LOCK/owner" ]; then
        { read -r holder_pid && read -r holder_started; } <"$LOCK/owner" || true
    fi
    case "$holder_pid" in
        "" | *[!0-9]*) return 1 ;;
    esac
    if kill -0 "$holder_pid" 2>/dev/null; then return 1; fi
    echo "remote lane: removing the stale lock on $NAME (pid $holder_pid is gone)" >&2
    rm -rf "$LOCK"
    if claim_lock; then return 0; fi
    return 1
}

mkdir -p "$LOCK_DIR"
waited=0
until take_lock; do
    if [ "$waited" = 0 ]; then
        if [ "$WAIT_S" = 0 ]; then then_what="not waiting (THREETEARS_LANE_WAIT_S=0)"; else then_what="waiting"; fi
        echo "remote lane: $NAME is already being tested from ${holder_pid:-an unknown pid} (started ${holder_started:-just now}); $then_what" >&2
    fi
    if [ "$waited" -ge "$WAIT_S" ]; then
        echo "remote lane: gave up on $NAME after ${waited}s, still held by ${holder_pid:-an unknown pid} ($LOCK)" >&2
        exit 75
    fi
    sleep 5
    waited=$((waited + 5))
done

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

if [ ! -f "$DISPATCHER_ROOT/scripts/remote-test.sh" ]; then
    echo "remote lane: no runner beside the dispatcher at $DISPATCHER_ROOT/scripts/remote-test.sh" >&2
    exit 66
fi

run ssh $SSH_OPTS "$hostname" "mkdir -p 'testruns-3tears/$NAME'"
# No -l/-o/-g/-D: the shim refuses links, ownership and devices on the receiver.
# The excludes also shield the far side's copies from --delete: .venv survives
# (so the next uv sync is a no-op) and so does the runner shipped below.
run rsync -rtpz --delete \
    --exclude .git --exclude .venv --exclude .worktrees --exclude '**/__pycache__' \
    --exclude .pytest_cache --exclude .mypy_cache --exclude .ruff_cache --exclude '*.egg-info' \
    --exclude /scripts/remote-test.sh \
    -e "ssh $SSH_OPTS" "$REPO_ROOT/" "$hostname:testruns-3tears/$NAME/"
# The dispatcher's own runner, whatever the tree carries (see the header).
run rsync -rtpz --include /scripts/ --include /scripts/remote-test.sh --exclude '*' \
    -e "ssh $SSH_OPTS" "$DISPATCHER_ROOT/" "$hostname:testruns-3tears/$NAME/"
# The release gates (tests/enforcement/test_api_growth_requires_a_minor_bump.py)
# read tags and old trees from git, so the objects and tags travel too, into a
# bare-bones .git the tree rsync above leaves alone. Incremental after the first.
if common="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
    run rsync -rtpz --include /HEAD --include /packed-refs --include '/objects/***' --include /refs/ \
        --include '/refs/tags/***' --exclude '*' \
        -e "ssh $SSH_OPTS" "$common/" "$hostname:testruns-3tears/$NAME/.git/"
fi
run ssh $SSH_OPTS "$hostname" "docker exec -i $CONTAINER bash /testruns/$NAME/scripts/remote-test.sh$REMOTE_ARGS"
