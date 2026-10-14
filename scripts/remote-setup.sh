#!/usr/bin/env bash
# Provision (or re-provision) one remote test host for the 3tears test lane.
#
#     scripts/remote-setup.sh <admin-alias> <lane-pubkey-file>
#     THREETEARS_LANE_PUBKEY_FILE=… scripts/remote-setup.sh <admin-alias>
#
# Run by the owner from a plain checkout, on any macOS or Linux machine with
# bash 3.2+ and ssh; docker is needed on the HOST only. Idempotent: every run
# rebuilds the image (cached), recreates the container (a run in flight on it
# dies) and replaces the lane key's one authorized_keys line.
#
# <admin-alias> must be an UNRESTRICTED login: setup runs `docker build` and
# `docker run`, which the lane key's forced command refuses. <lane-pubkey-file>
# is the PUBLIC half of the dispatcher's key (~/.ssh/threetears-testlane.pub
# there); the private half never leaves the dispatcher and this script never
# generates or reads one.
#
# Sizing: THREETEARS_LANE_CPUSET (default: every CPU docker sees but the first
# two, when it sees more than four) and THREETEARS_LANE_MEMORY (default: three
# quarters of docker's memory). On Docker Desktop both index the VM, not the Mac.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${1:?usage: scripts/remote-setup.sh <admin-alias> <lane-pubkey-file>}"
PUB="${2:-${THREETEARS_LANE_PUBKEY_FILE:-}}"
CONTAINER=threetears-test
LANE_DIR='.threetears-testlane'
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
# A non-interactive ssh on macOS gets /usr/bin:/bin:/usr/sbin:/sbin, and Docker
# Desktop's `docker` lives in /usr/local/bin or /opt/homebrew/bin.
RPATH='PATH="$PATH:/usr/local/bin:/opt/homebrew/bin"; export PATH; '

die() {
    echo "remote-setup: $1" >&2
    exit 1
}

# The public key: one line, an ed25519 or rsa type, a base64 body. Its content is
# never echoed, so a private key passed here by mistake is refused unprinted.
[ -n "$PUB" ] && [ -f "$PUB" ] || die "lane public key file missing (arg 2 or THREETEARS_LANE_PUBKEY_FILE)"
[ "$(grep -c . "$PUB")" = 1 ] || die "$PUB is not a single-line public key"
read -r ktype kbody _ <"$PUB"
case "$ktype" in ssh-ed25519 | ssh-rsa) ;; *) die "$PUB is not an ssh-ed25519 or ssh-rsa public key" ;; esac
case "$kbody" in "" | *[!A-Za-z0-9+/=]*) die "$PUB has no base64 key body" ;; esac

case "$REMOTE" in *-admin) ;; *) echo "NOTE: '$REMOTE' does not end in -admin; checking it is unrestricted." ;; esac
answer="$($SSH "$REMOTE" 'echo threetears-admin-ok' 2>&1 || true)"
[ "$answer" = threetears-admin-ok ] || die "'$REMOTE' is not an unrestricted login (a lane alias runs a forced command). Use the admin alias. It answered: $answer"

echo "==> Reading $REMOTE"
facts="$($SSH "$REMOTE" "${RPATH}"'echo "$HOME"; id -u; id -g; docker info -f "{{.NCPU}} {{.MemTotal}}"')" \
    || die "could not read uid/gid/docker info on $REMOTE (is docker installed and running?)"
set -- $facts
[ $# = 5 ] || die "unexpected answer from $REMOTE: $facts"
RHOME="$1" RUID="$2" RGID="$3" NCPU="$4" MEMB="$5"
if [ "$NCPU" -gt 4 ]; then DEF_CPUSET="2-$((NCPU - 1))"; else DEF_CPUSET="0-$((NCPU - 1))"; fi
CPUSET="${THREETEARS_LANE_CPUSET:-$DEF_CPUSET}"
MEMORY="${THREETEARS_LANE_MEMORY:-$((MEMB / 4 * 3 / 1048576))m}"
case "$CPUSET" in "" | *[!0-9,-]*) die "THREETEARS_LANE_CPUSET '$CPUSET' is not a cpuset" ;; esac
case "$MEMORY" in [0-9]*[0-9][bkmg] | [0-9][bkmg]) ;; *) die "THREETEARS_LANE_MEMORY '$MEMORY' is not like 12g" ;; esac
case "$RHOME" in /*) ;; *) die "remote home '$RHOME' is not absolute" ;; esac
echo "    home $RHOME, uid $RUID:$RGID, docker sees $NCPU CPUs; cpuset $CPUSET, memory $MEMORY"

echo "==> Building image $CONTAINER:latest"
$SSH "$REMOTE" "mkdir -p ~/$LANE_DIR/build ~/testruns-3tears && cat > ~/$LANE_DIR/build/Dockerfile" \
    <"$REPO_ROOT/.devcontainer/Dockerfile"
$SSH "$REMOTE" "${RPATH}docker build -q -t $CONTAINER:latest --build-arg UID=$RUID --build-arg GID=$RGID ~/$LANE_DIR/build"

echo "==> (Re)creating container $CONTAINER"
# --init: docker's tini becomes PID 1 and reaps. Without it PID 1 is `sleep infinity`,
# which never waits, so every process the suite kills after its parent died stays a
# zombie with a live /proc entry for the container's lifetime (models'
# test_killing_a_cli_takes_its_children_with_it fails on exactly that), and the
# zombies pile up across runs in a container that is never recreated.
$SSH "$REMOTE" "${RPATH}docker rm -f $CONTAINER >/dev/null 2>&1; docker run -d --init --name $CONTAINER \
    --restart unless-stopped --cpuset-cpus $CPUSET --memory $MEMORY --memory-swap $MEMORY \
    --cap-drop ALL --security-opt no-new-privileges -v $RHOME/testruns-3tears:/testruns $CONTAINER:latest"

echo "==> Installing the forced command"
SHIM="$RHOME/$LANE_DIR/threetears-testlane-shim"
$SSH "$REMOTE" "cat > $SHIM.new && chmod 0555 $SHIM.new && mv -f $SHIM.new $SHIM" \
    <"$REPO_ROOT/scripts/threetears-testlane-shim"
OPTS="command=\"$SHIM\",no-pty,no-port-forwarding,no-agent-forwarding,no-X11-forwarding,no-user-rc"
printf '%s %s %s threetears-testlane\n' "$OPTS" "$ktype" "$kbody" | $SSH "$REMOTE" \
    'umask 077; mkdir -p ~/.ssh; ak=~/.ssh/authorized_keys; touch "$ak";
     { grep -v " threetears-testlane$" "$ak" || true; cat; } > "$ak.tmp" && mv -f "$ak.tmp" "$ak"'
echo "    authorized_keys: lane key $(ssh-keygen -lf "$PUB" 2>/dev/null | cut -d' ' -f2) -> $SHIM"

echo "==> Verifying through the shim, as the lane key would"
probe="$($SSH "$REMOTE" "SSH_ORIGINAL_COMMAND='docker inspect -f {{.State.Running}} $CONTAINER' $SHIM" || true)"
[ "$probe" = true ] || die "lane probe answered '$probe', not 'true'"
rc=0
$SSH "$REMOTE" "SSH_ORIGINAL_COMMAND=id $SHIM" >/dev/null 2>&1 || rc=$?
[ "$rc" = 77 ] || die "the shim did not refuse a shell command (exit $rc)"
echo "    probe: true; shell command: refused"
echo "Done. On the dispatcher: THREETEARS_TEST_REMOTE=<lane alias for this host> ./scripts/test.sh"
