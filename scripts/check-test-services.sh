#!/usr/bin/env bash
# Fail unless this machine holds the services the integration tests start as child processes.
#
#   ./scripts/check-test-services.sh PG_MAJOR NATS_VERSION...
#   ./scripts/check-test-services.sh 16 2.14.2 2.12.6
#
# The integration suite on the testament pool starts its own Postgres and nats-server inside the
# project's container rather than in Docker, which the pool's image has none of. So the image must
# hold what the docker images the tests name hold: Postgres PG_MAJOR with pgvector, as in
# pgvector/pgvector:pg16, with the en_US.UTF-8 locale that image's initdb uses, and one nats-server per
# `nats:<version>-alpine` the tests pin, at /opt/nats/<version>/nats-server. Postgres's client tools
# (pg_dump, psql) must resolve to the same major, since backup matches its server to pg_dump's.
# docker/testament.Dockerfile runs this at build, so an image missing any of them never reaches a
# runner. It prints each version it finds, which puts the PGDG minor it installed into the build log
# (the packages are pinned by major only). tests/enforcement/test_testament_image_services.py holds
# the Dockerfile's arguments to what the tests name.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 PG_MAJOR NATS_VERSION..." >&2
    exit 2
fi
major="$1"
shift

bad=0
fail() {
    echo "check-test-services: $*" >&2
    bad=1
}

# "initdb (PostgreSQL) 16.11 (Debian 16.11-1.pgdg13+1)" -> the word after "(PostgreSQL)"
major_of() { sed -En 's/^[^ ]+ \(PostgreSQL\) ([0-9]+)\..*/\1/p' <<<"$1"; }

bin="/usr/lib/postgresql/${major}/bin"
for tool in initdb postgres pg_ctl pg_isready; do
    if [ ! -x "${bin}/${tool}" ]; then
        fail "${bin}/${tool} not found: install postgresql-${major}"
        continue
    fi
    said="$("${bin}/${tool}" --version)"
    echo "${tool}: ${said}"
    [ "$(major_of "$said")" = "$major" ] || fail "${bin}/${tool} is not Postgres ${major}: ${said}"
done
for tool in pg_dump pg_dumpall pg_restore psql; do
    if ! command -v "$tool" >/dev/null; then
        fail "${tool} not on PATH: install postgresql-client-${major}"
        continue
    fi
    said="$("$tool" --version)"
    echo "${tool} ($(command -v "$tool")): ${said}"
    [ "$(major_of "$said")" = "$major" ] || fail "${tool} on PATH is not Postgres ${major}: ${said}"
done

control="/usr/share/postgresql/${major}/extension/vector.control"
if [ -f "$control" ]; then
    echo "pgvector: $(sed -En "s/^default_version = '(.*)'/\1/p" "$control")"
else
    fail "${control} not found: install postgresql-${major}-pgvector"
fi

# read whole, then matched: a quiet grep at the end of a pipe exits on its first match, and under
# pipefail the writer's SIGPIPE would fail the match (tests/enforcement/test_no_grep_q_pipeline_under_pipefail.py)
locales="$(locale -a)"
if grep -qix 'en_US.utf8' <<<"$locales"; then
    echo "locale: en_US.UTF-8"
else
    fail "the en_US.UTF-8 locale is not generated (pgvector/pgvector:pg16's initdb runs in it)"
fi

for version in "$@"; do
    server="/opt/nats/${version}/nats-server"
    if [ ! -x "$server" ]; then
        fail "${server} not found"
        continue
    fi
    said="$("$server" --version)"
    echo "nats-server ${version}: ${said}"
    case "$said" in
        *"v${version}") ;;
        *) fail "${server} reports '${said}', not v${version}" ;;
    esac
done

du -sh /usr/lib/postgresql /opt/nats || true
exit "$bad"
