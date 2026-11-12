# The image testament's runners run the unit suite in (see .testament.toml). It holds only the
# toolchain, headless Chromium, and the Postgres and nats-server binaries the integration tests start
# as child processes: the workspace venv is built by the manifest's setup step, from the tree under test.
# uv by digest too (the multi-arch index): a tag can be pushed again, and the copied binary with it.
FROM ghcr.io/astral-sh/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c AS uv

# Pinned to one patch by digest (the multi-arch index, so every runner architecture resolves it).
# A floating `python:3.14-slim` moves the stdlib under the tests at each image rebuild: when it
# became 3.14.8, a robots.txt test that passes on the dev box's 3.14.6 failed on the pool. Move the
# pin deliberately, alongside the dev box.
FROM python:3.14.6-slim@sha256:7bec7ddcddeff7975d6ba9b4be7dd6f6b2f55e7491539145e2978f7f97ce9144

# build-essential and libzstd-dev: indexed-zstd (3tears-scrape -> camoufox -> fpgen) ships no
# linux/aarch64 wheel, so on an arm64 runner the setup step builds it from its sdist, against libzstd.
# Without them every batch planned onto an arm64 runner fails its env build and never reports.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates procps build-essential libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# Headless Chromium for the evals render tests (the `browser` marker), with the system libraries and fonts it needs,
# installed by the playwright the workspace locks: its browsers.json pins the Chromium build, and the browser must be
# the one that playwright drives. tests/enforcement/test_testament_image_playwright.py fails when this version and
# uv.lock's part. Only the headless shell: the tests launch headless, and the full browser would add ~150 MB. Root-owned
# under a fixed path, read by `lane` through PLAYWRIGHT_BROWSERS_PATH (here and in .testament.toml's env). Playwright
# builds the shell for linux x86_64 and arm64 alike, so every runner's architecture installs it.
# fonts-dejavu-core: the package's pages set a local font stack (IBM Plex where installed, then system-ui ...), so
# the faces they render in are the machine's. Playwright's own font set has no DejaVu, and without it fontconfig
# resolves sans-serif, system-ui and monospace to WenQuanYi Zen Hei, a CJK face whose taller boxes made a 390 px
# page's bar notes overlap where the dev box (DejaVu Sans) draws them clear. With it, fc-match gives the dev box's
# faces. fontconfig for fc-match: scripts/check-fallback-fonts.sh fails the build unless they are DejaVu.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright
COPY --chmod=0755 scripts/check-fallback-fonts.sh /usr/local/bin/check-fallback-fonts
RUN UV_NO_CACHE=1 uvx --from playwright==1.60.0 playwright install --with-deps --only-shell chromium \
    && apt-get install -y --no-install-recommends fonts-dejavu-core fontconfig \
    && check-fallback-fonts \
    && chmod -R a+rX /opt/ms-playwright \
    && rm -rf /var/lib/apt/lists/*

# The services the integration tests start as child processes (the pool's container runs no Docker daemon): what the
# docker images the tests name hold, installed after the browser layer so a change here keeps that one cached.
#
# Postgres 16 with pgvector, as pgvector/pgvector:pg16 (threetears.core.testing.fixtures' default image), from the
# PostgreSQL project's apt repository (PGDG), whose script Debian's postgresql-common ships for amd64 and arm64 alike.
# Pinned by major only: a rebuild may move the minor, and check-test-services prints it into the build log. The client
# tools come with it, so /usr/bin/pg_dump is 16 and backup's server matches it. No cluster is made at install
# (create_main_cluster = false, written once the PGDG postgresql-common is in, before the server): each test process
# initdb's its own, as `lane`. en_US.UTF-8, because the docker image's initdb runs in it and text sorts by it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl locales postgresql-common \
    && sed -i 's/^# *\(en_US.UTF-8 UTF-8\)$/\1/' /etc/locale.gen \
    && locale-gen \
    && /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y \
    && apt-get install -y --no-install-recommends postgresql-common \
    && sed -Ei 's/^#?[[:space:]]*create_main_cluster[[:space:]]*=.*/create_main_cluster = false/' \
        /etc/postgresql-common/createcluster.conf \
    && grep -qx 'create_main_cluster = false' /etc/postgresql-common/createcluster.conf \
    && apt-get install -y --no-install-recommends postgresql-16 postgresql-client-16 postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*

# nats-server, one per `nats:<version>-alpine` the tests pin (tests/enforcement/test_testament_image_services.py keeps
# this list complete), from the release tarballs, each checked against its sha256 for this architecture.
RUN set -eu; \
    arch="$(dpkg --print-architecture)"; \
    for pin in \
        "2.14.2 amd64 b3e7b14eb10c895fd90c2dacdb6b65bd3208adcc9524dd7689ba2c1024e6b97a" \
        "2.14.2 arm64 15fd0c3438e7178e5316e63be68373ad581c8d78db26e649113aa303b74e5e58" \
        "2.12.6 amd64 77fe7dd69ff5144126026b78355900be0ab0bb4339dc61a7621dcc9b9e9d07a6" \
        "2.12.6 arm64 fddaf3f223c7af3f4d0a0d2c2fc084406e6b3ec7adfd1b9e6a37fbd03bfe222f"; \
    do \
        set -- $pin; \
        [ "$2" = "$arch" ] || continue; \
        name="nats-server-v$1-linux-$2"; \
        curl -fsSL -o "/tmp/$name.tar.gz" "https://github.com/nats-io/nats-server/releases/download/v$1/$name.tar.gz"; \
        echo "$3  /tmp/$name.tar.gz" | sha256sum -c -; \
        mkdir -p "/opt/nats/$1"; \
        tar -xzf "/tmp/$name.tar.gz" -C "/opt/nats/$1" --strip-components=1 "$name/nats-server"; \
        rm "/tmp/$name.tar.gz"; \
    done

# Fails the build unless every service above is there at the version the tests expect, and prints what it found.
COPY --chmod=0755 scripts/check-test-services.sh /usr/local/bin/check-test-services
RUN check-test-services 16 2.14.2 2.12.6

RUN useradd --create-home --shell /bin/bash lane
USER lane

# PYTHONDONTWRITEBYTECODE is deliberately unset: bytecode caching saves ~31s per collection.
# CXX: this Python's build config names gcc as its C++ compiler, so setuptools would link a C++
# extension built from source (indexed-zstd, above) without libstdc++, and it fails at import.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    CXX=g++

CMD ["sleep", "infinity"]
