# The image testament's runners run the unit suite in (see .testament.toml). It holds only the
# toolchain: the workspace venv is built by the manifest's setup step, from the tree under test.
FROM ghcr.io/astral-sh/uv:0.11.32 AS uv

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

RUN useradd --create-home --shell /bin/bash lane
USER lane

# PYTHONDONTWRITEBYTECODE is deliberately unset: bytecode caching saves ~31s per collection.
# CXX: this Python's build config names gcc as its C++ compiler, so setuptools would link a C++
# extension built from source (indexed-zstd, above) without libstdc++, and it fails at import.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    CXX=g++

CMD ["sleep", "infinity"]
