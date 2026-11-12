# The image testament's runners run the unit suite in (see .testament.toml). It holds only the
# toolchain: the workspace venv is built by the manifest's setup step, from the tree under test.
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

RUN useradd --create-home --shell /bin/bash lane
USER lane

# PYTHONDONTWRITEBYTECODE is deliberately unset: bytecode caching saves ~31s per collection.
# CXX: this Python's build config names gcc as its C++ compiler, so setuptools would link a C++
# extension built from source (indexed-zstd, above) without libstdc++, and it fails at import.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    CXX=g++

CMD ["sleep", "infinity"]
