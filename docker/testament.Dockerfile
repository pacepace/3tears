# The image testament's runners run the unit suite in (see .testament.toml). It holds only the
# toolchain: the workspace venv is built by the manifest's setup step, from the tree under test.
FROM ghcr.io/astral-sh/uv:0.11.32 AS uv

FROM python:3.14-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

RUN useradd --create-home --shell /bin/bash lane
USER lane

# PYTHONDONTWRITEBYTECODE is deliberately unset: bytecode caching saves ~31s per collection.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

CMD ["sleep", "infinity"]
