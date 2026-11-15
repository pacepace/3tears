# The trial container: one docs build installed, a user that is not root, and nothing else.
#
# Built at set-up by packages/evals/onramp/isolation.py (`docker build`, with the wheel as the whole context), never
# inside a trial: a trial runs with no network. The build is installed into /opt/build, owned by root, so the agent's
# commands can read it and not change it; at run time the root filesystem is read-only besides.
#
# The base is pinned by digest, so every arm and every rerun of a baseline runs on the same interpreter.
ARG BASE=python:3.14-slim@sha256:cd89c30eda26ee911a4611b18680086e0ea42e6f5f8cbe653d07fa7958c9cec8
FROM ${BASE}

COPY *.whl /tmp/wheel/
RUN python -m venv /opt/build \
    && /opt/build/bin/pip install --no-cache-dir --disable-pip-version-check /tmp/wheel/*.whl \
    && rm -rf /tmp/wheel \
    && useradd --uid 10001 --user-group --no-create-home --home-dir /work --shell /usr/sbin/nologin agent \
    && mkdir /work

ENV PATH=/opt/build/bin:/usr/local/bin:/usr/bin:/bin \
    HOME=/work \
    LANG=C.UTF-8 \
    PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
WORKDIR /work
