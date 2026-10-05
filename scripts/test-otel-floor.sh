#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run threetears.observe's OpenTelemetry tests against the OpenTelemetry FLOOR that
# 3tears-observe declares, not the version uv.lock happens to pin.
#
# Why this exists: the workspace suite only ever runs against the locked OpenTelemetry, which is
# always recent, so a declared floor can fall behind the code without anything going red. It did:
# `_otel_internals` imports `LogRecordExporter` from `opentelemetry.sdk._logs.export`, which first
# shipped in opentelemetry-sdk 1.39.0, while the `otel` extra still said `>=1.28`. A consumer that
# pinned 1.36 (survey) resolved cleanly and then failed at import, in thirteen of its own tests.
#
# How: read the `otel` extra of packages/observe/pyproject.toml, turn every `>=X` floor into
# `==X`, and run the OpenTelemetry-touching observe tests in an isolated environment holding
# exactly those versions plus observe itself. The floors are read from the pyproject, never
# restated here, so raising a floor is what moves this check.
#
# The repo-root conftest is kept out on purpose (`-c` points pytest at observe's own pyproject,
# which makes packages/observe the rootdir): it pulls in threetears.core and langchain, which
# observe does not depend on, and installing them would let their own OpenTelemetry pins move the
# versions under test.
#
# Usage: ./scripts/test-otel-floor.sh [extra pytest args...]

OBSERVE_DIR="packages/observe"

FLOOR_PINS=()
while IFS= read -r pin; do
    FLOOR_PINS+=("--with" "$pin")
done < <(uv run --no-project --python 3.14 python - "$OBSERVE_DIR/pyproject.toml" <<'PY'
import re
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    extra = tomllib.load(handle)["project"]["optional-dependencies"]["otel"]
for requirement in extra:
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)>=([0-9][0-9A-Za-z.]*)(,<[0-9A-Za-z.]+)?", requirement)
    if match is None:
        sys.exit(f"cannot read a >= floor from {requirement!r} in the otel extra; declare one")
    print(f"{match.group(1)}=={match.group(2)}")
PY
)

if [ "${#FLOOR_PINS[@]}" -eq 0 ]; then
    echo "the otel extra of $OBSERVE_DIR/pyproject.toml declares nothing; refusing to report a pass" >&2
    exit 1
fi

echo "observe OpenTelemetry floor under test: ${FLOOR_PINS[*]}"

uv run --isolated --no-project --python 3.14 \
    --with "./$OBSERVE_DIR" \
    "${FLOOR_PINS[@]}" \
    --with "pytest>=8.0" \
    --with "pytest-asyncio>=0.23" \
    --with "pytest-timeout>=2.3" \
    pytest \
        -c "$OBSERVE_DIR/pyproject.toml" \
        --import-mode=importlib \
        -o asyncio_mode=auto \
        -p no:cacheprovider \
        "$OBSERVE_DIR/tests/test_otel_internals.py" \
        "$OBSERVE_DIR/tests/test_setup.py" \
        "$OBSERVE_DIR/tests/test_tracing.py" \
        "$OBSERVE_DIR/tests/test_smoke.py" \
        ${@+"$@"}
