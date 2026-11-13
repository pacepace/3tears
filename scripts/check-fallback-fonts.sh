#!/usr/bin/env bash
# Fail unless this machine's generic font families resolve to DejaVu, as the dev box's do.
#
#   ./scripts/check-fallback-fonts.sh
#
# The evals render tests (the `browser` marker) draw the package's report pages, whose stylesheet names
# a local font stack and no web font: --f-body and --f-display end in system-ui ... sans-serif, --f-mono
# in monospace (kit.css), and a page with no stack of its own falls to the browser's default, serif.
# IBM Plex is not installed on the dev box, the testament image or CI's runners, so those pages draw in
# whatever these families resolve to, and the checks measure that face: label overlaps, line counts and
# word breaks all move with it. Without DejaVu, Debian's fontconfig resolved them to WenQuanYi Zen Hei,
# a CJK face whose taller boxes failed 15 of the 73 render tests on the pool. Run where the tests run:
# docker/testament.Dockerfile at build, and CI's evals shards after their Chromium install.
set -euo pipefail

command -v fc-match >/dev/null || { echo "check-fallback-fonts: fc-match not found (install fontconfig)" >&2; exit 2; }

bad=0
for family in sans-serif system-ui serif monospace; do
    face="$(fc-match --format='%{family[0]}' "$family")"
    echo "${family} -> ${face}"
    case "$face" in
        "DejaVu "*) ;;
        *)
            echo "check-fallback-fonts: ${family} resolves to '${face}', not DejaVu: install fonts-dejavu-core" >&2
            bad=1
            ;;
    esac
done
exit "$bad"
