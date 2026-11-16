"""``python -m threetears.pwa export-static <dir>``: copy ``pwa.js`` and ``safe-area.css`` into an
app's static directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from threetears.pwa.assets import STATIC_FILES, export_static


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m threetears.pwa")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-static", help=f"copy {', '.join(STATIC_FILES)} into a directory")
    export.add_argument("dest", type=Path, help="the app's static directory for them")
    args = parser.parse_args(argv)
    written = export_static(args.dest)
    for path in written:
        sys.stdout.write(f"wrote {path}\n")
    if not written:
        sys.stdout.write(f"{args.dest}: up to date\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
