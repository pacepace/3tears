"""``python -m threetears.enforcement.release``: the same tool as ``threetears-release``."""

from __future__ import annotations

import sys

from threetears.enforcement.release.cli import main

__all__: list[str] = []

sys.exit(main())
