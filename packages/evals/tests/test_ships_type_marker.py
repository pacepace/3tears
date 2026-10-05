"""The package ships its PEP 561 marker, so an adopter's type checker reads its annotations.

Without ``py.typed`` a strict mypy run in an adopting app skips every ``threetears.evals`` import
as untyped, and the package's annotations check nothing on the adopter's side.
"""

from __future__ import annotations

from pathlib import Path

import threetears.evals


def test_the_package_carries_py_typed() -> None:
    """``py.typed`` sits beside the package's top-level ``__init__``."""
    package_dir = Path(threetears.evals.__file__).parent
    assert (package_dir / "py.typed").is_file()
