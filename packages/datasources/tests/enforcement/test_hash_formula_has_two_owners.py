"""the column-hash formula is written in its two SQL constants only, and every SQL constant cited exists.

three sides must agree byte for byte: ``column_hash_payload`` in python, and
the asyncpg and Redshift ``table_hashes`` SQL. a restated copy anywhere else
stops agreeing, silently, the next time the formula changes; and a docstring
that cites a constant by a name that no longer exists sends the next driver's
implementer to nothing.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[2]
_SRC = _PACKAGE / "src"
_THIS_FILE = Path(__file__).resolve()

#: the SQL spelling of the payload; only the two table-hash constants may carry it.
_FRAGMENT = "column_name || ':' || data_type"
_OWNERS = {
    "src/threetears/datasources/drivers/asyncpg_driver.py": 1,
    "src/threetears/datasources/drivers/redshift_driver.py": 1,
}

#: a module-level SQL constant's name, as prose and code cite it.
_SQL_CONSTANT = re.compile(r"(?<![\w.])(_[A-Z][A-Z0-9_]*_SQL(?:_TEMPLATE)?)\b")


def _text_files() -> list[Path]:
    """every python and markdown file in the package, this one excepted.

    :return: the files
    :rtype: list[Path]
    """
    return sorted(
        path
        for path in _PACKAGE.rglob("*")
        if path.suffix in {".py", ".md"} and path.is_file() and path.resolve() != _THIS_FILE
    )


def _module_level_names() -> set[str]:
    """every name assigned at module level anywhere in the package's source.

    :return: the names
    :rtype: set[str]
    """
    names: set[str] = set()
    for path in _SRC.rglob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign):
                names.update(target.id for target in node.targets if isinstance(target, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def test_the_formula_is_written_only_in_its_two_sql_constants() -> None:
    found = {str(path.relative_to(_PACKAGE)): path.read_text().count(_FRAGMENT) for path in _text_files()}

    assert {name: count for name, count in found.items() if count} == _OWNERS, (
        "describe the formula by pointing at threetears.datasources.introspection.column_hash_payload; "
        "a restated copy stops agreeing the next time the formula changes"
    )


def test_every_cited_sql_constant_exists() -> None:
    defined = _module_level_names()
    missing = sorted(
        f"{path.relative_to(_PACKAGE)}: {match.group(1)}"
        for path in _text_files()
        for match in _SQL_CONSTANT.finditer(path.read_text())
        if match.group(1) not in defined
    )

    assert missing == [], f"these cite SQL constants that do not exist: {missing}"
