"""AST-walker enforcement test for hardcoded pool / executor / timeout kwargs.

every concrete driver (asyncpg, redshift, snowflake, bigquery) MUST
pull its pool sizing / executor sizing / timeout knobs from the
relevant :class:`ConnectionConfig` member's documented defaults --
never from a literal at the call site. inline literals invite
silent drift between drivers; the enforcement test makes the rule
machine-checkable so the contract holds across future shards.

the walker targets specific kwarg NAMES (the actual hazard), not
"any integer literal" (which would false-positive on HTTP status
codes, retry counts, length math). the banned set is the constant
:data:`_BANNED_POOL_KWARGS` at the top of this file so future
reviewers can see the full list at a glance + extend it cleanly.

scope:

- walks every ``.py`` file under
  ``threetears/datasources/drivers/`` recursively
- flags every ``kwarg=Constant`` pair where ``kwarg`` is in the
  banned set
- ignores keyword args whose value is a Name / Attribute (config
  attribute access) -- that's the intended pattern

the drivers directory does NOT exist yet (lands in shard 09); the
walker treats absence as a no-op. when drivers ship, this test
becomes the load-bearing guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# banned kwarg NAMES. if a driver call site passes one of these with
# a Constant (literal) value, the test fails. extending the list is
# a one-line change visible in code review.
_BANNED_POOL_KWARGS: frozenset[str] = frozenset(
    {
        "min_size",
        "max_size",
        "max_workers",
        "timeout",
        "command_timeout",
        "connect_timeout",
        "query_timeout",
        "cache_size",
        "pool_size",
        "connection_cache_size",
        "executor_max_workers",
    }
)

# repo root path resolved relative to this file. allows pytest to run
# from anywhere.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DRIVERS_DIR = _PACKAGE_ROOT / "src" / "threetears" / "datasources" / "drivers"


def _iter_driver_modules() -> list[Path]:
    """return every ``.py`` file in the drivers/ tree (recursive).

    :return: sorted list of driver-module paths (empty when drivers/ doesn't exist yet)
    :rtype: list[Path]
    """
    if not _DRIVERS_DIR.is_dir():
        return []
    return sorted(_DRIVERS_DIR.rglob("*.py"))


def _find_banned_literals(path: Path) -> list[tuple[int, str, str]]:
    """walk one module for ``kwarg=Constant`` pairs in the banned set.

    :param path: driver-module path to walk
    :ptype path: Path
    :return: list of ``(lineno, kwarg, repr(literal))`` tuples
    :rtype: list[tuple[int, str, str]]
    """
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    hits: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg is None:
                # **kwargs splat -- not a literal kwarg, skip
                continue
            if kw.arg not in _BANNED_POOL_KWARGS:
                continue
            if isinstance(kw.value, ast.Constant):
                hits.append((kw.lineno, kw.arg, repr(kw.value.value)))
    return hits


@pytest.mark.parametrize("driver_module", _iter_driver_modules(), ids=lambda p: p.name)
def test_no_hardcoded_pool_params(driver_module: Path) -> None:
    """driver modules MUST NOT inline literal values for banned pool kwargs.

    if this test fails: replace the literal with a read off the
    relevant :class:`ConnectionConfig` member's field (defaults live
    on the pydantic model with documented descriptions).

    :param driver_module: driver-module path under test
    :ptype driver_module: Path
    """
    hits = _find_banned_literals(driver_module)
    if hits:
        rendered = "\n".join(
            f"  {driver_module.relative_to(_PACKAGE_ROOT)}:{lineno}: {kwarg}={value}" for lineno, kwarg, value in hits
        )
        raise AssertionError(
            f"banned literal pool/executor/timeout kwargs in driver module "
            f"(read from ConnectionConfig fields instead):\n{rendered}"
        )


def test_walker_loads_banned_set() -> None:
    """sanity: the banned set is non-empty and contains the documented kwargs.

    locks in the contract so a future "let me empty this set quickly"
    refactor surfaces in code review.
    """
    assert "min_size" in _BANNED_POOL_KWARGS
    assert "max_workers" in _BANNED_POOL_KWARGS
    assert "command_timeout" in _BANNED_POOL_KWARGS
    assert "connection_cache_size" in _BANNED_POOL_KWARGS
    assert "executor_max_workers" in _BANNED_POOL_KWARGS
