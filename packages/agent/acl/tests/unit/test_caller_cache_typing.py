"""a caller cache of any answer type binds to the access tables with no cast, as mypy checks it.

``CallerAccessCache`` is generic in its answer and invariant in it, so a binder typed for one answer type
refuses a cache of another: a host binding a ``CallerAccessCache[frozenset[str]]`` had to cast. This runs mypy
over such a binding.
"""

from __future__ import annotations

from pathlib import Path

from mypy import api

_PROGRAM = """
from threetears.agent.acl import CallerAccessCache, bind_caller_cache_to_access_tables
from threetears.core.collections.registry import CollectionRegistry

names: CallerAccessCache[frozenset[str]] = CallerAccessCache()
counts: CallerAccessCache[int] = CallerAccessCache()
bind_caller_cache_to_access_tables(CollectionRegistry(), names)
bind_caller_cache_to_access_tables(CollectionRegistry(), counts)
"""


def test_a_cache_of_any_answer_type_binds_without_a_cast(tmp_path: Path) -> None:
    program = tmp_path / "binds.py"
    program.write_text(_PROGRAM, encoding="utf-8")
    stdout, stderr, status = api.run(["--strict", "--no-incremental", "--follow-imports=silent", str(program)])
    assert status == 0, f"mypy refused the binding:\n{stdout}{stderr}"
