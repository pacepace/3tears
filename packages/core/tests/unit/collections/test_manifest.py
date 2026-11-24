"""the startup loader imports every listed collection module, and a lookup for an undeclared table is loud.

:mod:`threetears.core.collections.manifest` imports what each installed package lists under the
``threetears.collections`` entry points; :mod:`threetears.core.collections.undeclared_tables` reports a registry
lookup (tenancy, write generation, L2-first) for a table none of them declares: ERROR and a count, once per table and
lookup, with today's fallback kept this release.
"""

from __future__ import annotations

import importlib.metadata
import logging
import subprocess
import sys

import pytest
from prometheus_client import REGISTRY

from threetears.core.collections import load_declared_collections
from threetears.core.collections.generation import has_write_generation, tenancy_of_table, written_l2_first
from threetears.core.tenancy import TENANCY_UNDECLARED

_LOGGER = "threetears.core.collections.undeclared_tables"


def _count(table: str, lookup: str) -> float:
    value = REGISTRY.get_sample_value("threetears_undeclared_table_total", {"table": table, "lookup": lookup})
    return 0.0 if value is None else value


def test_the_loader_imports_every_listed_module_and_records_their_tables() -> None:
    listed = {entry.module for entry in importlib.metadata.entry_points(group="threetears.collections")}
    assert "threetears.core.coordination.tables" in listed, "core lists its own collection module"
    loaded = load_declared_collections()
    skipped = {each.split(" ")[0] for each in loaded.skipped}
    assert set(loaded.modules) | skipped == listed
    assert all(module in sys.modules for module in loaded.modules)
    assert "coordination_claims" in loaded.tables
    assert load_declared_collections() is loaded, "one load per process"


def test_a_lookup_for_an_undeclared_table_logs_and_counts_once_and_keeps_the_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    load_declared_collections()
    table = "no_such_table_manifest_lookups"
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        assert tenancy_of_table(table) is TENANCY_UNDECLARED
        assert tenancy_of_table(table) is TENANCY_UNDECLARED
        assert has_write_generation(table) is False
        assert has_write_generation(table) is False
        assert written_l2_first(table) is False
        assert written_l2_first(table) is False
    reports = [r for r in caplog.records if r.name == _LOGGER and table in r.getMessage()]
    assert [r.levelno for r in reports] == [logging.ERROR] * 3, "one ERROR per table and lookup"
    assert {getattr(r, "extra_data", {}).get("lookup") for r in reports} == {"tenancy", "write_generation", "l2_first"}
    assert [_count(table, lookup) for lookup in ("tenancy", "write_generation", "l2_first")] == [1.0, 1.0, 1.0]


def test_a_declared_table_without_the_feature_is_not_reported(caplog: pytest.LogCaptureFixture) -> None:
    loaded = load_declared_collections()
    assert "coordination_claims" in loaded.tables
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        written_l2_first("coordination_claims")
    assert [r for r in caplog.records if r.name == _LOGGER] == []
    assert _count("coordination_claims", "l2_first") == 0.0


@pytest.mark.uncacheable("a fresh interpreter is the point: this one has run the loader")
def test_a_process_that_never_ran_the_loader_reports_that() -> None:
    script = """
import logging, sys
from threetears.core.collections.generation import tenancy_of_table
records = []
class Keep(logging.Handler):
    def emit(self, record):
        records.append(record)
logging.getLogger("threetears.core.collections.undeclared_tables").addHandler(Keep())
tenancy_of_table("any_table")
print([r.extra_data["loader_ran"] for r in records])
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.strip().splitlines()[-1] == "[False]"


@pytest.mark.xfail(
    strict=True,
    reason="NEXT RELEASE: a lookup for a table no listed module declares is refused, not answered from the fallback; "
    "when the refusal lands this passes and strict xfail fails, so the marker comes off with it",
)
def test_an_undeclared_table_is_refused() -> None:
    load_declared_collections()
    with pytest.raises(LookupError):
        tenancy_of_table("no_such_table_manifest_refused")
