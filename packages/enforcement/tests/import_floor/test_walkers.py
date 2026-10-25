"""the import-floor probe and allowlist, against the standard library's own optional import.

``bisect`` imports its C accelerator ``_bisect`` under a guard and runs without it, which is the shape
an optional extra's module takes: refused, the probe measures the module without it; unguarded, a
refused root fails the probe.
"""

from __future__ import annotations

import pytest

from threetears.enforcement.import_floor import (
    ImportFloorConfig,
    ProbeError,
    modules_loaded_by_importing,
    outside_floor,
    run_import_floor_enforcement,
)


def _floor(base_modules: tuple[str, ...] = ("bisect",), refused_roots: tuple[str, ...] = ()) -> ImportFloorConfig:
    return ImportFloorConfig(
        base_modules=base_modules,
        third_party_roots=frozenset({"pydantic"}),
        family_prefixes=("threetears.iam",),
        refused_roots=refused_roots,
    )


def test_the_probe_reports_what_the_import_loaded() -> None:
    assert "bisect" in modules_loaded_by_importing("bisect")


def test_a_refused_root_behind_a_guard_is_left_out_of_what_loads() -> None:
    assert "_bisect" in modules_loaded_by_importing("bisect")
    assert "_bisect" not in modules_loaded_by_importing("bisect", refused=("_bisect",))


def test_a_refused_root_imported_without_a_guard_fails_the_probe() -> None:
    with pytest.raises(ProbeError, match="importing bisect failed"):
        modules_loaded_by_importing("bisect", refused=("bisect",))


def test_the_floor_permits_the_stdlib_its_roots_its_family_and_its_namespaces_by_exact_name() -> None:
    floor = _floor()
    loaded = [
        "json",
        "_sysconfigdata__linux_x86_64-linux-gnu",
        "pydantic.fields",
        "threetears",
        "threetears.iam",
        "threetears.iam.totp",
        "threetears.core",
        "threetears.iamposter",
        "sqlalchemy.orm",
    ]
    assert outside_floor(floor, loaded) == ["sqlalchemy.orm", "threetears.core", "threetears.iamposter"]


def test_a_floor_with_nothing_to_probe_is_refused() -> None:
    with pytest.raises(ValueError, match="no base modules"):
        _floor(base_modules=())


def test_a_root_both_refused_and_permitted_is_refused() -> None:
    with pytest.raises(ValueError, match="both refused and permitted"):
        _floor(refused_roots=("pydantic",))


def test_the_runner_passes_a_module_within_its_floor() -> None:
    run_import_floor_enforcement(_floor(refused_roots=("_bisect",)), "bisect")


def test_the_runner_fails_a_module_that_loads_past_its_floor() -> None:
    # this package is on no floor here but the namespace above it, so importing it loads past the floor
    floor = _floor(base_modules=("threetears.enforcement",))
    with pytest.raises(pytest.fail.Exception, match=r"outside its floor: \['threetears.enforcement'\]"):
        run_import_floor_enforcement(floor, "threetears.enforcement")


def test_the_runner_refuses_a_module_its_floor_does_not_name() -> None:
    with pytest.raises(ValueError, match="not one of the floor's base modules"):
        run_import_floor_enforcement(_floor(), "email.message")
