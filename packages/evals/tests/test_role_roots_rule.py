"""The re-root's rule places every public name of today on exactly one of the five role roots, and its map is sound.

``scripts/role_roots.py`` holds the rule (design-reroot-cutover.md §1.1: rules 1 to 6, observed use, reached by
definition, the switch for unreached provisional names) and ``scripts/reroot_map.py`` renders it as the map the
rewrite script reads. This holds, under BOTH settings of the switch (the owner's answer is the generator's default, and
no test here depends on which it is):

- every name of the API snapshot is placed exactly once, so a name that merges late is placed, or this fails at once;
- each placement follows the rule and its listed exceptions (§1.2, §4.1, §4.2, §9 Q2);
- no two names share a new spelling, and every stored document class and action result is public after the cut;
- the map keeps its invariant (no new value equals or extends an old one), round-trips through its own reader, and
  renders byte-identically twice;
- every name the recorded host's production code imports has a map line or keeps its spelling, and none leaves.
"""

from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.script_loader import load_script

surface = load_script("public_surface.py")
role_roots = load_script("role_roots.py")
reroot_map = load_script("reroot_map.py")

#: A fixed epoch, so a rendered map does not depend on the checkout's ``HEAD``.
EPOCH = "0123456789abcdef0123456789abcdef01234567"

SETTINGS = ("leave", "stay")


@pytest.fixture(scope="module")
def plans() -> dict[str, Any]:
    return {setting: role_roots.plan(setting) for setting in SETTINGS}


@pytest.fixture(params=SETTINGS)
def planned(request: pytest.FixtureRequest, plans: dict[str, Any]) -> Any:
    return plans[request.param]


def _root(spelling: str) -> str:
    return spelling.rsplit(".", 1)[0]


def test_every_name_of_the_snapshot_is_placed_exactly_once(planned: Any) -> None:
    olds = Counter(placement.old for placement in planned.placements)
    assert not [old for old, count in olds.items() if count > 1]
    assert set(olds) == set(surface.load_snapshot())


def test_each_placement_follows_the_rule(planned: Any) -> None:
    for placement in planned.placements:
        if placement.record in ("demoted", "left"):
            assert placement.rule is None and placement.new == placement.home, placement
            assert placement.note.strip(), f"{placement.old} leaves with no reason or replacement"
            continue
        assert placement.new.rsplit(".", 1)[0] == role_roots.RULE_ROOTS[placement.rule], placement
        if placement.old in role_roots.METADATA:
            assert placement.rule == 1, placement
        elif placement.kind == "typed":
            assert placement.rule == 2, placement
        elif placement.kind == "tests":
            assert placement.rule == 3, placement
        elif placement.old in role_roots.PLACED_BY_RULING:
            assert placement.rule == role_roots.PLACED_BY_RULING[placement.old], placement
        elif any(placement.home.startswith(f"{prefix}.") for prefix in role_roots.ACTIONS_HOMES):
            assert placement.rule in (2, 3, 4), placement
        elif placement.kind in ("bound", "invoked", "action"):
            assert placement.rule == 5, placement
        else:
            assert placement.kind == "received" and placement.rule in (5, 6), placement
        if placement.record == "kept":
            assert placement.new == placement.old


def test_the_listed_exceptions_hold(planned: Any) -> None:
    new = {placement.old: placement for placement in planned.placements}
    package = role_roots.PACKAGE
    assert new[f"{package}.providers.Model"].new == f"{package}.Model"
    assert (
        new[f"{package}.ops.EvalSummary"].new
        == new[f"{package}.analysis.EvalSummary"].new
        == f"{package}.shapes.EvalSummary"
    )
    assert new[f"{package}.ROOT_EXTRAS"].new == f"{package}.EXTRAS"
    assert new[f"{package}.RootExtra"].new == f"{package}.Extra"
    assert new[f"{package}.transports.fastmcp.mount_fastmcp"].new == f"{package}.mount_fastmcp"
    assert new[f"{package}.providers.ProviderError"].new == f"{package}.host.ProviderError"
    for kept in role_roots.KEPT_AS_TESTS:
        assert _root(new[kept].new) == role_roots.TESTING, kept
    for demoted, replacement in role_roots.DEMOTED.items():
        assert (new[demoted].record, new[demoted].note) == ("demoted", replacement)
    for name in role_roots.KIND_EXCEPTIONS:
        assert new[name].record not in ("demoted", "left"), name


def test_the_switch_moves_only_unreached_provisional_names(plans: dict[str, Any]) -> None:
    leave = {placement.old: placement for placement in plans["leave"].placements}
    stay = {placement.old: placement for placement in plans["stay"].placements}
    assert not [old for old, placement in stay.items() if placement.record == "left"]
    for old, placement in leave.items():
        if placement.record == "left":
            assert placement.tier == "provisional", old
            assert stay[old].record in ("name", "kept"), old
        else:
            assert stay[old] == placement, old
    for setting, planned in plans.items():
        lines = role_roots.describe(planned)
        assert lines[0].endswith(setting)
        assert all(any(line.startswith(f"{root}\t") for line in lines) for root in role_roots.ROLE_ROOTS)


def test_no_two_names_share_a_spelling_and_everything_received_by_definition_is_public(planned: Any) -> None:
    assert role_roots.spelling_clashes(planned) == []
    objects = surface_objects(planned)
    public = {
        id(objects[placement.old]) for placement in planned.placements if placement.record not in ("demoted", "left")
    }
    added = {added.home for added in planned.added}
    for klass, why in _by_definition():
        home = f"{klass.__module__}.{klass.__qualname__}"
        assert id(klass) in public or any(spelling.endswith(f".{klass.__qualname__}") for spelling in added), (
            f"{home} ({why}) is not public after the cut"
        )


def surface_objects(planned: Any) -> dict[str, Any]:
    kinds = load_script("surface_kinds.py")
    return dict(kinds.objects_of([placement.old for placement in planned.placements]))


def _by_definition() -> list[tuple[type, str]]:
    found = [(klass, f"stored document class ({doc_type})") for doc_type, klass in role_roots.stored_classes().items()]
    found += [(klass, f"action result ({name})") for name, klass in role_roots.action_results().items()]
    return found


def test_the_map_keeps_its_invariant_reads_back_and_renders_the_same_twice(plans: dict[str, Any]) -> None:
    for setting, planned in plans.items():
        text = reroot_map.render_map(planned, EPOCH)
        parsed = reroot_map.parse_map(text)
        assert [epoch.sha for epoch in parsed.epochs] == [EPOCH]
        assert reroot_map.invariant_problems(parsed.lines, [EPOCH]) == []
        assert text.splitlines()[0] == "# reroot-map 1"
        assert reroot_map.render_map(role_roots.plan(setting), EPOCH) == text, f"{setting}: not deterministic"
        olds = {line.old for line in parsed.lines if line.record not in ("module", "root")}
        moved = {placement.old for placement in planned.placements if placement.record != "kept"}
        assert olds == moved


def test_a_map_that_rewrites_its_own_output_is_refused() -> None:
    line = reroot_map.Line
    overlapping = [
        line(
            EPOCH, "name", "threetears.evals.run.drift", "threetears.evals.host.drift", "threetears.evals.run.x.drift"
        ),
        line(EPOCH, "module", "threetears.evals.host", "threetears.evals.hosting"),
    ]
    assert reroot_map.invariant_problems(overlapping, [EPOCH])
    equal = [
        line(EPOCH, "name", "threetears.evals.ops.a", "threetears.evals.host.a", "threetears.evals.ops.x.a"),
        line(EPOCH, "name", "threetears.evals.host.a", "threetears.evals.shapes.a", "threetears.evals.ops.y.a"),
    ]
    assert reroot_map.invariant_problems(equal, [EPOCH])
    left = [line(EPOCH, "left", "threetears.evals.ops.a", "threetears.evals.ops.x.a", " ")]
    assert reroot_map.invariant_problems(left, [EPOCH])


def test_the_written_map_is_a_sound_map() -> None:
    parsed = reroot_map.parse_map(reroot_map.MAP_PATH.read_text(encoding="utf-8"))
    assert len(parsed.epochs) == 1
    assert {line.record for line in parsed.lines} <= set(reroot_map.RECORDS)


def test_the_hosts_production_imports_all_map(plans: dict[str, Any]) -> None:
    production = role_roots.host_import_names()
    assert production, "no recorded host import list"
    for setting, planned in plans.items():
        by_old = {placement.old: placement for placement in planned.placements}
        former = set(planned.roots)
        assert set(role_roots.MODULE_RENAMES) <= {line.old for line in _module_lines(planned)}
        for host, names in production.items():
            for name in sorted(names):
                if name in by_old:
                    assert by_old[name].record in ("kept", "name", "renamed"), (
                        f"{setting}: {host}'s production code imports {name}, which leaves ({by_old[name].record})"
                    )
                    continue
                # A module binding: a role root, a former root (a ``root`` line), or a module, which the cut-over
                # moves only by a ``module`` line.
                assert _is_module(name), f"{setting}: {host} imports {name}, which neither the map nor a module holds"
                assert name in former or name in role_roots.ROLE_ROOTS or name not in role_roots.MODULE_RENAMES


def _module_lines(planned: Any) -> list[Any]:
    return [line for line in reroot_map.map_lines(planned, EPOCH) if line.record == "module"]


def _is_module(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def test_observed_use_reads_what_a_host_raises_subclasses_and_passes() -> None:
    source = """
from threetears.evals.kernel import ValidationFailedError, StorageError, SEED_REMEDY
from threetears.evals import ops
from threetears.evals.ops import report_read, run_get

class MyStorageError(StorageError):
    pass

def go(host):
    host.call(report_read)
    ops.campaigns_list(host, SEED_REMEDY)
    raise ValidationFailedError("no")

def not_passed():
    print(run_get)
"""
    found = role_roots.observed_uses(
        source, "host.py", lambda path: path in ("threetears.evals", "threetears.evals.ops")
    )
    assert found == {
        ("threetears.evals.kernel.StorageError", "subclassed"),
        ("threetears.evals.kernel.ValidationFailedError", "raised"),
        ("threetears.evals.kernel.SEED_REMEDY", "passed"),
    }


def test_the_recorded_observed_use_names_only_package_names() -> None:
    for name, _use in role_roots.recorded_observed_uses():
        assert name.startswith(f"{role_roots.PACKAGE}."), name
    assert Path(role_roots.HOST_OBSERVED).is_dir()
