"""What importing the contracts and the stages costs, as an allowlist (search-spec.md §2, §4.10 d).

``test_package_boundaries`` and ``test_contract_discipline`` already name modules the
leaf must never reach for. This test asks the harder question in the other direction:
not "did it avoid the things we thought of" but **"is everything it loaded on the
permitted list"** -- stdlib, pydantic, and ``3tears-media-contracts``, and nothing
else. A denylist only ever catches the weight somebody predicted; the next dependency
to arrive here will be one nobody wrote down.

That distinction is the leaf's whole promise to a constrained host. D24's floor is
pydantic plus two dependency-free family leaves, and a consumer that takes
``threetears.search.contracts`` to speak the vocabulary -- discodon copying the
``BudgetPort`` shape, samsung on a Pi -- pays exactly that and no more.

The same promise now covers **Aggregate and Select**, which is why this file probes
more than the vocabulary. ``docs/search-task-02-aggregate-and-select.md`` §5 records
that the stages were verified to load nothing beyond that floor and that the check was
"a fact somebody checked rather than a test that holds it". This is the test. It
matters because the stages are what a producer-seam consumer would actually import,
and success check 5 -- *a Pi deployment installs it without torch* -- is decided by
that import, not by the vocabulary's.

Measured in a fresh interpreter, because whatever the pytest process already imported
would otherwise hide the cost entirely.
"""

from __future__ import annotations

import json
import subprocess
import sys
from functools import cache

import pytest

#: Third-party import roots the ruled floor pays for: pydantic and the support tree
#: pydantic's own distribution installs. Enumerated rather than derived so a *change*
#: in what pydantic drags in is a visible event -- this list is the weight of the
#: floor, and a pydantic upgrade that widens it should be read, not absorbed.
_PERMITTED_THIRD_PARTY_ROOTS = frozenset(
    {
        "pydantic",
        "pydantic_core",
        "annotated_types",
        "typing_extensions",
        "typing_inspection",
    }
)

#: Family modules the contracts may load: the namespace packages on the way down,
#: this package's own ``__init__`` (which imports nothing on purpose), and the two
#: contracts trees. Exact names and prefixes are kept apart deliberately --
#: ``threetears.search`` is permitted while ``threetears.search.call`` is not, so a
#: lazy top-level surface that quietly turned eager fails here.
_PERMITTED_FAMILY_MODULES = frozenset({"threetears", "threetears.media", "threetears.search"})
_PERMITTED_FAMILY_PREFIXES = ("threetears.media.contracts", "threetears.search.contracts")

#: The stage modules of ``docs/search-task-02-aggregate-and-select.md`` §5. Listed by
#: name rather than discovered by walking the package, so that a *new* stage module is
#: a deliberate addition to this list -- discovery would silently extend the guarantee
#: to code nobody decided to make cheap.
_STAGE_MODULES = ("threetears.search.aggregate", "threetears.search.select")

#: CPython writes its build configuration into a module whose name embeds the ABI
#: flags and platform (``_sysconfigdata__darwin_darwin``), so it is the interpreter
#: rather than a dependency, yet it is absent from ``sys.stdlib_module_names``. Only
#: the prefix is stable across platforms, which is why this is a prefix test.
_INTERPRETER_ARTIFACT_PREFIXES = ("_sysconfigdata",)

_PROBE = """
import json
import sys

before = set(sys.modules)
import {module}  # noqa: F401
print(json.dumps(sorted(set(sys.modules) - before)))
"""


@cache
def _modules_loaded_by_importing(module: str) -> tuple[str, ...]:
    """Every module a fresh interpreter loads for *module*'s import.

    Cached across the session: the subprocess is still a fresh interpreter, so the
    measurement keeps its meaning, and the result for a given module is deterministic.
    Without it each assertion below pays its own interpreter start-up.

    :param module: the dotted module name to import in the probe
    :ptype module: str
    :return: the module names loaded, sorted
    :rtype: tuple[str, ...]
    """
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"probe for {module} failed:\n{result.stderr}"
    loaded: list[str] = json.loads(result.stdout.strip())
    return tuple(loaded)


def _is_permitted(module: str, *, also: tuple[str, ...] = ()) -> bool:
    """Whether the floor pays for *module*.

    :param module: a dotted module name from ``sys.modules``
    :ptype module: str
    :param also: family modules permitted on top of the floor -- the stage under probe
    :ptype also: tuple[str, ...]
    :return: true when the module is stdlib, pydantic's tree, or permitted family
    :rtype: bool
    """
    root = module.split(".")[0]
    if root in sys.stdlib_module_names or module.startswith(_INTERPRETER_ARTIFACT_PREFIXES):
        return True
    if root in _PERMITTED_THIRD_PARTY_ROOTS:
        return True
    if any(module == prefix or module.startswith(f"{prefix}.") for prefix in _PERMITTED_FAMILY_PREFIXES):
        return True
    return module in _PERMITTED_FAMILY_MODULES or module in also


def _modules_loaded_by_importing_the_contracts() -> tuple[str, ...]:
    """Every module a fresh interpreter loads for the contracts import.

    :return: the module names, sorted
    :rtype: tuple[str, ...]
    """
    return _modules_loaded_by_importing("threetears.search.contracts")


class TestContractsImportCost:
    def test_the_probe_actually_imported_the_contracts(self) -> None:
        """An allowlist over an empty set passes for the wrong reason."""
        loaded = _modules_loaded_by_importing_the_contracts()

        assert "threetears.search.contracts" in loaded
        assert "threetears.media.contracts" in loaded, (
            "the facets vocabulary is a module-top import; if it stopped loading, this test's "
            "permitted-family half is no longer being exercised"
        )

    def test_nothing_outside_the_permitted_floor_is_loaded(self) -> None:
        """§2: stdlib, pydantic, and media-contracts types -- an allowlist, not a denylist."""
        offenders = sorted(m for m in _modules_loaded_by_importing_the_contracts() if not _is_permitted(m))

        assert offenders == [], (
            "importing threetears.search.contracts loaded modules outside D24's floor: "
            f"{offenders}. Either the import belongs behind TYPE_CHECKING / inside a function, "
            "or the floor itself is changing -- in which case the ruling, the package's "
            "dependencies, and this list move together"
        )

    def test_no_module_of_the_working_layers_rides_along(self) -> None:
        """The contracts are the leaf within the leaf: adapters, Call and Bind stay unloaded.

        A consumer can take the vocabulary alone -- to satisfy ``BudgetPort`` by shape, or
        to read a metadata projection -- without importing a line of provider logic. Stated
        separately from the allowlist above because this is the property a *split* of the
        package would rest on (D23's non-breaking-move discipline), and it should fail by
        name if it is ever lost.
        """
        rode_along = sorted(
            module
            for module in _modules_loaded_by_importing_the_contracts()
            if module.startswith("threetears.search.") and not module.startswith("threetears.search.contracts")
        )

        assert rode_along == [], f"the contracts import pulled working layers: {rode_along}"


@pytest.mark.parametrize("stage", _STAGE_MODULES)
class TestStageImportCost:
    """Aggregate and Select cost the same floor the vocabulary does.

    §5 of the task doc states this as verified-by-hand and owes the test. The
    constraint it protects is a consumer's, not ours: samsung has already recorded
    package rejections for ``3tears-core`` and ``3tears-models`` on ``MemoryMax``
    grounds, and the seam is only cheap to adopt if importing a stage does not drag
    the framework in behind it.
    """

    def test_the_probe_actually_imported_the_stage(self, stage: str) -> None:
        """An allowlist over an empty set passes for the wrong reason."""
        assert stage in _modules_loaded_by_importing(stage)

    def test_nothing_outside_the_permitted_floor_is_loaded(self, stage: str) -> None:
        """No core, no agent, no httpx, no torch -- stated as an allowlist, not a denylist."""
        offenders = sorted(m for m in _modules_loaded_by_importing(stage) if not _is_permitted(m, also=(stage,)))

        assert offenders == [], (
            f"importing {stage} loaded modules outside D24's floor: {offenders}. A stage is "
            "the surface a constrained consumer imports, so weight arriving here is weight "
            "that deployment pays -- either the import belongs behind TYPE_CHECKING / inside "
            "a function, or the floor itself is changing and the ruling moves with it"
        )

    def test_no_other_working_layer_rides_along(self, stage: str) -> None:
        """A stage reaches for the contracts and for nothing else in the package.

        Call, Bind, the adapters, Extract and the limiter all stay unloaded -- and so
        does the *other* stage. Aggregate and Select are independent of each other by
        construction (§3.4, §3.6); one importing the other would make the pair a unit
        and is worth failing by name rather than absorbing.
        """
        rode_along = sorted(
            module
            for module in _modules_loaded_by_importing(stage)
            if module.startswith("threetears.search.")
            and module != stage
            and not module.startswith("threetears.search.contracts")
        )

        assert rode_along == [], f"importing {stage} pulled other working layers: {rode_along}"
