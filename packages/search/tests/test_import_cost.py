"""What importing the contracts costs, stated as an allowlist (search-spec.md §2, §4.10 d).

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

Measured in a fresh interpreter, because whatever the pytest process already imported
would otherwise hide the cost entirely.
"""

from __future__ import annotations

import json
import subprocess
import sys

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

#: CPython writes its build configuration into a module whose name embeds the ABI
#: flags and platform (``_sysconfigdata__darwin_darwin``), so it is the interpreter
#: rather than a dependency, yet it is absent from ``sys.stdlib_module_names``. Only
#: the prefix is stable across platforms, which is why this is a prefix test.
_INTERPRETER_ARTIFACT_PREFIXES = ("_sysconfigdata",)

_PROBE = """
import json
import sys

before = set(sys.modules)
import threetears.search.contracts  # noqa: F401
print(json.dumps(sorted(set(sys.modules) - before)))
"""


def _modules_loaded_by_importing_the_contracts() -> list[str]:
    """Every module a fresh interpreter loads for the contracts import.

    :return: the module names, sorted
    :rtype: list[str]
    """
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"probe failed:\n{result.stderr}"
    loaded: list[str] = json.loads(result.stdout.strip())
    return loaded


def _is_permitted(module: str) -> bool:
    """Whether the floor pays for *module*.

    :param module: a dotted module name from ``sys.modules``
    :ptype module: str
    :return: true when the module is stdlib, pydantic's tree, or family contracts
    :rtype: bool
    """
    root = module.split(".")[0]
    if root in sys.stdlib_module_names or module.startswith(_INTERPRETER_ARTIFACT_PREFIXES):
        return True
    if root in _PERMITTED_THIRD_PARTY_ROOTS:
        return True
    if any(module == prefix or module.startswith(f"{prefix}.") for prefix in _PERMITTED_FAMILY_PREFIXES):
        return True
    return module in _PERMITTED_FAMILY_MODULES


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
