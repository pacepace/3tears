"""The import check a host runs over its own tree: every reach into ``threetears.evals`` goes through a public root.

Each refusal is driven by a file that makes it, beside a file of the public forms the same check admits,
so a check that refused everything (or nothing) cannot pass. Every finding names its rule, and each rule
is driven here: static imports, dynamic imports, attribute chains, private reach and patches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.evals.testing import BOUNDARY_RULES, BoundaryFinding, nonpublic_evals_imports

_PUBLIC = """\
from __future__ import annotations

import json
import threetears.evals
import threetears.evals.run
from threetears.evals import PUBLIC_ROOTS, run
from threetears.evals.schema import AsyncDeliveryStatus, EvalRun, SCALES
from threetears.evals.kernel import TURN_BUDGET_ENDED_KEY, blended_cost_roles
from threetears.evals.schema import eval_trace_doc_id
from threetears.evals.schema import Firings
from threetears.evals.kernel import MATCH_MEASURE, CONFUSION_CELL_MEASURE
from threetears.evals.kernel.host import UNSEATED_LEVEL
from threetears.evals.run import EvalRunCostCap, resolve_ceiling_origin, resolve_effective_ceiling
from threetears.evals.run import stamp_witnessed_judge
from threetears.evals.kernel import host
from threetears.evals.kernel.host import HostProfile
from threetears.evals.testing import nonpublic_evals_imports
from pydantic import BaseModel
"""

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _only(findings: tuple[BoundaryFinding, ...]) -> BoundaryFinding:
    (finding,) = findings
    return finding


def test_the_public_forms_pass(tmp_path: Path) -> None:
    """Including every name an adopter was found reaching below a root for, each now exported from one."""
    _write(tmp_path, "public.py", _PUBLIC)

    assert nonpublic_evals_imports(tmp_path) == ()


def test_the_package_s_own_example_passes() -> None:
    assert nonpublic_evals_imports(_EXAMPLES) == ()


def test_a_module_below_a_root_is_refused_and_the_public_home_named(tmp_path: Path) -> None:
    path = _write(tmp_path, "deep.py", "from threetears.evals.schema.models import AsyncDeliveryStatus\n")

    finding = _only(nonpublic_evals_imports(tmp_path))

    assert (finding.path, finding.line, finding.rule, finding.target) == (
        path,
        1,
        "static_import",
        "threetears.evals.schema.models.AsyncDeliveryStatus",
    )
    assert "below a public root" in finding.reason
    assert "import it from threetears.evals.schema" in finding.reason


def test_a_name_no_root_exports_is_refused_saying_so(tmp_path: Path) -> None:
    _write(tmp_path, "deep.py", "from threetears.evals.schema.models import SCALE_LEVELS\n")

    assert "no public root exports it" in _only(nonpublic_evals_imports(tmp_path)).reason


def test_a_name_a_root_does_not_declare_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, "undeclared.py", "from threetears.evals.schema import annotations\n")

    finding = _only(nonpublic_evals_imports(tmp_path))

    assert finding.reason.startswith("annotations is not in threetears.evals.schema.__all__")


def test_a_name_declared_by_another_root_is_refused_naming_that_root(tmp_path: Path) -> None:
    _write(tmp_path, "elsewhere.py", "from threetears.evals.run import EvalRun\n")

    finding = _only(nonpublic_evals_imports(tmp_path))

    assert "EvalRun is not in threetears.evals.run.__all__; it is public from threetears.evals.schema" in (
        finding.reason
    )


def test_a_star_import_from_a_root_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, "star.py", "from threetears.evals.schema import *\n")

    assert "star import" in _only(nonpublic_evals_imports(tmp_path)).reason


def test_importing_a_module_below_a_root_is_refused_naming_the_root_above(tmp_path: Path) -> None:
    _write(tmp_path, "module.py", "import threetears.evals.schema.models\n")

    finding = _only(nonpublic_evals_imports(tmp_path))

    assert (finding.rule, finding.target) == ("static_import", "threetears.evals.schema.models")
    assert finding.reason == (
        "threetears.evals.schema.models is below a public root; import from threetears.evals.schema instead"
    )


def test_a_name_from_the_package_root_that_is_not_a_root_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, "pkg.py", "from threetears.evals import world_session_helpers\n")

    assert "is not in threetears.evals.__all__" in _only(nonpublic_evals_imports(tmp_path)).reason


def test_imports_inside_functions_and_type_checking_blocks_are_read(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "nested/late.py",
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from threetears.evals.schema.call_ledger import CallLedger\n"
        "def f() -> None:\n"
        "    from threetears.evals.kernel.spend import ExternalRateTable\n",
    )

    assert [(finding.line, finding.target) for finding in nonpublic_evals_imports(tmp_path)] == [
        (3, "threetears.evals.schema.call_ledger.CallLedger"),
        (5, "threetears.evals.kernel.spend.ExternalRateTable"),
    ]


def test_a_file_source_is_read_and_findings_render_as_one_line(tmp_path: Path) -> None:
    path = _write(tmp_path, "one.py", "from threetears.evals.kernel.spend import ExternalRateTable\n")

    finding = _only(nonpublic_evals_imports(path))

    assert str(finding).startswith(f"{path}:1: static_import: threetears.evals.kernel.spend.ExternalRateTable — ")


def test_a_source_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nothing under it could be checked"):
        nonpublic_evals_imports(tmp_path / "missing")


def _rules(tmp_path: Path, source: str) -> list[tuple[int, str, str]]:
    """The findings over one file holding ``source``, as ``(line, rule, target)``."""
    _write(tmp_path, "reach.py", source)
    return [(finding.line, finding.rule, finding.target) for finding in nonpublic_evals_imports(tmp_path)]


# --- rule: dynamic_import ------------------------------------------------------------------------------


def test_a_literal_dynamic_import_below_a_root_is_refused(tmp_path: Path) -> None:
    source = (
        "import importlib\n"
        "import importlib as il\n"
        "from importlib import import_module as im\n"
        "importlib.import_module('threetears.evals.schema.models')\n"
        "im('threetears.evals.transports')\n"
        "il.import_module('.models', package='threetears.evals.schema')\n"
        "__import__('threetears.evals.kernel.spend')\n"
        "__import__('threetears.evals.schema', fromlist=['models'])\n"
    )

    assert _rules(tmp_path, source) == [
        (4, "dynamic_import", "threetears.evals.schema.models"),
        (5, "dynamic_import", "threetears.evals.transports"),
        (6, "dynamic_import", "threetears.evals.schema.models"),
        (7, "dynamic_import", "threetears.evals.kernel.spend"),
        (8, "dynamic_import", "threetears.evals.schema.models"),
    ]


def test_a_dynamic_import_of_a_root_a_computed_name_or_another_package_passes(tmp_path: Path) -> None:
    source = (
        "import importlib\n"
        "name = 'threetears.evals.schema.models'\n"
        "importlib.import_module('threetears.evals.run')\n"
        "importlib.import_module('threetears.evals')\n"
        "importlib.import_module(name)\n"
        "importlib.import_module(f'threetears.evals.{name}')\n"
        "importlib.import_module('json')\n"
        "__import__('threetears.evals.schema', fromlist=['EvalRun'])\n"
    )

    assert _rules(tmp_path, source) == []


# --- rule: attribute_reach -----------------------------------------------------------------------------


def test_an_attribute_a_bound_root_does_not_declare_is_refused(tmp_path: Path) -> None:
    source = (
        "import threetears.evals\n"
        "import threetears.evals.kernel as k\n"
        "from threetears.evals import run\n"
        "run.metering\n"
        "k.spend.ExternalRateTable\n"
        "threetears.evals.schema.models\n"
        "getattr(run, 'metering')\n"
        "hasattr(k, 'spend')\n"
    )

    found = _rules(tmp_path, source)

    assert found == [
        (4, "attribute_reach", "threetears.evals.run.metering"),
        (5, "attribute_reach", "threetears.evals.kernel.spend"),
        (6, "attribute_reach", "threetears.evals.schema.models"),
        (7, "attribute_reach", "threetears.evals.run.metering"),
        (8, "attribute_reach", "threetears.evals.kernel.spend"),
    ]


def test_an_attribute_reach_names_where_the_name_is_public(tmp_path: Path) -> None:
    _write(tmp_path, "reach.py", "from threetears.evals import run\nrun.EvalRun\n")

    assert "EvalRun is not in threetears.evals.run.__all__; it is public from threetears.evals.schema" in (
        _only(nonpublic_evals_imports(tmp_path)).reason
    )


def test_declared_names_roots_dunders_and_other_packages_pass(tmp_path: Path) -> None:
    source = (
        "import os.path\n"
        "import threetears.evals\n"
        "import threetears.evals.kernel as k\n"
        "import threetears.observe\n"
        "from threetears.evals import run\n"
        "run.EvalRunCostCap.model_fields\n"
        "threetears.evals.run.EvalRunCostCap\n"
        "threetears.evals.PUBLIC_ROOTS\n"
        "k.host.HostProfile\n"
        "run.__name__\n"
        "getattr(run, '__all__')\n"
        "getattr(run, attribute_name)\n"
        "threetears.observe.get_logger\n"
        "os.path.join\n"
    )

    assert _rules(tmp_path, source) == []


# --- rule: private_reach -------------------------------------------------------------------------------


def test_a_private_name_on_a_bound_chain_is_refused(tmp_path: Path) -> None:
    source = (
        "import threetears.evals.kernel as k\n"
        "from threetears.evals import run\n"
        "from threetears.evals.schema import EvalRun\n"
        "EvalRun._private\n"
        "k._helper\n"
        "run.EvalRunCostCap._limit\n"
        "getattr(EvalRun, '_cache')\n"
    )

    assert _rules(tmp_path, source) == [
        (4, "private_reach", "threetears.evals.schema.EvalRun._private"),
        (5, "private_reach", "threetears.evals.kernel._helper"),
        (6, "private_reach", "threetears.evals.run.EvalRunCostCap._limit"),
        (7, "private_reach", "threetears.evals.schema.EvalRun._cache"),
    ]


def test_dunders_and_private_names_off_the_package_pass(tmp_path: Path) -> None:
    source = (
        "from threetears.evals.schema import EvalRun\n"
        "EvalRun.__name__\n"
        "EvalRun.__dict__\n"
        "getattr(EvalRun, '__doc__')\n"
        "class Host:\n"
        "    def f(self, other):\n"
        "        return self._x, other._y, getattr(other, '_z')\n"
    )

    assert _rules(tmp_path, source) == []


def test_bindings_under_type_checking_and_inside_functions_are_read(tmp_path: Path) -> None:
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    import threetears.evals.kernel as k\n"
        "def annotated(value: k._Private) -> None:\n"
        "    pass\n"
        "def late() -> None:\n"
        "    from threetears.evals import run\n"
        "    run.metering\n"
    )

    assert _rules(tmp_path, source) == [
        (4, "private_reach", "threetears.evals.kernel._Private"),
        (8, "attribute_reach", "threetears.evals.run.metering"),
    ]


# --- rule: patch ---------------------------------------------------------------------------------------


def test_patching_into_the_package_is_refused_in_every_form(tmp_path: Path) -> None:
    source = (
        "from unittest import mock\n"
        "from unittest.mock import patch\n"
        "import threetears.evals.kernel as k\n"
        "from threetears.evals import run\n"
        "from threetears.evals.schema import EvalRun\n"
        "def test(monkeypatch, mocker):\n"
        "    monkeypatch.setattr('threetears.evals.run.metering.open_cell', None)\n"
        "    monkeypatch.setattr(run, 'run_campaign', None)\n"
        "    monkeypatch.delattr(k, 'blended_cost_roles')\n"
        "    monkeypatch.setitem(EvalRun.model_fields, 'x', None)\n"
        "    mock.patch('threetears.evals.kernel.spend.ExternalRateTable')\n"
        "    patch.dict('threetears.evals.kernel.TABLE', {})\n"
        "    mocker.patch.object(run, 'run_campaign')\n"
        "    setattr(run, 'run_campaign', None)\n"
        "    run.run_campaign = None\n"
        "    del k.blended_cost_roles\n"
        "    k._cache = None\n"
        "@patch.object(EvalRun, 'model_validate')\n"
        "def test_decorated(validate):\n"
        "    pass\n"
    )

    assert _rules(tmp_path, source) == [
        (7, "patch", "threetears.evals.run.metering.open_cell"),
        (8, "patch", "threetears.evals.run.run_campaign"),
        (9, "patch", "threetears.evals.kernel.blended_cost_roles"),
        (10, "patch", "threetears.evals.schema.EvalRun.model_fields"),
        (11, "patch", "threetears.evals.kernel.spend.ExternalRateTable"),
        (12, "patch", "threetears.evals.kernel.TABLE"),
        (13, "patch", "threetears.evals.run.run_campaign"),
        (14, "patch", "threetears.evals.run.run_campaign"),
        (15, "patch", "threetears.evals.run.run_campaign"),
        (16, "patch", "threetears.evals.kernel.blended_cost_roles"),
        (17, "patch", "threetears.evals.kernel._cache"),
        (18, "patch", "threetears.evals.schema.EvalRun.model_validate"),
    ]


def test_patching_a_host_s_own_names_passes(tmp_path: Path) -> None:
    source = (
        "import os\n"
        "import requests\n"
        "from unittest.mock import patch\n"
        "from threetears.evals.schema import EvalRun\n"
        "def test(monkeypatch, host):\n"
        "    monkeypatch.setattr('myhost.reports.EvalRun', EvalRun)\n"
        "    monkeypatch.setattr(host, 'run', None)\n"
        "    patch('myhost.reports.render')\n"
        "    patch.dict(os.environ, {'A': '1'})\n"
        "    requests.patch('https://example.invalid/threetears.evals')\n"
        "    setattr(host, 'x', 1)\n"
        "    host.run = None\n"
    )

    assert _rules(tmp_path, source) == []


def test_every_finding_names_one_of_the_stable_rules(tmp_path: Path) -> None:
    source = (
        "import importlib\n"
        "from threetears.evals.schema.models import SCALE_LEVELS\n"
        "from threetears.evals import run\n"
        "importlib.import_module('threetears.evals.schema.models')\n"
        "run.metering\n"
        "run._hidden\n"
        "run.run_campaign = None\n"
    )

    rules = [rule for _line, rule, _target in _rules(tmp_path, source)]

    assert rules == list(BOUNDARY_RULES)
