"""The import check a host runs over its own tree: every ``threetears.evals`` import comes from a public root.

Each refusal is driven by a file that makes it, beside a file of the public forms the same check admits,
so a check that refused everything (or nothing) cannot pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.evals.testing import NonPublicImport, nonpublic_evals_imports

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


def _only(findings: tuple[NonPublicImport, ...]) -> NonPublicImport:
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

    assert (finding.path, finding.line, finding.module, finding.name) == (
        path,
        1,
        "threetears.evals.schema.models",
        "AsyncDeliveryStatus",
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

    assert finding.name is None
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

    assert [(finding.line, finding.name) for finding in nonpublic_evals_imports(tmp_path)] == [
        (3, "CallLedger"),
        (5, "ExternalRateTable"),
    ]


def test_a_file_source_is_read_and_findings_render_as_one_line(tmp_path: Path) -> None:
    path = _write(tmp_path, "one.py", "from threetears.evals.kernel.spend import ExternalRateTable\n")

    finding = _only(nonpublic_evals_imports(path))

    assert str(finding).startswith(f"{path}:1: from threetears.evals.kernel.spend import ExternalRateTable — ")


def test_a_source_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nothing under it could be checked"):
        nonpublic_evals_imports(tmp_path / "missing")
