"""The report writer's test path stands apart from the planner's and the memo's fixtures, so deleting them breaks none of it.

Each module on the writer's path is imported first in a fresh interpreter: rung 5's test, its case export, the writer
host, the case bank's seed, the recorded writer's support, the writer's acceptance and calibration scripts. None of
them may load the planner's and the memo's fixtures (``ops_support``, ``toyhost_memo``, ``toyhost_plan``), which go when
the planner and the memo generator are deleted.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Final

import pytest

from packages.evals.tests.test_package_matrix import REPO_ROOT

#: The writer's path, by module under ``packages.evals.tests``.
WRITER_PATH: Final = (
    "test_rung_five_writer",
    "fixtures.report_plans.export_writer_cases",
    "writer_host_support",
    "writer_case_support",
    "writer_eval_support",
    "report_writer_support",
    "fixtures.report_plans.analyst_cli",
    "fixtures.report_plans.writer_smoke",
    "fixtures.report_plans.writer_calibration",
    "fixtures.report_plans.writer_acceptance",
)

#: The planner's and the memo's fixtures, which the writer's path never loads.
RETIRING: Final = ("ops_support", "toyhost_memo", "toyhost_plan")

_PROBE = (
    "import importlib, sys\n"
    "importlib.import_module({module!r})\n"
    "print(' '.join(name for name in {retiring!r} if 'packages.evals.tests.' + name in sys.modules))\n"
)


@pytest.mark.parametrize("module", WRITER_PATH)
def test_a_module_on_the_writers_path_loads_no_planner_or_memo_fixture(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=f"packages.evals.tests.{module}", retiring=RETIRING)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, f"importing {module} first, in a fresh interpreter, failed:\n{proc.stderr}"
    assert proc.stdout.strip() == "", f"{module} loads {proc.stdout.strip()}"
