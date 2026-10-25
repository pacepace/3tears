"""runtime audit: the package roots import every driver, and import without an extra that is not installed.

Owner ruling, 2026-10-24: no lazy imports in serving code. A driver built on the event loop used to import its
module, and the backend library behind it, the first time a datasource of its type was queried, holding the
loop for the whole import. So importing :mod:`threetears.datasources.drivers.factory` imports every driver
module, and building a driver loads nothing (this replaces DS-09-09's contract that the roots load no driver).

What stays from DS-09-09 is that an extra is optional: without the redshift extra the roots still import, and
a Redshift config raises the ImportError naming the extra. The Snowflake and BigQuery drivers are stubs that
import no backend library, so neither library is ever loaded.

Each case runs in a clean subprocess, so it audits a fresh interpreter rather than this one, whose
``sys.modules`` every other test has filled.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)


def test_the_package_roots_import_every_driver_so_building_one_loads_nothing() -> None:
    result = _run(
        """
import sys
import threetears.datasources  # noqa: F401
import threetears.datasources.drivers.factory  # noqa: F401
drivers = {
    'threetears.datasources.drivers.asyncpg_driver',
    'threetears.datasources.drivers.redshift_driver',
    'threetears.datasources.drivers.snowflake_driver',
    'threetears.datasources.drivers.bigquery_driver',
}
missing = drivers - set(sys.modules)
banned = {'snowflake.connector', 'google.cloud.bigquery'} & set(sys.modules)
if missing or banned:
    print(f'MISSING: {sorted(missing)} BANNED: {sorted(banned)}', file=sys.stderr)
    sys.exit(1)
print('imported')
"""
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "imported" in result.stdout


def test_without_the_redshift_extra_the_roots_import_and_a_redshift_config_names_it() -> None:
    result = _run(
        """
import sys
sys.modules['redshift_connector'] = None  # the redshift extra is not installed
import threetears.datasources  # noqa: F401
from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.drivers.factory import create_driver
from threetears.datasources.entities import DataSourceType
config = RedshiftConnectionConfig(datasource_type=DataSourceType.REDSHIFT, host='cluster.example.com', database='a')
try:
    create_driver(config)
except ImportError as exc:
    print('refused:', exc)
    sys.exit(0)
print('a Redshift driver was built without its extra', file=sys.stderr)
sys.exit(1)
"""
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "3tears-datasources[redshift]" in result.stdout


def test_one_module_imports_opentelemetry_for_the_drivers() -> None:
    """``drivers/base.py`` owns the probe; a driver that imported it too could reach a name its host lacks."""
    drivers = Path(__file__).resolve().parents[2] / "src" / "threetears" / "datasources" / "drivers"
    importers = sorted(
        path.name
        for path in drivers.glob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Import | ast.ImportFrom)
        and any((getattr(node, "module", None) or alias.name).startswith("opentelemetry") for alias in node.names)
    )
    assert importers == ["base.py"], importers
