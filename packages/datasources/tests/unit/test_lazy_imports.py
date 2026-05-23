"""runtime audit: importing package roots MUST NOT load extras backend libs.

DS-09-09 (as scoped for shard 09): importing
:mod:`threetears.datasources`, :mod:`threetears.datasources.drivers`,
or :mod:`threetears.datasources.drivers.factory` MUST leave
``redshift_connector``, ``snowflake.connector``, and
``google.cloud.bigquery`` out of ``sys.modules``. those are extras-
keyed backend libs whose import cost is the contract's main concern.

``asyncpg`` is intentionally NOT in the audited banned set, even though
the shard doc lists it. the upstream :mod:`threetears.core` package
(consumed transitively via :class:`BaseEntity` in
:mod:`threetears.datasources.entities`) imports ``asyncpg`` at module
top -- that's not something this shard can fix, and it matches the
package's documented "asyncpg is a HARD dependency" stance from
``pyproject.toml`` (the postgres / yugabyte / agent_internal
coverage is baseline for every 3tears consumer). when
:mod:`threetears.core` decouples from asyncpg we can re-tighten this
audit; until then, listing asyncpg here would always fail.

verified in a clean subprocess so we audit a fresh interpreter rather
than relying on the current pytest process whose ``sys.modules`` is
polluted by every other test.
"""

from __future__ import annotations

import subprocess
import sys


def test_package_roots_dont_load_extras_backend_libs() -> None:
    """fresh-interpreter audit of the three package roots."""
    script = """
import sys
import threetears.datasources  # noqa: F401
import threetears.datasources.drivers  # noqa: F401
import threetears.datasources.drivers.factory  # noqa: F401
# asyncpg is intentionally NOT in this set; see the module docstring.
banned = {'redshift_connector', 'snowflake.connector', 'google.cloud.bigquery'}
loaded = banned & set(sys.modules)
if loaded:
    print(f'BANNED LOADED: {sorted(loaded)}', file=sys.stderr)
    sys.exit(1)
print('clean')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"lazy-import audit failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "clean" in result.stdout
