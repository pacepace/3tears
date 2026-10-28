"""a conftest's fixtures reach every test under it, whatever order the run names their files in.

pytest 9.1 scoped a conftest's fixtures to the Directory node collected for its directory. A file
argument that sits directly in a parent directory re-collects that parent, which builds fresh child
nodes the fixtures were never attached to. So a run whose arguments go ``unit/tools/test_a.py``,
``unit/test_b.py``, ``unit/tools/test_c.py`` errored every ``test_c`` test that asked for a
``unit/tools/conftest.py`` fixture with "fixture ... not found", and ran them without that conftest's
autouse fixtures. testament's batches name their files in exactly such an interleaved order, so
agent-workspace's tool tests, channels' presence tests and core's git-backend tests failed on the
pool whenever a run was wider than their own directory (#999; pytest-dev/pytest#14635, fixed in
9.2.0). The root ``pyproject.toml`` keeps the workspace off the 9.1 line; this holds it there.
"""

from __future__ import annotations

import pytest

_CONFTEST = """
import pytest


@pytest.fixture
def shared() -> str:
    return "from the tools conftest"


@pytest.fixture(autouse=True)
def installed() -> None:
    pass
"""

_TOOL_TEST = """
import pytest


def test_it_gets_the_fixture(shared: str) -> None:
    assert shared == "from the tools conftest"


def test_it_gets_the_autouse_fixture(request: pytest.FixtureRequest) -> None:
    assert "installed" in request.fixturenames
"""


def test_a_conftest_fixture_survives_its_parent_directory_being_collected_again(
    pytester: pytest.Pytester,
) -> None:
    for package in ("unit", "unit/tools"):
        pytester.mkpydir(package)
    (pytester.path / "unit" / "tools" / "conftest.py").write_text(_CONFTEST)
    (pytester.path / "unit" / "tools" / "test_first.py").write_text(_TOOL_TEST)
    (pytester.path / "unit" / "test_beside.py").write_text("def test_beside() -> None:\n    pass\n")
    (pytester.path / "unit" / "tools" / "test_second.py").write_text(_TOOL_TEST)

    result = pytester.runpytest_inprocess(
        "-p",
        "no:cacheprovider",
        "-p",
        "no:asyncio",
        "unit/tools/test_first.py",
        "unit/test_beside.py",
        "unit/tools/test_second.py",
    )

    result.assert_outcomes(passed=5)
