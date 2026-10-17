"""
enforcement: no test container is handed a file through a bind mount.

``DockerContainer.with_volume_mapping(source, target)`` mounts ``source`` as the
Docker DAEMON sees it, not as the test process does. Where the daemon runs on
another filesystem -- Docker Desktop reached through ``/var/run/docker.sock`` from
a dev container, a remote ``DOCKER_HOST`` -- a config the test wrote to its temp
directory is not on the daemon's side. Docker then mounts an empty directory (or
nothing), and the container fails in a way that names neither: NATS exited with
``open /etc/nats/nats.conf: no such file or directory`` and the test timed out
120s later waiting for a ready line, across 17 live suites and the session NATS
fixture; SearXNG's entrypoint would quietly have written default settings with
JSON off.

Give the container its configuration another way: pass the text in an
environment variable and have the image's shell write it out before ``exec``-ing
the server (``packages/nats/tests/integration/nats_support.py``, and the
``nats_container`` / ``searxng_container`` fixtures in
``threetears.core.testing.fixtures``).

Scope: every python file under a package's ``tests`` tree, and every file under a
``testing`` package inside a package's ``src``. A call the rule genuinely cannot
apply to is exempted IN PLACE, by a marker on the line that names
``with_volume_mapping``::

    .with_volume_mapping(...)  # bind-mount-exempt: <why this mount works on a daemon elsewhere>

The rationale clears the same bar as every other exemption in the repo (at least
30 characters, not a blanket phrase).
"""

from __future__ import annotations

import ast
from pathlib import Path

from threetears.enforcement.common.exemptions import rationale_defect

_REPO_ROOT = Path(__file__).resolve().parents[2]
_METHOD = "with_volume_mapping"
_MARKER = "# bind-mount-exempt:"


def _in_scope(relative: Path) -> bool:
    """whether ``relative`` is a test-tree file or a ``src/**/testing/**`` file.

    :param relative: a repo-relative path under ``packages/``
    :ptype relative: Path
    :return: whether the rule applies to it
    :rtype: bool
    """
    parts = relative.parts
    if ".venv" in parts:
        return False
    if "tests" in parts:
        return True
    return "src" in parts and "testing" in parts[parts.index("src") + 1 :]


def _sources() -> list[Path]:
    """every in-scope python file, repo-relative.

    :return: the files, sorted
    :rtype: list[Path]
    """
    return sorted(
        relative
        for relative in (path.relative_to(_REPO_ROOT) for path in (_REPO_ROOT / "packages").rglob("*.py"))
        if _in_scope(relative)
    )


def _mount_lines(tree: ast.AST) -> list[int]:
    """the line of every ``with_volume_mapping`` call, as the method name's own line.

    :param tree: a parsed module
    :ptype tree: ast.AST
    :return: 1-based line numbers
    :rtype: list[int]
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == _METHOD:
            lines.append(func.end_lineno or node.lineno)
        elif isinstance(func, ast.Name) and func.id == _METHOD:
            lines.append(node.lineno)
    return lines


def test_no_test_container_is_given_a_bind_mount() -> None:
    sources = _sources()
    assert len(sources) > 100, "found too few test sources; the walk is not seeing the test trees"

    offenders: list[str] = []
    for path in sources:
        text = (_REPO_ROOT / path).read_text(encoding="utf-8")
        if _METHOD not in text:
            continue
        lines = text.splitlines()
        for lineno in _mount_lines(ast.parse(text, filename=str(path))):
            line = lines[lineno - 1]
            if _MARKER not in line:
                offenders.append(f"{path}:{lineno}")
                continue
            defect = rationale_defect(line.split(_MARKER, 1)[1].strip())
            if defect is not None:
                offenders.append(f"{path}:{lineno} ({_MARKER} rationale {defect})")

    assert offenders == [], (
        "these test containers are given a file by bind mount. The mount source is resolved on the "
        "Docker daemon's host, which need not be this filesystem (Docker Desktop behind a dev "
        "container's socket, a remote DOCKER_HOST), so a mount from a temp dir silently mounts "
        "nothing and the container starts without its config. Give the container its config another "
        "way -- an environment variable its shell writes out before exec-ing the server, as "
        "packages/nats/tests/integration/nats_support.py does -- or, where the mount genuinely works "
        f"on any daemon, mark the call's line `{_MARKER} <specific rationale>`: {offenders}"
    )


def test_the_walk_sees_a_mount_however_it_is_spelled() -> None:
    """guards the detector itself: a chained, a direct, and a wrapped call are all found."""
    source = (
        "c = DockerContainer('x').with_volume_mapping('/a', '/b')\n"
        "c.with_volume_mapping('/a', '/b')\n"
        "c = (\n"
        "    DockerContainer('x')\n"
        "    .with_exposed_ports(1)\n"
        "    .with_volume_mapping('/a', '/b')\n"
        ")\n"
    )
    assert sorted(_mount_lines(ast.parse(source))) == [1, 2, 6]


def test_scope_is_test_trees_and_testing_packages() -> None:
    assert _in_scope(Path("packages/nats/tests/integration/x_live.py"))
    assert _in_scope(Path("packages/core/src/threetears/core/testing/fixtures.py"))
    assert _in_scope(Path("packages/agent/tools/src/threetears/agent/tools/testing/x.py"))
    assert not _in_scope(Path("packages/core/src/threetears/core/data/store.py"))
    assert not _in_scope(Path("packages/core/.venv/lib/tests/x.py"))
