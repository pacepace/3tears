"""enforcement walker: every nats-py usage routes through threetears.nats.

flags any production source file that:

- imports from ``nats`` directly (``import nats``)
- imports from ``nats.aio`` directly (``from nats.aio import ...``)
- imports from ``nats.errors`` / ``nats.js`` directly (with the
  carve-out for the wrapper itself which legitimately wraps the
  library's exception types and JetStream KV api)

allowlist (always-permitted import sites):

- ``threetears.nats.client`` and ``threetears.nats.kv`` — the wrapper
  IS the canonical nats-py consumer
- the per-package exemption file ``_nats_exemptions.txt`` — entries
  require a ``# rationale: <specific reason>`` line on the immediately
  preceding line

mode is controlled by the ``NATS_ENFORCEMENT_MODE`` env var:

- ``strict`` (default after nats-migration-agents-task-01 closeout):
  violations fail the test. enabled once the migration sweep
  emptied the catalog across all four repos.
- ``report``: violations log; test still passes. retained for
  short-term cleanup windows when a new ad-hoc nats-py site
  surfaces during refactor work.

mirrors the partition walker / underscore walker discipline. exemption
rationales like "internal access needed" / "tests need this" are
rejected; specific rationales required.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
"""path to ``3tears/`` repo root (``.../3tears``).

resolved by walking up four parents from this test file:
enforcement -> tests -> nats -> packages -> repo root.
"""

ENFORCEMENT_MODE = os.environ.get("NATS_ENFORCEMENT_MODE", "strict")


# wrapper modules that LEGITIMATELY consume nats-py
_WRAPPER_MODULES: set[str] = {
    "threetears.nats.client",
    "threetears.nats.kv",
}


def _scan_file(path: Path) -> list[str]:
    """find direct nats-py imports in a single python source file.

    :param path: path to python source file
    :ptype path: Path
    :return: list of human-readable violation strings
    :rtype: list[str]
    """
    violations: list[str] = []
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return violations

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "nats" or alias.name.startswith("nats."):
                    violations.append(
                        f"{path}:{node.lineno}: `import {alias.name}` — "
                        f"use `from threetears.nats import NatsClient`"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "nats" or mod.startswith("nats."):
                violations.append(
                    f"{path}:{node.lineno}: `from {mod} import ...` — "
                    f"use `from threetears.nats import NatsClient, Subjects`"
                )
    return violations


def _module_name_for(path: Path, src_root: Path) -> str | None:
    """derive dotted module name for a source file under ``src_root``.

    :param path: python source file path
    :ptype path: Path
    :param src_root: source root the file lives under
    :ptype src_root: Path
    :return: dotted module name, or ``None`` if path not under src_root
    :rtype: str | None
    """
    try:
        rel = path.relative_to(src_root)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    result: str | None
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    result = ".".join(parts) if parts else None
    return result


def _collect_python_sources(roots: list[Path]) -> list[Path]:
    """collect every .py file under each src root.

    :param roots: list of source root directories
    :ptype roots: list[Path]
    :return: list of python source file paths
    :rtype: list[Path]
    """
    sources: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        sources.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sources


def _load_exemptions(path: Path) -> set[str]:
    """load exemption file (one path per line, ``# rationale: ...`` discipline).

    :param path: path to exemption file
    :ptype path: Path
    :return: set of repo-relative paths to exempt
    :rtype: set[str]
    """
    if not path.exists():
        return set()
    exemptions: set[str] = set()
    last_was_rationale = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            last_was_rationale = False
            continue
        if line.startswith("#"):
            if line.startswith("# rationale:") or line.startswith("#rationale:"):
                if line.startswith("# rationale:"):
                    rationale = line.removeprefix("# rationale:").strip()
                else:
                    rationale = line.removeprefix("#rationale:").strip()
                # blanket rationales are rejected
                blanket = {
                    "internal access needed",
                    "tests need access",
                    "tests need this",
                    "same-file colocation",
                }
                if rationale.lower() in blanket:
                    raise ValueError(
                        f"exemption rationale rejected (blanket phrase): {rationale!r} in {path}"
                    )
                last_was_rationale = True
            continue
        if not last_was_rationale:
            raise ValueError(
                f"exemption {line!r} in {path} missing preceding `# rationale: <reason>` line"
            )
        exemptions.add(line)
        last_was_rationale = False
    return exemptions


@pytest.fixture(scope="module")
def src_roots() -> list[Path]:
    """source roots scanned by the walker.

    :return: list of source root directories
    :rtype: list[Path]
    """
    pkgs = REPO_ROOT / "packages"
    return [
        pkg / "src"
        for pkg in pkgs.iterdir()
        if pkg.is_dir() and (pkg / "src").exists()
    ]


@pytest.fixture(scope="module")
def exemptions() -> set[str]:
    """load per-repo exemptions.

    :return: set of repo-relative paths permitted to import nats-py directly
    :rtype: set[str]
    """
    return _load_exemptions(Path(__file__).parent / "_nats_exemptions.txt")


def test_no_direct_nats_imports_in_3tears_packages(
    src_roots: list[Path],
    exemptions: set[str],
) -> None:
    """every direct nats-py import in 3tears packages routes through the wrapper or carries a rationale.

    :param src_roots: list of source root directories to scan
    :ptype src_roots: list[Path]
    :param exemptions: set of repo-relative paths permitted to import nats-py
    :ptype exemptions: set[str]
    :return: nothing
    :rtype: None
    :raises AssertionError: in strict mode, when a violation is found and not exempted
    """
    sources = _collect_python_sources(src_roots)
    violations: list[str] = []

    for path in sources:
        rel = path.relative_to(REPO_ROOT).as_posix()
        # wrapper modules: skip
        for src_root in src_roots:
            module_name = _module_name_for(path, src_root)
            if module_name in _WRAPPER_MODULES:
                break
        else:
            module_name = None
        if module_name in _WRAPPER_MODULES:
            continue
        # exempted paths: skip
        if rel in exemptions:
            continue
        violations.extend(_scan_file(path))

    if violations and ENFORCEMENT_MODE == "strict":
        formatted = "\n".join(violations)
        raise AssertionError(
            f"{len(violations)} direct nats-py import(s) found outside the wrapper. "
            f"migrate to `from threetears.nats import NatsClient, Subjects` or add an "
            f"exemption with a specific rationale to "
            f"`packages/nats/tests/enforcement/_nats_exemptions.txt`.\n\n"
            f"{formatted}"
        )
    elif violations:
        # report mode — log and pass
        print(  # noqa: T201 — diag
            f"NATS_ENFORCEMENT_MODE=report: {len(violations)} violation(s) "
            f"would fail in strict mode.\n" + "\n".join(violations)
        )


def test_walker_self_test_detects_violation(tmp_path: Path) -> None:
    """walker self-test: deliberate violation is detected.

    :param tmp_path: pytest-managed temporary directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    bad = tmp_path / "bad.py"
    bad.write_text("import nats\n")
    violations = _scan_file(bad)
    assert len(violations) == 1
    assert "import nats" in violations[0]


def test_walker_self_test_detects_aio_import(tmp_path: Path) -> None:
    """walker self-test: from-import variants are detected.

    :param tmp_path: pytest-managed temporary directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    bad = tmp_path / "bad.py"
    bad.write_text("from nats.aio.client import Client\n")
    violations = _scan_file(bad)
    assert len(violations) == 1
    assert "nats.aio.client" in violations[0]


def test_walker_self_test_clean_file_passes(tmp_path: Path) -> None:
    """walker self-test: clean file produces no violations.

    :param tmp_path: pytest-managed temporary directory
    :ptype tmp_path: Path
    :return: nothing
    :rtype: None
    """
    clean = tmp_path / "clean.py"
    clean.write_text("from threetears.nats import NatsClient, Subjects\n")
    violations = _scan_file(clean)
    assert violations == []
