"""enforcement: no hardcoded MANAGED-model id literals outside the registry.

the single source of truth for current model ids is
:mod:`threetears.models.defaults` (``DEFAULT_CHAT_MODEL`` /
``DEFAULT_FAST_MODEL`` / ``DEFAULT_LARGE_MODEL`` / ``DEFAULT_EMBEDDING_MODEL``).
every source default and every test that needs "the current model" imports a
constant from there instead of pinning a raw id inline, so a model rev is one
edit in that module and nothing else changes.

this walker AST-scans every ``.py`` under each ``packages/*/src`` and
``packages/*/tests`` tree for string literals shaped like a MANAGED model id
and fails listing any that is not allowed. it deliberately flags ONLY the
providers the platform manages and pins -- Anthropic (``claude-*``) and
VoyageAI (``voyage-*``):

- OpenAI ids (``gpt-*`` / ``text-embedding-*``) are NOT flagged: the platform
  pins no OpenAI constant (no key to verify the lineup live), so an OpenAI id
  in a provider-handling test is not our currency problem.
- access-pattern GLOBS (any literal containing ``*``, e.g. ``claude-*``) are
  NOT flagged: a glob is an access selector, not a model id.

allowed (never a violation):

1. ``threetears/models/defaults.py`` -- the constants themselves.
2. anything under ``threetears/models/providers/`` -- the capability registry.
3. ``_generated_models.py`` -- codegen output.
4. a ``relative/path::reason`` line in the sibling
   ``_model_literal_exemptions.txt`` -- deliberate literals only (a retired-id
   regression pin, the namespace-sanitizer demo, a static catalog mirroring the
   registry), each with a specific reason.

ONE test, no per-file parametrization: an exempt/allowed file is simply skipped
in the walk, never a ``pytest.skip`` -- so the suite reports zero skips and the
exemption count is only ever visible by reading the (small) exemption file.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: per-package source-tree subdirs scanned (every ``.py`` under each,
#: recursively). the 3tears repo is a monorepo of ``packages/<name>/{src,tests}``
#: trees.
_PACKAGE_SUBDIRS = ("src", "tests")

#: managed-provider model-id shapes. a string literal whose value STARTS with
#: one of these -- and does NOT contain ``*`` (which would make it an access
#: glob, not an id) -- is a candidate violation.
_MODEL_ID_PATTERNS = (
    re.compile(r"^claude-"),
    re.compile(r"^voyage-"),
)

#: path fragments that are always allowed (the registry + codegen).
_ALWAYS_ALLOWED_FRAGMENTS = (
    "threetears/models/defaults.py",
    "threetears/models/providers/",
    "_generated_models.py",
)

_EXEMPTIONS_PATH = _PROJECT_ROOT / "tests" / "enforcement" / "_model_literal_exemptions.txt"


def _is_model_literal(value: str) -> bool:
    """report whether a string value is a MANAGED model-id literal.

    a glob selector (contains ``*``) is an access pattern, not an id, and is
    never a violation.

    :param value: string-constant value pulled from an AST node
    :ptype value: str
    :return: ``True`` when value is a bare claude-/voyage- model id
    :rtype: bool
    """
    if "*" in value:
        return False
    return any(pattern.match(value) for pattern in _MODEL_ID_PATTERNS)


def _load_exemptions() -> set[str]:
    """parse ``_model_literal_exemptions.txt`` into a set of relative paths.

    each non-comment, non-blank line is ``relative/path::reason``; the reason
    is mandatory (a bare path with no ``::reason`` is rejected so every
    exemption carries a justification).

    :return: set of repo-relative posix paths that are exempt
    :rtype: set[str]
    :raises ValueError: if an exemption line lacks a ``::reason`` justification
    """
    exemptions: set[str] = set()
    if _EXEMPTIONS_PATH.exists():
        for raw in _EXEMPTIONS_PATH.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "::" not in line:
                raise ValueError(
                    f"exemption line missing '::reason': {line!r} in {_EXEMPTIONS_PATH.name}",
                )
            path_part, reason_part = line.split("::", 1)
            if not reason_part.strip():
                raise ValueError(
                    f"exemption line has empty reason: {line!r} in {_EXEMPTIONS_PATH.name}",
                )
            exemptions.add(path_part.strip())
    return exemptions


def _is_always_allowed(rel_posix: str) -> bool:
    """report whether path is in the never-a-violation registry set.

    :param rel_posix: repo-relative posix path of the file
    :ptype rel_posix: str
    :return: ``True`` when path matches a registry / codegen fragment
    :rtype: bool
    """
    return any(fragment in rel_posix for fragment in _ALWAYS_ALLOWED_FRAGMENTS)


def _collect_files() -> list[Path]:
    """collect every ``.py`` under each ``packages/*/{src,tests}`` tree.

    :return: sorted list of python source paths to walk
    :rtype: list[Path]
    """
    files: list[Path] = []
    packages_root = _PROJECT_ROOT / "packages"
    if packages_root.exists():
        for package_dir in sorted(packages_root.iterdir()):
            if not package_dir.is_dir():
                continue
            for subdir in _PACKAGE_SUBDIRS:
                tree_root = package_dir / subdir
                if tree_root.exists():
                    files.extend(p for p in tree_root.rglob("*.py") if "__pycache__" not in str(p))
    return sorted(files)


def _violations_in_file(src_file: Path) -> list[str]:
    """AST-walk one file and return managed-model-literal violation descriptions.

    :param src_file: python source file to scan
    :ptype src_file: Path
    :return: list of ``line N: <value>`` violation strings (empty if none)
    :rtype: list[str]
    """
    tree = ast.parse(src_file.read_text(encoding="utf-8"), filename=str(src_file))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _is_model_literal(node.value):
                violations.append(f"  line {node.lineno}: {node.value!r}")
    return violations


def test_no_hardcoded_model_literals() -> None:
    """no raw managed-model-id literals outside the registry + exemptions.

    single pass over every scanned file; allowed/exempt files are skipped in
    the walk (no ``pytest.skip``), so the suite never reports skips for this
    guard. fails with every offending file + line when any bare ``claude-`` /
    ``voyage-`` id is found outside the registry.

    :raises AssertionError: when a managed-model-id literal is found
    """
    exemptions = _load_exemptions()
    offenders: dict[str, list[str]] = {}
    for src_file in _collect_files():
        rel_posix = src_file.relative_to(_PROJECT_ROOT).as_posix()
        if _is_always_allowed(rel_posix) or rel_posix in exemptions:
            continue
        found = _violations_in_file(src_file)
        if found:
            offenders[rel_posix] = found
    assert not offenders, (
        "hardcoded managed-model-id literal(s) found; import the canonical "
        "constant from threetears.models (DEFAULT_CHAT_MODEL / DEFAULT_FAST_MODEL "
        "/ DEFAULT_LARGE_MODEL / DEFAULT_EMBEDDING_MODEL) instead, or add a "
        "justified line to tests/enforcement/_model_literal_exemptions.txt "
        "(relative/path::reason):\n"
        + "\n".join(f"{path}:\n" + "\n".join(lines) for path, lines in sorted(offenders.items()))
    )
