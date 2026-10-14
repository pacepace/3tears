"""Structural gate: the provider adapter and the engine meet only through public names, in one direction.

``threetears.evals.providers`` is the ``[models]`` extra: the one place the package names a provider. Two
properties keep it an adapter rather than a dependency:

* **The engine never imports it.** No module of the package outside ``providers/`` names it, by an import
  statement or by a string a dynamic import could use, and importing every other public root in a fresh
  interpreter loads neither the adapter nor 3tears-models. A host that brings its own client installs the
  engine without models and never pays for it. ``tests/test_package_matrix.py`` holds the same at package level
  (the adapter's column is empty); this holds it name by name, strings included.
* **The adapter imports only public names.** Every reach into ``threetears.evals`` from the adapter goes through
  a public root's ``__all__`` (:func:`nonpublic_engine_reaches`), so the adapter is held to exactly what a host
  is. Every reach into
  ``threetears.models`` names no underscored module, name or attribute: models' leading underscores are its
  stability contract as this package's are, and an adapter built on them would break on a models release that
  changed nothing public.

Each rule has a planted fixture proving it refuses what it should, so a walk that read nothing cannot pass.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import threetears.evals
from packages.evals.tests.import_resolution import absolute_module

#: The package's ``src``: the directory holding the ``threetears`` namespace.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"

#: The package's source tree.
EVAL_ROOT = SOURCE_ROOT / "threetears" / "evals"

#: The adapter's tree.
PROVIDERS_ROOT = EVAL_ROOT / "providers"

#: The adapter's root, as a module name.
PROVIDERS = "threetears.evals.providers"

#: The engine's package.
EVALS = "threetears.evals"

#: The provider layer the adapter is built over.
MODELS = "threetears.models"

#: The repository root, which the probe runs from.
REPO_ROOT = Path(__file__).resolve().parents[3]


def _is_private(segment: str) -> bool:
    return segment.startswith("_") and not (segment.startswith("__") and segment.endswith("__"))


def _names_the_adapter(text: str) -> bool:
    return text == PROVIDERS or text.startswith(f"{PROVIDERS}.")


#: The package-level declarations that name every public root by string: ``PUBLIC_ROOTS``, and ``ROOT_EXTRAS``, which
#: names the roots that need an extra. Naming the adapter's root there is declaring it, not reaching it.
_ROOT_DECLARATIONS = frozenset({"PUBLIC_ROOTS", "ROOT_EXTRAS"})


def _public_roots_declaration(tree: ast.Module) -> set[int]:
    """The ids of the string constants inside the package's root declarations (:data:`_ROOT_DECLARATIONS`)."""
    declared: set[int] = set()
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(target, ast.Name) and target.id in _ROOT_DECLARATIONS for target in targets):
            declared |= {id(constant) for constant in ast.walk(node) if isinstance(constant, ast.Constant)}
    return declared


def engine_reaches(source_root: Path) -> Iterator[str]:
    """Every place a module outside ``providers/`` names the adapter: an import, or a string naming its module.

    Args:
        source_root: The directory holding the ``threetears`` namespace; ``threetears/evals`` below it is walked.

    Yields:
        ``path:line: what`` for each reach.
    """
    root = source_root / "threetears" / "evals"
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.parts[0] == "providers":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        declaration = _public_roots_declaration(tree) if relative == Path("__init__.py") else set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                yield from (
                    f"{relative}:{node.lineno}: import {alias.name}"
                    for alias in node.names
                    if _names_the_adapter(alias.name)
                )
            elif isinstance(node, ast.ImportFrom):
                module = absolute_module(path, node, root=source_root)
                if _names_the_adapter(module) or any(
                    _names_the_adapter(f"{module}.{alias.name}") for alias in node.names
                ):
                    yield f"{relative}:{node.lineno}: from {module} import …"
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and _names_the_adapter(node.value)
                and id(node) not in declaration
            ):
                yield f"{relative}:{node.lineno}: the string {node.value!r}"


def _public_names(root: str) -> frozenset[str]:
    """The names public root ``root`` declares in its ``__all__``."""
    return frozenset(importlib.import_module(root).__all__)


def nonpublic_engine_reaches(path: Path) -> Iterator[str]:
    """Every name ``path`` imports from ``threetears.evals`` below a public root's ``__all__``, as ``module.name``.

    A static walk: ``from M import n`` is a reach unless ``M`` is a public root and ``n`` is in its ``__all__``, and
    ``import M`` is one unless ``M`` is a public root or the package itself.

    Args:
        path: One source file, or a directory whose ``.py`` files are walked.

    Yields:
        The dotted name of each reach.
    """
    files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
    for file in files:
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                yield from (
                    alias.name
                    for alias in node.names
                    if alias.name.startswith(f"{EVALS}.") and alias.name not in threetears.evals.PUBLIC_ROOTS
                )
            elif isinstance(node, ast.ImportFrom):
                module = absolute_module(file, node, root=SOURCE_ROOT) if node.level else (node.module or "")
                if module != EVALS and not module.startswith(f"{EVALS}."):
                    continue
                public = _public_names(module) if module in threetears.evals.PUBLIC_ROOTS else frozenset()
                yield from (f"{module}.{alias.name}" for alias in node.names if alias.name not in public)


def private_models_reaches(path: Path) -> Iterator[str]:
    """Every reach into ``threetears.models`` in ``path`` that names an underscored module, name or attribute.

    Args:
        path: An adapter source file.

    Yields:
        ``line: what`` for each reach.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == MODELS or alias.name.startswith(f"{MODELS}."):
                    if any(_is_private(part) for part in alias.name.split(".")):
                        yield f"{node.lineno}: import {alias.name}"
                    bound.add(alias.asname or alias.name.split(".")[0])
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == MODELS or node.module.startswith(f"{MODELS}."))
        ):
            if any(_is_private(part) for part in node.module.split(".")):
                yield f"{node.lineno}: from {node.module} import …"
            for alias in node.names:
                if _is_private(alias.name):
                    yield f"{node.lineno}: from {node.module} import {alias.name}"
                bound.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _is_private(node.attr):
            base: ast.expr = node.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in bound:
                yield f"{node.lineno}: .{node.attr} on {base.id}"


# --- the gates over today's tree -------------------------------------------------------------------------


def test_the_engine_never_names_the_adapter() -> None:
    """No module outside ``providers/`` imports the adapter or names its module in a string."""
    reaches = list(engine_reaches(SOURCE_ROOT))
    assert not reaches, (
        "The engine names the provider adapter, which must stay an extra a host can leave out. Take what is "
        "needed as a value the host passes in, or move it below the engine:\n  " + "\n  ".join(reaches)
    )


#: The one reach below a public root the adapter makes, and why: the package's logger. Every module of the package
#: takes its stdlib logger from :mod:`threetears.evals.schema.logs` (the repository's conventions gate refuses
#: ``logging.getLogger`` everywhere else, and ``tests/test_package_matrix.py`` lets every package import it), and the
#: adapter logs what the engine does not store: a reply read as ``truncated`` with the provider's raw stop reason,
#: and every pricing failure. A host logs through loggers of its own and never needs it, which is why it is not a
#: public name; the adapter is the package's own code, held to the package's logging convention.
PACKAGE_LOGGING = "threetears.evals.schema.logs.get_logger"


def test_the_adapter_reaches_the_engine_only_through_public_roots() -> None:
    """Every reach into ``threetears.evals`` from the adapter is a public root's declared name, as for any host.

    The one exception is :data:`PACKAGE_LOGGING`, the package's logger, named exactly.
    """
    findings = [
        reach
        for reach in nonpublic_engine_reaches(PROVIDERS_ROOT)
        if not _names_the_adapter(reach) and reach != PACKAGE_LOGGING
    ]
    assert not findings, "The adapter reaches below a public root of the engine:\n  " + "\n  ".join(findings)


def test_the_logging_exception_is_the_one_reach_it_names() -> None:
    """Non-vacuity: the exception is in use, as an import of exactly that name."""
    assert PACKAGE_LOGGING in set(nonpublic_engine_reaches(PROVIDERS_ROOT))


def test_the_adapter_reaches_models_only_through_public_names() -> None:
    """No module, name or attribute of ``threetears.models`` the adapter names starts with an underscore."""
    reaches = [
        f"{path.relative_to(PROVIDERS_ROOT)}:{reach}"
        for path in sorted(PROVIDERS_ROOT.rglob("*.py"))
        for reach in private_models_reaches(path)
    ]
    assert not reaches, (
        "The adapter binds to a private name of 3tears-models, which may move in any release. Use a public name, "
        "or ask models to promote it:\n  " + "\n  ".join(reaches)
    )


def test_the_walks_read_the_adapter() -> None:
    """Non-vacuity: the adapter exists, names models, and is a public root."""
    files = sorted(PROVIDERS_ROOT.rglob("*.py"))
    assert files, "the walk found no adapter module"
    assert any(f"from {MODELS}" in path.read_text(encoding="utf-8") for path in files)
    assert PROVIDERS in threetears.evals.PUBLIC_ROOTS


#: Imports every public root but the adapter in a fresh interpreter, then reports whether the adapter or
#: 3tears-models was loaded along the way.
_OTHER_ROOTS_PROBE = (
    "import importlib, sys, threetears.evals as e\n"
    "for root in e.PUBLIC_ROOTS:\n"
    f"    if root != {PROVIDERS!r}:\n"
    "        importlib.import_module(root)\n"
    f"loaded = sorted(n for n in sys.modules if n == {PROVIDERS!r} or n.startswith({PROVIDERS + '.'!r})\n"
    f"                or n == {MODELS!r} or n.startswith({MODELS + '.'!r}))\n"
    "assert not loaded, loaded\n"
)


def test_importing_every_other_root_loads_neither_the_adapter_nor_models() -> None:
    """The transitive half, which no per-file walk can report: the engine's roots load no provider layer."""
    proc = subprocess.run(
        [sys.executable, "-c", _OTHER_ROOTS_PROBE],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, f"importing the engine's roots loaded the provider layer:\n{proc.stderr}"


# --- each rule refuses what it should ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("planted", "source", "found"),
    [
        ("run/planted.py", "from threetears.evals.providers import models_client\n", True),
        ("run/planted.py", "import threetears.evals.providers.adapter\n", True),
        ("run/planted.py", "from threetears.evals import providers\n", True),
        ("run/planted.py", "from .. import providers\n", True),
        ("run/planted.py", "import importlib\nimportlib.import_module('threetears.evals.providers')\n", True),
        ("run/planted.py", "from threetears.evals.schema import CompletionClient\n", False),
        ("run/planted.py", "NOTE = 'the providers root is optional'\n", False),
        ("providers/planted.py", "from threetears.evals.providers.adapter import models_client\n", False),
        ("__init__.py", "PUBLIC_ROOTS: tuple[str, ...] = ('threetears.evals.providers',)\n", False),
        ("__init__.py", "ROOT_EXTRAS = {'threetears.evals.providers': RootExtra('models', ('anthropic',))}\n", False),
        ("__init__.py", "OTHER = 'threetears.evals.providers'\n", True),
    ],
)
def test_the_engine_rule(tmp_path: Path, planted: str, source: str, found: bool) -> None:
    """An import, a relative import or a module string naming the adapter is found; a mention in prose is not."""
    path = tmp_path / "threetears" / "evals" / planted
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    assert bool(list(engine_reaches(tmp_path))) is found


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("from threetears.models import create_chat_model\n", False),
        ("from threetears.models.price_lookup import HttpClient\n", False),
        ("from threetears.models.providers._name_translation_mixin import X\n", True),
        ("from threetears.models.factory import _CallbacksKeptBinding\n", True),
        ("import threetears.models.providers._junk_tool_calls\n", True),
        ("import threetears.models as m\nm._REGISTRY\n", True),
        ("from threetears.models import price_lookup\nprice_lookup._REGISTRY.sources\n", True),
        ("from threetears.models import CircuitBreaker\nCircuitBreaker.__name__\n", False),
        ("import langchain_core\nlangchain_core._private\n", False),
    ],
)
def test_the_models_rule(tmp_path: Path, source: str, found: bool) -> None:
    """A private module, name or attribute of models is found; public names and dunders are not; nor is another package."""
    path = tmp_path / "planted.py"
    path.write_text(source, encoding="utf-8")
    assert bool(list(private_models_reaches(path))) is found


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("from threetears.evals.schema.completion import CompletionClient\n", True),
        ("import threetears.evals.schema.completion\n", True),
        ("from threetears.evals.schema import _private_name\n", True),
        ("from threetears.evals.schema import CompletionClient\n", False),
        ("import threetears.evals.schema\n", False),
        ("from threetears.models import create_chat_model\n", False),
    ],
)
def test_the_adapter_side_rule(tmp_path: Path, source: str, found: bool) -> None:
    """A submodule or a name outside a root's ``__all__`` is a reach; a public root's declared name is not."""
    path = tmp_path / "planted.py"
    path.write_text(source, encoding="utf-8")
    assert bool(list(nonpublic_engine_reaches(path))) is found
