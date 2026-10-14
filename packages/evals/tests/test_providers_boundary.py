"""Structural gate: the completion client and the engine meet only through public names, in one direction.

``threetears.evals.providers`` is the shipped completion client: the one place the package names a provider. Three
properties keep it a client the engine never depends on:

* **The engine never imports it.** No module of the package outside ``providers/`` names it, by an import
  statement or by a string a dynamic import could use, except the package root's declarations (``PUBLIC_ROOTS``, and
  ``_LAZY_EXPORTS``, which loads ``Model`` by name on first read) and ``quick``'s imports inside a function body (the
  batteries build the client's ``Model`` where an eval names a model by id, and only then); and importing every other
  public root in a fresh interpreter does not load it. A host that brings its own client never loads this one.
* **It imports only public names of the engine.** Every reach into ``threetears.evals`` from it goes through a public
  root's ``__all__`` (:func:`nonpublic_engine_reaches`), so the client is held to exactly what a host is.
* **It depends on nothing but ``httpx`` and pydantic.** No provider SDK, no LangChain, and no other 3tears package:
  every module it names is the standard library, the package itself, ``httpx`` or ``pydantic``.

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

#: The client's tree.
PROVIDERS_ROOT = SOURCE_ROOT / "threetears" / "evals" / "providers"

#: The client's root, as a module name.
PROVIDERS = "threetears.evals.providers"

#: The engine's package.
EVALS = "threetears.evals"

#: The repository root, which the probe runs from.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: What the client may name past the standard library and the package itself.
CLIENT_DEPENDENCIES = ("httpx", "pydantic")


def _names_the_client(text: str) -> bool:
    return text == PROVIDERS or text.startswith(f"{PROVIDERS}.")


#: The package-level declarations that name a public root by string: ``PUBLIC_ROOTS``, ``ROOT_EXTRAS``, and
#: ``_LAZY_EXPORTS``, the names the package exports from a root by loading it on first read. Naming the client's root
#: there is declaring it, not reaching it.
_ROOT_DECLARATIONS = frozenset({"PUBLIC_ROOTS", "ROOT_EXTRAS", "_LAZY_EXPORTS"})


def _root_declarations(tree: ast.Module) -> set[int]:
    """The ids of the string constants inside the package's root declarations (:data:`_ROOT_DECLARATIONS`)."""
    declared: set[int] = set()
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(target, ast.Name) and target.id in _ROOT_DECLARATIONS for target in targets):
            declared |= {id(constant) for constant in ast.walk(node) if isinstance(constant, ast.Constant)}
    return declared


def _under_type_checking(tree: ast.Module) -> set[int]:
    """The ids of the import nodes inside a module-level ``if TYPE_CHECKING:`` block: a name for the checker only."""
    guarded: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            guarded |= {id(inner) for inner in ast.walk(node) if isinstance(inner, ast.Import | ast.ImportFrom)}
    return guarded


def _inside_functions(tree: ast.Module) -> set[int]:
    """The ids of the import nodes inside a function body: run when the function is called, never on import."""
    nested: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            nested |= {id(inner) for inner in ast.walk(node) if isinstance(inner, ast.Import | ast.ImportFrom)}
    return nested


def engine_reaches(source_root: Path) -> Iterator[str]:
    """Every place a module outside ``providers/`` names the client: an import, or a string naming its module.

    The package root's declarations are not reaches, nor is an import the root makes for the type checker alone, nor
    one ``quick`` makes inside a function body: the batteries build the client's ``Model`` where an eval names a model
    by id, so importing ``quick`` loads no client (:func:`test_importing_every_other_root_loads_no_client`).

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
        is_root = relative == Path("__init__.py")
        declaration = _root_declarations(tree) if is_root else set()
        checker_only = _under_type_checking(tree) if is_root else set()
        on_call = _inside_functions(tree) if relative.parts[0] == "quick" else set()
        for node in ast.walk(tree):
            if id(node) in checker_only or id(node) in on_call:
                continue
            if isinstance(node, ast.Import):
                yield from (
                    f"{relative}:{node.lineno}: import {alias.name}"
                    for alias in node.names
                    if _names_the_client(alias.name)
                )
            elif isinstance(node, ast.ImportFrom):
                module = absolute_module(path, node, root=source_root)
                if _names_the_client(module) or any(
                    _names_the_client(f"{module}.{alias.name}") for alias in node.names
                ):
                    yield f"{relative}:{node.lineno}: from {module} import …"
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and _names_the_client(node.value)
                and id(node) not in declaration
            ):
                yield f"{relative}:{node.lineno}: the string {node.value!r}"


def outside_dependencies(path: Path, *, root: Path) -> Iterator[str]:
    """Every module ``path`` names that is neither the standard library, the package, nor a client dependency.

    Args:
        path: A client source file.
        root: The directory its dotted names are rooted at.

    Yields:
        ``line: module`` for each.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            named = [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            named = [(node.lineno, absolute_module(path, node, root=root))]
        else:
            continue
        for line, module in named:
            top = module.split(".")[0]
            if top in sys.stdlib_module_names or top in CLIENT_DEPENDENCIES:
                continue
            if module == "threetears.evals" or module.startswith("threetears.evals."):
                continue
            yield f"{line}: {module}"


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


# --- the gates over today's tree -------------------------------------------------------------------------


def test_the_engine_never_names_the_client() -> None:
    """No module outside ``providers/`` imports the client or names its module in a string, but root declarations."""
    reaches = list(engine_reaches(SOURCE_ROOT))
    assert not reaches, (
        "The engine names the completion client, which a host that brings its own never loads. Take what is needed "
        "as a value the host passes in, or move it below the engine:\n  " + "\n  ".join(reaches)
    )


#: The one reach below a public root the client makes, and why: the package's logger. Every module of the package
#: takes its stdlib logger from :mod:`threetears.evals.schema.logs`, and the client logs what the engine does not
#: store: a reply read as ``truncated`` with the provider's raw stop reason, an unread catalogue, a setting sent as
#: the closest the model takes. A host logs through loggers of its own and never needs it.
PACKAGE_LOGGING = "threetears.evals.schema.logs.get_logger"


def test_the_client_reaches_the_engine_only_through_public_roots() -> None:
    """Every reach into ``threetears.evals`` from the client is a public root's declared name, as for any host."""
    findings = [
        reach
        for reach in nonpublic_engine_reaches(PROVIDERS_ROOT)
        if not _names_the_client(reach) and reach != PACKAGE_LOGGING
    ]
    assert not findings, "The client reaches below a public root of the engine:\n  " + "\n  ".join(findings)


def test_the_logging_exception_is_the_one_reach_it_names() -> None:
    """Non-vacuity: the exception is in use, as an import of exactly that name."""
    assert PACKAGE_LOGGING in set(nonpublic_engine_reaches(PROVIDERS_ROOT))


def test_the_client_names_no_sdk_and_no_other_3tears_package() -> None:
    """Every module the client names is the standard library, the package, ``httpx`` or pydantic."""
    found = [
        f"{path.relative_to(PROVIDERS_ROOT)}:{reach}"
        for path in sorted(PROVIDERS_ROOT.rglob("*.py"))
        for reach in outside_dependencies(path, root=SOURCE_ROOT)
    ]
    assert not found, "The client names a dependency beyond httpx and pydantic:\n  " + "\n  ".join(found)


def test_the_walks_read_the_client() -> None:
    """Non-vacuity: the client exists, speaks httpx, is a public root, and is the package root's lazy ``Model``."""
    files = sorted(PROVIDERS_ROOT.rglob("*.py"))
    assert files, "the walk found no client module"
    assert any("import httpx" in path.read_text(encoding="utf-8") for path in files)
    assert PROVIDERS in threetears.evals.PUBLIC_ROOTS
    from threetears.evals.providers import Model

    assert threetears.evals.Model is Model


#: Imports every public root but the client in a fresh interpreter, and the package root, then reports whether the
#: client was loaded along the way.
_OTHER_ROOTS_PROBE = (
    "import importlib, sys, threetears.evals as e\n"
    "for root in e.PUBLIC_ROOTS:\n"
    f"    if root != {PROVIDERS!r}:\n"
    "        importlib.import_module(root)\n"
    f"loaded = sorted(n for n in sys.modules if n == {PROVIDERS!r} or n.startswith({PROVIDERS + '.'!r}))\n"
    "assert not loaded, loaded\n"
)


def test_importing_every_other_root_loads_no_client() -> None:
    """The transitive half, which no per-file walk can report: the engine's roots, and the package, load no client."""
    proc = subprocess.run(
        [sys.executable, "-c", _OTHER_ROOTS_PROBE],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, f"importing the engine's roots loaded the completion client:\n{proc.stderr}"


# --- each rule refuses what it should ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("planted", "source", "found"),
    [
        ("run/planted.py", "from threetears.evals.providers import Model\n", True),
        ("run/planted.py", "import threetears.evals.providers.model\n", True),
        ("run/planted.py", "from threetears.evals import providers\n", True),
        ("run/planted.py", "from .. import providers\n", True),
        ("run/planted.py", "import importlib\nimportlib.import_module('threetears.evals.providers')\n", True),
        ("run/planted.py", "from threetears.evals.schema import CompletionClient\n", False),
        ("run/planted.py", "NOTE = 'the providers root is a client'\n", False),
        ("providers/planted.py", "from threetears.evals.providers.model import Model\n", False),
        ("__init__.py", "PUBLIC_ROOTS: tuple[str, ...] = ('threetears.evals.providers',)\n", False),
        ("__init__.py", "_LAZY_EXPORTS = {'Model': 'threetears.evals.providers'}\n", False),
        (
            "__init__.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from threetears.evals.providers import Model\n",
            False,
        ),
        ("__init__.py", "from threetears.evals.providers import Model\n", True),
        ("quick/planted.py", "def build():\n    from threetears.evals.providers import Model\n", False),
        ("quick/planted.py", "from threetears.evals.providers import Model\n", True),
        ("run/planted.py", "def build():\n    from threetears.evals.providers import Model\n", True),
        ("__init__.py", "OTHER = 'threetears.evals.providers'\n", True),
    ],
)
def test_the_engine_rule(tmp_path: Path, planted: str, source: str, found: bool) -> None:
    """An import, a relative import or a module string naming the client is found; a declaration or prose is not."""
    path = tmp_path / "threetears" / "evals" / planted
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    assert bool(list(engine_reaches(tmp_path))) is found


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("import httpx\nimport json\n", False),
        ("from pydantic import BaseModel\n", False),
        ("from threetears.evals.schema import ProviderFailure\n", False),
        ("import anthropic\n", True),
        ("from langchain_core.messages import AIMessage\n", True),
        ("from threetears.models import create_chat_model\n", True),
        ("import openrouter\n", True),
    ],
)
def test_the_dependency_rule(tmp_path: Path, source: str, found: bool) -> None:
    """An SDK, LangChain or another 3tears package is found; httpx, pydantic, the stdlib and the package are not."""
    path = tmp_path / "threetears" / "evals" / "providers" / "planted.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    assert bool(list(outside_dependencies(path, root=tmp_path))) is found


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("from threetears.evals.schema.completion import CompletionClient\n", True),
        ("import threetears.evals.schema.completion\n", True),
        ("from threetears.evals.schema import _private_name\n", True),
        ("from threetears.evals.schema import CompletionClient\n", False),
        ("import threetears.evals.schema\n", False),
        ("import httpx\n", False),
    ],
)
def test_the_client_side_rule(tmp_path: Path, source: str, found: bool) -> None:
    """A submodule or a name outside a root's ``__all__`` is a reach; a public root's declared name is not."""
    path = tmp_path / "planted.py"
    path.write_text(source, encoding="utf-8")
    assert bool(list(nonpublic_engine_reaches(path))) is found
