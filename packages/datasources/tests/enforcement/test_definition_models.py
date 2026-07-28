"""enforcement tests for the dataset-definition model package.

Two properties, both AST- or manifest-level and both under a second:

- every pydantic model under ``threetears/datasources/definition/`` sets
  ``extra="forbid"``. An ignored field in a definition is a silently
  dropped semantic, which is exactly the class of bug the corpus gate
  exists to prevent.
- ``3tears-datasources`` gained no runtime dependency. The Hub-side
  compiler split exists solely to protect this: ``sqlglot`` is a
  dependency of no 3tears package, and every secondary backend already
  sits behind an extras key, so a parser dependency here would tax every
  consumer.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DEFINITION_DIR = _PACKAGE_ROOT / "src" / "threetears" / "datasources" / "definition"

# the runtime dependency list ``dsm-task-01a`` inherited and must not grow.
_EXPECTED_RUNTIME_DEPENDENCIES = {
    "3tears",
    "3tears-observe",
    "pydantic",
    "asyncpg",
}


def _definition_modules() -> list[Path]:
    """every python module in the definition package.

    :returns: sorted module paths
    :rtype: list[pathlib.Path]
    """
    return sorted(_DEFINITION_DIR.glob("*.py"))


def _model_classes(tree: ast.Module) -> list[ast.ClassDef]:
    """pydantic ``BaseModel`` subclasses declared in one module.

    :param tree: parsed module
    :ptype tree: ast.Module
    :returns: the model class nodes
    :rtype: list[ast.ClassDef]
    """
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases)
    ]


def _forbids_extra(node: ast.ClassDef) -> bool:
    """whether a class body assigns ``model_config`` with ``extra="forbid"``.

    :param node: the model class node
    :ptype node: ast.ClassDef
    :returns: True when the class forbids extra fields
    :rtype: bool
    """
    forbids = False
    for statement in node.body:
        if not isinstance(statement, ast.Assign):
            continue
        targets = [t.id for t in statement.targets if isinstance(t, ast.Name)]
        if "model_config" not in targets or not isinstance(statement.value, ast.Call):
            continue
        for keyword in statement.value.keywords:
            if keyword.arg == "extra" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "forbid":
                forbids = True
    return forbids


def _all_model_classes() -> list[tuple[str, ast.ClassDef]]:
    """every definition model paired with its module name.

    :returns: ``(module_name, class_node)`` pairs
    :rtype: list[tuple[str, ast.ClassDef]]
    """
    collected: list[tuple[str, ast.ClassDef]] = []
    for module_path in _definition_modules():
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        collected.extend((module_path.name, node) for node in _model_classes(tree))
    return collected


class TestDefinitionModelsForbidExtra:
    """``extra="forbid"`` on every definition model."""

    def test_the_definition_package_exists(self) -> None:
        assert _DEFINITION_DIR.is_dir()

    def test_models_were_found(self) -> None:
        assert _all_model_classes() != []

    @pytest.mark.parametrize(
        ("module_name", "node"),
        _all_model_classes(),
        ids=lambda value: value if isinstance(value, str) else value.name,
    )
    def test_model_forbids_extra(self, module_name: str, node: ast.ClassDef) -> None:
        assert _forbids_extra(node), f"{module_name}:{node.name} does not set extra='forbid'"


class TestNoNewRuntimeDependency:
    """the definition model added nothing to the wheel's dependency list."""

    def test_runtime_dependencies_are_unchanged(self) -> None:
        manifest = tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        declared = {
            requirement.split(">=")[0].split("==")[0].split("[")[0].strip()
            for requirement in manifest["project"]["dependencies"]
        }
        assert declared == _EXPECTED_RUNTIME_DEPENDENCIES

    def test_sqlglot_is_not_a_dependency(self) -> None:
        manifest = tomllib.loads((_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        runtime = manifest["project"]["dependencies"]
        optional = manifest["project"].get("optional-dependencies", {})
        every = list(runtime) + [item for group in optional.values() for item in group]
        assert [item for item in every if "sqlglot" in item] == []
