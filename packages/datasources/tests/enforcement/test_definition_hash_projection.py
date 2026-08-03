"""enforcement tests for the content hash's exclusion set and its escape hatches.

AST-level and under a second. Three properties, and each one produces a
silent, expensive failure months later if it regresses:

- **the hash exclusion set is declared, complete, and applied BEFORE
  serialization.** A grant, a physical layout, or a pinned upstream run
  leaking back into the hash mints a version over policy, fires
  ``definition_drift`` on every historical run, and -- versions being
  immutable -- makes every referenced draft permanently unreapable.
  Filtering a serialized blob instead would miss a nested occurrence, so
  the projection is asserted to walk ``model_fields`` and skip a declared
  set rather than to post-process a dump.
- **``RawSelect.provenance`` is required.** An optional one reopens the
  escape hatch ``custom_audience_units/`` was, which is the entire reason
  the type exists (D9a).
- **``DerivedColumn.expression`` stays OPEN.** A later reviewer tightening
  it into a closed kind union makes one real shipped deliverable --
  concatenate two labels, trim, then classify on the exact value of that
  concatenation -- inexpressible, and pushes it into ``raw:``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DEFINITION_DIR = _PACKAGE_ROOT / "src" / "threetears" / "datasources" / "definition"

_REQUIRED_EXCLUSIONS: dict[str, frozenset[str]] = {
    "UpstreamPin": frozenset({"run"}),
    "DeliverySpec": frozenset({"grants", "layout"}),
    "ArtifactSpec": frozenset({"layout"}),
    "DatasetDefinition": frozenset({"version"}),
}


def _module_tree(name: str) -> ast.Module:
    """parse one module of the definition package.

    :param name: module file name, e.g. ``dataset.py``
    :ptype name: str
    :returns: parsed module
    :rtype: ast.Module
    """
    return ast.parse((_DEFINITION_DIR / name).read_text(encoding="utf-8"))


def _class_node(name: str) -> ast.ClassDef:
    """locate one class anywhere in the definition package.

    :param name: class name
    :ptype name: str
    :returns: the class node
    :rtype: ast.ClassDef
    :raises AssertionError: the class is not declared in the package
    """
    for module_path in sorted(_DEFINITION_DIR.glob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        found = next(
            (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name),
            None,
        )
        if found is not None:
            return found
    raise AssertionError(f"{name} is not declared in the definition package")


def _is_docstring(statement: ast.stmt) -> bool:
    """whether a statement is a bare string expression.

    :param statement: one statement of a function body
    :ptype statement: ast.stmt
    :returns: True for a docstring
    :rtype: bool
    """
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)


def _declared_exclusions(node: ast.ClassDef) -> frozenset[str]:
    """field names a class declares as hash-excluded.

    :param node: the model class node
    :ptype node: ast.ClassDef
    :returns: declared excluded field names
    :rtype: frozenset[str]
    """
    declared: set[str] = set()
    for statement in node.body:
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
            continue
        if statement.target.id != "hash_excluded_fields" or statement.value is None:
            continue
        for element in ast.walk(statement.value):
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                declared.add(element.value)
    return frozenset(declared)


def _field_names(node: ast.ClassDef) -> frozenset[str]:
    """annotated field names a model declares.

    :param node: the model class node
    :ptype node: ast.ClassDef
    :returns: declared field names, excluding class-level configuration
    :rtype: frozenset[str]
    """
    return frozenset(
        statement.target.id
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id not in {"hash_excluded_fields", "model_config"}
    )


class TestHashExclusionSetIsDeclared:
    """every policy field is declared excluded on the model that owns it."""

    def test_every_required_exclusion_is_declared(self) -> None:
        missing = {
            name: sorted(expected - _declared_exclusions(_class_node(name)))
            for name, expected in _REQUIRED_EXCLUSIONS.items()
            if expected - _declared_exclusions(_class_node(name))
        }
        assert missing == {}

    def test_no_exclusion_names_a_field_that_does_not_exist(self) -> None:
        stale = {
            name: sorted(_declared_exclusions(_class_node(name)) - _field_names(_class_node(name)))
            for name in _REQUIRED_EXCLUSIONS
            if _declared_exclusions(_class_node(name)) - _field_names(_class_node(name))
        }
        assert stale == {}

    def test_no_exclusion_is_over_broad(self) -> None:
        over_broad = {
            name: sorted(_declared_exclusions(_class_node(name)) - expected)
            for name, expected in _REQUIRED_EXCLUSIONS.items()
            if _declared_exclusions(_class_node(name)) - expected
        }
        assert over_broad == {}


class TestExclusionHappensBeforeSerialization:
    """the projection walks fields; it never filters a serialized blob."""

    def test_the_projection_skips_declared_exclusions_while_walking_fields(self) -> None:
        tree = _module_tree("dataset.py")
        projector = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_canonical_model"
        )
        source = ast.unparse(projector)
        assert "hash_excluded_fields" in source
        assert "model_fields" in source
        assert "not in excluded" in source

    def test_the_digest_never_post_processes_a_dump(self) -> None:
        tree = _module_tree("dataset.py")
        digest = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "content_hash"
        )
        source = "\n".join(ast.unparse(statement) for statement in digest.body if not _is_docstring(statement))
        for banned in ("model_dump", ".replace(", ".pop(", "del "):
            assert banned not in source


class TestEscapeHatchesStayClosed:
    """two fields a later reviewer would plausibly and wrongly relax."""

    def test_raw_select_provenance_is_required(self) -> None:
        node = _class_node("RawSelect")
        provenance = next(
            statement
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "provenance"
        )
        assert provenance.value is None, "RawSelect.provenance gained a default and is no longer required"
        assert ast.unparse(provenance.annotation) == "ProvenanceSpec"

    def test_derived_column_expression_stays_open(self) -> None:
        node = _class_node("DerivedColumn")
        expression = next(
            statement
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "expression"
        )
        assert ast.unparse(expression.annotation) == "Expression"

    def test_the_artifact_enum_carries_the_fifth_member(self) -> None:
        node = _class_node("ArtifactKind")
        members = {
            statement.targets[0].id
            for statement in node.body
            if isinstance(statement, ast.Assign) and statement.targets and isinstance(statement.targets[0], ast.Name)
        }
        assert members == {"LONG", "QUALIFIED", "WIDE", "PROVENANCE", "RELATIONSHIP_UNION"}
