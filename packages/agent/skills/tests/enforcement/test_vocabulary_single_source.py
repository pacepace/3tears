"""AST-walker enforcement test for single-sourced value vocabularies.

``types.py`` is the one place in this package where an enumerated value
set may be written down. Every other module annotates with the alias.

This exists because re-spelling one diverged, silently and expensively.
``OutcomeSource`` is ``Literal["agent_marker", "agent_tool",
"user_feedback"]`` in ``types.py``; ``api_models.SkillInvocationResponse``
re-spelled it as ``Literal["agent_marker", "user_feedback"]``. The value
it omitted, ``'agent_tool'``, is the only one anything writes
(``tools.py``, in ``skill_report_outcome``) -- ``'agent_marker'`` is a
retired path. So the response model would have raised
``ValidationError`` on every real invocation row, and nothing caught it
because the re-spelling is locally well-typed: mypy checks a
``Literal`` against itself and has no idea another module meant the
same field.

Three checks:

- **no inline string ``Literal``** in an annotation outside
  ``types.py``. This is the shape that diverged.
- **no re-spelled value-set constant** -- a ``frozenset``/``set``/
  ``tuple``/``list`` of string literals at module scope whose contents
  match an alias in ``types.py``. ``tools._VALID_PROMPT_MODES`` was one
  of these; it is now ``frozenset(get_args(PromptMode))``.
- **every alias in ``types.py`` is exported** in its ``__all__``, so a
  new alias is importable by the modules required to use it.

Numeric and boolean ``Literal`` annotations are out of scope: the
failure mode is a vocabulary of named string values drifting between
two spellings of the same field.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# package root resolved relative to this file so pytest can run from
# any working directory.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PACKAGE_ROOT / "src" / "threetears" / "agent" / "skills"
_TYPES_MODULE = _SRC_ROOT / "types.py"

# minimum members before a string collection counts as a "value set"
# worth comparing against the aliases. a one-element collection is far
# more likely to be a config default than a vocabulary.
_MIN_VALUE_SET_SIZE = 2


def _iter_source_modules() -> list[Path]:
    """return every ``.py`` file under the package src tree except ``types.py``.

    :return: sorted list of source-module paths, ``types.py`` excluded
    :rtype: list[Path]
    """
    return sorted(path for path in _SRC_ROOT.rglob("*.py") if path != _TYPES_MODULE)


def _string_literal_values(node: ast.expr) -> tuple[str, ...] | None:
    """return the string members of a ``Literal[...]`` subscript.

    :param node: annotation expression to inspect
    :ptype node: ast.expr
    :return: the string members, or None if not an all-string ``Literal``
    :rtype: tuple[str, ...] | None
    """
    if not isinstance(node, ast.Subscript):
        return None
    base = node.value
    is_literal = (isinstance(base, ast.Name) and base.id == "Literal") or (
        isinstance(base, ast.Attribute) and base.attr == "Literal"
    )
    if not is_literal:
        return None
    elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    values: list[str] = []
    for element in elements:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            return None
        values.append(element.value)
    return tuple(values)


def _inline_literals(tree: ast.Module) -> list[tuple[int, str]]:
    """find every inline string-``Literal`` annotation in a module.

    walks the whole tree, so class-body fields, function parameters,
    return annotations and bare annotated assignments are all covered.

    :param tree: parsed module AST
    :ptype tree: ast.Module
    :return: list of ``(lineno, rendered_literal)`` tuples
    :rtype: list[tuple[int, str]]
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        values = _string_literal_values(node)
        if values is not None:
            hits.append((node.lineno, ast.unparse(node)))
    return hits


def _alias_value_sets() -> dict[str, frozenset[str]]:
    """return ``{alias_name: value_set}`` for every alias in ``types.py``.

    :return: mapping of alias name to its frozen set of string values
    :rtype: dict[str, frozenset[str]]
    """
    tree = ast.parse(_TYPES_MODULE.read_text(), filename=str(_TYPES_MODULE))
    aliases: dict[str, frozenset[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        values = _string_literal_values(node.value)
        if values is not None:
            aliases[target.id] = frozenset(values)
    return aliases


def _module_scope_string_collections(tree: ast.Module) -> list[tuple[int, str, frozenset[str]]]:
    """find module-scope collections built from string literals.

    covers the bare ``{...}`` / ``[...]`` / ``(...)`` displays and the
    ``frozenset({...})`` / ``set([...])`` call wrappers.

    :param tree: parsed module AST
    :ptype tree: ast.Module
    :return: list of ``(lineno, name, value_set)`` tuples
    :rtype: list[tuple[int, str, frozenset[str]]]
    """
    found: list[tuple[int, str, frozenset[str]]] = []
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, node.value
        else:
            continue
        if value is None:
            continue
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id not in {"frozenset", "set", "tuple", "list"} or len(value.args) != 1:
                continue
            value = value.args[0]
        if not isinstance(value, ast.Set | ast.List | ast.Tuple):
            continue
        members: list[str] = []
        for element in value.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                members = []
                break
            members.append(element.value)
        if len(members) >= _MIN_VALUE_SET_SIZE:
            found.append((node.lineno, name, frozenset(members)))
    return found


@pytest.mark.parametrize("source_module", _iter_source_modules(), ids=lambda p: p.name)
def test_no_inline_string_literal_annotations(source_module: Path) -> None:
    """string ``Literal`` value sets live only in ``types.py``.

    if this test fails: import the matching alias from
    :mod:`threetears.agent.skills.types` and annotate with it. if no
    alias fits, add one there first -- a new vocabulary belongs in the
    single source too, not inline at its first call site.

    :param source_module: source-module path under test
    :ptype source_module: Path
    """
    tree = ast.parse(source_module.read_text(), filename=str(source_module))
    hits = _inline_literals(tree)
    if hits:
        rendered = "\n".join(
            f"  {source_module.relative_to(_PACKAGE_ROOT)}:{lineno}: {literal}" for lineno, literal in hits
        )
        raise AssertionError(
            f"inline string `Literal` annotations outside `types.py` "
            f"(re-spelling one is how `outcome_source` lost `agent_tool`):\n{rendered}"
        )


@pytest.mark.parametrize("source_module", _iter_source_modules(), ids=lambda p: p.name)
def test_no_respelled_value_set_constants(source_module: Path) -> None:
    """a module-scope string collection MUST NOT duplicate an alias's values.

    if this test fails: derive the constant from the alias with
    ``typing.get_args`` -- ``frozenset(get_args(PromptMode))`` -- so a
    value added to the alias reaches the runtime check without a second
    edit anyone has to remember.

    :param source_module: source-module path under test
    :ptype source_module: Path
    """
    aliases = _alias_value_sets()
    tree = ast.parse(source_module.read_text(), filename=str(source_module))
    failures: list[str] = []

    for lineno, name, values in _module_scope_string_collections(tree):
        duplicated = sorted(alias for alias, alias_values in aliases.items() if alias_values == values)
        if duplicated:
            failures.append(
                f"  {source_module.relative_to(_PACKAGE_ROOT)}:{lineno}: "
                f"{name} re-spells {', '.join(duplicated)}; use get_args() instead"
            )

    if failures:
        rendered = "\n".join(failures)
        raise AssertionError(f"module-scope constants duplicating a `types.py` alias:\n{rendered}")


def test_every_types_alias_is_exported() -> None:
    """every alias defined in ``types.py`` MUST appear in its ``__all__``.

    an alias the other modules cannot import is an alias they will
    re-spell instead, which is the failure this suite exists to stop.
    """
    tree = ast.parse(_TYPES_MODULE.read_text(), filename=str(_TYPES_MODULE))
    exported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "__all__" and isinstance(node.value, ast.List):
                exported = {
                    element.value
                    for element in node.value.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, str)
                }
    missing = sorted(set(_alias_value_sets()) - exported)
    assert not missing, f"aliases defined in types.py but absent from its __all__: {', '.join(missing)}"


def test_walker_detects_the_outcome_source_divergence() -> None:
    """sanity: the walker trips on the exact defect that motivated it.

    locks in the detection of an inline ``Literal`` and of a re-spelled
    value-set constant, so a future edit cannot loosen either check into
    a no-op without this failing.
    """
    source = (
        "class Response(BaseModel):\n"
        '    outcome_source: Literal["agent_marker", "user_feedback"] | None\n'
        "\n"
        '_VALID: frozenset[str] = frozenset({"additive", "replace"})\n'
    )
    tree = ast.parse(source)

    hits = _inline_literals(tree)
    assert [literal for _, literal in hits] == ["Literal['agent_marker', 'user_feedback']"]

    collections = _module_scope_string_collections(tree)
    assert [(name, values) for _, name, values in collections] == [("_VALID", frozenset({"additive", "replace"}))]
    assert collections[0][2] == _alias_value_sets()["PromptMode"]
