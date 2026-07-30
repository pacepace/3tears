"""AST-walker enforcement test for enum-vs-docstring parity.

a ``StrEnum`` member set is written down three times in this package:
once as the enum body, once as the ``:cvar`` block in the class
docstring, and -- for the enums a module summarises up top -- once more
as a ``- :class:`Name` -- ``a`` / ``b``` bullet in the MODULE docstring.
the enum body is the only one the interpreter reads, so the other two
rot silently.

that is not hypothetical. :class:`DataSourceAccessMode` grew ``build``
and ``publish``; the class docstring gained both ``:cvar`` entries and
the module docstring did not, so the summary read ``read`` / ``write``
/ ``readwrite`` while the enum below it carried five members. a reader
who trusted the summary would conclude the build and publish access
surfaces did not exist.

two checks, both derived from the enum body:

- **member-documentation parity** -- IF a ``StrEnum`` documents ANY of
  its members, it must document ALL of them. two conventions are in use
  in this package and both count: a ``:cvar NAME:`` line in the class
  docstring, and a ``#:`` comment directly above the member. an enum
  that documents none of its members is not gated; one that documents
  some is, because a partial list reads as complete.
- **module-summary parity** -- IF a module docstring carries a
  ``- :class:`Name` --`` bullet for a ``StrEnum`` in that module AND
  that bullet quotes at least one member VALUE in double backticks,
  then it must quote EVERY member value. opt-in by construction: a
  module that does not summarise its enums is not gated, and a bullet
  that only prose-describes the enum (quoting no values) is not gated
  either.

why a gate and not a derived docstring: a module docstring must be a
string LITERAL to be visible to ``ast``, to Sphinx autodoc, and to
ruff's ``D`` rules -- all three read the source, not the imported
module. building it at import time (``__doc__ = f"..."``) would hide it
from every one of them, including the AST enforcement suites this
codebase relies on. the duplication is therefore structural; the fix is
to make it falsifiable, not to remove it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# package root resolved relative to this file so pytest can run from
# any working directory.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PACKAGE_ROOT / "src" / "threetears" / "datasources"

# a module-docstring summary bullet: ``- :class:`Name` --`` at the start
# of a line. the trailing text runs to the next such bullet or to the
# end of the docstring.
_BULLET_RE = re.compile(r"^- :class:`(?P<name>\w+)` --", re.MULTILINE)

# a member value quoted in a summary bullet. the value must fill the
# whole double-backtick span, so ``kind='datasource'`` (a prose example
# of a FIELD comparison, not a member of the enum being summarised) does
# not match, while ``agent_internal`` does.
_QUOTED_VALUE_RE = re.compile(r"``([a-z][a-z0-9_]*)``")

# a ``:cvar NAME:`` entry in a class docstring.
_CVAR_RE = re.compile(r"^\s*:cvar\s+(?P<name>\w+):", re.MULTILINE)

# the Sphinx ``#:`` member-comment prefix, the other documentation
# convention in use in this package (see ``geo_config.GeometryKind``).
_MEMBER_COMMENT_PREFIX = "#:"


def _iter_source_modules() -> list[Path]:
    """return every ``.py`` file under the package src tree (recursive).

    :return: sorted list of source-module paths
    :rtype: list[Path]
    """
    return sorted(_SRC_ROOT.rglob("*.py"))


def _is_str_enum(class_node: ast.ClassDef) -> bool:
    """return True iff the class declares ``StrEnum`` as a base.

    resolves both the bare ``StrEnum`` and the ``enum.StrEnum``
    qualified form.

    :param class_node: class-definition node to inspect
    :ptype class_node: ast.ClassDef
    :return: True iff ``StrEnum`` is among the declared bases
    :rtype: bool
    """
    found = False
    for base in class_node.bases:
        if isinstance(base, ast.Name) and base.id == "StrEnum":
            found = True
        elif isinstance(base, ast.Attribute) and base.attr == "StrEnum":
            found = True
    return found


def _enum_members(class_node: ast.ClassDef) -> dict[str, str]:
    """return the ``{member_name: member_value}`` map of a StrEnum body.

    only plain ``NAME = "value"`` assignments count -- a member built
    from a call or a computed expression is not a literal the docstring
    could name, so it is skipped rather than reported as missing.

    :param class_node: StrEnum class-definition node
    :ptype class_node: ast.ClassDef
    :return: mapping of member name to its string value
    :rtype: dict[str, str]
    """
    members: dict[str, str] = {}
    for body_node in class_node.body:
        if not isinstance(body_node, ast.Assign):
            continue
        if len(body_node.targets) != 1:
            continue
        target = body_node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id.startswith("_"):
            continue
        value = body_node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            members[target.id] = value.value
    return members


def _comment_documented_members(class_node: ast.ClassDef, source_lines: list[str]) -> set[str]:
    """return the members carrying a ``#:`` Sphinx comment above them.

    the comment is not in the AST, so this reads the physical line above
    each member assignment. a member documented across several ``#:``
    lines still has one directly above it, so a single-line lookback is
    sufficient.

    :param class_node: StrEnum class-definition node
    :ptype class_node: ast.ClassDef
    :param source_lines: the module's source split into lines
    :ptype source_lines: list[str]
    :return: names of members carrying a ``#:`` comment
    :rtype: set[str]
    """
    documented: set[str] = set()
    for body_node in class_node.body:
        if not isinstance(body_node, ast.Assign):
            continue
        target = body_node.targets[0] if len(body_node.targets) == 1 else None
        if not isinstance(target, ast.Name):
            continue
        above = body_node.lineno - 2
        if above >= 0 and source_lines[above].strip().startswith(_MEMBER_COMMENT_PREFIX):
            documented.add(target.id)
    return documented


def _str_enums(tree: ast.Module) -> list[ast.ClassDef]:
    """return every ``StrEnum`` class definition in a parsed module.

    :param tree: parsed module AST
    :ptype tree: ast.Module
    :return: list of StrEnum class-definition nodes, source order
    :rtype: list[ast.ClassDef]
    """
    return [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and _is_str_enum(node)]


def _summary_bullets(module_doc: str) -> dict[str, str]:
    """split a module docstring into its ``- :class:`Name` --`` bullets.

    a bullet's text runs from its own header to the start of the next
    bullet, or to the end of the docstring for the last one, so a
    wrapped continuation line stays with the bullet that owns it.

    :param module_doc: module docstring text (empty string if absent)
    :ptype module_doc: str
    :return: mapping of class name to that bullet's full text
    :rtype: dict[str, str]
    """
    matches = list(_BULLET_RE.finditer(module_doc))
    bullets: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(module_doc)
        bullets[match.group("name")] = module_doc[match.start() : end]
    return bullets


@pytest.mark.parametrize("source_module", _iter_source_modules(), ids=lambda p: p.name)
def test_str_enum_documents_all_members_or_none(source_module: Path) -> None:
    """a ``StrEnum`` that documents ANY member MUST document every member.

    if this test fails: document the named members the same way the
    others already are -- a ``:cvar NAME: description`` line in the
    class docstring, or a ``#:`` comment directly above the member. an
    enum that documents some members and not others is worse than one
    that documents none, because the partial list reads as complete.

    :param source_module: source-module path under test
    :ptype source_module: Path
    """
    source = source_module.read_text()
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=str(source_module))
    failures: list[str] = []

    for class_node in _str_enums(tree):
        members = _enum_members(class_node)
        if not members:
            continue
        class_doc = ast.get_docstring(class_node) or ""
        documented = {match.group("name") for match in _CVAR_RE.finditer(class_doc)}
        documented |= _comment_documented_members(class_node, source_lines)
        if not documented:
            continue  # documents nothing: not gated
        missing = sorted(set(members) - documented)
        if missing:
            failures.append(
                f"  {source_module.relative_to(_PACKAGE_ROOT)}:{class_node.lineno}: "
                f"{class_node.name} documents some members but not {', '.join(missing)}"
            )

    if failures:
        rendered = "\n".join(failures)
        raise AssertionError(f"StrEnums documenting only SOME of their members:\n{rendered}")


@pytest.mark.parametrize("source_module", _iter_source_modules(), ids=lambda p: p.name)
def test_module_summary_names_every_enum_member(source_module: Path) -> None:
    """a module-docstring enum summary MUST name every member value.

    the gate is opt-in: it fires only on a ``- :class:`Name` --``
    bullet that already quotes at least one member value in double
    backticks. if this test fails: add the missing value to the bullet,
    or drop every quoted value from it and describe the enum in prose
    instead.

    :param source_module: source-module path under test
    :ptype source_module: Path
    """
    tree = ast.parse(source_module.read_text(), filename=str(source_module))
    bullets = _summary_bullets(ast.get_docstring(tree) or "")
    failures: list[str] = []

    for class_node in _str_enums(tree):
        bullet = bullets.get(class_node.name)
        if bullet is None:
            continue
        members = _enum_members(class_node)
        quoted = {match.group(1) for match in _QUOTED_VALUE_RE.finditer(bullet)}
        if not quoted & set(members.values()):
            continue  # prose-only bullet, quotes no member: not gated
        missing = sorted(set(members.values()) - quoted)
        if missing:
            failures.append(
                f"  {source_module.relative_to(_PACKAGE_ROOT)}:{class_node.lineno}: "
                f"{class_node.name} module-docstring summary omits {', '.join(missing)}"
            )

    if failures:
        rendered = "\n".join(failures)
        raise AssertionError(
            f"module-docstring enum summaries that name only SOME members "
            f"(a partial list reads as complete):\n{rendered}"
        )


def test_walker_detects_a_stale_module_summary() -> None:
    """sanity: the walker trips on the exact defect that motivated it.

    reproduces the :class:`DataSourceAccessMode` shape -- a five-member
    enum summarised with three -- so a future edit cannot loosen the
    check into a no-op without this failing.
    """
    source = (
        '"""module.\n'
        "\n"
        "- :class:`Mode` -- ``read`` / ``write`` / ``readwrite``\n"
        '"""\n'
        "\n"
        "class Mode(StrEnum):\n"
        '    """m.\n'
        "\n"
        "    :cvar READ: r\n"
        "    :cvar WRITE: w\n"
        "    :cvar READWRITE: rw\n"
        '    """\n'
        '    READ = "read"\n'
        '    WRITE = "write"\n'
        '    READWRITE = "readwrite"\n'
        '    BUILD = "build"\n'
        '    PUBLISH = "publish"\n'
    )
    tree = ast.parse(source)
    enums = _str_enums(tree)
    assert [node.name for node in enums] == ["Mode"]

    members = _enum_members(enums[0])
    assert set(members.values()) == {"read", "write", "readwrite", "build", "publish"}

    bullet = _summary_bullets(ast.get_docstring(tree) or "")["Mode"]
    quoted = {match.group(1) for match in _QUOTED_VALUE_RE.finditer(bullet)}
    assert sorted(set(members.values()) - quoted) == ["build", "publish"]

    class_doc = ast.get_docstring(enums[0]) or ""
    documented = {match.group("name") for match in _CVAR_RE.finditer(class_doc)}
    documented |= _comment_documented_members(enums[0], source.splitlines())
    assert sorted(set(members) - documented) == ["BUILD", "PUBLISH"]


def test_walker_ignores_a_prose_only_bullet() -> None:
    """sanity: a bullet that quotes no member value is not gated.

    the opt-in rule is what keeps this gate from forcing every module
    docstring in the package to become a value inventory.
    """
    source = (
        '"""module.\n'
        "\n"
        "- :class:`Mode` -- the access-mode axis; see the class docstring\n"
        '"""\n'
        "\n"
        "class Mode(StrEnum):\n"
        '    READ = "read"\n'
        '    WRITE = "write"\n'
    )
    tree = ast.parse(source)
    enums = _str_enums(tree)
    bullet = _summary_bullets(ast.get_docstring(tree) or "")["Mode"]
    quoted = {match.group(1) for match in _QUOTED_VALUE_RE.finditer(bullet)}
    assert not quoted & set(_enum_members(enums[0]).values())
