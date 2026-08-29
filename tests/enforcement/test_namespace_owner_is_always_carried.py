"""every namespace value reaching the evaluator must carry its owner.

``threetears.agent.acl.types.Namespace`` gained ``owner_namespace``, which
is the key the evaluator's ownership short-circuit reads. The field is
OPTIONAL and defaults to ``None``, which is deliberate -- a construction
site with no owner to supply must deny rather than guess.

That default is also the hazard this walker exists for. A construction
site that simply FORGETS the field is indistinguishable, at runtime, from
one that has no owner: both produce a namespace nobody owns, and the
symptom is an ordinary deny with nothing pointing at the missing
argument. That failure mode was not hypothetical -- it was hit three
separate times while the field was being introduced, once on the path an
agent takes to its own storage, where it surfaced as an agent being
refused WRITE on its own namespace.

So the rule is: passing the field is REQUIRED, and passing ``None``
deliberately is fine. What is refused is silence.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: the value type whose construction this walker polices, under either
#: the plain name or the ``AclNamespace`` alias every consumer imports it
#: as.
_NAMESPACE_CONSTRUCTORS = frozenset({"Namespace", "AclNamespace"})

#: the field every construction must name.
_OWNER_FIELD = "owner_namespace"

#: the other fields a namespace value carries. a call naming none of
#: these is some OTHER ``Namespace`` -- there are unrelated classes by
#: that name -- so it is not this walker's business.
_DISCRIMINATING_FIELDS = frozenset({"namespace_type", "owner_agent_id"})

_PACKAGES = Path(__file__).resolve().parents[2] / "packages"


def _sources() -> list[Path]:
    """collect every production python file under ``packages/``.

    :return: source paths, tests and virtualenvs excluded
    :rtype: list[Path]
    """
    return [path for path in _PACKAGES.rglob("*.py") if ".venv" not in path.parts and "tests" not in path.parts]


def _offending_calls(tree: ast.AST) -> list[int]:
    """line numbers of namespace constructions that omit the owner.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: line numbers, ascending
    :rtype: list[int]
    """
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name not in _NAMESPACE_CONSTRUCTORS:
            continue
        keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
        # ``**row`` and friends: the walker cannot see what a splat
        # carries, so it does not guess.
        splatted = any(kw.arg is None for kw in node.keywords)
        if not (keywords & _DISCRIMINATING_FIELDS):
            continue
        if _OWNER_FIELD not in keywords and not splatted:
            offenders.append(node.lineno)
    return sorted(offenders)


def test_every_namespace_construction_names_its_owner() -> None:
    """a construction that omits ``owner_namespace`` fails here, not in production."""
    violations: list[str] = []
    for path in _sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        for lineno in _offending_calls(tree):
            violations.append(f"{path}:{lineno}")
    assert not violations, (
        "these namespace constructions reach the rbac evaluator without an "
        "owner, so every owner they describe is refused access to its own "
        "storage:\n  " + "\n  ".join(violations) + "\n\n"
        "pass owner_namespace explicitly. passing None is fine when the site "
        "genuinely has no owner to supply -- what is refused is omitting it, "
        "because that is indistinguishable from an unowned namespace at "
        "runtime and denies in the shape of an ordinary access failure."
    )


def test_the_walker_detects_an_omission() -> None:
    """the walker must fail a real omission, or it guards nothing."""
    omitted = ast.parse(
        "AclNamespace(id=x, customer_id=c, namespace_type='tool', owner_agent_id=a)",
    )
    assert _offending_calls(omitted) == [1]


def test_the_walker_accepts_an_explicit_none() -> None:
    """passing the field deliberately as None is not a violation."""
    explicit = ast.parse(
        "AclNamespace(id=x, customer_id=c, namespace_type='tool', owner_agent_id=a, owner_namespace=None)",
    )
    assert _offending_calls(explicit) == []


def test_the_walker_ignores_an_unrelated_namespace_class() -> None:
    """other classes are called Namespace; this walker is not about them."""
    unrelated = ast.parse("Namespace(prefix='x', separator='.')")
    assert _offending_calls(unrelated) == []
