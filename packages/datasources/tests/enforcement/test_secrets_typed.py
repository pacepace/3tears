"""AST-walker enforcement test for secret-typed pydantic fields.

every pydantic field whose NAME suggests it holds a credential
(``password``, ``secret``, ``key``, ``token``) MUST be typed
:class:`pydantic.SecretStr` -- never plain ``str``. ``SecretStr`` is
opaque by construction: ``repr()`` and ``str()`` redact to
``'**********'`` so the resolved value can't leak through tracebacks,
log lines, or pydantic ``ValidationError`` chains.

field-name EXEMPTION: names ending in ``_ref`` are credential
REFERENCES -- a ``scheme://locator`` string (e.g.
``"env://OTS_REDSHIFT_PASSWORD"`` or
``"k8s://central-reporting/password"``), not the secret itself. those
stay typed ``str`` because the reference is safe to surface in logs.
see :class:`PostgresConnectionConfig.password_ref` for the canonical
shape -- the reference is the field, and a :meth:`resolve_password`
method returns the ``SecretStr`` at use time (resolved by
:mod:`threetears.datasources.secrets`).

scope:

- walks every ``.py`` file under ``threetears/datasources/`` (not
  just ``drivers/``) -- secret fields can appear in any module that
  models a credential
- flags every pydantic-style field assignment
  (``name: AnnotationLike``) where ``name`` matches the credential
  patterns AND does NOT end with ``_ref`` AND the annotation is not
  ``SecretStr``

this catches the most common drift: a future contributor adds a
``password: str`` field to a ConnectionConfig (instead of
``password_ref: str``) "for convenience," opening a credential-leak
hole.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# field-name patterns that suggest a credential.
#
# ``password``, ``secret``, ``token`` match anywhere because the
# words are unambiguously credential-related (no domain-overload).
# ``key`` only matches when it stands alone or appears as a trailing
# ``_key`` -- ``api_key``, ``private_key``, ``secret_key`` are
# credentials; ``primary_key_field`` / ``natural_key_column`` /
# ``partition_key`` are database-modelling names where ``key`` means
# DB index, not credential.
#
# the `(?i)` inline flag makes both alternations case-insensitive.
_CREDENTIAL_NAME_RE = re.compile(
    r"(?i)(password|secret|token|(^|_)key$)",
)

# exemption suffix: names ending in ``_ref`` carry a scheme://locator
# credential reference, not the secret itself. typed ``str`` is correct.
_REF_SUFFIX = "_ref"

# package root resolved relative to this file so pytest can run from
# any working directory.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PACKAGE_ROOT / "src" / "threetears" / "datasources"


def _iter_source_modules() -> list[Path]:
    """return every ``.py`` file under the package src tree (recursive).

    :return: sorted list of source-module paths
    :rtype: list[Path]
    """
    return sorted(_SRC_ROOT.rglob("*.py"))


def _annotation_is_secret_str(annotation: ast.expr | None) -> bool:
    """return True iff the annotation expression resolves to ``SecretStr``.

    handles bare ``SecretStr`` and ``SecretStr | None`` / ``Optional[SecretStr]``
    shapes. resolves both ``SecretStr`` and
    ``pydantic.SecretStr``-qualified forms.

    :param annotation: AST annotation node (or None for unannotated)
    :ptype annotation: ast.expr | None
    :return: True iff annotation includes ``SecretStr``
    :rtype: bool
    """
    if annotation is None:
        return False
    if isinstance(annotation, ast.Name) and annotation.id == "SecretStr":
        return True
    if isinstance(annotation, ast.Attribute) and annotation.attr == "SecretStr":
        return True
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        # ``X | Y`` union: either side may carry SecretStr
        return _annotation_is_secret_str(annotation.left) or _annotation_is_secret_str(annotation.right)
    if isinstance(annotation, ast.Subscript):
        # ``Optional[SecretStr]`` / ``Annotated[SecretStr, ...]``
        if isinstance(annotation.slice, ast.Tuple):
            return any(_annotation_is_secret_str(elt) for elt in annotation.slice.elts)
        return _annotation_is_secret_str(annotation.slice)
    return False


def _find_unprotected_credentials(path: Path) -> list[tuple[int, str, str]]:
    """walk one module for credential-named fields that aren't ``SecretStr``.

    :param path: source-module path to walk
    :ptype path: Path
    :return: list of ``(lineno, field_name, annotation_repr)`` tuples
    :rtype: list[tuple[int, str, str]]
    """
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    hits: list[tuple[int, str, str]] = []

    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue
        # only inspect class-body annotated assignments (the pydantic
        # field-declaration shape). top-level annotated assignments
        # are module-level constants like _CREDENTIAL_NAME_RE.
        for body_node in class_node.body:
            if not isinstance(body_node, ast.AnnAssign):
                continue
            if not isinstance(body_node.target, ast.Name):
                continue
            name = body_node.target.id
            if name.endswith(_REF_SUFFIX):
                continue  # scheme://locator reference, not the secret
            if not _CREDENTIAL_NAME_RE.search(name):
                continue
            if _annotation_is_secret_str(body_node.annotation):
                continue
            hits.append(
                (
                    body_node.lineno,
                    name,
                    ast.unparse(body_node.annotation) if body_node.annotation else "<unannotated>",
                ),
            )
    return hits


@pytest.mark.parametrize("source_module", _iter_source_modules(), ids=lambda p: p.name)
def test_credential_fields_are_secret_str(source_module: Path) -> None:
    """credential-named pydantic fields MUST be typed ``SecretStr``.

    if this test fails: change the field's annotation to
    :class:`SecretStr`, OR rename the field with ``_ref`` suffix if
    the value is actually a ``scheme://locator`` credential reference
    (the resolved secret then lives in a :meth:`resolve_*` method that
    returns ``SecretStr``).

    :param source_module: source-module path under test
    :ptype source_module: Path
    """
    hits = _find_unprotected_credentials(source_module)
    if hits:
        rendered = "\n".join(
            f"  {source_module.relative_to(_PACKAGE_ROOT)}:{lineno}: {name}: {ann}" for lineno, name, ann in hits
        )
        raise AssertionError(
            f"credential-named pydantic fields MUST be typed `SecretStr` "
            f"(or renamed with `_ref` suffix if they carry a "
            f"scheme://locator credential reference):\n"
            f"{rendered}"
        )


def test_walker_correctly_classifies_ref_suffix_as_safe() -> None:
    """sanity: walker logic exempts ``*_ref`` field names.

    locks in the reference-string vs secret-value distinction so a
    future edit can't silently flip the exemption.
    """
    source = (
        "class C:\n"
        "    password_ref: str\n"  # exempt: scheme://locator reference
        "    primary_key_field: str\n"  # exempt: DB primary-key concept, not credential
        "    api_key: str\n"  # SHOULD trip: trailing _key is a credential
        "    api_token: str\n"  # SHOULD trip: credential w/o SecretStr
    )
    tree = ast.parse(source)
    hits: list[tuple[int, str, str]] = []
    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue
        for body_node in class_node.body:
            if not isinstance(body_node, ast.AnnAssign):
                continue
            if not isinstance(body_node.target, ast.Name):
                continue
            name = body_node.target.id
            if name.endswith(_REF_SUFFIX):
                continue
            if not _CREDENTIAL_NAME_RE.search(name):
                continue
            if _annotation_is_secret_str(body_node.annotation):
                continue
            hits.append((body_node.lineno, name, "str"))
    assert hits == [(4, "api_key", "str"), (5, "api_token", "str")], hits
