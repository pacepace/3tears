"""Structural enforcement: :mod:`threetears.iam.tokens` must pin its signing algorithms and
never disable a verification check.

Algorithm confusion (``alg=none``, HS/RS substitution) is the canonical JWS forgery, and
``test_tokens.py`` already proves the pin holds behaviourally. This test proves a future edit
cannot quietly REMOVE it -- widen the ``algorithms`` allow-list, disable signature or expiry
verification, or repoint the pinned constants -- without tripping the build.

Two algorithms are legitimate here, unlike in a single-scheme module: EdDSA for tokens other
services verify from a JWKS, HS256 for a service that both mints and verifies its own. What
is enforced is that each decode names exactly ONE of them, from a module constant, so the
algorithm can never be chosen by anything the token itself carries.
"""

from __future__ import annotations

import ast
from pathlib import Path

_MODULE = Path(__file__).resolve().parents[2] / "src" / "threetears" / "iam" / "tokens.py"

#: The pinned algorithm constants, by name, and the only values they may hold.
_PINNED_CONSTANTS = {"_EDDSA": "EdDSA", "_HS256": "HS256"}

#: Algorithm names this module must never mention. HS256 is absent deliberately -- it is a
#: supported scheme here. HS384/HS512 are banned because nothing should be reaching for a
#: wider HMAC family than the one pinned constant provides.
_BANNED_ALGORITHMS = {
    "none",
    "HS384",
    "HS512",
    "RS256",
    "RS384",
    "RS512",
    "ES256",
    "ES256K",
    "ES384",
    "ES512",
    "PS256",
    "PS384",
    "PS512",
}


def _tree() -> ast.AST:
    return ast.parse(_MODULE.read_text())


def _pinned_values(tree: ast.AST) -> dict[str, str]:
    """Resolve the module's algorithm constants from their AST assignments.

    Handles the annotated form (``_EDDSA: Final[str] = "EdDSA"``) as well as a plain
    assignment, so the pin cannot be hidden by dropping the annotation.
    """
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            targets: list[ast.expr] = [node.target]
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
        else:
            continue
        if node.value is None or not isinstance(node.value, ast.Constant):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in _PINNED_CONSTANTS:
                found[target.id] = node.value.value
    return found


def _decode_calls(tree: ast.AST) -> list[ast.Call]:
    """Every decode call node: ``jwt.decode(...)`` / ``<alias>.decode(...)`` AND a bare
    ``decode(...)``, so a future edit cannot dodge the assertions below by switching to the
    bare-name form."""
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "decode":
            calls.append(node)
        elif isinstance(func, ast.Name) and func.id == "decode":
            calls.append(node)
    return calls


def _kwarg(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def test_module_exists() -> None:
    assert _MODULE.is_file(), f"tokens module missing at {_MODULE}"


def test_algorithm_constants_are_hard_pinned() -> None:
    """The constants' literal values, read from the AST rather than imported, so the check
    cannot be satisfied by anything computed at runtime."""
    assert _pinned_values(_tree()) == _PINNED_CONSTANTS


def test_every_decode_pins_exactly_one_allowed_algorithm() -> None:
    tree = _tree()
    pinned = _pinned_values(tree)
    calls = _decode_calls(tree)
    assert calls, "expected at least one jwt.decode call in the verifiers"
    for call in calls:
        algorithms = _kwarg(call, "algorithms")
        assert algorithms is not None, "jwt.decode must pass an explicit algorithms allow-list"
        assert isinstance(algorithms, ast.List), "algorithms must be a literal list (statically auditable)"
        # Each element is either a string literal or one of the pinned module constants;
        # anything else (a parameter, an attribute lookup, a dict read) would mean the
        # algorithm could be chosen at runtime, which is the whole thing being prevented.
        values: set[str] = set()
        for element in algorithms.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                values.add(element.value)
            elif isinstance(element, ast.Name) and element.id in pinned:
                values.add(pinned[element.id])
            else:
                raise AssertionError(f"algorithms entry {ast.dump(element)} is not a statically pinned name")
        assert len(values) == 1, f"each decode must name exactly one algorithm; found {values}"
        assert values <= set(_PINNED_CONSTANTS.values()), f"unexpected algorithm {values}"


def test_no_decode_disables_signature_or_expiry() -> None:
    for call in _decode_calls(_tree()):
        options = _kwarg(call, "options")
        if isinstance(options, ast.Dict):
            for key, value in zip(options.keys, options.values, strict=True):
                if isinstance(key, ast.Constant) and key.value in {"verify_signature", "verify_exp"}:
                    assert not (isinstance(value, ast.Constant) and value.value is False), (
                        f"decode must not disable {key.value}"
                    )
        for keyword in call.keywords:
            if keyword.arg in {"verify", "verify_signature", "verify_exp"}:
                assert not (isinstance(keyword.value, ast.Constant) and keyword.value.value is False), (
                    f"decode must not pass {keyword.arg}=False"
                )


def test_module_never_names_a_weak_algorithm() -> None:
    """Belt and braces: the source must not contain a weak algorithm name anywhere, so one
    cannot be smuggled in through a variable the checks above do not follow."""
    found = {
        node.value
        for node in ast.walk(_tree())
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in _BANNED_ALGORITHMS
    }
    assert not found, f"tokens.py must not reference weak algorithms: {sorted(found)}"


def test_jwt_decode_is_not_imported_as_a_bare_name() -> None:
    """``from jwt import decode`` would let a future edit call ``decode(...)`` unprefixed.
    The matcher above catches bare-name calls, but forbidding the import outright keeps the
    audited ``jwt.decode`` call sites the only decode path."""
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module == "jwt":
            imported = {alias.name for alias in node.names}
            assert "decode" not in imported, "do not import jwt.decode as a bare name; call jwt.decode"
