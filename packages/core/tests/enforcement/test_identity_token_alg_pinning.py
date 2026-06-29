"""Enforcement test: the identity-token verifier must pin EdDSA and never disable checks.

Algorithm confusion (``alg=none``, HS/RS substitution) is the canonical JWS forgery. The
defence is structural, not behavioural: every ``jwt.decode`` in the identity-token module
MUST pass ``algorithms=["EdDSA"]`` and MUST NOT disable signature/expiry verification. A
unit test proves it holds today; this AST test proves a future edit can't quietly remove the
pin (e.g. widen ``algorithms`` to include ``"none"``/``"HS256"`` or pass
``verify_signature=False``) without tripping the build.
"""

from __future__ import annotations

import ast
from pathlib import Path

_MODULE = (
    Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "core" / "security" / "identity_token.py"
)

_ALLOWED_ALGS = {"EdDSA"}


def _decode_calls(tree: ast.AST) -> list[ast.Call]:
    """every decode call node: ``jwt.decode(...)`` / ``<alias>.decode(...)`` AND a bare

    ``decode(...)`` (e.g. via ``from jwt import decode``) — so a future edit can't dodge the
    algorithm-pin assertion by switching to the bare-name form.
    """
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
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def test_module_exists() -> None:
    assert _MODULE.is_file(), f"identity_token module missing at {_MODULE}"


def test_every_decode_pins_eddsa_algorithms() -> None:
    tree = ast.parse(_MODULE.read_text())
    calls = _decode_calls(tree)
    assert calls, "expected at least one jwt.decode call in the verifier"
    for call in calls:
        algs = _kwarg(call, "algorithms")
        assert algs is not None, "jwt.decode must pass an explicit algorithms allow-list"
        assert isinstance(algs, ast.List), "algorithms must be a literal list (statically auditable)"
        values = {elt.value for elt in algs.elts if isinstance(elt, ast.Constant)}
        assert len(values) == len(algs.elts), "algorithms list must be string literals only"
        assert values <= _ALLOWED_ALGS, f"only EdDSA may be allowed; found {values}"
        assert values, "algorithms list must not be empty"


def test_no_decode_disables_signature_or_expiry() -> None:
    tree = ast.parse(_MODULE.read_text())
    for call in _decode_calls(tree):
        # no verify_signature=False (in options dict or as a kwarg)
        opts = _kwarg(call, "options")
        if isinstance(opts, ast.Dict):
            for k, v in zip(opts.keys, opts.values):
                if isinstance(k, ast.Constant) and k.value in {"verify_signature", "verify_exp"}:
                    assert not (isinstance(v, ast.Constant) and v.value is False), (
                        f"identity-token decode must not disable {k.value}"
                    )
        for kw in call.keywords:
            if kw.arg in {"verify", "verify_signature", "verify_exp"}:
                assert not (isinstance(kw.value, ast.Constant) and kw.value.value is False), (
                    f"identity-token decode must not pass {kw.arg}=False"
                )


def test_module_never_names_a_weak_algorithm() -> None:
    # belt-and-suspenders: the source must not contain "none"/HS*/RS*/ES*/PS* as an algorithm
    # string anywhere (e.g. smuggled via a variable). EdDSA is the only algorithm name allowed.
    source = _MODULE.read_text()
    tree = ast.parse(source)
    banned = {
        "none",
        "HS256",
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
    found = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in banned
    }
    assert not found, f"identity-token module must not reference weak algorithms: {found}"


def test_jwt_decode_is_not_imported_as_a_bare_name() -> None:
    # ``from jwt import decode`` would let a future edit call ``decode(...)`` without the
    # ``jwt.`` prefix; the matcher above now catches bare-name calls, but forbidding the import
    # outright keeps the single audited ``jwt.decode`` call site the only decode path.
    tree = ast.parse(_MODULE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "jwt":
            imported = {alias.name for alias in node.names}
            assert "decode" not in imported, "do not import jwt.decode as a bare name; call jwt.decode"
