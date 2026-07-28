"""The JWT alg-pinning walkers, driven by synthesised modules.

Every test here is a bypass someone could actually write. The gate previously had no tests at
all, and an adversarial review then produced seven one-line edits that defeated it -- each of
those is now a case below, because a gate whose holes are unknown is worse than no gate: it
makes the next review LOOK clean.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.jwt_alg_pinning import (
    JwtAlgPinningConfig,
    PinnedModule,
    find_alg_pinning_violations,
)

_CLEAN = """
import jwt

_EDDSA = "EdDSA"


def verify(token, key, issuer, audience):
    return jwt.decode(
        token,
        key=key,
        algorithms=[_EDDSA],
        issuer=issuer,
        audience=audience,
        options={"require": ["iss", "aud"]},
    )
"""


def _module(tmp_path: Path, source: str, **overrides: object) -> JwtAlgPinningConfig:
    path = tmp_path / "subject.py"
    path.write_text(source)
    kwargs: dict[str, object] = {
        "path": path,
        "allowed_algorithms": frozenset({"EdDSA"}),
        "pinned_constants": {"_EDDSA": "EdDSA"},
        "require_audience": True,
    }
    kwargs.update(overrides)
    return JwtAlgPinningConfig(repo_root=tmp_path, modules=(PinnedModule(**kwargs),))  # type: ignore[arg-type]


def _categories(config: JwtAlgPinningConfig) -> set[str]:
    return {v.category for v in find_alg_pinning_violations(config)}


def test_a_correctly_pinned_module_passes(tmp_path: Path) -> None:
    """The positive control. Without it, every test below could pass on a walker that
    reports everything."""
    assert find_alg_pinning_violations(_module(tmp_path, _CLEAN)) == []


def test_an_empty_module_list_is_itself_a_violation(tmp_path: Path) -> None:
    """A shell that resolves no modules has silently stopped enforcing anything."""
    config = JwtAlgPinningConfig(repo_root=tmp_path, modules=())
    assert "jwt_alg_pinning.no_modules" in _categories(config)


def test_a_missing_module_is_a_violation(tmp_path: Path) -> None:
    config = JwtAlgPinningConfig(
        repo_root=tmp_path,
        modules=(PinnedModule(path=tmp_path / "gone.py", allowed_algorithms=frozenset({"EdDSA"})),),
    )
    assert "jwt_alg_pinning.missing_module" in _categories(config)


def test_a_module_with_no_decode_is_a_violation(tmp_path: Path) -> None:
    """Either it stopped verifying, or the walker has gone blind to how it does."""
    assert "jwt_alg_pinning.no_decode" in _categories(_module(tmp_path, "import jwt\n"))


# --- algorithm pinning ------------------------------------------------------------------


def test_a_widened_allow_list_is_caught(tmp_path: Path) -> None:
    source = _CLEAN.replace("algorithms=[_EDDSA],", 'algorithms=[_EDDSA, "HS256"],')
    assert "jwt_alg_pinning.algorithms" in _categories(_module(tmp_path, source))


def test_a_repointed_constant_is_caught(tmp_path: Path) -> None:
    source = _CLEAN.replace('_EDDSA = "EdDSA"', '_EDDSA = "HS256"')
    assert "jwt_alg_pinning.constant" in _categories(_module(tmp_path, source))


def test_algorithms_from_a_runtime_value_is_caught(tmp_path: Path) -> None:
    """The whole point: the algorithm must never come from anything the token influences."""
    source = _CLEAN.replace("algorithms=[_EDDSA],", "algorithms=[header['alg']],")
    assert "jwt_alg_pinning.algorithms" in _categories(_module(tmp_path, source))


def test_a_non_literal_algorithms_list_is_caught(tmp_path: Path) -> None:
    source = _CLEAN.replace("algorithms=[_EDDSA],", "algorithms=permitted,")
    assert "jwt_alg_pinning.algorithms" in _categories(_module(tmp_path, source))


def test_a_weak_algorithm_named_anywhere_is_caught(tmp_path: Path) -> None:
    source = _CLEAN.replace('_EDDSA = "EdDSA"', '_EDDSA = "EdDSA"\n_FALLBACK = "none"')
    assert "jwt_alg_pinning.weak_algorithm" in _categories(_module(tmp_path, source))


# --- disabled checks: the seven proven bypasses -------------------------------------------


def test_a_literal_disabled_check_is_caught(tmp_path: Path) -> None:
    source = _CLEAN.replace('options={"require": ["iss", "aud"]},', 'options={"verify_signature": False},')
    assert "jwt_alg_pinning.disabled_check" in _categories(_module(tmp_path, source))


def test_options_hidden_behind_a_name_is_caught(tmp_path: Path) -> None:
    """`_OPTS = {"verify_signature": False}` then `options=_OPTS` -- the flags move somewhere
    the walker cannot read, which is indistinguishable from disabling them."""
    source = _CLEAN.replace('_EDDSA = "EdDSA"', '_EDDSA = "EdDSA"\n_OPTS = {"verify_signature": False}').replace(
        'options={"require": ["iss", "aud"]},', "options=_OPTS,"
    )
    assert "jwt_alg_pinning.opaque_options" in _categories(_module(tmp_path, source))


def test_options_built_by_a_call_is_caught(tmp_path: Path) -> None:
    source = _CLEAN.replace('options={"require": ["iss", "aud"]},', "options=dict(verify_signature=False),")
    assert "jwt_alg_pinning.opaque_options" in _categories(_module(tmp_path, source))


def test_options_spread_from_another_mapping_is_caught(tmp_path: Path) -> None:
    source = _CLEAN.replace('_EDDSA = "EdDSA"', '_EDDSA = "EdDSA"\n_BASE = {"verify_signature": False}').replace(
        'options={"require": ["iss", "aud"]},', 'options={**_BASE, "require": ["iss"]},'
    )
    assert "jwt_alg_pinning.opaque_options" in _categories(_module(tmp_path, source))


def test_a_disabled_check_via_a_name_is_caught(tmp_path: Path) -> None:
    """`_OFF = False` then `verify_signature=_OFF`."""
    source = _CLEAN.replace('_EDDSA = "EdDSA"', '_EDDSA = "EdDSA"\n_OFF = False').replace(
        "issuer=issuer,", "issuer=issuer,\n        verify_signature=_OFF,"
    )
    assert "jwt_alg_pinning.disabled_check" in _categories(_module(tmp_path, source))


def test_a_disabled_check_in_the_options_dict_via_a_name_is_caught(tmp_path: Path) -> None:
    source = _CLEAN.replace('_EDDSA = "EdDSA"', '_EDDSA = "EdDSA"\n_OFF = False').replace(
        'options={"require": ["iss", "aud"]},', 'options={"verify_exp": _OFF},'
    )
    assert "jwt_alg_pinning.disabled_check" in _categories(_module(tmp_path, source))


def test_a_missing_audience_is_caught_when_required(tmp_path: Path) -> None:
    """PyJWT skips audience validation entirely when no `audience` is supplied, so deleting
    the argument is equivalent to `verify_aud=False`."""
    source = _CLEAN.replace("        audience=audience,\n", "")
    assert "jwt_alg_pinning.disabled_check" in _categories(_module(tmp_path, source))


def test_a_missing_audience_is_allowed_for_a_format_that_has_none(tmp_path: Path) -> None:
    """DPoP proofs carry no `aud` at all; requiring one there would be noise."""
    source = _CLEAN.replace("        audience=audience,\n", "")
    assert find_alg_pinning_violations(_module(tmp_path, source, require_audience=False)) == []


# --- reaching decode by another name ------------------------------------------------------


def test_decode_complete_is_checked_too(tmp_path: Path) -> None:
    source = _CLEAN.replace("jwt.decode(", "jwt.decode_complete(").replace(
        'options={"require": ["iss", "aud"]},', 'options={"verify_signature": False},'
    )
    assert "jwt_alg_pinning.disabled_check" in _categories(_module(tmp_path, source))


def test_decode_reached_through_an_alias_is_checked(tmp_path: Path) -> None:
    """`_dec = jwt.decode` then `_dec(...)`. The legitimate decode elsewhere keeps the
    "no decode found" check quiet, so without this the module looks fully pinned."""
    source = (
        _CLEAN
        + """

_dec = jwt.decode


def sneaky(token, key):
    return _dec(token, key=key, algorithms=["none"])
"""
    )
    categories = _categories(_module(tmp_path, source))
    assert "jwt_alg_pinning.algorithms" in categories or "jwt_alg_pinning.weak_algorithm" in categories


def test_a_bare_decode_import_is_caught(tmp_path: Path) -> None:
    source = _CLEAN.replace("import jwt", "import jwt\nfrom jwt import decode")
    assert "jwt_alg_pinning.bare_import" in _categories(_module(tmp_path, source))


@pytest.mark.parametrize("name", ["decode", "decode_complete"])
def test_a_bare_import_of_either_entry_point_is_caught(tmp_path: Path, name: str) -> None:
    source = _CLEAN.replace("import jwt", f"import jwt\nfrom jwt import {name}")
    assert "jwt_alg_pinning.bare_import" in _categories(_module(tmp_path, source))
