"""AST walkers for JWT algorithm-pinning enforcement.

Each walker answers one question about a pinned module, and each maps to a failure that has
actually shipped in real systems: an algorithm read from the token, a verification check
turned off "temporarily", a weak algorithm smuggled in through a variable.
"""

from __future__ import annotations

import ast

from threetears.enforcement.common import Violation, parse_python_file

from threetears.enforcement.jwt_alg_pinning.config import JwtAlgPinningConfig, PinnedModule

__all__ = ["find_alg_pinning_violations"]

#: Decoding entry points. ``decode_complete`` matters as much as ``decode`` -- it takes the
#: identical ``algorithms``/``options`` arguments, so a walker that only knew about ``decode``
#: could be sidestepped by a one-word edit.
_DECODE_NAMES: frozenset[str] = frozenset({"decode", "decode_complete"})

#: Options that must never be disabled.
_REQUIRED_CHECKS: frozenset[str] = frozenset({"verify_signature", "verify_exp", "verify_aud"})


def find_alg_pinning_violations(config: JwtAlgPinningConfig) -> list[Violation]:
    """check every configured module and return the violations found.

    :param config: per-repo enforcement config
    :ptype config: JwtAlgPinningConfig
    :return: violations, in module then line order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    if not config.modules:
        violations.append(
            Violation(
                category="jwt_alg_pinning.no_modules",
                file=config.repo_root,
                line=0,
                symbol="(config)",
                reason="no modules configured; this shell is not enforcing anything",
            )
        )
        return violations

    for module in config.modules:
        violations.extend(_check_module(module, config.banned_algorithms))
    return violations


def _check_module(module: PinnedModule, banned: frozenset[str]) -> list[Violation]:
    if not module.path.is_file():
        return [
            Violation(
                category="jwt_alg_pinning.missing_module",
                file=module.path,
                line=0,
                symbol=module.path.name,
                reason="pinned module does not exist; the pin is silently unenforced",
            )
        ]
    tree = parse_python_file(module.path)
    if tree is None:
        return [
            Violation(
                category="jwt_alg_pinning.unparseable",
                file=module.path,
                line=0,
                symbol=module.path.name,
                reason="pinned module could not be parsed",
            )
        ]

    resolved = _resolve_constants(tree)
    violations: list[Violation] = []
    violations.extend(_check_constants(module, resolved))
    violations.extend(_check_decodes(module, tree, resolved))
    violations.extend(_check_banned_names(module, tree, banned))
    violations.extend(_check_bare_decode_import(module, tree))
    return violations


def _resolve_constants(tree: ast.Module) -> dict[str, str]:
    """module-level string constants, from annotated or plain assignment.

    Both forms are read so the pin cannot be hidden by dropping the ``Final`` annotation.
    """
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            targets: list[ast.expr] = [node.target]
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
        else:
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                found[target.id] = node.value.value
    return found


def _check_constants(module: PinnedModule, resolved: dict[str, str]) -> list[Violation]:
    violations: list[Violation] = []
    for name, expected in sorted(module.pinned_constants.items()):
        actual = resolved.get(name)
        if actual != expected:
            violations.append(
                Violation(
                    category="jwt_alg_pinning.constant",
                    file=module.path,
                    line=0,
                    symbol=name,
                    reason=f"must be the literal {expected!r}; found {actual!r}",
                )
            )
    return violations


def _decode_aliases(tree: ast.Module) -> set[str]:
    """names bound to a decode function, e.g. ``_dec = jwt.decode``.

    Without this, rebinding the entry point to a local name hides every call behind it from
    the matcher below -- and the legitimate decodes elsewhere keep the "no decode found"
    check quiet, so the module looks fully pinned while an unpinned path runs.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
            if node.value.attr in _DECODE_NAMES:
                aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            if node.value.id in _DECODE_NAMES:
                aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return aliases


def _decode_calls(tree: ast.Module) -> list[ast.Call]:
    """every decode call: ``jwt.decode(...)``, ``<alias>.decode_complete(...)``, the bare
    ``decode(...)`` form, AND any name bound to one of those, so neither switching call style
    nor rebinding the function can dodge the checks."""
    names = _DECODE_NAMES | _decode_aliases(tree)
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in names:
            calls.append(node)
        elif isinstance(func, ast.Name) and func.id in names:
            calls.append(node)
    return calls


def _kwarg(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _check_decodes(module: PinnedModule, tree: ast.Module, resolved: dict[str, str]) -> list[Violation]:
    violations: list[Violation] = []
    calls = _decode_calls(tree)
    if not calls:
        violations.append(
            Violation(
                category="jwt_alg_pinning.no_decode",
                file=module.path,
                line=0,
                symbol=module.path.name,
                reason="no decode call found; the module no longer verifies, or the walker is blind to how it does",
            )
        )
    for call in calls:
        violations.extend(_check_one_decode(module, call, resolved))
    return violations


def _check_one_decode(module: PinnedModule, call: ast.Call, resolved: dict[str, str]) -> list[Violation]:
    violations: list[Violation] = []
    algorithms = _kwarg(call, "algorithms")
    if algorithms is None:
        violations.append(
            Violation(
                category="jwt_alg_pinning.algorithms",
                file=module.path,
                line=call.lineno,
                symbol="decode",
                reason="decode must pass an explicit algorithms allow-list",
            )
        )
    elif not isinstance(algorithms, ast.List):
        violations.append(
            Violation(
                category="jwt_alg_pinning.algorithms",
                file=module.path,
                line=call.lineno,
                symbol="decode",
                reason="algorithms must be a literal list, so the allow-list is statically auditable",
            )
        )
    else:
        violations.extend(_check_algorithm_entries(module, call, algorithms, resolved))

    options = _kwarg(call, "options")
    if options is not None:
        if not isinstance(options, ast.Dict):
            # `options=SOME_NAME` or `options=dict(...)` puts the verification flags somewhere
            # this walker cannot read. That is indistinguishable from disabling them.
            violations.append(
                Violation(
                    category="jwt_alg_pinning.opaque_options",
                    file=module.path,
                    line=call.lineno,
                    symbol="options",
                    reason="decode options must be a literal dict, so the verification flags are auditable",
                )
            )
        else:
            violations.extend(_check_options_dict(module, call, options))

    for keyword in call.keywords:
        if keyword.arg not in {"verify", *_REQUIRED_CHECKS}:
            continue
        if not isinstance(keyword.value, ast.Constant):
            violations.append(
                Violation(
                    category="jwt_alg_pinning.disabled_check",
                    file=module.path,
                    line=call.lineno,
                    symbol=str(keyword.arg),
                    reason=f"{keyword.arg} must be a literal, not a name whose value this walker cannot read",
                )
            )
        elif keyword.value.value is False:
            violations.append(
                Violation(
                    category="jwt_alg_pinning.disabled_check",
                    file=module.path,
                    line=call.lineno,
                    symbol=str(keyword.arg),
                    reason=f"decode must not pass {keyword.arg}=False",
                )
            )

    # 4. Audience validation is skipped entirely by PyJWT when no `audience` is supplied, so
    #    deleting the argument is equivalent to `verify_aud=False` and must be caught too.
    if module.require_audience and _kwarg(call, "audience") is None:
        violations.append(
            Violation(
                category="jwt_alg_pinning.disabled_check",
                file=module.path,
                line=call.lineno,
                symbol="audience",
                reason="decode must pass an audience; PyJWT skips aud validation entirely without one",
            )
        )
    return violations


def _check_options_dict(module: PinnedModule, call: ast.Call, options: ast.Dict) -> list[Violation]:
    """every entry of a literal options dict, including ``**spread`` which hides its keys."""
    violations: list[Violation] = []
    for key, value in zip(options.keys, options.values, strict=True):
        if key is None:
            violations.append(
                Violation(
                    category="jwt_alg_pinning.opaque_options",
                    file=module.path,
                    line=call.lineno,
                    symbol="**",
                    reason="decode options must not be spread from another mapping; the flags become unreadable",
                )
            )
            continue
        if not (isinstance(key, ast.Constant) and key.value in _REQUIRED_CHECKS):
            continue
        if not isinstance(value, ast.Constant):
            violations.append(
                Violation(
                    category="jwt_alg_pinning.disabled_check",
                    file=module.path,
                    line=call.lineno,
                    symbol=str(key.value),
                    reason=f"{key.value} must be a literal, not a name whose value this walker cannot read",
                )
            )
        elif value.value is False:
            violations.append(
                Violation(
                    category="jwt_alg_pinning.disabled_check",
                    file=module.path,
                    line=call.lineno,
                    symbol=str(key.value),
                    reason=f"decode must not disable {key.value}",
                )
            )
    return violations


def _check_algorithm_entries(
    module: PinnedModule, call: ast.Call, algorithms: ast.List, resolved: dict[str, str]
) -> list[Violation]:
    values: set[str] = set()
    for element in algorithms.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            values.add(element.value)
        elif isinstance(element, ast.Name) and element.id in resolved:
            values.add(resolved[element.id])
        else:
            return [
                Violation(
                    category="jwt_alg_pinning.algorithms",
                    file=module.path,
                    line=call.lineno,
                    symbol="decode",
                    reason="every algorithms entry must be a string literal or a module constant, "
                    "never a value chosen at runtime",
                )
            ]
    if len(values) != 1:
        return [
            Violation(
                category="jwt_alg_pinning.algorithms",
                file=module.path,
                line=call.lineno,
                symbol="decode",
                reason=f"each decode must name exactly one algorithm; found {sorted(values)}",
            )
        ]
    unexpected = values - module.allowed_algorithms
    if unexpected:
        return [
            Violation(
                category="jwt_alg_pinning.algorithms",
                file=module.path,
                line=call.lineno,
                symbol="decode",
                reason=f"algorithm {sorted(unexpected)} is not permitted here "
                f"(allowed: {sorted(module.allowed_algorithms)})",
            )
        ]
    return []


def _check_banned_names(module: PinnedModule, tree: ast.Module, banned: frozenset[str]) -> list[Violation]:
    """belt and braces: a weak algorithm name must not appear anywhere in the module, so one
    cannot be smuggled through a variable the decode checks do not follow."""
    forbidden = banned - module.allowed_algorithms
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in forbidden:
            violations.append(
                Violation(
                    category="jwt_alg_pinning.weak_algorithm",
                    file=module.path,
                    line=node.lineno,
                    symbol=node.value,
                    reason="pinned module must not reference this algorithm name",
                )
            )
    return violations


def _check_bare_decode_import(module: PinnedModule, tree: ast.Module) -> list[Violation]:
    """``from jwt import decode`` would allow an unprefixed call. The matcher above catches
    bare-name calls, but forbidding the import keeps the audited call sites the only path."""
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "jwt":
            for alias in node.names:
                if alias.name in _DECODE_NAMES:
                    violations.append(
                        Violation(
                            category="jwt_alg_pinning.bare_import",
                            file=module.path,
                            line=node.lineno,
                            symbol=alias.name,
                            reason=f"do not import jwt.{alias.name} as a bare name; call it qualified",
                        )
                    )
    return violations
