"""KV bucket open discipline (coll-task-04a).

Two rules, each pinning a decision whose violation is invisible at runtime.

(a) **Nothing opens a KV bucket outside the reconcile primitive.** ``kv.py`` is the
    sanctioned opener: it builds the KV stream shape itself, reconciles the fields
    that must not drift, and reports the ones it deliberately does not. A direct
    ``js.create_key_value(...)`` / ``js.key_value(...)`` elsewhere gets none of
    that, and the failure is silent in the worst way -- the bucket opens, works,
    and carries somebody else's configuration. That is exactly what the platform
    had: a second opener asking for ``ttl=7200s`` against a bucket created with
    ``ttl=60s`` got a handle REPORTING ``2:00:00`` while the server still said 60,
    with a ``log.debug`` as the only trace.

(b) **``KvConfigMismatch`` never appears in an ``except`` clause beside ``KvError``.**
    ``KvError`` is what ``BaseCollection``'s L2 accessors catch and degrade on, and
    bucket resolution sits inside those catches. Widening one of them to cover the
    mismatch reads as defensive tidying and restores the exact silent degradation
    the distinct type exists to prevent: the fleet runs with L2 off, at WARNING,
    and the misconfigured bucket is never mentioned. It is the one specific way
    this fix gets undone without anyone noticing.

Mode via ``KV_OPEN_ENFORCEMENT_MODE`` (default ``strict``), mirroring the sibling
guards. Exemptions live in ``_kv_open_exemptions.txt`` and require a specific
``# rationale:`` line.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from threetears.enforcement.common import (
    MODE_REPORT,
    MODE_STRICT,
    Violation,
    apply_exemptions,
    emit_report,
    find_local_src_roots,
    iter_python_files,
    parse_exemptions_with_rationale,
    parse_python_file,
    resolve_mode,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXEMPTIONS_PATH = _REPO_ROOT / "tests" / "enforcement" / "_kv_open_exemptions.txt"
_MODE_ENV_VAR = "KV_OPEN_ENFORCEMENT_MODE"

_UNSANCTIONED_OPEN_CATEGORY = "kv_open.outside_the_reconcile_primitive"
_MISMATCH_SWALLOW_CATEGORY = "kv_open.mismatch_caught_with_kv_error"

#: the nats-py methods that open a KV bucket. both are wrapped by the primitive.
_KV_OPENERS: frozenset[str] = frozenset({"create_key_value", "key_value"})

#: the ONE module allowed to call them: it is the reconcile primitive.
#:
#: A path suffix rather than a module name, so a copy of the logic under a
#: different package does not inherit the exemption by being called ``kv.py``.
_SANCTIONED_MODULE = "threetears/nats/kv.py"

#: the mismatch type, and its module-qualified spelling.
_MISMATCH_NAMES: frozenset[str] = frozenset({"KvConfigMismatch"})

#: the transport error whose handlers degrade to a warning.
_DEGRADING_ERROR = "KvError"


def _called_name(func: ast.expr) -> str | None:
    """the bare callee name of a call target, however it was reached.

    :param func: the ``ast.Call.func`` node under inspection
    :ptype func: ast.expr
    :return: the callee's final name segment, or ``None`` when it is not a plain reference
    :rtype: str | None
    """
    result: str | None = None
    if isinstance(func, ast.Name):
        result = func.id
    elif isinstance(func, ast.Attribute):
        result = func.attr
    return result


def find_unsanctioned_kv_opens(scan_roots: tuple[Path, ...]) -> list[Violation]:
    """flag every direct KV-bucket open outside the reconcile primitive.

    :param scan_roots: src roots to scan for violations
    :ptype scan_roots: tuple[Path, ...]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in scan_roots:
        for source in iter_python_files(root):
            if source.as_posix().endswith(_SANCTIONED_MODULE):
                continue
            tree = parse_python_file(source)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _called_name(node.func)
                if name not in _KV_OPENERS:
                    continue
                violations.append(
                    Violation(
                        category=_UNSANCTIONED_OPEN_CATEGORY,
                        file=source,
                        line=node.lineno,
                        symbol=name,
                        reason=(
                            f"'{name}' opens a KV bucket directly, bypassing the create-or-reconcile "
                            "primitive; the requested bucket configuration is then silently discarded "
                            "whenever the bucket already exists. open it through "
                            "NatsClient.kv_bucket / NatsClient.ensure_kv_bucket instead"
                        ),
                    )
                )
    return violations


def _handler_names(handler: ast.ExceptHandler) -> set[str]:
    """the exception type names one ``except`` clause catches.

    :param handler: the except-handler node
    :ptype handler: ast.ExceptHandler
    :return: caught type names, attribute-qualified ones reduced to their final segment
    :rtype: set[str]
    """
    if handler.type is None:
        return set()
    candidates = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    names: set[str] = set()
    for candidate in candidates:
        name = _called_name(candidate)
        if name is not None:
            names.add(name)
    return names


def find_mismatch_caught_with_kv_error(scan_roots: tuple[Path, ...]) -> list[Violation]:
    """flag any ``except`` clause catching the mismatch alongside ``KvError``.

    :param scan_roots: src roots to scan for violations
    :ptype scan_roots: tuple[Path, ...]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in scan_roots:
        for source in iter_python_files(root):
            tree = parse_python_file(source)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                caught = _handler_names(node)
                shared = caught & _MISMATCH_NAMES
                if not shared or _DEGRADING_ERROR not in caught:
                    continue
                violations.append(
                    Violation(
                        category=_MISMATCH_SWALLOW_CATEGORY,
                        file=source,
                        line=node.lineno,
                        symbol=sorted(shared)[0],
                        reason=(
                            f"'{sorted(shared)[0]}' is caught alongside {_DEGRADING_ERROR}; those "
                            "handlers degrade to a warning, so a misconfigured bucket would be "
                            "swallowed and the fleet would run with L2 silently off"
                        ),
                    )
                )
    return violations


def _assert_clean(violations: list[Violation], domain: str) -> None:
    """apply exemptions + mode and fail in strict mode with a rendered report.

    :param violations: raw walker output
    :ptype violations: list[Violation]
    :param domain: report domain label
    :ptype domain: str
    :return: nothing
    :rtype: None
    :raises pytest.fail.Exception: in strict mode with surviving violations
    """
    exemptions = parse_exemptions_with_rationale(_EXEMPTIONS_PATH)
    filtered = apply_exemptions(violations, exemptions, _REPO_ROOT)
    mode = resolve_mode(_MODE_ENV_VAR, default=MODE_STRICT)
    report = emit_report(filtered, (_REPO_ROOT,), exemptions, mode, _REPO_ROOT, domain=domain)
    print(report, file=sys.stderr)
    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(f"{domain} found {len(filtered)} violation(s):\n{report}")


class TestKvOpenDiscipline:
    """the two rules over the 3tears package tree."""

    def test_no_production_code_opens_a_bucket_outside_the_primitive(self) -> None:
        violations = find_unsanctioned_kv_opens(find_local_src_roots(_REPO_ROOT))
        _assert_clean(violations, _UNSANCTIONED_OPEN_CATEGORY)

    def test_the_mismatch_is_never_caught_with_kv_error(self) -> None:
        violations = find_mismatch_caught_with_kv_error(find_local_src_roots(_REPO_ROOT))
        _assert_clean(violations, _MISMATCH_SWALLOW_CATEGORY)


class TestWalkersFlagPlantedViolations:
    """each walker must fire on a planted violation (self-test of the guard).

    A guard whose walker silently matches nothing reads exactly like a guard with
    nothing to report -- the same failure mode the enforcement suite exists to
    catch elsewhere. Each rule is therefore exercised against source that SHOULD
    trip it, and against source that should not, since a walker that flags
    everything is equally useless.
    """

    def test_open_walker_flags_a_direct_create(self, tmp_path: Path) -> None:
        source = (
            "async def start(nc, bucket):\n"
            "    js = nc.jetstream_context()\n"
            "    return await js.create_key_value(bucket=bucket)\n"
        )
        (tmp_path / "planted_create.py").write_text(source, encoding="utf-8")

        violations = find_unsanctioned_kv_opens((tmp_path,))

        assert [v.symbol for v in violations] == ["create_key_value"]

    def test_open_walker_flags_a_direct_bind(self, tmp_path: Path) -> None:
        source = "async def start(js, bucket):\n    return await js.key_value(bucket)\n"
        (tmp_path / "planted_bind.py").write_text(source, encoding="utf-8")

        violations = find_unsanctioned_kv_opens((tmp_path,))

        assert [v.symbol for v in violations] == ["key_value"]

    def test_open_walker_accepts_the_wrapper(self, tmp_path: Path) -> None:
        source = (
            "async def start(nc):\n"
            "    await nc.ensure_kv_bucket(name='collections', direct=True)\n"
            "    return await nc.kv_bucket(name='collections')\n"
        )
        (tmp_path / "sanctioned_open.py").write_text(source, encoding="utf-8")

        assert find_unsanctioned_kv_opens((tmp_path,)) == []

    def test_mismatch_walker_flags_a_widened_handler(self, tmp_path: Path) -> None:
        source = (
            "from threetears.nats.errors import KvConfigMismatch, KvError\n\n\n"
            "async def read(self, entity_id):\n"
            "    try:\n"
            "        kv = await self._ensure_kv()\n"
            "        return await kv.get(key=self.l2_key(entity_id))\n"
            "    except (KvError, KvConfigMismatch):\n"
            "        return None\n"
        )
        (tmp_path / "planted_handler.py").write_text(source, encoding="utf-8")

        violations = find_mismatch_caught_with_kv_error((tmp_path,))

        assert [v.symbol for v in violations] == ["KvConfigMismatch"]

    def test_mismatch_walker_accepts_a_kv_error_only_handler(self, tmp_path: Path) -> None:
        source = (
            "from threetears.nats.errors import KvError\n\n\n"
            "async def read(self, entity_id):\n"
            "    try:\n"
            "        kv = await self._ensure_kv()\n"
            "        return await kv.get(key=self.l2_key(entity_id))\n"
            "    except KvError:\n"
            "        return None\n"
        )
        (tmp_path / "sanctioned_handler.py").write_text(source, encoding="utf-8")

        assert find_mismatch_caught_with_kv_error((tmp_path,)) == []
