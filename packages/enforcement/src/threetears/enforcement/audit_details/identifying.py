"""no producer passes a details key that names a person: audit rows hold ids (Pace, 2026-11-11).

``threetears.agent.audit`` refuses such a key when an event is published or stored (``IdentifyingAuditDetail``); this
gate finds it where it is written, before a run does: every ``details=`` an audit event is built with
(:data:`~threetears.enforcement.audit_details.config.DEFAULT_AUDIT_CONSTRUCTORS`), every repo wrapper that takes
``details=`` and hands it on (declared ``forwarders``, and every ``emit_*`` function with a ``details`` parameter), read
by the same walker as the classification gate, every key at every depth. A key the event type's family is allowed (a
workspace's own ``name``) passes; an unreadable ``details`` is the classification gate's finding.

The rule's data is injected, as the classification gate's is: a shell passes
``threetears.agent.audit.identifying_keys_for``. A consumer shell::

    from threetears.agent.audit import identifying_keys_for
    from threetears.enforcement.audit_details import IdentifyingDetailsConfig, identifying_detail_findings
    from threetears.enforcement.common import find_local_src_roots

    def test_no_producer_passes_an_identifying_key() -> None:
        config = IdentifyingDetailsConfig(
            repo_root=REPO_ROOT, src_roots=find_local_src_roots(REPO_ROOT), identifying_keys_for=identifying_keys_for
        )
        assert identifying_detail_findings(config) == []
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from threetears.enforcement.common import iter_python_files, parse_python_file

from threetears.enforcement.audit_details.config import DEFAULT_AUDIT_CONSTRUCTORS, AuditDetailsConfig
from threetears.enforcement.audit_details.walkers import collect_audit_details_sites

__all__ = ["IdentifyingDetailsConfig", "identifying_detail_findings"]

#: a repo function whose name starts so and which takes ``details`` is read as a wrapper handing it to an event
_EMIT_PREFIX = "emit_"


@dataclass(frozen=True)
class IdentifyingDetailsConfig:
    """per-repo config for the "no producer passes an identifying key" gate.

    :ivar repo_root: the consumer repo's root
    :ivar src_roots: the source trees to scan
    :ivar identifying_keys_for: the keys refused for an event type (its family's allowances taken out); pass
        ``threetears.agent.audit.identifying_keys_for``. Called with ``""`` for a construction whose event type does
        not resolve, which allows nothing
    :ivar constructors: callee names that build an audit event
    :ivar forwarders: the repo's own wrappers that take ``details=`` and hand it on, beside every ``emit_*`` function
        with a ``details`` parameter, found on their own
    """

    repo_root: Path
    src_roots: tuple[Path, ...]
    identifying_keys_for: Callable[[str], frozenset[str]]
    constructors: frozenset[str] = DEFAULT_AUDIT_CONSTRUCTORS
    forwarders: frozenset[str] = frozenset()


def _emitters(src_roots: tuple[Path, ...]) -> frozenset[str]:
    """every ``emit_*`` function under the roots that takes a ``details`` parameter."""
    found: set[str] = set()
    for root in src_roots:
        for path in iter_python_files(root):
            tree = parse_python_file(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(_EMIT_PREFIX):
                    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                    if any(argument.arg == "details" for argument in arguments):
                        found.add(node.name)
    return frozenset(found)


def identifying_detail_findings(config: IdentifyingDetailsConfig) -> list[str]:
    """each details key, at any depth, that names a person for the event type(s) its site resolves to.

    :param config: per-repo config
    :ptype config: IdentifyingDetailsConfig
    :return: ``<path>:<line>: details key <dotted path> names a person (<event types>) -- ...``, sorted; empty
        when no producer passes one
    :rtype: list[str]
    """
    if not config.src_roots:
        return [f"{config.repo_root}: no source roots configured; this gate is not enforcing anything"]
    reading = AuditDetailsConfig(
        repo_root=config.repo_root,
        src_roots=config.src_roots,
        safe_keys_for=lambda _event_type: frozenset(),
        is_classified=lambda _key, *, event_type: True,
        constructors=config.constructors,
        forwarders=config.forwarders | _emitters(config.src_roots),
    )
    findings: list[str] = []
    for path, sites in collect_audit_details_sites(reading).items():
        relative = path.relative_to(config.repo_root) if path.is_relative_to(config.repo_root) else path
        for site in sites:
            event_types = sorted(site.event_types) or [""]
            for key_path in sorted(site.keys):
                named = [
                    key
                    for key in key_path
                    if any(key in config.identifying_keys_for(event_type) for event_type in event_types)
                ]
                if named:
                    findings.append(
                        f"{relative}:{site.lineno}: details key {'.'.join(key_path)} names a person "
                        f"({', '.join(event_type or '<unresolved>' for event_type in event_types)}) -- an audit row "
                        "holds ids: write the id, or allow the key for a family whose value there is no person's "
                        "(threetears.agent.audit.anonymize._BUILT_IN_FAMILY_IDENTIFYING_ALLOWED)"
                    )
    return sorted(set(findings))
