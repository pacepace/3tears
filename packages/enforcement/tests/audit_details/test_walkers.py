"""The audit-details walker, driven by synthesised modules.

Every case is a shape a producer actually writes -- or a shape that would slip a key past
the classification if the walker narrowed. The walker passes the 3tears tree as it stands,
so a reader that found nothing would pass it too; these cases are what make a narrowed
reader fail.

The classification here is synthetic (a fixed safe set, one family declaration, a fixed
personal set) so the walker is tested without the audit package it is normally handed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from threetears.enforcement.audit_details import (
    AuditDetailsConfig,
    AuditDetailsSite,
    collect_audit_details_sites,
    find_audit_details_violations,
    read_audit_details_sites,
    unclassified_detail_paths,
)

_SAFE = frozenset({"tool_name", "files_changed", "platform_repointed", "member_count"})
_PERSONAL = frozenset({"failure_reason", "value"})
_FAMILY = "gatesynthetic"
_FAMILY_KEYS = frozenset({"gate_declared_count"})


def _safe_keys_for(event_type: str) -> frozenset[str]:
    """
    the synthetic safe lookup: the platform set, plus one family's key under its prefix.

    :param event_type: dotted event type
    :ptype event_type: str
    :return: the safe keys for it
    :rtype: frozenset[str]
    """
    in_family = event_type == _FAMILY or event_type.startswith(f"{_FAMILY}.")
    return _SAFE | _FAMILY_KEYS if in_family else _SAFE


def _read(source: str, *, forwarders: frozenset[str] = frozenset()) -> AuditDetailsSite:
    """
    read the single site in a synthetic module.

    :param source: module source holding exactly one audit call
    :ptype source: str
    :param forwarders: wrapper helpers to declare
    :ptype forwarders: frozenset[str]
    :return: its site
    :rtype: AuditDetailsSite
    """
    (site,) = read_audit_details_sites(ast.parse(source), forwarders=forwarders)
    return site


def _unclassified(site: AuditDetailsSite) -> list[str]:
    """
    the unclassified key paths at a site, dotted.

    :param site: a read site
    :ptype site: AuditDetailsSite
    :return: dotted paths
    :rtype: list[str]
    """
    return [
        ".".join(path)
        for path in unclassified_detail_paths(site, safe_keys_for=_safe_keys_for, personal_keys=_PERSONAL)
    ]


class TestKeysAreRead:
    """every literal-keyed way of building details is followed."""

    def test_a_fully_classified_literal_is_clean(self) -> None:
        """the positive control: without it every case below could pass on a walker that reports everything."""
        site = _read("AuditEvent(event_type='x.y', details={'tool_name': t, 'failure_reason': r})")

        assert _unclassified(site) == []
        assert site.unreadable == ()

    def test_an_unclassified_literal_key_is_reported(self) -> None:
        """the base case: a new key in a literal."""
        site = _read("AuditEvent(event_type='x.y', details={'tool_name': t, 'mystery_key': m})")

        assert _unclassified(site) == ["mystery_key"]
        assert site.unreadable == ()

    def test_a_key_added_by_subscript_is_reported(self) -> None:
        """the tool server's shape: a literal, then one key added conditionally."""
        site = _read(
            "def f():\n"
            "    details = {'tool_name': t}\n"
            "    if r:\n"
            "        details['mystery_key'] = r\n"
            "    AuditEvent(event_type='x.y', details=details)\n"
        )

        assert _unclassified(site) == ["mystery_key"]

    def test_keys_added_by_update_and_setdefault_are_reported(self) -> None:
        """update with a literal and with keywords, setdefault, and ``|=``."""
        site = _read(
            "def f():\n"
            "    details: dict[str, object] = {}\n"
            "    details.update({'mystery_a': 1}, mystery_b=2)\n"
            "    details.setdefault('mystery_c', 3)\n"
            "    details |= {'mystery_d': 4}\n"
            "    AuditEvent(event_type='x.y', details=details)\n"
        )

        assert _unclassified(site) == ["mystery_a", "mystery_b", "mystery_c", "mystery_d"]

    def test_an_attribute_qualified_constructor_is_found(self) -> None:
        """``audit.AuditEvent(...)`` is the same construction, and ``dict(k=v)`` a literal."""
        site = _read("audit.AuditEvent(event_type='x.y', details=dict(mystery_key=1))")

        assert _unclassified(site) == ["mystery_key"]

    def test_a_construction_without_details_carries_no_keys(self) -> None:
        """an event with no details is readable and empty."""
        site = _read("AuditEvent(event_type='x.y', action='a')")

        assert site.keys == frozenset()
        assert site.unreadable == ()


class TestNestedKeys:
    """nested literal keys are judged exactly where the anonymization rule consults them."""

    def test_a_nested_key_under_a_safe_key_is_reported(self) -> None:
        """a dict under a safe key is judged key by key, so its keys must be classified."""
        site = _read(
            "AuditEvent(event_type='x.y', details={'platform_repointed': {'member_count': 1, 'mystery_key': 2}})"
        )

        assert _unclassified(site) == ["platform_repointed.mystery_key"]

    def test_a_nested_key_in_a_list_under_a_safe_key_is_reported(self) -> None:
        """a list does not add a path segment; the dict inside it is judged by its own keys."""
        site = _read("AuditEvent(event_type='x.y', details={'files_changed': [{'mystery_key': 1}]})")

        assert _unclassified(site) == ["files_changed.mystery_key"]

    def test_a_nested_key_under_an_unsafe_key_is_not_consulted(self) -> None:
        """under an unsafe key the whole value is anonymized, keys included; its keys need no classification."""
        site = _read("AuditEvent(event_type='x.y', details={'value': {'anything_a_user_typed': 1}})")

        assert _unclassified(site) == []

    def test_a_nested_key_under_an_unclassified_key_is_not_reported_twice(self) -> None:
        """the unclassified parent is the finding; what it holds is anonymized whole."""
        site = _read("AuditEvent(event_type='x.y', details={'mystery_key': {'inner_key': 1}})")

        assert _unclassified(site) == ["mystery_key"]


class TestFamilies:
    """a family declaration is credited to its own event types and no others."""

    def test_a_family_declaration_is_credited_to_its_family_only(self) -> None:
        """a declared key passes for its own family and fails for any other."""
        own = _read(f"AuditEvent(event_type='{_FAMILY}.x', details={{'gate_declared_count': 1}})")
        other = _read("AuditEvent(event_type='other.x', details={'gate_declared_count': 1})")

        assert _unclassified(own) == []
        assert _unclassified(other) == ["gate_declared_count"]

    def test_a_computed_event_type_gets_no_family_credit(self) -> None:
        """when the walker cannot read the event type it cannot credit a declaration."""
        site = _read("AuditEvent(event_type=kind, details={'gate_declared_count': 1})")

        assert site.event_type is None
        assert _unclassified(site) == ["gate_declared_count"]


class TestUnreadableShapesAreRefused:
    """what the walker cannot see is reported as unreadable, never passed as clean."""

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param("AuditEvent(event_type='x.y', details=build())", id="details-from-a-call"),
            pytest.param("AuditEvent(event_type='x.y', details={**extra})", id="spread-in-literal"),
            pytest.param("AuditEvent(event_type='x.y', details={key: 1})", id="computed-literal-key"),
            pytest.param("AuditEvent(event_type='x.y', details=dict(other))", id="copy-of-an-unknown-mapping"),
            pytest.param("AuditEvent(event_type='x.y', **fields)", id="spread-into-constructor"),
            pytest.param(
                "def f():\n    details = {}\n    details[k] = 1\n    AuditEvent(event_type='x.y', details=details)\n",
                id="computed-subscript-key",
            ),
            pytest.param(
                "def f():\n    details = {}\n    details.setdefault(k, 1)\n    AuditEvent(event_type='x.y', details=details)\n",
                id="computed-setdefault-key",
            ),
        ],
    )
    def test_a_shape_the_walker_cannot_read_is_refused(self, source: str) -> None:
        """every shape here hides at least one key from the classification.

        :param source: a synthetic module with one unreadable details argument
        :ptype source: str
        """
        assert _read(source).unreadable


_WRAPPER = (
    "def publish_wrapped(*, event_type, details=None):\n"
    "    AuditEvent(event_type=event_type, details=details)\n"
    "\n"
    "def caller():\n"
    "    publish_wrapped(event_type='x.y', details={'tool_name': t, 'mystery_key': m})\n"
)


class TestForwarders:
    """a consumer's wrapper helper is refused until declared, then read at its call sites."""

    def test_an_undeclared_wrapper_is_refused(self) -> None:
        """the wrapper's hand-off of its parameter is unreadable, and its callers' keys are unseen."""
        sites = read_audit_details_sites(ast.parse(_WRAPPER))

        assert len(sites) == 1
        assert sites[0].unreadable

    def test_a_declared_wrapper_is_read_at_its_call_sites(self) -> None:
        """declared, the hand-off is accepted and the caller's literal keys are judged."""
        sites = read_audit_details_sites(ast.parse(_WRAPPER), forwarders=frozenset({"publish_wrapped"}))

        hand_off, call_site = sites
        assert hand_off.unreadable == ()
        assert call_site.unreadable == ()
        assert call_site.event_type == "x.y"
        assert _unclassified(call_site) == ["mystery_key"]

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param("AuditEvent(event_type=e, details=details or {})", id="defaulted"),
            pytest.param("AuditEvent(event_type=e, details=dict(details))", id="copied"),
            pytest.param("AuditEvent(event_type=e, details=details if details is not None else {})", id="ifexp"),
            pytest.param("AuditEvent(event_type=e, **kwargs)", id="kwargs-passed-through"),
            pytest.param(
                "merged = {**(details or {}), 'tool_name': t}\n    AuditEvent(event_type=e, details=merged)",
                id="spread-and-extended",
            ),
            pytest.param(
                "details['tool_name'] = t\n    AuditEvent(event_type=e, details=details)",
                id="parameter-extended-in-place",
            ),
        ],
    )
    def test_a_declared_wrapper_may_default_copy_or_extend_its_parameter(self, body: str) -> None:
        """a forwarder handles its parameter in the ways wrappers do, and nothing it adds goes unread.

        :param body: the forwarder's body
        :ptype body: str
        """
        source = f"def publish_wrapped(*, e, details=None, **kwargs):\n    {body}\n"
        (site,) = read_audit_details_sites(ast.parse(source), forwarders=frozenset({"publish_wrapped"}))

        assert site.unreadable == ()

    def test_a_key_a_declared_wrapper_adds_is_still_judged(self) -> None:
        """extending the forwarded parameter with a literal key reads that key."""
        source = (
            "def publish_wrapped(*, e, details=None):\n"
            "    merged = {**(details or {}), 'mystery_key': 1}\n"
            "    AuditEvent(event_type=e, details=merged)\n"
        )
        (site,) = read_audit_details_sites(ast.parse(source), forwarders=frozenset({"publish_wrapped"}))

        assert _unclassified(site) == ["mystery_key"]

    def test_a_closure_inside_a_declared_wrapper_forwards_its_parameter(self) -> None:
        """the fire-and-forget shape: the hand-off happens in a task defined inside the wrapper."""
        source = (
            "def emit_wrapped(*, e, details=None):\n"
            "    async def run():\n"
            "        await publish(AuditEvent(event_type=e, details=details))\n"
            "    spawn(run())\n"
        )
        (site,) = read_audit_details_sites(ast.parse(source), forwarders=frozenset({"emit_wrapped"}))

        assert site.unreadable == ()

    def test_a_forwarder_declaration_does_not_launder_an_unrelated_function(self) -> None:
        """only the declared function's own parameters are accepted."""
        source = "def other(details):\n    AuditEvent(event_type='x.y', details=details)\n"
        (site,) = read_audit_details_sites(ast.parse(source), forwarders=frozenset({"publish_wrapped"}))

        assert site.unreadable


class TestTheTreeWalk:
    """the file-level entry points."""

    def _config(self, tmp_path: Path, *src_roots: Path) -> AuditDetailsConfig:
        """
        a config over synthetic source roots.

        :param tmp_path: the repo root
        :ptype tmp_path: Path
        :param src_roots: the roots to scan
        :ptype src_roots: Path
        :return: the config
        :rtype: AuditDetailsConfig
        """
        return AuditDetailsConfig(
            repo_root=tmp_path,
            src_roots=src_roots,
            safe_keys_for=_safe_keys_for,
            personal_keys=_PERSONAL,
        )

    def test_a_tree_with_an_unclassified_key_yields_a_violation(self, tmp_path: Path) -> None:
        """the violation names the file, the line and the key path."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "producer.py").write_text("\n\nAuditEvent(event_type='x.y', details={'mystery_key': 1})\n")

        violations = find_audit_details_violations(self._config(tmp_path, src))

        assert [(v.category, v.file.name, v.line, v.symbol) for v in violations] == [
            ("audit_details.unclassified", "producer.py", 3, "mystery_key")
        ]

    def test_an_unreadable_site_yields_a_violation(self, tmp_path: Path) -> None:
        """an unreadable details argument is a violation of its own category."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "producer.py").write_text("AuditEvent(event_type='x.y', details=build())\n")

        violations = find_audit_details_violations(self._config(tmp_path, src))

        assert [v.category for v in violations] == ["audit_details.unreadable"]

    def test_a_clean_tree_yields_nothing_and_its_sites_are_collected(self, tmp_path: Path) -> None:
        """the positive control for the tree walk, and the collector a shell pins non-vacuity on."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "producer.py").write_text("AuditEvent(event_type='x.y', details={'tool_name': 1})\n")

        config = self._config(tmp_path, src)

        assert find_audit_details_violations(config) == []
        assert list(collect_audit_details_sites(config)) == [src / "producer.py"]

    def test_no_source_roots_is_itself_a_violation(self, tmp_path: Path) -> None:
        """a shell that scans nothing reports what a clean repo reports, unless this refuses it."""
        violations = find_audit_details_violations(self._config(tmp_path))

        assert [v.category for v in violations] == ["audit_details.no_src_roots"]
