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

        assert site.event_types == frozenset()
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
        assert call_site.event_types == frozenset({"x.y"})
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


def _tree(tmp_path: Path, files: dict[str, str]) -> AuditDetailsConfig:
    """
    write a synthetic source tree under ``src/`` and configure the walker over it.

    :param tmp_path: the repo root
    :ptype tmp_path: Path
    :param files: path under ``src/`` -> module source
    :ptype files: dict[str, str]
    :return: a config scanning that tree
    :rtype: AuditDetailsConfig
    """
    src = tmp_path / "src"
    for relative, source in files.items():
        path = src / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return AuditDetailsConfig(
        repo_root=tmp_path, src_roots=(src,), safe_keys_for=_safe_keys_for, personal_keys=_PERSONAL
    )


def _findings(config: AuditDetailsConfig) -> list[tuple[str, str]]:
    """
    the violations over a synthetic tree, as ``(category, symbol)``.

    :param config: the config
    :ptype config: AuditDetailsConfig
    :return: the findings, in walker order
    :rtype: list[tuple[str, str]]
    """
    return [(violation.category, violation.symbol) for violation in find_audit_details_violations(config)]


_FAMILY_KEY_SITE = "AuditEvent(event_type={event_type}, details={{'gate_declared_count': 1}})"
_UNCLASSIFIED = [("audit_details.unclassified", "gate_declared_count")]


class TestEventTypeConstants:
    """an ``event_type`` named by a constant resolves to the constant's value."""

    def test_a_module_level_constant_in_the_same_module(self) -> None:
        """``EVENT = "..."`` at module level, then ``event_type=EVENT``."""
        site = _read(f"EVENT = '{_FAMILY}.x'\n" + _FAMILY_KEY_SITE.format(event_type="EVENT"))

        assert site.event_types == frozenset({f"{_FAMILY}.x"})
        assert _unclassified(site) == []

    def test_an_annotated_module_constant(self) -> None:
        """``EVENT: Final = "..."`` is the same binding."""
        site = _read(f"EVENT: Final = '{_FAMILY}.x'\n" + _FAMILY_KEY_SITE.format(event_type="EVENT"))

        assert _unclassified(site) == []

    def test_a_module_name_bound_twice_is_not_trusted(self) -> None:
        """two bindings means the value at the call is not knowable statically."""
        site = _read(f"EVENT = '{_FAMILY}.x'\nEVENT = 'other.y'\n" + _FAMILY_KEY_SITE.format(event_type="EVENT"))

        assert site.event_types == frozenset()
        assert _unclassified(site) == ["gate_declared_count"]

    def test_a_local_that_shadows_the_constant_is_not_the_constant(self) -> None:
        """a function-local assignment of the same name is a runtime value."""
        site = _read(
            f"EVENT = '{_FAMILY}.x'\n"
            "def f(kind):\n"
            "    EVENT = kind\n"
            "    " + _FAMILY_KEY_SITE.format(event_type="EVENT") + "\n"
        )

        assert _unclassified(site) == ["gate_declared_count"]

    def test_a_constant_imported_from_another_module(self, tmp_path: Path) -> None:
        """``from pkg.events import EVENT`` resolves through the import to its definition."""
        config = _tree(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/events.py": f"EVENT = '{_FAMILY}.x'\n",
                "pkg/producer.py": "from pkg.events import EVENT\n" + _FAMILY_KEY_SITE.format(event_type="EVENT"),
            },
        )

        assert _findings(config) == []

    def test_a_constant_imported_relatively_and_under_an_alias(self, tmp_path: Path) -> None:
        """``from .events import EVENT as KIND`` is the same resolution."""
        config = _tree(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/events.py": f"EVENT = '{_FAMILY}.x'\n",
                "pkg/producer.py": "from .events import EVENT as KIND\n" + _FAMILY_KEY_SITE.format(event_type="KIND"),
            },
        )

        assert _findings(config) == []

    def test_a_constant_re_exported_by_a_package(self, tmp_path: Path) -> None:
        """a package ``__init__`` importing the constant is followed to where it is defined."""
        config = _tree(
            tmp_path,
            {
                "pkg/__init__.py": "from pkg.events import EVENT\n",
                "pkg/events.py": f"EVENT = '{_FAMILY}.x'\n",
                "other/producer.py": "from pkg import EVENT\n" + _FAMILY_KEY_SITE.format(event_type="EVENT"),
            },
        )

        assert _findings(config) == []

    def test_an_attribute_on_an_imported_module(self, tmp_path: Path) -> None:
        """``events.EVENT`` with ``from pkg import events``, and ``pkg.events.EVENT`` with ``import pkg.events``."""
        config = _tree(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/events.py": f"EVENT = '{_FAMILY}.x'\n",
                "pkg/a.py": "from pkg import events\n" + _FAMILY_KEY_SITE.format(event_type="events.EVENT"),
                "pkg/b.py": "import pkg.events\n" + _FAMILY_KEY_SITE.format(event_type="pkg.events.EVENT"),
            },
        )

        assert _findings(config) == []

    def test_an_import_from_outside_the_scanned_roots_is_not_resolved(self, tmp_path: Path) -> None:
        """a module the walker cannot read proves nothing about the value."""
        config = _tree(
            tmp_path,
            {"pkg/producer.py": "from elsewhere import EVENT\n" + _FAMILY_KEY_SITE.format(event_type="EVENT")},
        )

        assert _findings(config) == _UNCLASSIFIED

    def test_a_non_string_constant_is_not_an_event_type(self) -> None:
        """only a ``str`` literal binding resolves."""
        site = _read("EVENT = 3\n" + _FAMILY_KEY_SITE.format(event_type="EVENT"))

        assert site.event_types == frozenset()


_HELPER = "def emit(request, *, event_type):\n    " + _FAMILY_KEY_SITE.format(event_type="event_type") + "\n"


class TestHelperParameters:
    """``event_type=<parameter>`` resolves to what the helper's callers pass; every value must credit the key."""

    def test_callers_passing_literals_of_one_family_credit_the_key(self) -> None:
        """two callers, two event types of the same family: the key is safe for both."""
        site = _read(
            _HELPER + f"emit(r, event_type='{_FAMILY}.a')\n" + f"emit(r, event_type='{_FAMILY}.b')\n",
        )

        assert site.event_types == frozenset({f"{_FAMILY}.a", f"{_FAMILY}.b"})
        assert _unclassified(site) == []

    def test_a_key_safe_for_only_one_callers_family_is_reported(self) -> None:
        """the credit is the intersection: a value safe for one caller's family and not the other's is not safe."""
        site = _read(_HELPER + f"emit(r, event_type='{_FAMILY}.a')\n" + "emit(r, event_type='unrelated.b')\n")

        assert site.event_types == frozenset({f"{_FAMILY}.a", "unrelated.b"})
        assert _unclassified(site) == ["gate_declared_count"]

    def test_a_caller_passing_an_unresolvable_value_is_reported(self) -> None:
        """one caller the walker cannot read makes the whole parameter unknown."""
        site = _read(_HELPER + f"emit(r, event_type='{_FAMILY}.a')\n" + "emit(r, event_type=pick())\n")

        assert site.event_types == frozenset()
        assert _unclassified(site) == ["gate_declared_count"]

    def test_a_helper_with_no_caller_is_reported(self) -> None:
        """nothing sets the parameter, so nothing is known about it."""
        site = _read(_HELPER)

        assert _unclassified(site) == ["gate_declared_count"]

    def test_a_caller_passing_the_parameter_positionally_is_reported(self) -> None:
        """the walker does not map positions to parameters, so a positional caller is unresolved."""
        source = (
            "def emit(request, event_type):\n    "
            + _FAMILY_KEY_SITE.format(event_type="event_type")
            + f"\nemit(r, '{_FAMILY}.a')\n"
        )

        assert _unclassified(_read(source)) == ["gate_declared_count"]

    def test_a_caller_omitting_a_defaulted_parameter_takes_the_default(self) -> None:
        """a keyword-only default is what an omitting caller passes."""
        source = (
            f"def emit(request, *, event_type='{_FAMILY}.a'):\n    "
            + _FAMILY_KEY_SITE.format(event_type="event_type")
            + "\nemit(r)\n"
        )

        assert _unclassified(_read(source)) == []

    def test_a_parameter_the_helper_reassigns_is_reported(self) -> None:
        """a rebound parameter no longer holds what the callers passed."""
        source = (
            "def emit(request, *, event_type):\n"
            "    event_type = derive(event_type)\n    "
            + _FAMILY_KEY_SITE.format(event_type="event_type")
            + f"\nemit(r, event_type='{_FAMILY}.a')\n"
        )

        assert _unclassified(_read(source)) == ["gate_declared_count"]

    def test_a_caller_passing_its_own_parameter_is_followed_to_its_callers(self) -> None:
        """two helpers deep, the literals at the outer callers decide."""
        source = _HELPER + "def route(r, *, kind):\n    emit(r, event_type=kind)\n" + f"route(r, kind='{_FAMILY}.a')\n"

        assert _unclassified(_read(source)) == []

    def test_mutual_recursion_is_not_a_hang(self) -> None:
        """a cycle of helpers passing the parameter round resolves to nothing, and terminates."""
        source = (
            _HELPER
            + "def ping(r, *, kind):\n    emit(r, event_type=kind)\n    pong(r, kind=kind)\n"
            + "def pong(r, *, kind):\n    ping(r, kind=kind)\n"
        )

        assert _unclassified(_read(source)) == ["gate_declared_count"]

    def test_a_method_called_through_an_attribute_with_imported_constants(self, tmp_path: Path) -> None:
        """the hub's knowledge shape: ``self._sink.publish(event_type=EVENT_X)``, constant imported."""
        config = _tree(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/events.py": f"EVENT_A = '{_FAMILY}.a'\nEVENT_B = '{_FAMILY}.b'\n",
                "pkg/sink.py": (
                    "class Sink:\n"
                    "    async def publish(self, *, event_type):\n        "
                    + _FAMILY_KEY_SITE.format(event_type="event_type")
                    + "\n"
                ),
                "pkg/a.py": "from pkg.events import EVENT_A\nasync def f(s):\n    await s._sink.publish(event_type=EVENT_A)\n",
                "pkg/b.py": "from pkg import events\nasync def g(s):\n    await s.sink.publish(event_type=events.EVENT_B)\n",
            },
        )

        assert _findings(config) == []

    def test_a_same_named_function_elsewhere_only_narrows_the_credit(self, tmp_path: Path) -> None:
        """callers are matched by name; an unrelated ``emit`` passing another family can only withhold credit."""
        config = _tree(
            tmp_path,
            {
                "pkg/helper.py": _HELPER + f"emit(r, event_type='{_FAMILY}.a')\n",
                "pkg/unrelated.py": "emit(r, event_type='unrelated.b')\n",
            },
        )

        assert _findings(config) == _UNCLASSIFIED


class TestConditionalEventTypes:
    """``event_type=A if cond else B`` resolves to both branches; a key must be safe for each."""

    def test_both_branches_in_the_family_credit_the_key(self) -> None:
        """the approvals shape: two module constants of one family."""
        site = _read(
            f"_APPROVED = '{_FAMILY}.approved'\n_DENIED = '{_FAMILY}.denied'\n"
            + _FAMILY_KEY_SITE.format(event_type="_APPROVED if verdict == 'approve' else _DENIED")
        )

        assert site.event_types == frozenset({f"{_FAMILY}.approved", f"{_FAMILY}.denied"})
        assert _unclassified(site) == []

    def test_a_branch_in_a_family_where_the_key_is_not_safe_is_reported(self) -> None:
        """the credit is the intersection over the branches, never the union."""
        site = _read(_FAMILY_KEY_SITE.format(event_type=f"'{_FAMILY}.a' if ok else 'unrelated.b'"))

        assert site.event_types == frozenset({f"{_FAMILY}.a", "unrelated.b"})
        assert _unclassified(site) == ["gate_declared_count"]

    def test_an_unresolvable_branch_leaves_the_whole_expression_unresolved(self) -> None:
        """one branch the walker cannot read makes the value unknown."""
        site = _read(_FAMILY_KEY_SITE.format(event_type=f"'{_FAMILY}.a' if ok else pick()"))

        assert site.event_types == frozenset()
        assert _unclassified(site) == ["gate_declared_count"]


_WRAPPED_CALLER = "def caller(r):\n    wrapped(r, details={'gate_declared_count': 1})\n"


def _forwarded(inner_body: str, *, signature: str = "request, *, details=None") -> AuditDetailsSite:
    """
    the call site of a declared forwarder ``wrapped`` whose body is ``inner_body``.

    :param inner_body: the forwarder's body, one statement per line, unindented
    :ptype inner_body: str
    :param signature: the forwarder's parameter list
    :ptype signature: str
    :return: the site of the call to ``wrapped`` in ``caller``
    :rtype: AuditDetailsSite
    """
    body = "".join(f"    {line}\n" for line in inner_body.splitlines())
    source = f"def wrapped({signature}):\n{body}" + _WRAPPED_CALLER
    sites = read_audit_details_sites(ast.parse(source), forwarders=frozenset({"wrapped", "emit"}))
    (call_site,) = [site for site in sites if ("gate_declared_count",) in site.keys]
    return call_site


class TestAForwarderThatFixesItsEventType:
    """a call site that passes no ``event_type`` is credited with what the forwarder's own construction resolves."""

    def test_an_inner_construction_with_a_literal(self) -> None:
        """the forwarder builds the event itself, with a fixed type."""
        site = _forwarded(f"AuditEvent(event_type='{_FAMILY}.x', details=details)")

        assert site.event_types == frozenset({f"{_FAMILY}.x"})
        assert _unclassified(site) == []

    def test_an_inner_call_to_another_forwarder(self) -> None:
        """the hub's ``_audited_failure`` shape: the forwarder hands on to a declared forwarder."""
        site = _forwarded(
            f"emit(request, event_type='{_FAMILY}.x', details={{**(details or {{}}), 'interrupted_by': 'E'}})"
        )

        assert site.event_types == frozenset({f"{_FAMILY}.x"})
        assert _unclassified(site) == []

    def test_an_inner_construction_that_resolves_to_nothing_is_reported(self) -> None:
        """if the forwarder's own type cannot be shown, the call site gets no family credit."""
        site = _forwarded("emit(request, event_type=pick(), details=details)")

        assert site.event_types == frozenset()
        assert _unclassified(site) == ["gate_declared_count"]

    def test_one_unresolvable_inner_construction_among_resolvable_ones_is_reported(self) -> None:
        """a partial answer is not an answer: the unreadable construction may publish any type."""
        site = _forwarded(
            f"AuditEvent(event_type='{_FAMILY}.x', details=details)\nemit(request, event_type=pick(), details=details)"
        )

        assert site.event_types == frozenset()
        assert _unclassified(site) == ["gate_declared_count"]

    def test_two_inner_constructions_in_different_families_are_intersected(self) -> None:
        """a forwarder that may publish either type credits only what is safe for both."""
        site = _forwarded(
            f"AuditEvent(event_type='{_FAMILY}.x', details=details)\nAuditEvent(event_type='unrelated.y', details=details)"
        )

        assert site.event_types == frozenset({f"{_FAMILY}.x", "unrelated.y"})
        assert _unclassified(site) == ["gate_declared_count"]

    def test_the_forwarders_own_defaulted_parameter(self) -> None:
        """an inner ``event_type=event_type`` with a default the call site leaves in place."""
        site = _forwarded(
            "AuditEvent(event_type=event_type, details=details)",
            signature=f"request, *, details=None, event_type='{_FAMILY}.x'",
        )

        assert _unclassified(site) == []

    def test_a_call_site_that_passes_its_own_event_type_wins(self) -> None:
        """an explicit ``event_type`` at the call site is what is published, whatever the forwarder's default."""
        source = (
            f"def wrapped(request, *, details=None, event_type='{_FAMILY}.x'):\n"
            "    AuditEvent(event_type=event_type, details=details)\n"
            "def caller(r):\n"
            "    wrapped(r, event_type='unrelated.y', details={'gate_declared_count': 1})\n"
        )
        sites = read_audit_details_sites(ast.parse(source), forwarders=frozenset({"wrapped"}))
        (call_site,) = [site for site in sites if ("gate_declared_count",) in site.keys]

        assert call_site.event_types == frozenset({"unrelated.y"})
        assert _unclassified(call_site) == ["gate_declared_count"]

    def test_a_forwarder_the_walker_cannot_find_is_reported(self) -> None:
        """a declared forwarder defined outside the scanned roots fixes nothing it can see."""
        sites = read_audit_details_sites(ast.parse(_WRAPPED_CALLER), forwarders=frozenset({"wrapped"}))

        assert [_unclassified(site) for site in sites] == [["gate_declared_count"]]

    def test_a_forwarder_calling_itself_terminates(self) -> None:
        """a recursive forwarder resolves to nothing rather than looping."""
        site = _forwarded("wrapped(request, details=details)")

        assert _unclassified(site) == ["gate_declared_count"]
