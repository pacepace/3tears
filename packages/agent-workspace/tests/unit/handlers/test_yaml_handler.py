"""unit tests for YamlHandler: round-trip preservation, path ops, registration.

covers WS-06-01 through WS-06-10 success criteria from workspace-task-06
shard. round-trip preservation uses real audience YAML fixtures from the
bluelabs audience_builder_tool to exercise comments, key order, anchors,
and quote-style preservation under realistic content.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from threetears.core.serialization import _HANDLERS

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """snapshot and restore module-level handler registry around test.

    mirrors the fixture pattern in packages/core/tests/unit/
    test_format_handler_registry.py to avoid cross-test contamination
    when tests exercise self-registration side effects.

    :return: iterator yielding once while registry is cleared, then
        restoring prior snapshot
    :rtype: Iterator[None]
    """
    snapshot = dict(_HANDLERS)
    _HANDLERS.clear()
    try:
        yield
    finally:
        _HANDLERS.clear()
        _HANDLERS.update(snapshot)


@pytest.fixture
def handler() -> Any:
    """construct fresh YamlHandler for use in a test.

    :return: YamlHandler instance
    :rtype: YamlHandler
    """
    from threetears.agent.workspace.handlers.yaml_handler import YamlHandler

    return YamlHandler()


class TestExtensions:
    """YamlHandler declares yaml and yml extensions."""

    def test_extensions_tuple(self, handler: Any) -> None:
        assert handler.extensions == (".yaml", ".yml")


class TestLoadDump:
    """load parses into CommentedMap/CommentedSeq; dump produces text."""

    def test_load_returns_commented_map_for_mapping_root(
        self, handler: Any
    ) -> None:
        tree = handler.load("key: value\n")
        assert isinstance(tree, CommentedMap)
        assert tree["key"] == "value"

    def test_load_returns_commented_seq_for_sequence_root(
        self, handler: Any
    ) -> None:
        tree = handler.load("- a\n- b\n")
        assert isinstance(tree, CommentedSeq)
        assert list(tree) == ["a", "b"]

    def test_dump_returns_string(self, handler: Any) -> None:
        tree = handler.load("key: value\n")
        text = handler.dump(tree)
        assert isinstance(text, str)
        assert "key" in text
        assert "value" in text

    def test_round_trip_scalar_types_preserved(self, handler: Any) -> None:
        source = "n: 1\nf: 1.5\nb: true\ns: hello\n"
        tree = handler.load(source)
        assert tree["n"] == 1
        assert tree["f"] == 1.5
        assert tree["b"] is True
        assert tree["s"] == "hello"


class TestRoundTripPreservation:
    """load -> dump -> load preserves comments, order, anchors, quotes."""

    def test_comments_preserved_line_and_eol(self, handler: Any) -> None:
        source = (
            "# top level comment\n"
            "key1: value1  # eol comment\n"
            "# between comment\n"
            "key2: value2\n"
        )
        dumped = handler.dump(handler.load(source))
        assert "# top level comment" in dumped
        assert "# eol comment" in dumped
        assert "# between comment" in dumped

    def test_key_order_preserved(self, handler: Any) -> None:
        source = "zebra: 1\napple: 2\nmango: 3\n"
        dumped = handler.dump(handler.load(source))
        z_pos = dumped.index("zebra")
        a_pos = dumped.index("apple")
        m_pos = dumped.index("mango")
        assert z_pos < a_pos < m_pos

    def test_anchors_and_aliases_preserved(self, handler: Any) -> None:
        source = (
            "defaults: &defaults\n"
            "  timeout: 30\n"
            "  retries: 3\n"
            "service:\n"
            "  <<: *defaults\n"
            "  name: api\n"
        )
        dumped = handler.dump(handler.load(source))
        assert "&defaults" in dumped
        assert "*defaults" in dumped

    def test_single_quotes_preserved(self, handler: Any) -> None:
        source = "key: 'single'\n"
        dumped = handler.dump(handler.load(source))
        assert "'single'" in dumped

    def test_double_quotes_preserved(self, handler: Any) -> None:
        source = 'key: "double"\n'
        dumped = handler.dump(handler.load(source))
        assert '"double"' in dumped

    def test_bare_strings_stay_bare(self, handler: Any) -> None:
        source = "key: bare\n"
        dumped = handler.dump(handler.load(source))
        assert "key: bare" in dumped
        assert "'bare'" not in dumped
        assert '"bare"' not in dumped

    def test_audience_settings_round_trip_preserves_structure(
        self, handler: Any
    ) -> None:
        """round-trip of real audience_settings.yaml keeps all keys and values.

        source indent is 4-space mapping; handler normalizes to 2-space, so
        exact-text equality is not expected. structural equality and presence
        of quote-style tokens are the invariants.
        """
        text = (FIXTURES_DIR / "audience_settings.yaml").read_text()
        tree = handler.load(text)
        dumped = handler.dump(tree)
        reloaded = handler.load(dumped)
        assert reloaded == tree
        assert "'>= 1000'" in dumped
        assert "= 0" in dumped

    def test_standard_audience_units_round_trip_byte_stable(
        self, handler: Any
    ) -> None:
        """standard_audience_units.yaml uses 2-space mapping indent -- matches.

        this fixture was written in the same indent style the handler uses,
        so the dump should equal the source modulo a trailing newline.
        """
        text = (FIXTURES_DIR / "standard_audience_units.yaml").read_text()
        dumped = handler.dump(handler.load(text))
        assert dumped.strip() == text.strip()

    def test_linkedin_audience_units_round_trip_byte_stable(
        self, handler: Any
    ) -> None:
        """linkedin_audience_units.yaml uses 2-space mapping indent -- matches."""
        text = (FIXTURES_DIR / "linkedin_audience_units.yaml").read_text()
        dumped = handler.dump(handler.load(text))
        assert dumped.strip() == text.strip()


class TestGet:
    """get resolves jsonpath expressions; returns None on no match."""

    def test_get_single_match_returns_value(self, handler: Any) -> None:
        tree = handler.load("a:\n  b: 42\n")
        assert handler.get(tree, "$.a.b") == 42

    def test_get_nested_via_bracket_notation(self, handler: Any) -> None:
        tree = handler.load("a:\n  - x: 1\n  - x: 2\n")
        assert handler.get(tree, "$.a[0].x") == 1
        assert handler.get(tree, "$.a[1].x") == 2

    def test_get_multiple_matches_returns_list(self, handler: Any) -> None:
        tree = handler.load("a:\n  - x: 1\n  - x: 2\n  - x: 3\n")
        result = handler.get(tree, "$.a[*].x")
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_get_no_match_returns_none(self, handler: Any) -> None:
        tree = handler.load("a: 1\n")
        assert handler.get(tree, "$.nonexistent") is None

    def test_get_invalid_jsonpath_raises(self, handler: Any) -> None:
        tree = handler.load("a: 1\n")
        with pytest.raises(Exception):
            handler.get(tree, "$[[[")

    def test_get_filter_expression_uses_ext_parser(self, handler: Any) -> None:
        """filter expressions like [?(@.x > 1)] require jsonpath_ng.ext."""
        tree = handler.load("a:\n  - x: 1\n  - x: 2\n  - x: 3\n")
        result = handler.get(tree, "$.a[?(@.x > 1)].x")
        assert sorted(result) == [2, 3]


class TestSet:
    """set mutates tree, creates missing paths, returns tree."""

    def test_set_existing_path_mutates_value(self, handler: Any) -> None:
        tree = handler.load("a:\n  b: 1\n")
        result = handler.set(tree, "$.a.b", 99)
        assert result is tree
        assert tree["a"]["b"] == 99

    def test_set_creates_missing_intermediate_path(self, handler: Any) -> None:
        tree = handler.load("a: 1\n")
        handler.set(tree, "$.new.nested.key", "value")
        assert tree["new"]["nested"]["key"] == "value"

    def test_set_array_index(self, handler: Any) -> None:
        tree = handler.load("a:\n  - x: 1\n  - x: 2\n")
        handler.set(tree, "$.a[0].x", 42)
        assert tree["a"][0]["x"] == 42

    def test_set_preserves_comments_on_unaffected_portions(
        self, handler: Any
    ) -> None:
        source = (
            "# header comment\n"
            "a:\n"
            "  b: 1  # inline comment\n"
            "c: 2\n"
        )
        tree = handler.load(source)
        handler.set(tree, "$.c", 99)
        dumped = handler.dump(tree)
        assert "# header comment" in dumped
        assert "# inline comment" in dumped
        assert "c: 99" in dumped

    def test_set_on_audience_fixture_preserves_comments_and_structure(
        self, handler: Any
    ) -> None:
        """set on a real audience fixture leaves surrounding structure intact."""
        text = (FIXTURES_DIR / "audience_settings.yaml").read_text()
        tree = handler.load(text)
        handler.set(tree, "$.audience_units[0].vb_candidates", 5)
        dumped = handler.dump(tree)
        reloaded = handler.load(dumped)
        assert reloaded["audience_units"][0]["vb_candidates"] == 5
        assert reloaded["audience_units"][0]["audience_unit"] == "knowwho_all"
        assert reloaded["audience_units"][3]["audience_unit"] == "donors"
        assert (
            reloaded["audience_units"][3]["relationships"]["donors_to_candidate"][
                "committee_transaction_amt"
            ]
            == ">= 1000"
        )


class TestMerge:
    """merge deep-merges mappings, replaces lists and scalars."""

    def test_merge_adds_new_top_level_key(self, handler: Any) -> None:
        tree = handler.load("a: 1\n")
        handler.merge(tree, {"b": 2})
        assert tree["a"] == 1
        assert tree["b"] == 2

    def test_merge_overwrites_existing_scalar(self, handler: Any) -> None:
        tree = handler.load("a: 1\n")
        handler.merge(tree, {"a": 99})
        assert tree["a"] == 99

    def test_merge_recursively_merges_nested_mapping(
        self, handler: Any
    ) -> None:
        tree = handler.load("a:\n  b: 1\n  c: 2\n")
        handler.merge(tree, {"a": {"c": 99, "d": 3}})
        assert tree["a"]["b"] == 1
        assert tree["a"]["c"] == 99
        assert tree["a"]["d"] == 3

    def test_merge_replaces_list_wholesale(self, handler: Any) -> None:
        tree = handler.load("x:\n  - 1\n  - 2\n  - 3\n")
        handler.merge(tree, {"x": [9]})
        assert list(tree["x"]) == [9]

    def test_merge_returns_tree(self, handler: Any) -> None:
        tree = handler.load("a: 1\n")
        result = handler.merge(tree, {"b": 2})
        assert result is tree

    def test_merge_replaces_mapping_with_scalar(self, handler: Any) -> None:
        tree = handler.load("a:\n  b: 1\n")
        handler.merge(tree, {"a": "scalar"})
        assert tree["a"] == "scalar"

    def test_merge_does_not_touch_unrelated_keys(self, handler: Any) -> None:
        tree = handler.load("keep: true\ntouch:\n  inner: 1\n")
        handler.merge(tree, {"touch": {"inner": 2}})
        assert tree["keep"] is True
        assert tree["touch"]["inner"] == 2


class TestSelfRegistration:
    """YamlHandler self-registers on module import via register_handler."""

    def test_import_triggers_registration(self, clean_registry: None) -> None:
        """importing yaml_handler module must register a YamlHandler."""
        import sys

        sys.modules.pop("threetears.agent.workspace.handlers.yaml_handler", None)
        import threetears.agent.workspace.handlers.yaml_handler as mod
        from threetears.core.serialization import handler_for

        resolved = handler_for("foo.yaml")
        assert isinstance(resolved, mod.YamlHandler)

    def test_package_import_triggers_registration(
        self, clean_registry: None
    ) -> None:
        """importing threetears.agent.workspace top-level also registers.

        simulates a fresh process by evicting the package, its handlers
        subpackage, and the yaml_handler module from sys.modules, then
        importing only the top-level package and verifying the side
        effect fired through the handlers submodule chain.
        """
        import sys

        for mod_name in (
            "threetears.agent.workspace",
            "threetears.agent.workspace.handlers",
            "threetears.agent.workspace.handlers.yaml_handler",
        ):
            sys.modules.pop(mod_name, None)

        import threetears.agent.workspace  # noqa: F401
        from threetears.core.serialization import handler_for

        resolved = handler_for("foo.yaml")
        from threetears.agent.workspace.handlers.yaml_handler import YamlHandler

        assert isinstance(resolved, YamlHandler)

    def test_registered_for_both_yaml_and_yml(
        self, clean_registry: None
    ) -> None:
        import sys

        sys.modules.pop("threetears.agent.workspace.handlers.yaml_handler", None)
        import threetears.agent.workspace.handlers.yaml_handler as mod
        from threetears.core.serialization import handler_for

        assert isinstance(handler_for("x.yaml"), mod.YamlHandler)
        assert isinstance(handler_for("x.yml"), mod.YamlHandler)
