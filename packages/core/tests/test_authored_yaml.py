"""A duplicated key in a hand-edited file must be refused, not silently resolved.

PyYAML's ``safe_load`` resolves a duplicated mapping key by last-wins in silence.
For a file a human maintains that is the worst available behaviour: the botched edit
parses, every downstream validation reports success, and the value the author wrote
is gone with nothing left to notice it by.

The baseline test below pins ``yaml.safe_load``'s own behaviour rather than
describing it, so the reason this module exists stays visible and checkable.
"""

from __future__ import annotations

import io

import pytest
import yaml

from threetears.core.authored_yaml import DuplicateKeyError, safe_load_authored


class TestTheHazardThisGuards:
    def test_pyyaml_itself_silently_keeps_the_last_value(self) -> None:
        """the behaviour being guarded against, pinned rather than asserted in prose."""
        assert yaml.safe_load("a: 1\na: 2\n") == {"a": 2}


class TestSafeLoadAuthored:
    def test_a_duplicated_key_is_refused(self) -> None:
        with pytest.raises(DuplicateKeyError):
            safe_load_authored("a: 1\na: 2\n")

    def test_the_refusal_names_the_key(self) -> None:
        """an operator has to be able to find the line; a bare "invalid YAML" cannot."""
        with pytest.raises(DuplicateKeyError, match="'a'"):
            safe_load_authored("a: 1\na: 2\n")

    def test_a_duplicate_nested_inside_a_list_item_is_refused(self) -> None:
        """the real shape of the accident: one entry in a list gets a field twice.

        This is the case that bit the SDK's authored knowledge files, and it is why
        the guard has to live in the loader -- by the time a model validates the
        entry, the mapping has already collapsed and the duplicate is unobservable.
        """
        document = "entries:\n  - name: x\n    body: first\n    body: second\n"
        with pytest.raises(DuplicateKeyError, match="'body'"):
            safe_load_authored(document)

    def test_a_clean_document_parses_normally(self) -> None:
        assert safe_load_authored("a: 1\nb:\n  - 2\n  - 3\n") == {"a": 1, "b": [2, 3]}

    def test_the_same_key_in_two_different_mappings_is_fine(self) -> None:
        """uniqueness is per mapping, not per document -- the obvious over-reach."""
        document = "one:\n  name: x\ntwo:\n  name: y\n"

        assert safe_load_authored(document) == {"one": {"name": "x"}, "two": {"name": "y"}}

    def test_an_empty_document_is_none_not_an_error(self) -> None:
        """matches yaml.safe_load, so this is a drop-in replacement at a call site."""
        assert safe_load_authored("") is None

    def test_an_open_handle_is_accepted(self) -> None:
        """callers pass a handle so PyYAML's problem mark names the real path."""
        assert safe_load_authored(io.StringIO("a: 1\n")) == {"a": 1}

    def test_malformed_yaml_still_raises_a_yaml_error(self) -> None:
        """the guard adds a refusal; it does not swallow the ordinary syntax failure."""
        with pytest.raises(yaml.YAMLError):
            safe_load_authored("a: [1, 2\n")

    def test_the_refusal_is_distinguishable_from_a_syntax_error(self) -> None:
        """a caller reporting "you duplicated a key" must not have to match on text."""
        with pytest.raises(DuplicateKeyError):
            safe_load_authored("a: 1\na: 2\n")
        with pytest.raises(yaml.YAMLError) as syntax_error:
            safe_load_authored("a: [1, 2\n")
        assert not isinstance(syntax_error.value, DuplicateKeyError)
