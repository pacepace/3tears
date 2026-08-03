"""unit tests for the ``datasets/*.dataset.yaml`` envelope and its parser.

Every test here runs with nothing running -- no Hub, no NATS, no
database -- because offline is the whole point of the file surface. One
test disables :func:`socket.socket` outright rather than merely leaving
the platform unconfigured, since "offline" degrades quietly the first
time someone adds a convenience lookup.

The four rejections the acceptance names, plus the two the knowledge
validator cannot make:

- a duplicate name across files
- ``scope: platform``
- malformed YAML, named by file and line
- a misspelled field -- the gap ``aibots knowledge validate`` has,
  because its ``enforcement`` field is an untyped dict; ``DatasetDefinition``
  is typed end to end with ``extra="forbid"`` and this asserts it stays so
- ``visibility`` / ``grants``, which per D21 are administered rows

And the property that makes the file surface safe to author against: a
file hashes to exactly what the equivalent in-memory model hashes to. A
serialisation that drifts from the model mints versions silently, and a
minted version is immutable.
"""

from __future__ import annotations

import ast
import json
import socket
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from threetears.datasources.definition import DatasetDefinition
from threetears.datasources.definition.expression import LiteralExpression, LiteralType
from threetears.datasources.definition.file_model import (
    DATASET_FILE_SUFFIX,
    DATASETS_DIR_NAME,
    DatasetDriftReport,
    DatasetFile,
    DatasetFileError,
    dataset_drift,
    discover_dataset_files,
    load_dataset_file,
    load_dataset_files,
    render_dataset_file,
    unloaded_dataset_files,
    warn_unloaded_dataset_files,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_JSON_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "definition_minimal.json"

_MINIMAL_BODY = """\
  datasource: influencers-build
  grain:
    entity_column: voterbase_id
  units:
    - name: academy_members
      resolutions:
        - source:
            raw_sql: SELECT voterbase_id FROM academy
            projection: [voterbase_id]
            provenance:
              grain: [voterbase_id]
              columns:
                - name: voterbase_id
                  expression: resolved.voterbase_id
  artifacts:
    - artifact: long
      columns: [unit, voterbase_id]
"""

_MINIMAL_FILE = f"name: academy_influencers\ndefinition:\n{_MINIMAL_BODY}"


def _equivalent_payload() -> dict[str, object]:
    """the in-memory spelling of :data:`_MINIMAL_FILE`, authored by hand.

    Deliberately NOT derived from the YAML: deriving it would make the
    hash comparison compare a value with itself.

    :returns: raw definition payload
    :rtype: dict[str, object]
    """
    return {
        "name": "academy_influencers",
        "datasource": "influencers-build",
        "grain": {"entity_column": "voterbase_id"},
        "units": [
            {
                "name": "academy_members",
                "resolutions": [
                    {
                        "source": {
                            "raw_sql": "SELECT voterbase_id FROM academy",
                            "projection": ["voterbase_id"],
                            "provenance": {
                                "grain": ["voterbase_id"],
                                "columns": [{"name": "voterbase_id", "expression": "resolved.voterbase_id"}],
                            },
                        }
                    }
                ],
            }
        ],
        "artifacts": [{"artifact": "long", "columns": ["unit", "voterbase_id"]}],
    }


def _write(directory: Path, stem: str, text: str) -> Path:
    """write one dataset file into a directory.

    :param directory: directory to write into, created when absent
    :ptype directory: pathlib.Path
    :param stem: file stem, without the dataset suffix
    :ptype stem: str
    :param text: file body
    :ptype text: str
    :returns: path written
    :rtype: pathlib.Path
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}{DATASET_FILE_SUFFIX}"
    path.write_text(text, encoding="utf-8")
    return path


def _with_qualification(arm: str) -> str:
    """the minimal file, carrying one definition-level qualification arm.

    :param arm: indented YAML for the arm's predicate block
    :ptype arm: str
    :returns: dataset file text
    :rtype: str
    """
    return (
        f"name: academy_influencers\ndefinition:\n{_MINIMAL_BODY}"
        "  qualification:\n"
        "    - name: scored\n"
        "      applies_to: [academy_members]\n"
        "      predicate:\n"
        "        compare:\n"
        "          left: entity.score\n"
        "          op: '>='\n"
        f"          right:\n{arm}"
    )


class TestTheEnvelopeIsThin:
    """the file model delegates; it is not a second copy of the schema."""

    def test_the_envelope_is_a_name_a_scope_and_a_delegated_body(self) -> None:
        assert set(DatasetFile.model_fields) == {"name", "scope", "definition"}

    def test_the_body_is_the_definition_model_itself(self) -> None:
        assert DatasetFile.model_fields["definition"].annotation is DatasetDefinition

    def test_the_file_name_is_the_definition_name(self, tmp_path: Path) -> None:
        loaded = load_dataset_file(_write(tmp_path, "academy", _MINIMAL_FILE))
        assert loaded.name == "academy_influencers"
        assert loaded.definition.name == "academy_influencers"

    def test_a_body_repeating_the_name_is_refused(self, tmp_path: Path) -> None:
        text = _MINIMAL_FILE.replace("definition:\n", "definition:\n  name: academy_influencers\n")
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(_write(tmp_path, "academy", text))
        assert "name" in str(excinfo.value)

    def test_rejects_an_unknown_envelope_field(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFileError):
            load_dataset_file(_write(tmp_path, "academy", f"{_MINIMAL_FILE}customer: acme\n"))

    def test_the_suffix_and_directory_name_are_declared(self) -> None:
        assert DATASET_FILE_SUFFIX == ".dataset.yaml"
        assert DATASETS_DIR_NAME == "datasets"


class TestScopeIsCustomerOnly:
    """``scope: platform`` is a hard error, matching the knowledge layer."""

    def test_an_omitted_scope_is_the_customer_default(self, tmp_path: Path) -> None:
        loaded = load_dataset_file(_write(tmp_path, "academy", _MINIMAL_FILE))
        assert loaded.scope is None

    def test_an_explicit_customer_scope_is_admitted(self, tmp_path: Path) -> None:
        text = f"scope: customer\n{_MINIMAL_FILE}"
        assert load_dataset_file(_write(tmp_path, "academy", text)).scope == "customer"

    def test_platform_scope_is_a_hard_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "academy", f"scope: platform\n{_MINIMAL_FILE}")
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(path)
        message = str(excinfo.value)
        assert "platform" in message
        assert str(path) in message

    def test_platform_scope_is_caught_case_insensitively(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFileError):
            load_dataset_file(_write(tmp_path, "academy", f"scope: PLATFORM\n{_MINIMAL_FILE}"))

    def test_platform_scope_inside_the_body_is_a_hard_error(self, tmp_path: Path) -> None:
        text = _MINIMAL_FILE.replace("definition:\n", "definition:\n  scope: platform\n")
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(_write(tmp_path, "academy", text))
        assert "platform" in str(excinfo.value)

    def test_any_other_scope_is_refused_too(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(_write(tmp_path, "academy", f"scope: user\n{_MINIMAL_FILE}"))
        assert "user" in str(excinfo.value)


class TestPlatformStateIsNotAuthorable:
    """per D21 visibility and grants are administered rows, never file content."""

    def test_rejects_a_file_level_visibility(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(_write(tmp_path, "academy", f"visibility: public\n{_MINIMAL_FILE}"))
        assert "visibility" in str(excinfo.value)

    def test_rejects_a_file_level_grants(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(_write(tmp_path, "academy", f"grants: [acme]\n{_MINIMAL_FILE}"))
        assert "grants" in str(excinfo.value)

    def test_rejects_visibility_inside_the_definition_body(self, tmp_path: Path) -> None:
        text = _MINIMAL_FILE.replace("definition:\n", "definition:\n  visibility: restricted\n")
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(_write(tmp_path, "academy", text))
        assert "visibility" in str(excinfo.value)

    def test_rejects_dataset_grants_inside_the_definition_body(self, tmp_path: Path) -> None:
        text = _MINIMAL_FILE.replace("definition:\n", "definition:\n  grants: [acme]\n")
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(_write(tmp_path, "academy", text))
        assert "grants" in str(excinfo.value)

    def test_the_rejection_points_at_the_admin_path(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(_write(tmp_path, "academy", f"visibility: public\n{_MINIMAL_FILE}"))
        assert "admin" in str(excinfo.value).lower()

    def test_the_warehouse_grant_under_delivery_is_still_authorable(self, tmp_path: Path) -> None:
        # DeliverySpec.grants is the warehouse GRANT, not the D21 dataset
        # grant, and it is excluded from the content hash. Rejecting it here
        # would strip a delivered contract to close an unrelated hole.
        text = (
            f"name: academy_influencers\ndefinition:\n{_MINIMAL_BODY}"
            "  delivery:\n"
            "    grants:\n"
            "      - privilege: select\n"
            "        grantee_kind: group\n"
            "        grantee: influencers\n"
        )
        loaded = load_dataset_file(_write(tmp_path, "academy", text))
        assert [grant.grantee for grant in loaded.definition.delivery.grants] == ["influencers"]


class TestMalformedInput:
    """what a parser owes an author who mistypes."""

    def test_rejects_malformed_yaml_naming_the_file_and_line(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "academy", "name: academy\ndefinition:\n  units: [\n   - broken\n")
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(path)
        message = str(excinfo.value)
        assert str(path) in message
        assert "line 4, column 4" in message
        # PyYAML's own problem mark must name the path too, not
        # "<unicode string>": in a directory of files the mark is the only
        # part an author reads.
        assert message.count(str(path)) == 2

    def test_rejects_a_document_that_is_not_a_mapping(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(_write(tmp_path, "academy", "- one\n- two\n"))
        assert "mapping" in str(excinfo.value)

    def test_rejects_an_empty_document(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFileError):
            load_dataset_file(_write(tmp_path, "academy", "\n"))

    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFileError):
            load_dataset_file(tmp_path / f"absent{DATASET_FILE_SUFFIX}")

    def test_rejects_a_misspelled_field(self, tmp_path: Path) -> None:
        # the gap `aibots knowledge validate` carries: its `enforcement` field
        # is an untyped dict, so a misspelling passes. DatasetDefinition is
        # typed end to end with extra="forbid" and must stay that way.
        text = _MINIMAL_FILE.replace("  artifacts:", "  artefacts: []\n  artifacts:")
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(_write(tmp_path, "academy", text))
        assert "artefacts" in str(excinfo.value)

    def test_rejects_a_misspelled_nested_field(self, tmp_path: Path) -> None:
        text = _MINIMAL_FILE.replace("entity_column:", "entity_colum:")
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(_write(tmp_path, "academy", text))
        assert "entity_colum" in str(excinfo.value)

    def test_the_definition_model_still_forbids_extra_fields(self) -> None:
        assert DatasetDefinition.model_config["extra"] == "forbid"
        assert DatasetFile.model_config["extra"] == "forbid"


class TestOffline:
    """validation reaches nothing, and the test proves it rather than assuming."""

    def test_parsing_touches_no_socket(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = _write(tmp_path, "academy", _MINIMAL_FILE)

        def _blocked(*_args: object, **_kwargs: object) -> socket.socket:
            raise OSError("network disabled for this test")

        monkeypatch.setattr(socket, "socket", _blocked)
        monkeypatch.setattr(socket, "create_connection", _blocked)
        assert load_dataset_file(path).name == "academy_influencers"

    def test_directory_loading_touches_no_socket(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        datasets = tmp_path / DATASETS_DIR_NAME
        _write(datasets, "academy", _MINIMAL_FILE)

        def _blocked(*_args: object, **_kwargs: object) -> socket.socket:
            raise OSError("network disabled for this test")

        monkeypatch.setattr(socket, "socket", _blocked)
        monkeypatch.setattr(socket, "create_connection", _blocked)
        assert list(load_dataset_files(datasets)) == ["academy_influencers"]


class TestDirectoryLoading:
    """files are name-keyed, and a repeated name is ambiguous."""

    def test_loads_every_file_keyed_by_definition_name(self, tmp_path: Path) -> None:
        datasets = tmp_path / DATASETS_DIR_NAME
        _write(datasets, "academy", _MINIMAL_FILE)
        _write(datasets, "donors", _MINIMAL_FILE.replace("academy_influencers", "donor_influencers"))
        loaded = load_dataset_files(datasets)
        assert sorted(loaded) == ["academy_influencers", "donor_influencers"]

    def test_rejects_a_duplicate_name_across_files(self, tmp_path: Path) -> None:
        datasets = tmp_path / DATASETS_DIR_NAME
        _write(datasets, "academy", _MINIMAL_FILE)
        _write(datasets, "academy_copy", _MINIMAL_FILE)
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_files(datasets)
        message = str(excinfo.value)
        assert "academy_influencers" in message
        assert f"academy{DATASET_FILE_SUFFIX}" in message
        assert f"academy_copy{DATASET_FILE_SUFFIX}" in message

    def test_a_single_file_path_loads_regardless_of_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "one.yaml"
        path.write_text(_MINIMAL_FILE, encoding="utf-8")
        assert discover_dataset_files(path) == [path]

    def test_an_empty_directory_is_an_error(self, tmp_path: Path) -> None:
        datasets = tmp_path / DATASETS_DIR_NAME
        datasets.mkdir()
        with pytest.raises(DatasetFileError) as excinfo:
            discover_dataset_files(datasets)
        assert DATASET_FILE_SUFFIX in str(excinfo.value)

    def test_an_absent_path_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFileError):
            discover_dataset_files(tmp_path / "nowhere")

    def test_a_misnamed_yaml_is_reported_as_unloaded(self, tmp_path: Path) -> None:
        datasets = tmp_path / DATASETS_DIR_NAME
        _write(datasets, "academy", _MINIMAL_FILE)
        stray = datasets / "donors.yaml"
        stray.write_text(_MINIMAL_FILE, encoding="utf-8")
        assert unloaded_dataset_files(datasets) == [stray]

    def test_the_unloaded_warning_names_the_required_suffix(self, tmp_path: Path) -> None:
        datasets = tmp_path / DATASETS_DIR_NAME
        _write(datasets, "academy", _MINIMAL_FILE)
        (datasets / "donors.yml").write_text(_MINIMAL_FILE, encoding="utf-8")
        lines: list[str] = []
        warn: Callable[[str], None] = lines.append
        assert warn_unloaded_dataset_files(datasets, warn) == 1
        assert DATASET_FILE_SUFFIX in lines[0]

    def test_a_single_file_path_has_no_unloaded_siblings(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "academy", _MINIMAL_FILE)
        assert unloaded_dataset_files(path) == []


class TestDrift:
    """absence is drift, and drift is reported rather than acted on."""

    def test_a_name_in_the_store_and_not_the_files_is_drift(self) -> None:
        report = dataset_drift(["academy_influencers"], ["academy_influencers", "retired_audience"])
        assert report.drift == ["retired_audience"]

    def test_a_name_in_both_is_not_drift(self) -> None:
        assert dataset_drift(["academy_influencers"], ["academy_influencers"]).drift == []

    def test_a_name_only_in_the_files_is_not_drift(self) -> None:
        assert dataset_drift(["academy_influencers", "new_audience"], ["academy_influencers"]).drift == []

    def test_drift_is_sorted_and_deduplicated(self) -> None:
        report = dataset_drift([], ["b", "a", "b"])
        assert report.drift == ["a", "b"]

    def test_the_report_carries_nothing_that_was_deleted(self) -> None:
        assert set(DatasetDriftReport.model_fields) == {"drift"}

    def test_the_module_issues_no_delete_call_at_all(self) -> None:
        # deleting a definition strands its runs' inventory, and the reaper
        # keys on that inventory, so the tables become unfindable objects.
        # the guard is structural: no delete verb may appear in this module.
        source = Path(__file__).resolve().parents[3] / "src" / "threetears" / "datasources" / "definition"
        tree = ast.parse((source / "file_model.py").read_text(encoding="utf-8"))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert called.isdisjoint({"delete", "unlink", "remove", "rmtree", "drop", "discard", "pop"})


class TestContentHashRoundTrip:
    """the content hash is the version, so a drifting serialisation mints one."""

    def test_a_file_hashes_identically_to_the_in_memory_model(self, tmp_path: Path) -> None:
        from_file = load_dataset_file(_write(tmp_path, "academy", _MINIMAL_FILE)).definition
        in_memory = DatasetDefinition.model_validate(_equivalent_payload())
        assert from_file.content_hash == in_memory.content_hash

    def test_a_file_is_structurally_identical_to_the_in_memory_model(self, tmp_path: Path) -> None:
        from_file = load_dataset_file(_write(tmp_path, "academy", _MINIMAL_FILE)).definition
        assert from_file == DatasetDefinition.model_validate(_equivalent_payload())

    def test_the_committed_definition_survives_a_yaml_round_trip(self, tmp_path: Path) -> None:
        payload = json.loads(_JSON_FIXTURE.read_text(encoding="utf-8"))
        name = payload.pop("name")
        original = DatasetDefinition.model_validate({"name": name, **payload})
        rendered = render_dataset_file(DatasetFile(name=name, definition=original))
        restored = load_dataset_file(_write(tmp_path, "universal", rendered)).definition
        assert restored.content_hash == original.content_hash

    def test_the_committed_definition_is_structurally_identical_after_a_round_trip(self, tmp_path: Path) -> None:
        payload = json.loads(_JSON_FIXTURE.read_text(encoding="utf-8"))
        name = payload.pop("name")
        original = DatasetDefinition.model_validate({"name": name, **payload})
        rendered = render_dataset_file(DatasetFile(name=name, definition=original))
        assert load_dataset_file(_write(tmp_path, "universal", rendered)).definition == original

    def test_the_rendered_file_carries_no_name_inside_the_body(self, tmp_path: Path) -> None:
        original = DatasetDefinition.model_validate(_equivalent_payload())
        rendered = render_dataset_file(DatasetFile(name=original.name, definition=original))
        assert load_dataset_file(_write(tmp_path, "academy", rendered)).name == "academy_influencers"

    def test_a_text_literal_and_a_decimal_literal_are_different_definitions(self, tmp_path: Path) -> None:
        # the defect dsm-task-01d found: a text '1.0' compared against a
        # numeric column hashed identically to the numeric 1.0, so the edit
        # between them minted no version at all.
        as_text = load_dataset_file(
            _write(tmp_path / "text", "academy", _with_qualification("            literal: '1.0'\n"))
        ).definition
        as_decimal = load_dataset_file(
            _write(tmp_path / "decimal", "academy", _with_qualification("            literal: 1.0\n"))
        ).definition
        assert as_text.content_hash != as_decimal.content_hash

    def test_a_yaml_round_trip_preserves_literal_type(self, tmp_path: Path) -> None:
        for spelling, expected, value in (
            ("'1.0'", LiteralType.TEXT, "1.0"),
            ("1.0", LiteralType.DECIMAL, Decimal("1.0")),
        ):
            original = load_dataset_file(
                _write(tmp_path / expected.value, "academy", _with_qualification(f"            literal: {spelling}\n"))
            ).definition
            rendered = render_dataset_file(DatasetFile(name=original.name, definition=original))
            restored = load_dataset_file(_write(tmp_path / f"{expected.value}_again", "academy", rendered)).definition
            literal = _sole_literal(restored)
            assert literal.literal_type is expected
            assert literal.literal == value
            assert restored.content_hash == original.content_hash

    def test_a_declared_decimal_tag_coerces_a_quoted_scalar(self, tmp_path: Path) -> None:
        # exactly the shape render_dataset_file emits: JSON mode writes a
        # Decimal as a string, and only the tag says it was a number.
        original = load_dataset_file(
            _write(
                tmp_path,
                "academy",
                _with_qualification("            literal: '1.0'\n            literal_type: decimal\n"),
            )
        ).definition
        literal = _sole_literal(original)
        assert literal.literal_type is LiteralType.DECIMAL
        assert literal.literal == Decimal("1.0")

    def test_a_declared_text_tag_will_not_reinterpret_a_yaml_number(self, tmp_path: Path) -> None:
        # the tag decides, but it decides by refusing a scalar that cannot
        # carry it -- not by silently restringifying one. a tag that could
        # rewrite the scalar would let a cosmetic edit change the SQL.
        with pytest.raises(DatasetFileError) as excinfo:
            load_dataset_file(
                _write(
                    tmp_path,
                    "academy",
                    _with_qualification("            literal: 1.0\n            literal_type: text\n"),
                )
            )
        assert "requires a string" in str(excinfo.value)


def _sole_literal(definition: DatasetDefinition) -> LiteralExpression:
    """the single literal operand of the qualification fixture.

    :param definition: definition parsed from a qualification fixture
    :ptype definition: DatasetDefinition
    :returns: right-hand literal of the sole comparison
    :rtype: LiteralExpression
    """
    predicate = definition.qualification[0].predicate
    assert predicate is not None
    comparison = predicate.compare
    assert comparison is not None
    right = comparison.right
    assert isinstance(right, LiteralExpression)
    return right
