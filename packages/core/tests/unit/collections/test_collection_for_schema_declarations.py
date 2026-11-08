"""a factory-built collection carries its schema's write-generation declaration, and is the caller's class.

``collection_for_schema`` builds a class per table described as data. The class reads its
``write_generation`` from the schema whichever declaration it is (switched on, kept in L2, opted out with
its reason), and says it is defined in the module that built it, so a census of a consumer's modules
counts it as the consumer's.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from threetears.core.collections.generation import (
    WRITE_GENERATION,
    NoWriteGeneration,
    UndeclaredWriteGeneration,
    tables_with_write_generation,
)
from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema, collection_for_schema
from threetears.enforcement.collection_census.census import census


def _schema(name: str, declared: object) -> TableSchema:
    return TableSchema(name=name, primary_key="id", columns=[Column("id", STRING_TYPE)], write_generation=declared)  # type: ignore[arg-type]


def test_an_opt_out_with_its_reason_reaches_the_built_class() -> None:
    built = collection_for_schema(_schema("factory_opted_out", NoWriteGeneration("written every second")))
    assert built.write_generation == NoWriteGeneration("written every second")


def test_a_switched_on_schema_s_class_is_switched_on_and_registered() -> None:
    built = collection_for_schema(_schema("factory_switched_on", WRITE_GENERATION))
    assert built.write_generation is WRITE_GENERATION
    assert "factory_switched_on" in tables_with_write_generation()


def test_an_undeclared_schema_s_class_is_undeclared() -> None:
    built = collection_for_schema(
        TableSchema(name="factory_undeclared", primary_key="id", columns=[Column("id", STRING_TYPE)])
    )
    assert isinstance(built.write_generation, UndeclaredWriteGeneration)


def test_the_built_class_is_the_caller_s() -> None:
    built = collection_for_schema(_schema("factory_module", WRITE_GENERATION))
    assert built.__module__ == __name__
    assert collection_for_schema(_schema("factory_named", WRITE_GENERATION), module="consumer.tables").__module__ == (
        "consumer.tables"
    )


def test_a_census_of_a_consumer_s_modules_counts_its_factory_built_classes(tmp_path: Path) -> None:
    package = tmp_path / "consumer_reports_census"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "tables.py").write_text(
        textwrap.dedent(
            """
            from threetears.core.collections.generation import NoWriteGeneration
            from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema, collection_for_schema

            Votes = collection_for_schema(
                TableSchema(
                    name="consumer_votes",
                    primary_key="id",
                    columns=[Column("id", STRING_TYPE)],
                    write_generation=NoWriteGeneration("reloaded whole every night"),
                )
            )
            """
        )
    )
    found = census([tmp_path])
    assert found["import_failures"] == []
    [record] = [record for record in found["classes"] if record["table"] == "consumer_votes"]
    assert record["name"].startswith("consumer_reports_census.tables.")
    assert (record["declaration"], record["reason"]) == ("opted_out", "reloaded whole every night")
