"""tests for :mod:`threetears.datasources.export`, the export wire and the drivers' ``unload``.

The export is a security surface: a caller's ``SELECT`` is embedded in a statement the warehouse
runs with write access to a bucket. These pin that the statement around it is fixed, that the
``SELECT`` cannot close its literal, that a destination cannot leave the configured prefix, that a
driver with no export configured (or no export at all) refuses, and that the wire omits the export
field from every other request so an older hub still reads them.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import uuid7

import pytest
from pydantic import ValidationError

from threetears.datasources.config import BigQueryConnectionConfig, RedshiftConnectionConfig
from threetears.datasources.drivers.bigquery_driver import BigQueryDriver
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.entities import DataSourceType
from threetears.datasources.export import (
    DriverExportUnsupportedError,
    ExportConfig,
    ExportLocation,
    ExportRefusedError,
    ExportResult,
    check_destination,
    redshift_unload_statement,
    sql_string_literal,
)
from threetears.datasources.query_client import (
    DatasourceExportDeleteRequest,
    DatasourceExportRequest,
    DatasourceQueryRequest,
    RelationFingerprintRequest,
)

_PASSWORD_ENV = "TEST_EXPORT_REDSHIFT_PW"
_CONFIG = ExportConfig(bucket="bl-eng-aibots-reports-export-dev")


@pytest.fixture(autouse=True)
def _password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_PASSWORD_ENV, "unit-test-password")


def _redshift(export: ExportConfig | None) -> RedshiftConnectionConfig:
    # left out rather than passed as None when absent, as a config read from a file leaves it out
    fields: dict[str, object] = {} if export is None else {"export": export}
    return RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host="rs.example.com",
        database="analytics",
        username="rs_user",
        password_ref=f"env://{_PASSWORD_ENV}",
        executor_max_workers=1,
        **fields,  # type: ignore[arg-type]
    )


def _connection(unload_count: int) -> MagicMock:
    conn = MagicMock(name="conn")
    cursor = MagicMock(name="cursor")
    cursor.fetchone = MagicMock(return_value=(unload_count,))
    cursor.description = []
    conn.cursor = MagicMock(return_value=cursor)
    conn.recorded_cursor = cursor
    return conn


class TestExportConfig:
    def test_defaults_are_the_exports_prefix_and_the_default_role(self) -> None:
        assert (_CONFIG.prefix, _CONFIG.iam_role) == ("exports/", "default")

    @pytest.mark.parametrize("bucket", ["Bad_Bucket", "s3://bucket", "a", "bucket/key", "a..b", "bucket-1\n"])
    def test_a_bucket_that_is_not_a_bucket_name_is_refused(self, bucket: str) -> None:
        with pytest.raises(ValidationError):
            ExportConfig(bucket=bucket)

    @pytest.mark.parametrize(
        "prefix", ["exports", "/exports/", "exports/../x/", "../", "exports//", "a/./", "exports/\n"]
    )
    def test_a_prefix_that_is_not_plain_segments_ending_in_a_slash_is_refused(self, prefix: str) -> None:
        with pytest.raises(ValidationError):
            ExportConfig(bucket="b-1-2", prefix=prefix)

    def test_a_role_is_default_or_an_arn(self) -> None:
        ExportConfig(bucket="b-1-2", iam_role="arn:aws:iam::924165706792:role/redshift-write")
        with pytest.raises(ValidationError):
            ExportConfig(bucket="b-1-2", iam_role="redshift-write' CREDENTIALS 'x")
        with pytest.raises(ValidationError):
            ExportConfig(bucket="b-1-2", iam_role="arn:aws:iam::924165706792:role/redshift-write\n")

    def test_the_cleanup_grant_is_a_pair_of_references_or_none(self) -> None:
        both = ExportConfig(
            bucket="b-1-2", cleanup_access_key_ref="env://CLEANUP_ID", cleanup_secret_key_ref="env://CLEANUP_SECRET"
        )
        assert both.cleanup_access_key_ref == "env://CLEANUP_ID"
        with pytest.raises(ValidationError):
            ExportConfig(bucket="b-1-2", cleanup_access_key_ref="env://CLEANUP_ID")
        with pytest.raises(ValidationError):
            ExportConfig(bucket="b-1-2", cleanup_access_key_ref="AKIAPLAINVALUE", cleanup_secret_key_ref="env://X")

    def test_the_connection_config_carries_it_and_refuses_an_unknown_key(self) -> None:
        assert _redshift(_CONFIG).export == _CONFIG
        assert _redshift(None).export is None
        with pytest.raises(ValidationError):
            RedshiftConnectionConfig.model_validate(
                {
                    "datasource_type": "redshift",
                    "host": "h",
                    "database": "d",
                    "export": {"bucket": "b-1-2", "allow_overwrite": True},
                }
            )

    def test_a_stored_config_without_an_export_does_not_name_one(self) -> None:
        """stored with exclude_unset, a config with no export reads on a release that predates it."""
        assert "export" not in json.loads(_redshift(None).model_dump_json(exclude_unset=True))


class TestUnloadStatement:
    def test_the_statement_is_fixed_around_the_select(self) -> None:
        statement, location = redshift_unload_statement("SELECT a FROM s.t", _CONFIG, "enr/run-1/VA")
        assert statement == (
            "UNLOAD ('SELECT a FROM s.t') TO 's3://bl-eng-aibots-reports-export-dev/exports/enr/run-1/VA/' "
            "IAM_ROLE default FORMAT AS PARQUET MANIFEST VERBOSE"
        )
        assert "ALLOWOVERWRITE" not in statement
        assert location == ExportLocation(
            bucket="bl-eng-aibots-reports-export-dev",
            object_prefix="exports/enr/run-1/VA/",
            manifest_path="exports/enr/run-1/VA/manifest",
        )

    def test_a_quote_in_the_select_cannot_close_the_literal(self) -> None:
        statement, _ = redshift_unload_statement("SELECT a FROM t WHERE s = 'VA'", _CONFIG, "x")
        assert "UNLOAD ('SELECT a FROM t WHERE s = ''VA''') TO" in statement

    def test_a_role_arn_is_quoted(self) -> None:
        arn = "arn:aws:iam::924165706792:role/redshift-s3-write"
        statement, _ = redshift_unload_statement("SELECT 1", ExportConfig(bucket="b-1-2", iam_role=arn), "x")
        assert f"IAM_ROLE '{arn}'" in statement

    @pytest.mark.parametrize("select", ["SELECT 'a\\') TO ''s3://evil/''--' FROM t", "SELECT 1\x00"])
    def test_a_select_a_literal_cannot_carry_is_refused(self, select: str) -> None:
        with pytest.raises(ExportRefusedError):
            redshift_unload_statement(select, _CONFIG, "x")

    @pytest.mark.parametrize(
        "destination", ["../other", "a/../b", "/abs", "s3://evil/x", "a/", "", "a//b", "a.b", "a b", "x'y", "a\n"]
    )
    def test_a_destination_outside_the_prefix_is_refused(self, destination: str) -> None:
        with pytest.raises(ExportRefusedError):
            check_destination(destination)
        with pytest.raises(ExportRefusedError):
            redshift_unload_statement("SELECT 1", _CONFIG, destination)

    def test_a_literal_doubles_its_quotes_and_refuses_a_backslash(self) -> None:
        assert sql_string_literal("O'Brien") == "'O''Brien'"
        with pytest.raises(ExportRefusedError):
            sql_string_literal("a\\")


class TestWire:
    def test_a_query_request_carries_no_export_field(self) -> None:
        """an older hub forbids unknown fields; a query must reach it as it always did."""
        query = DatasourceQueryRequest(correlation_id=uuid7(), identity_token="t", query="SELECT 1")
        fingerprint = DatasourceQueryRequest(
            correlation_id=uuid7(),
            identity_token="t",
            fingerprint=RelationFingerprintRequest(relation="s.t", key_columns=["a"]),
        )
        assert "export" not in json.loads(query.model_dump_json())
        assert "export" not in json.loads(fingerprint.model_dump_json())

    def test_an_export_request_round_trips(self) -> None:
        request = DatasourceQueryRequest(
            correlation_id=uuid7(),
            identity_token="t",
            export=DatasourceExportRequest(select="SELECT 1", destination="enr/x"),
        )
        wire = json.loads(request.model_dump_json())
        assert wire["export"] == {"select": "SELECT 1", "destination": "enr/x"}
        assert wire["identity_token"] == "t"
        assert DatasourceQueryRequest.model_validate_json(request.model_dump_json()).export == request.export

    def test_a_query_request_carries_no_export_delete_field_and_a_delete_round_trips(self) -> None:
        query = DatasourceQueryRequest(correlation_id=uuid7(), identity_token="t", query="SELECT 1")
        assert "export_delete" not in json.loads(query.model_dump_json())
        delete = DatasourceQueryRequest(
            correlation_id=uuid7(), identity_token="t", export_delete=DatasourceExportDeleteRequest(destination="enr/x")
        )
        assert json.loads(delete.model_dump_json())["export_delete"] == {"destination": "enr/x"}
        with pytest.raises(ValidationError):
            DatasourceExportDeleteRequest(destination="../x")

    def test_exactly_one_ask(self) -> None:
        with pytest.raises(ValidationError):
            DatasourceQueryRequest(
                correlation_id=uuid7(),
                identity_token="t",
                query="SELECT 1",
                export=DatasourceExportRequest(select="SELECT 1", destination="x"),
            )

    def test_an_export_binds_no_parameters(self) -> None:
        with pytest.raises(ValidationError):
            DatasourceQueryRequest(
                correlation_id=uuid7(),
                identity_token="t",
                params=["VA"],
                export=DatasourceExportRequest(select="SELECT 1 WHERE s = $1", destination="x"),
            )

    def test_the_request_refuses_a_destination_that_leaves_the_prefix(self) -> None:
        with pytest.raises(ValidationError):
            DatasourceExportRequest(select="SELECT 1", destination="../x")


class TestDriverUnload:
    @pytest.mark.asyncio
    async def test_a_redshift_datasource_with_no_export_configured_refuses(self) -> None:
        driver = RedshiftDriver(_redshift(None))
        with pytest.raises(DriverExportUnsupportedError):
            await driver.unload("SELECT 1", "x")
        assert driver.export_config is None

    @pytest.mark.asyncio
    async def test_a_driver_with_no_export_refuses(self) -> None:
        """the base answers every engine that cannot export, so a caller asks once and branches on the type."""
        driver = BigQueryDriver(
            BigQueryConnectionConfig(
                datasource_type=DataSourceType.BIGQUERY, project_id="p", credentials_json_ref="env://X"
            )
        )
        with pytest.raises(DriverExportUnsupportedError):
            await driver.unload("SELECT 1", "x")
        assert driver.export_config is None

    @pytest.mark.asyncio
    async def test_redshift_unloads_and_counts_on_one_session(self) -> None:
        conn = _connection(unload_count=7)
        with patch("threetears.datasources.drivers.redshift_driver.redshift_connector.connect", return_value=conn):
            driver = RedshiftDriver(_redshift(_CONFIG))
            result = await driver.unload("SELECT a FROM s.t WHERE s = 'VA'", "enr/r1/VA", timeout_seconds=100)
            await driver.close()
        statements = [c.args[0] for c in conn.recorded_cursor.execute.call_args_list if c.args]
        unloads = [s for s in statements if s.startswith("UNLOAD")]
        assert unloads == [
            "UNLOAD ('SELECT a FROM s.t WHERE s = ''VA''') TO "
            "'s3://bl-eng-aibots-reports-export-dev/exports/enr/r1/VA/' IAM_ROLE default FORMAT AS PARQUET "
            "MANIFEST VERBOSE"
        ]
        # the count is read right after, on the same cursor and session
        assert statements[statements.index(unloads[0]) + 1] == "SELECT pg_last_unload_count()"
        assert any(s.startswith("SET LOCAL statement_timeout") or "statement_timeout" in s for s in statements)
        assert result == ExportResult(
            row_count=7,
            bucket="bl-eng-aibots-reports-export-dev",
            object_prefix="exports/enr/r1/VA/",
            manifest_path="exports/enr/r1/VA/manifest",
        )
        assert driver.export_config == _CONFIG

    @pytest.mark.asyncio
    async def test_a_refused_destination_never_takes_a_connection(self) -> None:
        connect = MagicMock(side_effect=AssertionError("connected"))
        with patch("threetears.datasources.drivers.redshift_driver.redshift_connector.connect", connect):
            driver = RedshiftDriver(_redshift(_CONFIG))
            with pytest.raises(ExportRefusedError):
                await driver.unload("SELECT 1", "../elsewhere")
            await driver.close()
        connect.assert_not_called()
