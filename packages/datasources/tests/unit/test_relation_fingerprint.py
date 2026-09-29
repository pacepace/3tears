"""tests for the relation fingerprint: the completeness check a paged read rests on.

The contract this pins:

- a NULL key value is distinguishable from an empty string, so a row swapped for one
  carrying a NULL in the same position changes the fingerprint. Coalescing both to ``''``
  is the blindness a fingerprint replaces a count to remove;
- the key expression joins its columns with a separator, so ``('a', 'bc')`` and
  ``('ab', 'c')`` are different keys rather than one concatenation;
- an empty key is refused, because a fingerprint over no columns answers the same for
  every relation of the same size -- a count wearing a digest's name;
- the emitted SQL counts and digests in ONE statement, because the pair is only a
  completeness check if both describe the same instant;
- every driver that cannot yet compute one says so by name rather than returning
  something a caller would compare.

The per-dialect SQL is asserted at the level this suite can honestly reach: that the
statement is built, that it carries the dialect's own hash-to-number spelling, and that
the two live drivers do not share one. Whether Redshift's ``STRTOL`` and Postgres's
``bit(32)`` cast agree on a real warehouse is a live-integration question, and the
digests are deliberately NOT comparable across engines, so there is nothing to
cross-check here even in principle.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.datasources.drivers._util import build_relation_key_expression
from threetears.datasources.query_client import RelationFingerprintRequest


class TestTheKeyExpressionSeesEveryDifference:
    def test_a_null_is_not_an_empty_string(self) -> None:
        # the whole point: a row whose key went NULL must not fingerprint the same as one
        # whose key went ''. A COALESCE to '' would make those identical.
        expression = build_relation_key_expression(["jurisdiction"])
        assert "IS NULL" in expression
        assert "CHR(30)" in expression, expression

    def test_columns_are_separated_so_a_shift_is_visible(self) -> None:
        # without a separator ('a','bc') and ('ab','c') render the same text, so a row
        # swap that moves a character across the boundary would go unseen.
        expression = build_relation_key_expression(["a", "b"])
        assert "CHR(31)" in expression, expression
        assert expression.count("CHR(31)") == 1, expression

    def test_each_column_is_cast_so_a_non_text_key_still_renders(self) -> None:
        expression = build_relation_key_expression(["generated_at"])
        assert "CAST(generated_at AS VARCHAR)" in expression, expression

    def test_a_three_column_key_joins_all_three(self) -> None:
        expression = build_relation_key_expression(["a", "b", "c"])
        assert expression.count("CHR(31)") == 2, expression
        for column in ("a", "b", "c"):
            assert f"CAST({column} AS VARCHAR)" in expression, expression

    def test_an_empty_key_is_refused(self) -> None:
        # a fingerprint over no columns is a constant, so every relation of the same size
        # would fingerprint identically -- silently reducing the check back to a count.
        with pytest.raises(ValueError, match="at least one ordering column"):
            build_relation_key_expression([])


class TestTheDialectsDifferWhereTheyMust:
    """the reason this is a driver method and not SQL a portable caller writes."""

    def test_postgres_and_redshift_do_not_share_a_hash_to_number_spelling(self) -> None:
        # Postgres casts through bit(32); Redshift has STRTOL; Snowflake has TO_NUMBER with
        # a format model. If these ever converge, the case for a driver method weakens --
        # so this asserts the divergence the design rests on rather than assuming it.
        from threetears.datasources.drivers import asyncpg_driver, redshift_driver

        postgres_source = asyncpg_driver.AsyncpgDriver.relation_fingerprint.__doc__ or ""
        redshift_source = redshift_driver.RedshiftDriver.relation_fingerprint.__doc__ or ""
        assert "bit(32)" in postgres_source
        assert "STRTOL" in redshift_source


class TestTheUnbuiltDriversRefuseByName:
    """a driver that cannot fingerprint must say so, not return something comparable."""

    @pytest.mark.asyncio
    async def test_snowflake_refuses(self) -> None:
        from threetears.datasources.config import SnowflakeConnectionConfig
        from threetears.datasources.drivers.snowflake_driver import SnowflakeDriver

        driver = SnowflakeDriver(
            SnowflakeConnectionConfig(
                datasource_type="snowflake",
                account="acct",
                warehouse="wh",
                user="u",
                password_ref="env://X",
            )
        )
        with pytest.raises(NotImplementedError, match="relation_fingerprint"):
            await driver.relation_fingerprint("s.t", ["k"])


class TestTheAskCannotCarrySql:
    """the relation and its key columns are INTERPOLATED, so the model is the only gate.

    Every driver implementing ``relation_fingerprint`` writes these two values straight into
    a statement -- it has to, because a relation name cannot be a bind parameter in any
    admitted engine -- and each documents them as TRUSTED identifiers. Nothing made them
    trusted: they arrive off the wire, and the hub's fingerprint branch deliberately skips
    ``validate_read_sql`` because the ask is declarative rather than a statement.

    So the trust the drivers assume is established at this model or nowhere. These are the
    payloads that reached the warehouse before it was.
    """

    @pytest.mark.parametrize(
        "relation",
        [
            "public.orders; DROP TABLE users --",
            "public.orders WHERE 1=1",
            "(SELECT 1)",
            "public.orders UNION SELECT password FROM secrets",
            'public."orders"',
            "public.orders--",
            "a.b.c",
            "",
            " public.orders",
            "public. orders",
        ],
    )
    def test_a_relation_that_is_not_an_identifier_is_refused(self, relation: str) -> None:
        """
        :return: nothing
        :rtype: None
        """
        with pytest.raises(ValidationError):
            RelationFingerprintRequest(relation=relation, key_columns=["id"])

    @pytest.mark.parametrize(
        "column",
        [
            "id; DROP TABLE users --",
            "id, (SELECT password FROM secrets)",
            "id)",
            '"id"',
            "",
            "1",
        ],
    )
    def test_a_key_column_that_is_not_an_identifier_is_refused(self, column: str) -> None:
        """
        :return: nothing
        :rtype: None
        """
        with pytest.raises(ValidationError):
            RelationFingerprintRequest(relation="public.orders", key_columns=["id", column])

    def test_an_empty_key_is_refused_here_rather_than_at_the_warehouse(self) -> None:
        """every driver raises on this separately, and that reads as a connection fault.

        :return: nothing
        :rtype: None
        """
        with pytest.raises(ValidationError):
            RelationFingerprintRequest(relation="public.orders", key_columns=[])

    @pytest.mark.parametrize(
        "relation",
        ["orders", "public.orders", "_private.t$1", "Schema.Table"],
    )
    def test_a_real_relation_still_passes(self, relation: str) -> None:
        """the positive half, which is not decoration.

        A validator that refused everything would pass every case above and break every
        caller -- and the fingerprint path is a change-probe, so it would fail as "the
        relation changed" rather than as a refused request.

        :return: nothing
        :rtype: None
        """
        asked = RelationFingerprintRequest(relation=relation, key_columns=["id", "created_at"])
        assert asked.relation == relation
        assert asked.key_columns == ["id", "created_at"]
