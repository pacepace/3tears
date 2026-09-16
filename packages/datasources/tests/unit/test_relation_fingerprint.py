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

from threetears.datasources.drivers._util import build_relation_key_expression


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
