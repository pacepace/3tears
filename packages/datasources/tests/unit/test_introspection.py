"""tests for :mod:`threetears.datasources.introspection`.

three contracts under test:

1. :func:`compute_column_hash` produces the SAME byte output as the
   driver-side SQL ``MD5(STRING_AGG(column_name || ':' || data_type
   || ':' || COALESCE(is_nullable, ''), ',' ORDER BY ordinal_position
   ))``. byte-equivalence is verified against real warehouses in the
   driver integration tests (``test_asyncpg_driver_live`` /
   ``test_redshift_driver_live``); this file covers the python-side
   invariants: order-insensitivity, sorting by ordinal, raw
   ``is_nullable`` handling.

2. :class:`IntrospectionDiff` carries the right work lists +
   summary counts.

3. :func:`compute_introspection_diff` classifies every
   ``(warehouse_hashes, stored_hashes)`` case correctly per the
   ``datasource-task-03`` spec: matching = unchanged; differing =
   changed; null-stored = changed (forced re-introspect); warehouse-
   only = added; storage-only = removed.
"""

from __future__ import annotations

import hashlib

import pytest

from threetears.datasources.introspection import (
    IntrospectionDiff,
    column_hash_payload,
    compute_column_hash,
    compute_introspection_diff,
)


def _joined(*column_payloads: bytes) -> bytes:
    """the payload compute_column_hash now hashes, built from readable parts.

    Each column's ``name:type:nullable`` is hashed FIRST and the digests are
    joined -- see ``column_hash_payload`` for why (Redshift's LISTAGG refuses
    a result over 65535 bytes, and the raw join overflowed on a 1462-column
    table). Writing the expectation this way keeps the literals in these tests
    readable while pinning the real formula.

    :param column_payloads: per-column ``name:type:nullable`` byte strings
    :ptype column_payloads: bytes
    :return: the joined digest payload
    :rtype: bytes
    """
    return ",".join(
        hashlib.md5(part).hexdigest()  # noqa: S324 -- change-detect signal
        for part in column_payloads
    ).encode()


class TestComputeColumnHash:
    """Python-side hash MUST be byte-stable + match the SQL formula."""

    def test_empty_column_list_is_md5_of_empty_string(self) -> None:
        # ",".join(empty_iter) == "" ; md5("") == "d41d8cd98f00b204e9800998ecf8427e"
        assert compute_column_hash([]) == hashlib.md5(b"").hexdigest()  # noqa: S324

    def test_single_column_payload(self) -> None:
        cols = [
            {
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": "NO",
                "ordinal_position": 1,
            }
        ]
        expected = hashlib.md5(_joined(b"id:integer:NO")).hexdigest()  # noqa: S324
        assert compute_column_hash(cols) == expected

    def test_multi_column_payload_sorted_by_ordinal(self) -> None:
        # input rows in REVERSE ordinal order; hash MUST sort by ordinal
        cols = [
            {"column_name": "z", "data_type": "text", "is_nullable": "YES", "ordinal_position": 3},
            {"column_name": "a", "data_type": "integer", "is_nullable": "NO", "ordinal_position": 1},
            {"column_name": "m", "data_type": "text", "is_nullable": "YES", "ordinal_position": 2},
        ]
        expected = hashlib.md5(_joined(b"a:integer:NO", b"m:text:YES", b"z:text:YES")).hexdigest()  # noqa: S324
        assert compute_column_hash(cols) == expected

    def test_input_order_does_not_change_output(self) -> None:
        """sorting by ordinal_position means input list order is irrelevant."""
        cols_a = [
            {"column_name": "a", "data_type": "int", "is_nullable": "NO", "ordinal_position": 1},
            {"column_name": "b", "data_type": "int", "is_nullable": "NO", "ordinal_position": 2},
        ]
        cols_b = list(reversed(cols_a))
        assert compute_column_hash(cols_a) == compute_column_hash(cols_b)

    def test_none_is_nullable_renders_as_empty_string(self) -> None:
        """matches SQL's ``COALESCE(is_nullable, '')`` byte output."""
        cols = [
            {
                "column_name": "x",
                "data_type": "text",
                "is_nullable": None,
                "ordinal_position": 1,
            }
        ]
        # payload becomes 'x:text:' (trailing empty after the second colon)
        expected = hashlib.md5(_joined(b"x:text:")).hexdigest()  # noqa: S324
        assert compute_column_hash(cols) == expected

    def test_missing_is_nullable_renders_as_empty_string(self) -> None:
        """``.get('is_nullable') or ''`` handles missing keys gracefully."""
        cols = [
            {
                "column_name": "x",
                "data_type": "text",
                # is_nullable key absent
                "ordinal_position": 1,
            }
        ]
        expected = hashlib.md5(_joined(b"x:text:")).hexdigest()  # noqa: S324
        assert compute_column_hash(cols) == expected

    def test_raw_is_nullable_strings_are_used_verbatim(self) -> None:
        """``'YES'``/``'NO'`` byte values pass through unchanged.

        catches a regression where a future contributor normalizes
        the warehouse string to a bool ("YES" -> True) before
        hashing. that breaks byte-equivalence with the SQL.
        """
        cols_yes = [{"column_name": "x", "data_type": "text", "is_nullable": "YES", "ordinal_position": 1}]
        cols_yes_lower = [{"column_name": "x", "data_type": "text", "is_nullable": "yes", "ordinal_position": 1}]
        # different byte strings produce different hashes
        assert compute_column_hash(cols_yes) != compute_column_hash(cols_yes_lower)

    def test_input_not_mutated(self) -> None:
        """function is pure; doesn't mutate the input list or rows."""
        cols = [
            {"column_name": "b", "data_type": "int", "is_nullable": "NO", "ordinal_position": 2},
            {"column_name": "a", "data_type": "int", "is_nullable": "NO", "ordinal_position": 1},
        ]
        snapshot = [dict(c) for c in cols]
        compute_column_hash(cols)
        # original list + dicts unchanged
        assert cols == snapshot
        # order in the original list is unchanged
        assert cols[0]["column_name"] == "b"


# ---------------------------------------------------------------------------
# IntrospectionDiff dataclass shape + invariants
# ---------------------------------------------------------------------------


class TestIntrospectionDiffShape:
    """frozen dataclass + ``has_changes`` derivation."""

    def test_is_frozen(self) -> None:
        diff = IntrospectionDiff(
            tables_to_introspect=(),
            tables_to_delete=(),
            tables_checked=0,
            tables_unchanged=0,
            tables_changed=0,
            tables_added=0,
            tables_removed=0,
        )
        with pytest.raises(Exception):  # noqa: B017 -- dataclass FrozenInstanceError
            diff.tables_checked = 99  # type: ignore[misc]

    def test_has_changes_false_for_empty_work_lists(self) -> None:
        diff = IntrospectionDiff(
            tables_to_introspect=(),
            tables_to_delete=(),
            tables_checked=5,
            tables_unchanged=5,
            tables_changed=0,
            tables_added=0,
            tables_removed=0,
        )
        assert diff.has_changes is False

    def test_has_changes_true_when_to_introspect_non_empty(self) -> None:
        diff = IntrospectionDiff(
            tables_to_introspect=(("s", "t"),),
            tables_to_delete=(),
            tables_checked=1,
            tables_unchanged=0,
            tables_changed=0,
            tables_added=1,
            tables_removed=0,
        )
        assert diff.has_changes is True

    def test_has_changes_true_when_to_delete_non_empty(self) -> None:
        diff = IntrospectionDiff(
            tables_to_introspect=(),
            tables_to_delete=(("s", "t"),),
            tables_checked=0,
            tables_unchanged=0,
            tables_changed=0,
            tables_added=0,
            tables_removed=1,
        )
        assert diff.has_changes is True

    def test_columns_and_elapsed_default_zero(self) -> None:
        """columns_*, elapsed_ms are 0 from compute_introspection_diff."""
        diff = IntrospectionDiff(
            tables_to_introspect=(),
            tables_to_delete=(),
            tables_checked=0,
            tables_unchanged=0,
            tables_changed=0,
            tables_added=0,
            tables_removed=0,
        )
        assert diff.columns_added == 0
        assert diff.columns_removed == 0
        assert diff.columns_changed == 0
        assert diff.elapsed_ms == 0


# ---------------------------------------------------------------------------
# compute_introspection_diff classification rules
# ---------------------------------------------------------------------------


class TestComputeIntrospectionDiff:
    """classification of every (warehouse_hash, stored_hash) case."""

    def test_unchanged_when_hashes_match(self) -> None:
        wh = {("s", "t"): "abc123"}
        st = {("s", "t"): "abc123"}
        diff = compute_introspection_diff(wh, st)
        assert diff.tables_unchanged == 1
        assert diff.tables_changed == 0
        assert diff.tables_to_introspect == ()
        assert diff.has_changes is False

    def test_changed_when_hashes_differ(self) -> None:
        wh = {("s", "t"): "warehouse-hash"}
        st = {("s", "t"): "old-stored-hash"}
        diff = compute_introspection_diff(wh, st)
        assert diff.tables_changed == 1
        assert diff.tables_to_introspect == (("s", "t"),)

    def test_null_stored_hash_treated_as_changed(self) -> None:
        """``stored_hashes[key] is None`` forces re-introspect.

        regression guard for the migration backfill case: after the
        ``column_hash`` migration adds the column with NULL on
        existing rows, every table appears as "stored hash is null"
        and MUST get a forced re-introspect on the next probe so the
        hashes populate.
        """
        wh = {("s", "t"): "abc"}
        st = {("s", "t"): None}
        diff = compute_introspection_diff(wh, st)
        assert diff.tables_changed == 1
        assert diff.tables_to_introspect == (("s", "t"),)

    def test_table_added_when_in_warehouse_only(self) -> None:
        wh = {("s", "new"): "hash"}
        st = {}
        diff = compute_introspection_diff(wh, st)
        assert diff.tables_added == 1
        assert diff.tables_to_introspect == (("s", "new"),)

    def test_table_removed_when_in_storage_only(self) -> None:
        wh = {}
        st = {("s", "gone"): "hash"}
        diff = compute_introspection_diff(wh, st)
        assert diff.tables_removed == 1
        assert diff.tables_to_delete == (("s", "gone"),)

    def test_mixed_diff(self) -> None:
        """exercise all five classification paths in one call."""
        wh = {
            ("s", "unchanged"): "h1",
            ("s", "changed"): "h2-new",
            ("s", "null-stored"): "h3",
            ("s", "added"): "h4",
        }
        st = {
            ("s", "unchanged"): "h1",
            ("s", "changed"): "h2-old",
            ("s", "null-stored"): None,
            ("s", "removed"): "h5",
        }
        diff = compute_introspection_diff(wh, st)
        assert diff.tables_checked == 4
        assert diff.tables_unchanged == 1
        assert diff.tables_changed == 2  # changed + null-stored
        assert diff.tables_added == 1
        assert diff.tables_removed == 1
        # work lists -- sorted deterministically
        assert set(diff.tables_to_introspect) == {
            ("s", "changed"),
            ("s", "null-stored"),
            ("s", "added"),
        }
        assert diff.tables_to_delete == (("s", "removed"),)
        assert diff.has_changes is True

    def test_empty_inputs_produce_empty_diff(self) -> None:
        diff = compute_introspection_diff({}, {})
        assert diff.tables_checked == 0
        assert diff.tables_unchanged == 0
        assert diff.tables_changed == 0
        assert diff.tables_added == 0
        assert diff.tables_removed == 0
        assert diff.tables_to_introspect == ()
        assert diff.tables_to_delete == ()
        assert diff.has_changes is False

    def test_returns_deterministic_order(self) -> None:
        """work lists are sorted so test assertions don't depend on dict iteration."""
        wh = {
            ("s", "b"): "h",
            ("s", "a"): "h",
            ("s", "c"): "h",
        }
        st = {}  # everything is "added"
        diff = compute_introspection_diff(wh, st)
        # sorted ascending by (schema, table)
        assert diff.tables_to_introspect == (("s", "a"), ("s", "b"), ("s", "c"))


class TestTheHashPayloadIsBoundedByColumnCount:
    """Redshift's LISTAGG refuses a result over 65535 bytes, and it took the
    whole introspection with it.

    THE INCIDENT. Introspecting ripple's nine source schemas failed outright:

        IntrospectionError: failed to probe table hashes on datasource
        influencers-read: Result size exceeds LISTAGG limit ... limit: 65535

    Measured cause: ``influencers.test_targetsmart`` carries 1462 columns whose
    ``name:type:nullable`` payload concatenates to 69,210 bytes. ONE table over
    the limit fails the query for all 5,615 tables in scope, so a single wide
    table anywhere in an allow-listed schema disabled schema introspection
    entirely -- and the agent then has no catalog to resolve names against.

    THE FIX, and why this shape. Each column is hashed FIRST and the digests
    are concatenated, so the payload is 33 bytes per column regardless of how
    long the column names and type names are. Size now depends on column COUNT
    alone, which is the thing a warehouse actually bounds.

    The ceiling is still finite and worth stating: 65535 / 33 is about 1985
    columns. Redshift's own limit is 1600 columns per table, so the payload can
    no longer overflow on Redshift at all. That is the point of the shape.
    """

    def test_the_payload_is_fixed_width_per_column(self) -> None:
        """a very long column name must not enlarge the payload.

        This is the whole property. Under the old formula the payload grew
        with name and type lengths, so a wide table of verbosely-named columns
        overflowed while a wider table of short names did not -- which is why
        this surfaced on one warehouse and not another.

        :returns: none
        :rtype: None
        """
        short = [{"column_name": "a", "data_type": "int", "is_nullable": "YES", "ordinal_position": 1}]
        long = [
            {
                "column_name": "a" * 400,
                "data_type": "character varying" * 20,
                "is_nullable": "YES",
                "ordinal_position": 1,
            }
        ]
        # asserted on the PAYLOAD, not the digest. The digest is 32 chars
        # whatever the payload was, so a digest-length assertion would pass
        # against the very formula that overflowed.
        assert len(column_hash_payload(short)) == len(column_hash_payload(long)), (
            "the payload length still depends on column-name and type-name length. A wide table "
            "of verbosely-named columns will overflow Redshift's LISTAGG limit while a wider "
            "table of short names does not"
        )
        assert compute_column_hash(short) != compute_column_hash(long)

    def test_a_redshift_maximum_width_table_stays_under_the_limit(self) -> None:
        """1600 columns is Redshift's per-table ceiling; the payload must fit.

        Asserted on the payload SIZE the SQL side would build, not on the
        digest, because the digest is always 32 chars and would pass whatever
        the intermediate did.

        :returns: none
        :rtype: None
        """
        columns = [
            {
                "column_name": f"some_reasonably_long_column_name_{index}",
                "data_type": "character varying",
                "is_nullable": "YES",
                "ordinal_position": index,
            }
            for index in range(1, 1601)
        ]
        payload_bytes = len(column_hash_payload(columns).encode())
        assert payload_bytes < 65535, (
            f"a 1600-column table would build a {payload_bytes}-byte LISTAGG payload, over "
            f"Redshift's 65535 limit. The per-column pre-hash is what keeps this bounded"
        )
        assert len(compute_column_hash(columns)) == 32

    def test_order_still_matters(self) -> None:
        """the hash is a CHANGE probe, so a reordered column set must differ.

        :returns: none
        :rtype: None
        """
        first = [
            {"column_name": "a", "data_type": "int", "is_nullable": "YES", "ordinal_position": 1},
            {"column_name": "b", "data_type": "int", "is_nullable": "YES", "ordinal_position": 2},
        ]
        swapped = [
            {"column_name": "b", "data_type": "int", "is_nullable": "YES", "ordinal_position": 1},
            {"column_name": "a", "data_type": "int", "is_nullable": "YES", "ordinal_position": 2},
        ]
        assert compute_column_hash(first) != compute_column_hash(swapped)

    def test_a_type_change_still_changes_the_hash(self) -> None:
        """the probe must still notice the thing it exists to notice.

        :returns: none
        :rtype: None
        """
        before = [{"column_name": "a", "data_type": "smallint", "is_nullable": "YES", "ordinal_position": 1}]
        after = [{"column_name": "a", "data_type": "integer", "is_nullable": "YES", "ordinal_position": 1}]
        assert compute_column_hash(before) != compute_column_hash(after)
