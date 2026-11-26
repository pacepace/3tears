"""The cleanup floor the change log's owner publishes in its metadata is read only for the log's own incarnation.

A pod reads the restore window's cleanup floor F from the log's ``STREAM.INFO`` metadata (``ChangeLogState.cleanup_floor``)
because it may not read the log by sequence. A value published for another incarnation (a lost log's) is not this log's
floor, and an absent or malformed one is none: the pod's cleanup then removes nothing.
"""

from __future__ import annotations

from threetears.nats.change_log import cleanup_floor_metadata, published_cleanup_floor


def test_a_floor_published_for_this_incarnation_is_read() -> None:
    assert published_cleanup_floor(cleanup_floor_metadata(42 << 15, "inc-1"), "inc-1") == 42 << 15


def test_a_floor_of_another_incarnation_or_none_or_malformed_is_no_floor() -> None:
    """Flips if a lost log's floor, or a value that names no floor, is read as this log's F."""
    assert published_cleanup_floor(cleanup_floor_metadata(42, "lost"), "inc-1") is None
    assert published_cleanup_floor(None, "inc-1") is None
    assert published_cleanup_floor("", "inc-1") is None
    assert published_cleanup_floor("inc-1:", "inc-1") is None
    assert published_cleanup_floor("inc-1:-5", "inc-1") is None
    assert published_cleanup_floor("42", "inc-1") is None


def test_an_incarnation_with_a_colon_still_reads_its_floor() -> None:
    assert published_cleanup_floor(cleanup_floor_metadata(7, "created-2026:11:26"), "created-2026:11:26") == 7
