"""Smoke tests -- verify the package imports cleanly."""

from __future__ import annotations


def test_package_imports():
    """All public symbols import without error."""
    from threetears.langgraph import (
        CheckpointL1Cache,
        CheckpointL2Cache,
        ThreeTierCheckpointSaver,
    )

    assert ThreeTierCheckpointSaver is not None
    assert CheckpointL1Cache is not None
    assert CheckpointL2Cache is not None


def test_version():
    from threetears.langgraph import __version__

    assert __version__ == "0.10.2"


def test_protocols_are_runtime_checkable():
    from threetears.langgraph.protocols import CheckpointL1Cache, CheckpointL2Cache, FlushCallback

    # Verify the protocols have __protocol_attrs__
    assert hasattr(CheckpointL1Cache, "__protocol_attrs__") or hasattr(CheckpointL1Cache, "__abstractmethods__")
    assert hasattr(CheckpointL2Cache, "__protocol_attrs__") or hasattr(CheckpointL2Cache, "__abstractmethods__")
    assert hasattr(FlushCallback, "__protocol_attrs__") or hasattr(FlushCallback, "__abstractmethods__")
