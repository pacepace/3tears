"""Shared utilities for agent tools."""

from __future__ import annotations

__all__ = [
    "tool_error",
]


def tool_error(tool_name: str, action: str, error: str) -> str:
    """Format a standardized tool error message."""
    return f"[TOOL ERROR] {tool_name}: {action} failed — {error}"
