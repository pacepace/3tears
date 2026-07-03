"""Small shared helpers for the langgraph package.

Provider-agnostic utilities with no LangGraph-loop dependency. (Relocated from
the now-removed hand-rolled ``hooks`` module during the framework alignment.)
"""

from __future__ import annotations

from typing import Any

__all__ = ["summarize_args"]


def summarize_args(args: dict[str, Any], max_length: int = 100) -> str:
    """build a truncated summary of tool-call arguments for observation.

    publishes only the argument *keys* with elided values so a downstream
    observer (tool-call-start envelope, audit event, log line) sees the shape of
    the call without leaking sensitive contents (SQL fragments, passwords, PII).
    caller-tunable ``max_length`` clamps the summary length so a tool with many
    keys cannot blow up the wire envelope.

    :param args: tool-call arguments dict (typically the ``args`` key on a
        LangChain tool_call dict)
    :ptype args: dict[str, Any]
    :param max_length: maximum returned summary length in characters
    :ptype max_length: int
    :return: truncated string representation
    :rtype: str
    """
    keys = list(args.keys())
    if not keys:
        return "(no arguments)"
    summary = ", ".join(f"{k}=..." for k in keys[:3])
    if len(keys) > 3:
        summary += f" (+{len(keys) - 3} more)"
    return summary[:max_length]
