"""validation guards on :class:`McpTool` -- empty fields rejected at construction.

silent default-deny on an empty ``required_permission`` is the kind
of bug that doesn't surface until a real user is denied for an
unexplained reason. ``__post_init__`` makes the contract explicit.
"""

from __future__ import annotations

import pytest
from threetears.mcp.tool import McpTool


async def _noop(**_kwargs: object) -> str:
    return "ok"


class TestMcpToolValidation:
    def test_empty_name_rejected(self) -> None:
        """name="" raises ValueError at construction."""
        with pytest.raises(ValueError, match="non-empty"):
            McpTool(
                name="",
                description="d",
                input_schema={},
                required_permission="t.x.read",
                handler=_noop,
            )

    def test_empty_required_permission_rejected(self) -> None:
        """required_permission="" raises (silent default-deny otherwise)."""
        with pytest.raises(ValueError, match="non-empty"):
            McpTool(
                name="probe",
                description="d",
                input_schema={},
                required_permission="",
                handler=_noop,
            )

    def test_explicit_public_sentinel_accepted(self) -> None:
        """non-empty placeholder like 'public' constructs cleanly.

        the framework doesn't gate on the value -- the consuming
        Authorizer impl decides whether 'public' means
        always-allow.
        """
        tool = McpTool(
            name="probe",
            description="d",
            input_schema={},
            required_permission="public",
            handler=_noop,
        )
        assert tool.required_permission == "public"
