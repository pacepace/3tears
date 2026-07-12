"""unit tests for the face-flag columns on NamespaceCollection.

gu-task-02b (GU-02b-01): the shared ``NamespaceCollection`` schema
declares three ``BOOLEAN`` face columns -- ``face_api``, ``face_mcp``,
``face_platform_tool`` -- that the tool-registration path stamps from
each tool's manifest entry. The columns mirror the
``tool_eligible``/``skill_eligible`` eligibility precedent exactly:
``BOOL_TYPE`` column type, DB-side ``server_default`` matching the wire
default (``face_platform_tool`` TRUE, the other two FALSE) so pre-face
rows read as "platform-tool only" without a backfill.
"""

from __future__ import annotations

from threetears.core.collections.schema_backed import BOOL_TYPE
from threetears.agent.acl import NamespaceCollection

_FACE_COLUMNS = {
    "face_api": "false",
    "face_mcp": "false",
    "face_platform_tool": "true",
}


def _columns_by_name() -> dict[str, object]:
    """map every ``namespaces`` schema column by name.

    :return: column-name -> Column descriptor
    :rtype: dict[str, object]
    """
    return {col.name: col for col in NamespaceCollection.schema.columns}


class TestFaceColumnsDeclared:
    """GU-02b-01: the three face columns ride on the namespaces schema."""

    def test_all_three_face_columns_present(self) -> None:
        """``face_api`` / ``face_mcp`` / ``face_platform_tool`` all declared."""
        names = {col.name for col in NamespaceCollection.schema.columns}
        assert _FACE_COLUMNS.keys() <= names

    def test_face_columns_are_boolean_typed(self) -> None:
        """each face column carries ``BOOL_TYPE`` (column-type aligned)."""
        by_name = _columns_by_name()
        for name in _FACE_COLUMNS:
            assert by_name[name].column_type == BOOL_TYPE

    def test_face_platform_tool_defaults_true(self) -> None:
        """``face_platform_tool`` defaults TRUE (pre-face = platform tool)."""
        by_name = _columns_by_name()
        assert by_name["face_platform_tool"].server_default == "true"

    def test_face_api_defaults_false(self) -> None:
        """``face_api`` defaults FALSE (opt-in)."""
        by_name = _columns_by_name()
        assert by_name["face_api"].server_default == "false"

    def test_face_mcp_defaults_false(self) -> None:
        """``face_mcp`` defaults FALSE (opt-in)."""
        by_name = _columns_by_name()
        assert by_name["face_mcp"].server_default == "false"

    def test_face_columns_not_nullable(self) -> None:
        """face columns are NOT NULL (DB default backfills old rows)."""
        by_name = _columns_by_name()
        for name in _FACE_COLUMNS:
            assert by_name[name].nullable is False
