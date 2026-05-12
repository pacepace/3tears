"""integration: validator rejection is a clean ToolResult, no state mutated.

REALISM
-------

- **real :class:`FsWriteTool`**: same production tool the LLM calls.
- **real :func:`dispatch_validators`**: the production dispatcher; the
  rejection raises :class:`WorkspaceValidationError` which the tool
  traps and converts into ``ToolResult(success=False, error=...)``.
- **real :class:`WorkspaceSandbox`**: configured to allow writes on
  ``*.yaml``.
- **fake DB pool + fake store**: no journal / head update on validator
  rejection is asserted by inspecting the in-memory store directly.

the validator lives at
:mod:`tests.integration._strict_validator.reject_any_audience_units`
(module-level so :func:`_resolve_validator` can import it by dotted
path). it fails every payload that carries the ``audience_units:`` key
so the fixture ``audience_settings.yaml`` is rejected deterministically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from threetears.agent.workspace.config import (
    AllowConfig,
    ValidatorEntry,
    WorkspaceConfig,
)
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.agent.workspace.tools.fs_write import FsWriteTool


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


# ensure the directory holding _strict_validator.py is importable via
# dotted path. pytest --import-mode=importlib rewires sys.path for
# test modules but not for arbitrary siblings; we make it explicit.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


async def test_fs_write_on_rejected_content_returns_failure_and_leaves_store_untouched(
    workspace_with_audience_fixture: Any,
    permissive_acl_cache: Any,
) -> None:
    """
    validator rejects content; tool returns ``success=False``; no DB write.

    :param workspace_with_audience_fixture: pre-seeded fixture bag
    :ptype workspace_with_audience_fixture: WorkspaceFixture
    :return: None
    :rtype: None
    """
    fx = workspace_with_audience_fixture
    config = WorkspaceConfig(
        allow=AllowConfig(read=["**/*"], write=["**/*.yaml"]),
        validators=[
            ValidatorEntry(
                pattern="audience_settings.yaml",
                validator="_strict_validator.reject_any_audience_units",
            ),
        ],
    )
    sandbox = WorkspaceSandbox.from_config(config)

    from threetears.agent.workspace.pin import set_pin

    await set_pin(
        fx.context,
        workspace_id=fx.workspace_id,
        workspace_name=fx.workspace_name,
        pinned_by_actor_id=fx.agent_id,
    )

    # snapshot pre-call state so we can confirm zero mutations.
    pre_head = fx.store.files[(fx.workspace_id, "audience_settings.yaml")]
    pre_head_content = pre_head.content
    pre_head_version = pre_head.version
    pre_version_count = len([v for v in fx.store.versions if v.relative_path == "audience_settings.yaml"])

    fs_write = FsWriteTool(
        workspace_collection=fx.workspace_collection,
        workspace_file_collection=fx.file_collection,
        workspace_file_version_collection=fx.version_collection,
        sandbox=sandbox,
        context_provider=lambda: fx.context,
        agent_id=fx.agent_id,
        db_pool=fx.pool,
        validators=config.validators,
        acl_cache=permissive_acl_cache,
    )

    # load the fixture content (which contains audience_units:).
    fixture_bytes = (fx.fixture_path / "audience_settings.yaml").read_bytes()
    result = await fs_write.execute(
        relative_path="audience_settings.yaml",
        content=fixture_bytes.decode("utf-8"),
    )

    assert result.success is False
    assert result.error is not None
    assert "validation failed" in result.error

    # head row is unchanged
    post_head = fx.store.files[(fx.workspace_id, "audience_settings.yaml")]
    assert post_head.content == pre_head_content
    assert post_head.version == pre_head_version

    # no new journal row for this file
    post_version_count = len([v for v in fx.store.versions if v.relative_path == "audience_settings.yaml"])
    assert post_version_count == pre_version_count
