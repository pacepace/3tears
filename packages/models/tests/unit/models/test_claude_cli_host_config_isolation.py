"""The subscription CLI must not inherit the host machine's Claude Code configuration.

Found live. A subscription-backed turn spawns the bundled Claude Code CLI, and a
CLI launched with no configuration directory of its own reads the host's: every
installed plugin, their SessionStart hooks, user memory, and the project
``CLAUDE.md`` in whatever working directory it inherits. On a developer machine
an agent answered "You there?" with that developer's plugin session briefing,
verbatim, and called the developer by name.

Measured against the bundled CLI's own init message:

- as shipped: 1 plugin loaded, its 14 commands registered, 6 hook events fired
- ``--setting-sources=`` (empty): **identical** -- it is not the lever
- ``CLAUDE_CONFIG_DIR`` and ``cwd`` both pointed at empty directories: 0 / 0 / 0

Two more things ride on the same directory. With no token in the environment
the CLI does not fail -- it quietly authenticates with the host's own stored
login; isolated, it reports "Not logged in". And it writes a transcript of every
session under ``<config>/projects``; ``--no-session-persistence`` writes none,
which matters when one credential serves many people.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from threetears.models import DEFAULT_CHAT_MODEL
from threetears.models.providers._claude_cli import create_subscription_chat

_TOKEN_A = "sk-ant-oat01-faketokenfortest-aaaa"
_TOKEN_B = "sk-ant-oat01-faketokenfortest-bbbb"


def _options(token: str = _TOKEN_A, **kwargs: object):
    model = create_subscription_chat(DEFAULT_CHAT_MODEL, token, **kwargs)
    return model._build_options()  # noqa: SLF001 -- the method under test


class TestHostConfigIsolation:
    def test_the_cli_gets_a_configuration_directory_of_its_own(self) -> None:
        options = _options()
        config_dir = options.env.get("CLAUDE_CONFIG_DIR")
        assert config_dir, "no CLAUDE_CONFIG_DIR: the CLI reads the host's plugins, hooks, memory and login"
        assert Path(config_dir).is_dir()
        assert Path(config_dir).resolve() != (Path.home() / ".claude").resolve()

    def test_the_token_still_reaches_the_cli(self) -> None:
        """Isolating the directory must not displace the credential that authenticates the call."""
        assert _options().env.get("CLAUDE_CODE_OAUTH_TOKEN") == _TOKEN_A

    def test_the_cli_runs_somewhere_with_no_project_instructions(self) -> None:
        options = _options()
        assert options.cwd, "no cwd: the CLI inherits the host process's directory and its CLAUDE.md"
        cwd = Path(options.cwd)
        assert cwd.is_dir(), "the SDK refuses to start a CLI in a directory that does not exist"
        assert not (cwd / "CLAUDE.md").exists()

    def test_a_caller_that_names_its_own_working_directory_keeps_it(self, tmp_path: Path) -> None:
        assert Path(_options(cwd=str(tmp_path)).cwd) == tmp_path

    def test_sessions_are_not_written_to_disk(self) -> None:
        extra = _options().extra_args
        assert "no-session-persistence" in extra and extra["no-session-persistence"] is None

    def test_the_accounts_own_connectors_are_not_attached(self) -> None:
        """A logged-in CLI attaches the account's claude.ai connectors -- Gmail, Drive, Slack,
        Calendar -- regardless of the configuration directory. Either switch removes them all;
        both are set so one changing upstream cannot quietly reopen the door."""
        options = _options()
        assert options.env.get("ENABLE_CLAUDEAI_MCP_SERVERS") == "false"
        assert "strict-mcp-config" in options.extra_args

    def test_two_credentials_never_share_a_configuration_directory(self) -> None:
        """The CLI keeps its own bookkeeping in that directory; one person's must not be another's."""
        assert _options(_TOKEN_A).env["CLAUDE_CONFIG_DIR"] != _options(_TOKEN_B).env["CLAUDE_CONFIG_DIR"]

    def test_the_same_credential_reuses_its_directory(self) -> None:
        """One directory per credential, not one per call: a call must not leave a directory behind."""
        assert _options(_TOKEN_A).env["CLAUDE_CONFIG_DIR"] == _options(_TOKEN_A).env["CLAUDE_CONFIG_DIR"]

    def test_the_directory_name_does_not_carry_the_token(self) -> None:
        assert "faketokenfortest" not in _options().env["CLAUDE_CONFIG_DIR"]

    def test_isolation_survives_a_per_call_override(self) -> None:
        """``_build_options`` takes overrides on every call; none of them may reopen the host config."""
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, _TOKEN_A)
        options = model._build_options(permission_mode="default")  # noqa: SLF001
        assert options.env.get("CLAUDE_CONFIG_DIR")
        assert "no-session-persistence" in options.extra_args
