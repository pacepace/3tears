"""Keep a Claude Code CLI subprocess away from the host machine's Claude Code configuration.

A Claude Code CLI launched without a configuration directory of its own reads the host's: every
installed plugin and the SessionStart hooks they register, user memory, the stored login, and the
project ``CLAUDE.md`` of whatever working directory it inherits. Found live: on a developer
machine an agent answered a greeting with that developer's plugin session briefing, verbatim, and
called the developer by name.

Measured against the bundled CLI's own init message:

====================================  =======  ===============  ===========  =====================
launch                                plugins  plugin commands  hook events  with no token
====================================  =======  ===============  ===========  =====================
as shipped                            1        14               6            host's own login
``--setting-sources=`` (empty)        1        14               6            -- not the lever
config dir and cwd both empty         0        0                0            "Not logged in"
====================================  =======  ===============  ===========  =====================

The CLI also writes a transcript of every session under ``<config dir>/projects``;
``--no-session-persistence`` writes none. That matters twice over when one credential serves many
people, which is exactly the case a subscription token is.

And the configuration directory is not the only way in. A logged-in CLI attaches the *account's*
claude.ai connectors -- measured: Slack, Google Calendar, Google Drive, Gmail and ZoomInfo --
because they come with the login, not with the directory. An agent running under a person's token
would be handed that person's mail and files as tools. ``--strict-mcp-config`` (only the servers
the caller passes) and ``ENABLE_CLAUDEAI_MCP_SERVERS=false`` each remove all five; both are set,
so a CLI release that changes one of them does not quietly reopen the other.

Standard library only, so any consumer that spawns the CLI -- a chat model, a session pool -- can
apply the same isolation without importing the chat-model backend.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ClaudeCliIsolation", "claude_cli_isolation"]

#: The CLI flag that stops a session being written to disk.
NO_SESSION_PERSISTENCE_FLAG = "no-session-persistence"

#: The CLI flag that restricts MCP servers to the ones the caller passes explicitly.
STRICT_MCP_CONFIG_FLAG = "strict-mcp-config"

#: Per-credential isolation roots, created once per process. Keyed by a digest of the token, so the
#: token never appears in a path and two credentials never share even the CLI's own bookkeeping.
_ROOTS: dict[str, Path] = {}


@dataclass(frozen=True)
class ClaudeCliIsolation:
    """What a CLI launch needs in order to see none of the host's Claude Code configuration.

    :ivar env: environment entries to ADD to the subprocess's environment
    :ivar cwd: an empty working directory, so no project instructions are found
    :ivar extra_args: CLI flags, in ``ClaudeAgentOptions.extra_args`` form (``None`` = bare flag)
    """

    env: dict[str, str]
    cwd: str
    extra_args: dict[str, str | None] = field(default_factory=dict)


def claude_cli_isolation(token: str | None) -> ClaudeCliIsolation:
    """The isolation settings for a CLI that will run under ``token``.

    One pair of directories per credential for the life of the process: they stay empty apart from
    the CLI's own bookkeeping file (session persistence is off), so reusing them costs nothing,
    whereas a directory per call would leave one behind on disk for every turn.

    :param token: the credential the CLI will authenticate with, or ``None``
    :ptype token: str | None
    :return: the environment additions, working directory and flags to launch with
    :rtype: ClaudeCliIsolation
    """
    key = hashlib.sha256((token or "").encode("utf-8")).hexdigest()[:16]
    root = _ROOTS.get(key)
    if root is None or not root.is_dir():
        root = Path(tempfile.mkdtemp(prefix=f"threetears-claude-cli-{key}-"))
        _ROOTS[key] = root
    config_dir = root / "config"
    cwd = root / "cwd"
    config_dir.mkdir(exist_ok=True)
    cwd.mkdir(exist_ok=True)
    return ClaudeCliIsolation(
        env={"CLAUDE_CONFIG_DIR": str(config_dir), "ENABLE_CLAUDEAI_MCP_SERVERS": "false"},
        cwd=str(cwd),
        extra_args={NO_SESSION_PERSISTENCE_FLAG: None, STRICT_MCP_CONFIG_FLAG: None},
    )
