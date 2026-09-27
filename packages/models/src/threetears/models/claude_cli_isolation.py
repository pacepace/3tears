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

Standard library and ``threetears.observe`` only, so any consumer that spawns the CLI -- a chat
model, a session pool -- can apply the same isolation without importing the chat-model backend.
"""

from __future__ import annotations

import atexit
import hashlib
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from threetears.observe import BuildOnce

__all__ = ["ClaudeCliIsolation", "claude_cli_isolation"]

#: The CLI flag that stops a session being written to disk.
NO_SESSION_PERSISTENCE_FLAG = "no-session-persistence"

#: The CLI flag that restricts MCP servers to the ones the caller passes explicitly.
STRICT_MCP_CONFIG_FLAG = "strict-mcp-config"

#: Per-credential isolation roots, created once per process. Keyed by a digest of the token, so the
#: token never appears in a path and two credentials never share even the CLI's own bookkeeping.
#: Options are built on every call and a sync call runs on its caller's thread, so several threads
#: reach this at once for one credential; built through ``BuildOnce`` so the credential gets one
#: root, and rebuilt when something deleted the directory underneath it.
_ROOTS: BuildOnce[str, Path] = BuildOnce(is_current=Path.is_dir)


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


def _make_root(key: str) -> Path:
    """Create one credential's isolation root and have it removed when the process exits.

    :param key: the credential's digest, used in the directory name
    :ptype key: str
    :return: the new root
    :rtype: Path
    """
    root = Path(tempfile.mkdtemp(prefix=f"threetears-claude-cli-{key}-"))
    # One directory per credential per process start would otherwise accumulate in the temp
    # directory for the life of the host.
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


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
    root = _ROOTS.get(key, lambda: _make_root(key))
    config_dir = root / "config"
    cwd = root / "cwd"
    config_dir.mkdir(exist_ok=True)
    cwd.mkdir(exist_ok=True)
    return ClaudeCliIsolation(
        env={"CLAUDE_CONFIG_DIR": str(config_dir), "ENABLE_CLAUDEAI_MCP_SERVERS": "false"},
        cwd=str(cwd),
        extra_args={NO_SESSION_PERSISTENCE_FLAG: None, STRICT_MCP_CONFIG_FLAG: None},
    )
