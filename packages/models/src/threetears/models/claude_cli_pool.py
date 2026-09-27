"""Start Claude Code CLI subprocesses ahead of the calls that need them.

A Claude subscription credential (``sk-ant-oat…``) has no HTTP API: it is spent by driving the
bundled Claude Code CLI, and ``langchain-claude-code`` starts a fresh subprocess for every call.
Measured on a production deployment: ~2.4 s to start and ~2.7 s per call, against ~600 ms for the
same model over HTTP. This module starts CLIs before they are needed and hands each to exactly one
call. Design, with the evidence for every protocol claim: ``docs/claude-cli-session-pool-design.md``.

**A CLI is reset between calls by rewinding its conversation, never with ``/clear``.** The pool
used to clear a CLI with its local ``/clear`` and hand it to the next caller. ``/clear`` empties the
conversation but leaves the command itself in it -- a ``<local-command-caveat>``,
``<command-name>/clear</command-name>`` and an empty ``<local-command-stdout>`` -- and the next
caller's model reads that as the latest thing the person did (found live, 0.56.0: replies that
began "This is a local slash command (/clear)" and answered it, and "Nothing." for a message with
work in it; a reused session quoted those lines back on 10 calls of 10).

Now each call's message is sent with a uuid of the pool's own (:class:`LentClient`), and on return
the CLI's ``rewind_conversation`` control request cuts the conversation at that message, so the
next call starts on an empty one and nothing of the reset is in it. The CLI grants a rewind to the
FIRST message only when its server-side flag ``tengu_rewind_first_message`` is on; when it refuses
("no preceding assistant"), or the rewind fails any other way, the CLI is stopped instead and a
fresh one -- a *spare* -- is started in the background for the next call with the same launch
options. Nothing else resets a live CLI without a visible turn: a new input ``session_id`` keeps the
conversation, and ``end_session`` ends the process.

What can change on a live CLI and what cannot decides the key:

- the **system prompt** is NOT part of the key. A launch flag prompt never changes, so a CLI
  launches with none, and every system prompt its key has seen is defined on it as a named agent
  (:func:`agent_name`); a checkout switches to the call's prompt with the ``apply_flag_settings``
  control request (``{"agent": name}``, or ``null`` for a call with no prompt). Measured live
  (bundled CLI 2.1.207): the request the model gets under a switched agent carries exactly the
  system blocks and messages a CLI launched with that prompt sends; a switched agent holds across
  the rewind reset; and agents are fixed at launch -- defining one later is accepted, and
  switching to it answers 'Agent "..." not found'. So a CLI serves the prompts its key had seen
  when it started, and a call with a new prompt starts a CLI that defines it and every earlier one;
  a key holds at most :data:`_MAX_AGENTS_PER_KEY` prompts, the least recently used dropped first;
- the **JSON schema** is part of the key: it is the ``--json-schema`` launch flag, and setting it
  through ``apply_flag_settings`` was accepted and changed nothing (measured);
- the **model** changes per call with ``set_model``;
- the **bound tools** (an in-process MCP server) change per call with the CLI's
  ``mcp_set_servers`` control request -- ``reconnect_mcp_server`` refuses SDK servers.

Four things keep CLIs from piling up, because each of the others misses a case:

- every session's pid is tracked, and disposal kills the process and its descendants (see
  :func:`kill_process_tree` for why not the process group);
- the host closes the pool on a clean stop (:func:`close_claude_cli_pool`);
- a startup sweep kills CLIs a crashed previous process left behind
  (:func:`sweep_orphaned_claude_clis`), which is the shutdown a crash skips;
- an idle TTL and hard caps bound the live count while the process runs -- spares included.

The pool is per process. A host running N worker processes has a ceiling of N times its cap. It
serves one event loop at a time: the loop that first checks a session out of it, for as long as
that loop is open. A call on any other open loop -- a sync ``invoke`` runs its own -- runs on a CLI
of its own, as a call does when every session is busy. Once the served loop closes it can never
run a call again, so the next checkout's loop takes the pool over, and the CLIs the closed loop
held are stopped rather than handed on.

Standard library and lazy SDK imports only, so importing this module costs nothing on a host that
never spends a subscription.
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import contextvars
import dataclasses
import hashlib
import json
import os
import secrets
import signal
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uuid_utils import uuid7

from threetears.observe import BuildOnce, get_logger

__all__ = [
    "POOL_MARKER_ENV",
    "LentClient",
    "agent_name",
    "TOOL_SERVER_NAME",
    "ClaudeCliPool",
    "PooledCliSession",
    "ClaudeCliPoolExhausted",
    "ClaudeCliSessionError",
    "claude_cli_pool",
    "close_claude_cli_pool",
    "configure_claude_cli_pool",
    "kill_process_tree",
    "launch_fingerprint",
    "bind_tool_server_to_context",
    "launch_key",
    "poolable",
    "sweep_orphaned_claude_clis",
]

_logger = get_logger(__name__)

#: Stamped into every pooled CLI's environment as ``<owner_pid>:<owner_start_ticks>:<session_id>``.
#: The startup sweep finds orphans by this variable alone, so nothing unmarked is ever signalled.
POOL_MARKER_ENV = "THREETEARS_CLAUDE_CLI_POOL"

#: The in-process MCP server name ``langchain-claude-code`` binds tools under. Tool names reach the
#: model as ``mcp__<server>__<tool>``, so the name is part of every bound tool's identity.
TOOL_SERVER_NAME = "langchain-tools"

#: ``/proc`` is how a session's owner is proved alive and how descendants are found. A platform
#: without it makes the sweep a logged no-op, never a guess.
_PROC = Path("/proc")


class ClaudeCliSessionError(RuntimeError):
    """A pooled CLI session failed to start or prepare."""


class ClaudeCliPoolExhausted(RuntimeError):
    """No pooled session is free within the checkout timeout; run this call on its own CLI."""


# ---------------------------------------------------------------------------
# Process identity, liveness and killing -- all of it /proc-based
# ---------------------------------------------------------------------------


def _stat_tokens(pid: int) -> list[str] | None:
    """``/proc/<pid>/stat`` fields from field 3 on (the comm field is dropped).

    ``comm`` is parenthesised and may itself contain spaces and parentheses, so the split is on
    the LAST ``)``. The returned list is 0-indexed at field 3: ``ppid`` is index 1, ``starttime``
    index 19.

    :param pid: the process to read
    :ptype pid: int
    :return: the fields, or ``None`` when the process or ``/proc`` is gone
    :rtype: list[str] | None
    """
    try:
        raw = (_PROC / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
    except OSError, ValueError:
        # NOSILENT: a process that exited between listing and reading is the normal case in a
        # /proc walk, and None is the documented "gone" answer every caller already handles.
        return None
    close = raw.rfind(")")
    if close == -1:
        return None
    return raw[close + 2 :].split()


def _process_start_ticks(pid: int) -> int | None:
    """The process's start time in clock ticks since boot, or ``None``.

    Paired with the pid this identifies a process across pid reuse, which a restarted container
    makes likely rather than theoretical.

    :param pid: the process to read
    :ptype pid: int
    :return: the start time, or ``None`` when it cannot be read
    :rtype: int | None
    """
    tokens = _stat_tokens(pid)
    if tokens is None or len(tokens) < 20:
        return None
    try:
        return int(tokens[19])
    except ValueError:
        # NOSILENT: an unparseable start time means "cannot identify", the documented None.
        return None


def _parent_pid(pid: int) -> int | None:
    """The process's parent pid, or ``None`` when it cannot be read.

    :param pid: the process to read
    :ptype pid: int
    :return: the parent pid
    :rtype: int | None
    """
    tokens = _stat_tokens(pid)
    if tokens is None or len(tokens) < 2:
        return None
    try:
        return int(tokens[1])
    except ValueError:
        # NOSILENT: an unparseable parent pid means "not a child of anything we track", the documented None.
        return None


def _process_environ(pid: int) -> dict[str, str]:
    """The process's environment, or an empty mapping when it cannot be read.

    :param pid: the process to read
    :ptype pid: int
    :return: the environment
    :rtype: dict[str, str]
    """
    try:
        raw = (_PROC / str(pid) / "environ").read_bytes()
    except OSError, ValueError:
        # NOSILENT: another user's or an exited process's environment is unreadable by design; an
        # empty mapping means "carries no pool marker", so the sweep never touches it.
        return {}
    env: dict[str, str] = {}
    for chunk in raw.split(b"\0"):
        if not chunk:
            continue
        name, _, value = chunk.partition(b"=")
        env[name.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def _live_pids() -> list[int]:
    """Every pid currently in ``/proc``, or an empty list where there is none.

    :return: the pids
    :rtype: list[int]
    """
    try:
        return sorted(int(entry.name) for entry in _PROC.iterdir() if entry.name.isdigit())
    except OSError:
        return []


def _descendants(pid: int) -> list[int]:
    """Every descendant of ``pid``, deepest last.

    :param pid: the ancestor
    :ptype pid: int
    :return: the descendant pids
    :rtype: list[int]
    """
    children: dict[int, list[int]] = {}
    for candidate in _live_pids():
        parent = _parent_pid(candidate)
        if parent is not None:
            children.setdefault(parent, []).append(candidate)
    found: list[int] = []
    queue = deque(children.get(pid, ()))
    while queue:
        current = queue.popleft()
        if current in found or current == pid:
            continue
        found.append(current)
        queue.extend(children.get(current, ()))
    return found


def _signal(pid: int, sig: int) -> bool:
    """Send ``sig`` to ``pid``, reporting whether it landed.

    :param pid: the process to signal
    :ptype pid: int
    :param sig: the signal number
    :ptype sig: int
    :return: whether the signal was delivered
    :rtype: bool
    """
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:
        _logger.warning(
            "Not permitted to signal a pooled Claude CLI process",
            extra={"extra_data": {"pid": pid, "signal": sig}},
        )
        return False
    except OSError as exc:
        _logger.warning(
            "Signalling a pooled Claude CLI process failed",
            extra={"extra_data": {"pid": pid, "signal": sig, "error": str(exc)}},
        )
        return False
    return True


def _alive(pid: int) -> bool:
    """Whether ``pid`` names a running process.

    :param pid: the process to check
    :ptype pid: int
    :return: whether it is running
    :rtype: bool
    """
    return _signal(pid, 0)


def kill_process_tree(pid: int, *, grace_seconds: float = 2.0) -> int:
    """Stop ``pid`` and everything it started, SIGTERM then SIGKILL.

    **Not** ``killpg``. The Agent SDK starts the CLI with ``anyio.open_process(...)`` and no
    ``start_new_session``, so the CLI sits in the HOST process's own group -- signalling that group
    would take down the host and every other CLI it owns. The group is used only when the child is
    genuinely its own group leader, which is what a future SDK passing ``start_new_session`` would
    give; otherwise the descendant walk gives the same guarantee.

    :param pid: the CLI's process id
    :ptype pid: int
    :param grace_seconds: how long SIGTERM gets before SIGKILL
    :ptype grace_seconds: float
    :return: how many processes were still alive when SIGTERM was sent
    :rtype: int
    """
    targets = [pid, *_descendants(pid)]
    group = -1
    # NOSILENT: a process that already exited has no group to test; the descendant walk below
    # still runs, so nothing is skipped by staying silent here.
    with suppress(ProcessLookupError, PermissionError, OSError):
        if os.getpgid(pid) == pid:
            group = pid

    signalled = 0
    if group != -1 and _signal(-group, signal.SIGTERM):
        signalled = len(targets)
    else:
        for target in targets:
            if _signal(target, signal.SIGTERM):
                signalled += 1

    deadline = time.monotonic() + max(grace_seconds, 0.0)
    while time.monotonic() < deadline:
        if not any(_alive(target) for target in targets):
            return signalled
        time.sleep(0.05)

    for target in targets:
        if _alive(target):
            _signal(target, signal.SIGKILL)
    return signalled


def _marker(owner_pid: int | None = None) -> str:
    """A fresh session marker naming its owner and a unique session id.

    :param owner_pid: the owning process, defaulting to this one
    :ptype owner_pid: int | None
    :return: the marker value
    :rtype: str
    """
    pid = os.getpid() if owner_pid is None else owner_pid
    return f"{pid}:{_process_start_ticks(pid) or 0}:{secrets.token_hex(16)}"


def _owner_is_alive(marker: str) -> bool:
    """Whether the process that started a marked CLI is still running.

    Both halves matter: the pid must exist AND still report the start time the marker recorded, so
    a recycled pid cannot make a stale CLI look owned.

    :param marker: the :data:`POOL_MARKER_ENV` value read off the process
    :ptype marker: str
    :return: whether the owner is alive
    :rtype: bool
    """
    parts = marker.split(":")
    if len(parts) < 2:
        return False
    try:
        owner_pid = int(parts[0])
        owner_ticks = int(parts[1])
    except ValueError:
        return False
    if not _alive(owner_pid):
        return False
    if owner_ticks == 0:
        return True
    return _process_start_ticks(owner_pid) == owner_ticks


def sweep_orphaned_claude_clis(*, grace_seconds: float = 2.0) -> int:
    """Kill pooled CLIs whose owning process is gone. A host runs this at startup.

    The mechanism that actually saves a host: a crash, an OOM kill or a container kill skips every
    shutdown hook, and the CLIs it started are reparented to init and keep running. A marked CLI
    whose owner is still alive belongs to a sibling worker and is left alone.

    :param grace_seconds: how long each SIGTERM gets before SIGKILL
    :ptype grace_seconds: float
    :return: how many orphaned CLI processes were killed
    :rtype: int
    """
    pids = _live_pids()
    if not pids:
        _logger.info("No /proc: the orphaned Claude CLI sweep is a no-op on this platform")
        return 0

    self_pid = os.getpid()
    killed = 0
    for pid in pids:
        if pid == self_pid:
            continue
        marker = _process_environ(pid).get(POOL_MARKER_ENV)
        if not marker or _owner_is_alive(marker):
            continue
        _logger.warning(
            "Killing an orphaned Claude CLI left by a previous process",
            extra={"extra_data": {"pid": pid, "owner": marker.split(":")[0]}},
        )
        kill_process_tree(pid, grace_seconds=grace_seconds)
        killed += 1

    if killed:
        _logger.warning(
            "Orphaned Claude CLI sweep finished",
            extra={"extra_data": {"killed": killed, "scanned": len(pids)}},
        )
    else:
        _logger.info("Orphaned Claude CLI sweep found nothing", extra={"extra_data": {"scanned": len(pids)}})
    return killed


# ---------------------------------------------------------------------------
# The launch key
# ---------------------------------------------------------------------------

#: Options applied per checkout rather than at launch, so they are not part of the key. A string
#: system prompt is one too: it is switched per checkout as a named agent (see the module docstring),
#: and ``agents`` is how the pool defines those prompts on a CLI.
_PER_CHECKOUT_FIELDS = frozenset({"model", "mcp_servers", "agents"})

#: The most system prompts a key defines as agents on the CLIs it starts. A caller's stable prompt
#: is meant to be stable; this bounds a caller whose "stable" part is not.
_MAX_AGENTS_PER_KEY = 32

#: What a pooled system prompt's agent is described as. The CLI requires a description; nothing
#: the model reads carries it while its built-in tools are off.
_AGENT_DESCRIPTION = "a system prompt the Claude CLI pool switches to per call"


def agent_name(system_prompt: str) -> str:
    """The name a system prompt is defined under as an agent on a pooled CLI.

    :param system_prompt: the prompt
    :ptype system_prompt: str
    :return: a name derived from the prompt alone, so one prompt has one name on every CLI
    :rtype: str
    """
    return "prompt-" + hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:24]


def _switched_prompt(options: Any) -> tuple[bool, str | None]:
    """Whether the options' system prompt can be switched per checkout, and the prompt.

    A string prompt, or none, is switched as an agent. Any other shape -- a preset that appends to
    Claude Code's own prompt, a prompt file -- stays a launch flag and part of the key.

    :param options: the call's launch options
    :ptype options: Any
    :return: ``(switchable, prompt)``, the prompt ``None`` when the call has none
    :rtype: tuple[bool, str | None]
    """
    prompt = getattr(options, "system_prompt", None)
    switchable = prompt is None or isinstance(prompt, str)
    return switchable, (prompt or None) if switchable else None


#: Options carrying Python callables. A callable has no stable identity to key on and closes over
#: one caller's state -- the same hazard as a tool server -- so a call that sets any of them is not
#: pooled at all. ``debug_stderr`` is deliberately absent: it DEFAULTS to ``sys.stderr``, a stream
#: rather than a callback, and listing it here made every real call unpoolable -- found live, when
#: every call logged "carries callables" and ran on a CLI of its own. ``session_store`` is here
#: though it is an object, not a function: it is one caller's state, and it renders in a key by
#: memory address, so a new store at a reused address could otherwise share a CLI.
_CALLABLE_FIELDS = frozenset({"hooks", "can_use_tool", "stderr", "session_store"})

#: Options that ask the CLI to continue a stored session, which isolation disables and which cannot
#: be shared between callers.
_RESUME_FIELDS = frozenset({"resume", "continue_conversation", "fork_session", "session_id"})

#: The environment variable carrying the credential; keyed by digest, never by value.
_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"


def _option_fields(options: Any) -> list[str]:
    """Every option field the launch key considers.

    Every field of the options dataclass, so an option a caller can set per call cannot silently
    share a CLI launched without it -- the base package's ``_build_options`` accepts any field as an
    override. A non-dataclass stand-in falls back to its public attributes.

    :param options: the launch options
    :ptype options: Any
    :return: the field names, sorted
    :rtype: list[str]
    """
    if dataclasses.is_dataclass(options):
        names = [f.name for f in dataclasses.fields(options)]
    else:
        names = [n for n in vars(options) if not n.startswith("_")]
    per_checkout = _PER_CHECKOUT_FIELDS | ({"system_prompt"} if _switched_prompt(options)[0] else set())
    return sorted(n for n in names if n not in per_checkout)


def poolable(options: Any) -> bool:
    """Whether a call with these options may run on a pooled CLI, one started before the call.

    :param options: the call's launch options
    :ptype options: Any
    :return: ``False`` when a callable option or a resumed session is set
    :rtype: bool
    """
    for name in _CALLABLE_FIELDS | _RESUME_FIELDS:
        if getattr(options, name, None):
            return False
    return True


def _jsonable(value: Any) -> Any:
    """A deterministic JSON-safe rendering of an option value.

    :param value: an option value
    :ptype value: Any
    :return: something ``json.dumps`` renders the same way every time
    :rtype: Any
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    return str(value)


def _keyed_value(name: str, options: Any) -> Any:
    """One option's value as the key sees it: the credential in ``env`` digested, the pool marker gone.

    :param name: the field
    :ptype name: str
    :param options: the launch options
    :ptype options: Any
    :return: a JSON-safe rendering
    :rtype: Any
    """
    value = getattr(options, name, None)
    if name == "env" and isinstance(value, dict):
        value = {
            k: (hashlib.sha256(str(v).encode("utf-8")).hexdigest() if k == _TOKEN_ENV else v)
            for k, v in value.items()
            if k != POOL_MARKER_ENV
        }
    return _jsonable(value)


def launch_fingerprint(options: Any) -> dict[str, str]:
    """A short digest per launch-time field, for the log line of a session that starts.

    When two calls that look alike get different sessions, comparing these across the two
    "Started a pooled Claude CLI" lines names the field that differed -- without writing a system
    prompt or a tool list into the log.

    :param options: the launch options
    :ptype options: Any
    :return: ``{field: 8-hex-character digest}``
    :rtype: dict[str, str]
    """
    return {
        name: hashlib.sha256(json.dumps(_keyed_value(name, options), sort_keys=True).encode("utf-8")).hexdigest()[:8]
        for name in _option_fields(options)
    }


def launch_key(options: Any, token: str | None) -> str:
    """The pool key for a launch: a digest of the credential and every launch-time option.

    :param options: the ``ClaudeAgentOptions`` the CLI would start with
    :ptype options: Any
    :param token: the credential the CLI authenticates with
    :ptype token: str | None
    :return: a 32-hex-character key
    :rtype: str
    """
    material = {
        "credential": hashlib.sha256((token or "").encode("utf-8")).hexdigest(),
        **{name: _keyed_value(name, options) for name in _option_fields(options)},
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# One pooled session: a connected ClaudeSDKClient over one CLI subprocess
# ---------------------------------------------------------------------------


def _sdk_process_pid(client: Any) -> Any:
    """the CLI subprocess's pid as the SDK holds it, or ``None`` when its internals have moved.

    the SDK offers no public accessor: the process object sits on the private ``_transport`` and
    its private ``_process``. read here as attributes under a reasoned SLF001 pragma -- the
    spelling every check sees -- rather than through ``getattr`` with the names as strings, which
    hid the dependency from all of them. :func:`_discover_pid` falls back to ``/proc`` when this is
    ``None``.

    :param client: the connected ``ClaudeSDKClient``
    :ptype client: Any
    :return: whatever the SDK's process object reports as its pid, or ``None``
    :rtype: Any
    """
    result: Any = None
    try:
        result = client._transport._process.pid  # noqa: SLF001 -- the SDK exposes its CLI process nowhere else
    except AttributeError:
        # NOSILENT: the SDK's internals moved; the caller falls back to scanning /proc for this
        # session's marker, and warns only if that finds nothing either.
        result = None
    return result


def _discover_pid(client: Any, marker: str) -> int | None:
    """The CLI subprocess's pid, from the SDK if it will say, else from ``/proc``.

    The SDK's transport holds the process object two private attributes down. The marker is unique
    per session, so scanning ``/proc`` for it finds the same process and keeps pid tracking working
    if the SDK's internals move.

    :param client: the connected ``ClaudeSDKClient``
    :ptype client: Any
    :param marker: this session's :data:`POOL_MARKER_ENV` value
    :ptype marker: str
    :return: the pid, or ``None`` when it cannot be determined
    :rtype: int | None
    """
    pid = _sdk_process_pid(client)
    if isinstance(pid, int):
        return pid
    for candidate in _live_pids():
        if _process_environ(candidate).get(POOL_MARKER_ENV) == marker:
            return candidate
    _logger.warning(
        "Could not determine a pooled Claude CLI's pid; it can only be stopped by disconnecting",
        extra={"extra_data": {"marker_owner": marker.split(":")[0]}},
    )
    return None


class _ContextBoundServer:
    """An in-process MCP server whose tool calls run in one borrower's context.

    The SDK runs a tool call in a task spawned from its message-reader task, and that task was
    created when the CLI connected -- in whatever context STARTED the session. A pooled session is
    started before its borrower's call (a spare, in a context of its own), and when sessions were
    reused every later borrower's tool calls ran with the first borrower's context variables: an ``interrupt()`` was captured into the first caller's list and the graph never
    paused, tool-status events went to the first caller's callbacks, and the tool saw the first
    caller's runnable config. Found by review and reproduced.

    Each call runs in a fresh copy of the borrower's context: a copy because two tool calls in one
    turn may run at once and a context cannot be entered twice, and a copy still shares the
    borrower's mutable values -- the list an interrupt is appended to is the same list.
    """

    def __init__(self, server: Any, context: contextvars.Context) -> None:
        from mcp.types import CallToolRequest  # noqa: PLC0415 -- arrives with the claude-cli extra

        self._server = server
        self.name = server.name
        self.version = getattr(server, "version", None)
        self.request_handlers = dict(server.request_handlers)
        original = self.request_handlers.get(CallToolRequest)
        if original is not None:

            async def call_in_borrower_context(request: Any) -> Any:
                return await asyncio.create_task(original(request), context=context.copy())

            self.request_handlers[CallToolRequest] = call_in_borrower_context

    def __getattr__(self, name: str) -> Any:
        return getattr(self._server, name)


def bind_tool_server_to_context(server: Any, context: contextvars.Context | None) -> Any:
    """``server`` with its tool calls running in ``context``; ``server`` itself when there is none.

    :param server: an in-process MCP server instance (``create_sdk_mcp_server(...)["instance"]``)
    :ptype server: Any
    :param context: the borrower's context, captured when its call began
    :ptype context: contextvars.Context | None
    :return: the server to install on the session
    :rtype: Any
    """
    if server is None or context is None:
        return server
    return _ContextBoundServer(server, context)


class LentClient:
    """A pooled CLI's client as one call sees it: every message the call sends carries a known uuid.

    The pool rewinds the conversation to the call's first message when the call returns, and the
    CLI finds that message by the uuid it was sent with. A string prompt is sent as the SDK sends
    it, as one user message, with a uuid of the pool's own. Everything else is the client's.

    :param client: the connected ``ClaudeSDKClient``
    :ptype client: Any
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        #: The uuid the call's first message was sent with; ``None`` until the call sends one.
        self.first_message_uuid: str | None = None

    async def query(self, prompt: Any, session_id: str = "default") -> None:
        """Send the call's message, recording the uuid of the first one sent.

        :param prompt: a string, or the SDK's stream of message dicts
        :ptype prompt: Any
        :param session_id: the SDK's session identifier for the message
        :ptype session_id: str
        """
        messages = (
            [{"type": "user", "message": {"role": "user", "content": prompt}}] if isinstance(prompt, str) else None
        )

        async def tagged() -> AsyncIterator[dict[str, Any]]:
            source: Any = messages if messages is not None else prompt
            if isinstance(source, list):
                for message in source:
                    yield self._tag(message)
            else:
                async for message in source:
                    yield self._tag(message)

        await self.client.query(tagged(), session_id=session_id)

    def _tag(self, message: dict[str, Any]) -> dict[str, Any]:
        """``message`` with the fields the SDK adds, and a uuid when it has none.

        :param message: one message dict
        :ptype message: dict[str, Any]
        :return: the message as sent
        :rtype: dict[str, Any]
        """
        sent = {"parent_tool_use_id": None, **message}
        if sent.get("type") == "user":
            sent.setdefault("uuid", str(uuid7()))
            if self.first_message_uuid is None:
                self.first_message_uuid = str(sent["uuid"])
        return sent

    def receive_response(self) -> AsyncIterator[Any]:
        """The client's messages up to and including the call's ``ResultMessage``.

        :return: the messages
        :rtype: AsyncIterator[Any]
        """
        messages: AsyncIterator[Any] = self.client.receive_response()
        return messages

    def __getattr__(self, name: str) -> Any:
        """Anything else is the client's own.

        :param name: the attribute
        :ptype name: str
        :return: the client's attribute
        :rtype: Any
        """
        return getattr(self.client, name)


class PooledCliSession:
    """One live CLI subprocess, lent to exactly one caller at a time."""

    def __init__(self, client: Any, *, key: str, pid: int | None, marker: str) -> None:
        self.client = client
        self.key = key
        self.pid = pid
        self.marker = marker
        self.closed = False
        self._model: str | None = None
        #: The agents -- system prompts, by :func:`agent_name` -- this CLI was launched with.
        self.agents: frozenset[str] = frozenset()
        #: The agent the CLI is switched to; ``None`` is the launch prompt, which is empty.
        self.agent: str | None = None
        #: The CLI's start time, so disposal never signals a recycled pid that is not this CLI.
        self.start_ticks = _process_start_ticks(pid) if pid is not None else None

    @classmethod
    async def start(cls, options: Any, *, key: str) -> PooledCliSession:
        """Start a CLI with ``options`` and connect to it.

        :param options: launch options; this method adds the pool marker to their environment
        :ptype options: Any
        :param key: the launch key the session serves
        :ptype key: str
        :return: the connected session
        :rtype: PooledCliSession
        :raises ClaudeCliSessionError: when the CLI cannot be started
        """
        import claude_agent_sdk as sdk  # noqa: PLC0415 -- the claude-cli extra, imported only when used

        marker = _marker()
        options.env = {**(options.env or {}), POOL_MARKER_ENV: marker}
        prompts: dict[str, str] = dict(getattr(options, "agents", None) or {})
        if prompts:
            options.agents = {
                name: sdk.AgentDefinition(description=_AGENT_DESCRIPTION, prompt=prompt)
                for name, prompt in prompts.items()
            }
        client = sdk.ClaudeSDKClient(options=options)
        try:
            await client.connect()
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- any start failure (including an SDK whose surface moved) must become "no pooled session", so the call falls back to its own CLI instead of failing the turn
            # NOSILENT: best-effort cleanup of a CLI that never came up; the start failure itself is
            # raised immediately below as ClaudeCliSessionError and logged by the caller.
            with suppress(Exception):  # prawduct:allow prawduct/broad-except -- see NOSILENT above
                await asyncio.wait_for(client.disconnect(), timeout=5.0)
            raise ClaudeCliSessionError(f"could not start a Claude CLI: {exc}") from exc
        except BaseException:
            # A cancellation (a turn stopped while its CLI starts) must not leave a connected
            # subprocess behind -- disconnect before the cancellation continues.
            # NOSILENT: the cancellation being re-raised below is the event that matters; this only
            # makes sure the half-started CLI does not outlive it.
            with suppress(Exception):  # prawduct:allow prawduct/broad-except -- see NOSILENT above
                await asyncio.shield(asyncio.wait_for(client.disconnect(), timeout=5.0))
            raise
        session = cls(client, key=key, pid=_discover_pid(client, marker), marker=marker)
        session._model = getattr(options, "model", None)
        session.agents = frozenset(prompts)
        return session

    async def prepare(
        self,
        *,
        model: str | None,
        tool_server: Any | None,
        call_context: contextvars.Context | None = None,
        agent: str | None = None,
    ) -> None:
        """Point this session at the caller's system prompt, model and tools.

        The tool server is replaced on EVERY checkout, never reused: its handlers close over the
        caller's own tool objects, which belong to that caller's conversation. A stale server
        would run one conversation's tools on another's behalf.

        :param model: the model the call wants, or ``None`` to keep the current one
        :ptype model: str | None
        :param tool_server: the call's in-process MCP server instance, or ``None`` for no tools
        :ptype tool_server: Any | None
        :param call_context: the borrower's context, which every tool call on this checkout runs in
        :ptype call_context: contextvars.Context | None
        :param agent: the agent holding the call's system prompt, ``None`` for a call with none
        :ptype agent: str | None
        :raises ClaudeCliSessionError: when any change is refused
        """
        if self.closed:
            raise ClaudeCliSessionError("this Claude CLI session has been stopped")
        if agent is not None and agent not in self.agents:
            raise ClaudeCliSessionError(f"this Claude CLI was launched without the agent {agent!r}")
        try:
            if agent != self.agent:
                await self.client._query._send_control_request(  # noqa: SLF001 -- the SDK exposes no agent switch
                    {"subtype": "apply_flag_settings", "settings": {"agent": agent}}, timeout=30.0
                )
                self.agent = agent
            if model and model != self._model:
                await self.client.set_model(model)
                self._model = model
            query = self.client._query  # noqa: SLF001 -- the SDK exposes no public server swap
            await query._send_control_request({"subtype": "mcp_set_servers", "servers": {}}, timeout=30.0)  # noqa: SLF001
            query.sdk_mcp_servers.pop(TOOL_SERVER_NAME, None)
            if tool_server is not None:
                query.sdk_mcp_servers[TOOL_SERVER_NAME] = bind_tool_server_to_context(tool_server, call_context)
                await query._send_control_request(  # noqa: SLF001
                    {
                        "subtype": "mcp_set_servers",
                        "servers": {TOOL_SERVER_NAME: {"type": "sdk", "name": TOOL_SERVER_NAME}},
                    },
                    timeout=30.0,
                )
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- the SDK raises bare Exception for a refused control request; any failure here must dispose the session, never hand it on
            error = ClaudeCliSessionError(f"could not prepare the Claude CLI session: {exc}")
            #: A failure of the SDK surface itself (a private attribute or control request that
            #: moved) rather than of this one CLI; the pool counts these and stops pooling.
            error.structural = isinstance(exc, (AttributeError, TypeError, KeyError))  # type: ignore[attr-defined]
            raise error from exc

    async def release_tools(self, *, timeout: float) -> None:
        """Drop the last borrower's tool server, so an idle session holds none of its objects.

        :param timeout: seconds before the release is abandoned
        :ptype timeout: float
        :raises ClaudeCliSessionError: when the CLI refuses or times out; the pool disposes the session
        """
        if self.closed:
            raise ClaudeCliSessionError("this Claude CLI session has been stopped")
        try:
            query = self.client._query  # noqa: SLF001 -- the SDK exposes no public server swap
            query.sdk_mcp_servers.pop(TOOL_SERVER_NAME, None)
            await query._send_control_request({"subtype": "mcp_set_servers", "servers": {}}, timeout=timeout)  # noqa: SLF001
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- any failure means the session cannot be trusted idle; the pool disposes it
            raise ClaudeCliSessionError(f"could not release the Claude CLI's tools: {exc}") from exc

    async def rewind(self, first_message_uuid: str, *, timeout: float) -> None:
        """Cut the conversation at the call's first message, leaving it empty and leaving no trace.

        :param first_message_uuid: the uuid the call's first message was sent with
        :ptype first_message_uuid: str
        :param timeout: seconds before the rewind is abandoned
        :ptype timeout: float
        :raises ClaudeCliSessionError: when the CLI refuses the rewind or fails to answer; the pool
            stops the session rather than hand on a conversation it could not empty
        """
        if self.closed:
            raise ClaudeCliSessionError("this Claude CLI session has been stopped")
        try:
            query = self.client._query  # noqa: SLF001 -- the SDK exposes no rewind of the conversation
            answer = await query._send_control_request(  # noqa: SLF001
                {"subtype": "rewind_conversation", "target_message_uuid": first_message_uuid}, timeout=timeout
            )
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- any failure means the conversation may not be empty; the pool stops the session
            raise ClaudeCliSessionError(f"the Claude CLI failed to rewind: {exc}") from exc
        if not isinstance(answer, dict) or answer.get("rewound") is not True:
            reason = answer.get("error") if isinstance(answer, dict) else answer
            raise ClaudeCliSessionError(f"the Claude CLI refused to rewind: {reason}")

    async def dispose(self, *, grace_seconds: float) -> None:
        """Disconnect, then make sure the process and its children are gone. Idempotent.

        :param grace_seconds: how long SIGTERM gets before SIGKILL
        :ptype grace_seconds: float
        """
        if self.closed:
            return
        self.closed = True
        # NOSILENT: whatever disconnect raises, the process-tree kill below is what guarantees the CLI
        # is gone; a disconnect failure changes nothing about that outcome.
        with suppress(Exception):  # prawduct:allow prawduct/broad-except -- see NOSILENT above
            await asyncio.wait_for(self.client.disconnect(), timeout=5.0)
        await asyncio.to_thread(self._kill_if_still_ours, grace_seconds)

    async def abandon(self, *, grace_seconds: float) -> None:
        """Stop the CLI without talking to it, for a session whose event loop has closed. Idempotent.

        The client's reader task and its subprocess transport belong to the loop it connected
        on. Once that loop is closed, awaiting ``disconnect`` from another loop would drive
        objects a closed loop owns, so the process tree is killed directly instead -- which is
        the step of :meth:`dispose` that actually guarantees the CLI is gone.

        Safe to repeat, and it always attempts the kill even on a session already marked
        closed: an earlier stop that was interrupted before its kill finished leaves the CLI
        running, and a repeat is how the pool finishes it. A CLI already gone is a no-op.

        :param grace_seconds: how long SIGTERM gets before SIGKILL
        :ptype grace_seconds: float
        """
        self.closed = True
        await asyncio.to_thread(self._kill_if_still_ours, grace_seconds)

    def still_ours(self) -> bool:
        """Whether this session's CLI process is still running as the process it started as.

        :return: ``False`` when there is no pid, the process is gone, or its pid now names another process
        :rtype: bool
        """
        if self.pid is None or not _alive(self.pid):
            return False
        return self.start_ticks is None or _process_start_ticks(self.pid) == self.start_ticks

    def _kill_if_still_ours(self, grace_seconds: float) -> None:
        """Kill the CLI's process tree, unless it already exited or its pid now names another process.

        :param grace_seconds: how long SIGTERM gets before SIGKILL
        :ptype grace_seconds: float
        """
        # A CLI that exited, or whose pid now belongs to something else, is signalled nothing.
        if self.pid is not None and self.still_ours():
            kill_process_tree(self.pid, grace_seconds=grace_seconds)


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


async def _reacquire(condition: asyncio.Condition) -> None:
    """Take ``condition``'s lock back even if the caller is cancelled while waiting for it.

    The eviction path releases the lock around a slow disposal and re-takes it before continuing
    inside its ``async with``. A cancellation arriving while it waits would propagate with the lock
    NOT held, and the ``async with`` would then release a lock this task does not own. The waiting is
    done in a shielded task; if the caller is cancelled meanwhile, the lock is still taken (so the
    ``async with`` exits cleanly) and the cancellation is re-raised.

    :param condition: the pool's condition
    :ptype condition: asyncio.Condition
    """
    waiter = asyncio.ensure_future(condition.acquire())
    try:
        await asyncio.shield(waiter)
    except asyncio.CancelledError:
        await waiter
        raise


@dataclass
class _Idle:
    session: PooledCliSession
    since: float


#: Consecutive structural prepare failures after which the pool turns itself off.
_STRUCTURAL_FAILURE_LIMIT = 3


class ClaudeCliPool:
    """Bounded pools of started Claude CLI sessions, one pool per launch key; each serves one call.

    ``per_key`` defaults to the whole cap. A key once held one system prompt, so two sessions per
    key still let different stages run side by side; now one key holds every prompt of a credential
    and its options (see the module docstring), and two would cap an application at two pooled
    calls at once -- measured: a stage group of three ran one on a CLI of its own.
    """

    def __init__(
        self,
        *,
        max_sessions: int = 8,
        per_key: int = 8,
        idle_ttl_seconds: float = 300.0,
        checkout_timeout_seconds: float = 2.0,
        reset_timeout_seconds: float = 5.0,
        kill_grace_seconds: float = 2.0,
        session_factory: Callable[..., Awaitable[PooledCliSession]] | None = None,
    ) -> None:
        if max_sessions < 1 or per_key < 1:
            raise ValueError("max_sessions and per_key must both be at least 1")
        self._max_sessions = max_sessions
        self._per_key = per_key
        self._idle_ttl = idle_ttl_seconds
        self._checkout_timeout = checkout_timeout_seconds
        self._reset_timeout = reset_timeout_seconds
        self._kill_grace = kill_grace_seconds
        self._condition = asyncio.Condition()
        self._idle: dict[str, deque[_Idle]] = {}
        self._live: dict[str, int] = {}
        self._all: set[PooledCliSession] = set()
        self._total = 0
        self._closing = False
        self._reaper: asyncio.Task[None] | None = None
        #: Spares being started in the background (:meth:`_start_spare`), so a close can stop them.
        self._spares: set[asyncio.Task[None]] = set()
        #: Every system prompt each key has seen, by agent name, least recently used first. A CLI
        #: starts with all of them defined as agents (see the module docstring).
        self._prompts: dict[str, dict[str, str]] = {}
        #: How a session is started. Injected by tests; a real host never passes it.
        self._start = session_factory or PooledCliSession.start
        self._structural_failures = 0
        self._broken = False
        #: The event loop this pool serves, claimed by its first checkout. Everything the pool
        #: holds is tied to it: the condition binds to the loop that first waits on it, a pooled
        #: client's reader task runs on the loop it connected on, and the reaper is a task there.
        #: A ``BuildOnce`` because a sync ``invoke`` runs its own loop on its caller's thread, so
        #: two threads can make a first checkout at once and exactly one of them may win. A
        #: closed loop is not current: it can never run a call again, so keeping it would turn
        #: pooling off for the rest of the process -- a startup warm-up through a sync
        #: ``invoke`` claims the pool with a loop that closes when the warm-up returns. The next
        #: checkout's loop claims it instead (:meth:`_rebind_to`).
        self._owner_loop: BuildOnce[str, asyncio.AbstractEventLoop] = BuildOnce(
            is_current=lambda loop: not loop.is_closed()
        )
        #: Whether a loop has claimed this pool before, so a takeover is logged as one.
        self._served_a_loop = False
        #: Sessions taken out of the books to be stopped whose kill has not yet completed. A
        #: session leaves this set only AFTER its CLI is confirmed stopped, so a stop that is
        #: cancelled or fails part-way leaves it where :meth:`known_pids` -- the interpreter-exit
        #: backstop -- and the next takeover or close still find it.
        self._stopping: set[PooledCliSession] = set()

    @property
    def live_count(self) -> int:
        """How many CLI sessions this pool currently holds."""
        return self._total

    def known_pids(self) -> list[int]:
        """Every live session's pid, for a synchronous last-resort kill at interpreter exit.

        Includes the sessions being stopped whose stop has not completed, while their CLI is
        still the process it started as.

        :return: the pids
        :rtype: list[int]
        """
        live = [s.pid for s in list(self._all) if s.pid is not None and not s.closed]
        stopping = [s.pid for s in list(self._stopping) if s.pid is not None and s.still_ours()]
        return [*live, *stopping]

    @asynccontextmanager
    async def checkout(
        self,
        options: Any,
        *,
        token: str | None,
        tool_server: Any | None,
        call_context: contextvars.Context | None = None,
    ) -> AsyncIterator[Any]:
        """Borrow a connected CLI client for one call.

        The caller drives ``client.query`` and reads ``client.receive_response()`` to its
        ``ResultMessage``. A call that leaves normally has its conversation rewound to empty and
        its session re-pooled; when the rewind is refused the session is stopped and a spare started
        in its place (see the module docstring). One that leaves by any exception -- a failure, a
        timeout, a cancellation, a consumer that stopped reading -- is stopped with no spare,
        because an abandoned stream is the one way a later caller could read an earlier caller's
        answer, and a CLI that keeps failing must not be restarted in a loop.

        :param options: the launch options for a session this call could use
        :ptype options: Any
        :param token: the credential, for the key
        :ptype token: str | None
        :param tool_server: the call's in-process MCP server instance, or ``None``
        :ptype tool_server: Any | None
        :param call_context: the borrower's context; its tool calls run in a copy of it
        :ptype call_context: contextvars.Context | None
        :return: the connected client, as a :class:`LentClient`
        :rtype: AsyncIterator[Any]
        :raises ClaudeCliPoolExhausted: when no session frees up in time, or when the caller runs
            on an open event loop other than the one this pool serves
        :raises ClaudeCliSessionError: when a session cannot be started or prepared
        """
        if self._broken:
            raise ClaudeCliPoolExhausted("pooling is off: the Claude Agent SDK's surface failed repeatedly")
        if not poolable(options):
            raise ClaudeCliPoolExhausted("this call carries callables or a resumed session and cannot share a CLI")
        stranded = self._claim_loop()
        if stranded:
            # Shielded: a call cancelled here must not leave the closed loop's CLIs running with
            # nothing left that tracks them.
            await asyncio.shield(self._abandon(stranded))
        key = launch_key(options, token)
        agent = self._learn_prompt(key, options)
        session = await self._acquire(key, options, agent)
        lent = LentClient(session.client)
        clean = False
        try:
            try:
                await session.prepare(
                    model=getattr(options, "model", None),
                    tool_server=tool_server,
                    call_context=call_context,
                    agent=agent,
                )
            except ClaudeCliSessionError as exc:
                self._note_prepare_failure(exc)
                raise
            self._structural_failures = 0
            yield lent
            clean = True
        finally:
            await asyncio.shield(
                self._return(key, session, clean=clean, options=options, first_message_uuid=lent.first_message_uuid)
            )

    def _claim_loop(self) -> list[PooledCliSession]:
        """Serve the running loop if it is this pool's, claiming it when no open loop holds the pool.

        The pool is process-wide, but what it holds is not: its condition binds to the first loop
        that waits on it, a pooled client's reader task lives on the loop it connected on, and the
        reaper is a task on the first loop. A sync ``invoke`` runs its own loop on its caller's
        thread, so a consumer calling models from several threads reaches this one pool from
        several loops. Another loop touching that state raised ``RuntimeError`` (a condition
        "bound to a different event loop"), which no caller treats as "run on your own CLI" --
        the call failed instead of falling back. It is refused as exhaustion instead, before
        anything loop-bound is touched, so the call runs on a CLI of its own exactly as it does
        when every session is busy.

        A served loop that has CLOSED is not refused against: nothing can run on it again, so the
        running loop takes the pool over, and the sessions the closed loop held come back for the
        caller to stop. The takeover runs inside the ``BuildOnce`` build, so of several threads
        finding the served loop closed exactly one takes over and the rest are refused.

        :return: the sessions a closed loop left behind, for the caller to stop; empty unless this
            call took the pool over from a closed loop
        :rtype: list[PooledCliSession]
        :raises ClaudeCliPoolExhausted: when the caller's loop is not the one this pool serves
        """
        running = asyncio.get_running_loop()
        owner, stranded = self._take_over_if_unserved(running)
        if owner is not running:
            raise ClaudeCliPoolExhausted(
                "the Claude CLI pool serves another event loop; a call on this one runs on its own CLI"
            )
        return stranded

    def _take_over_if_unserved(
        self, running: asyncio.AbstractEventLoop
    ) -> tuple[asyncio.AbstractEventLoop, list[PooledCliSession]]:
        """The loop this pool serves, after ``running`` takes it over if no open loop held it.

        :param running: the caller's running loop
        :ptype running: asyncio.AbstractEventLoop
        :return: the loop the pool now serves, and the sessions to stop when ``running`` took it
            over from a closed loop (empty otherwise)
        :rtype: tuple[asyncio.AbstractEventLoop, list[PooledCliSession]]
        """
        stranded: list[PooledCliSession] = []

        def take_over() -> asyncio.AbstractEventLoop:
            stranded.extend(self._rebind_to(running))
            return running

        owner = self._owner_loop.get(_ONLY, take_over)
        return owner, stranded

    def _rebind_to(self, loop: asyncio.AbstractEventLoop) -> list[PooledCliSession]:
        """Give this pool fresh loop-bound state for ``loop``, returning every session to stop.

        Runs under the owner ``BuildOnce``'s lock, when no loop has claimed the pool or the one
        that did has closed. No code can be running on a closed loop, so nothing is using the
        state being replaced: the condition (bound to the closed loop once anything waited on it)
        is replaced, the reaper task (cancelled with its loop, or stranded on it) is dropped, and
        every session -- idle or still marked busy by a call its loop never finished -- is taken
        out of the books, because a client whose reader task lived on the closed loop can never
        serve a call again. ``_closing`` and the SDK-failure count are about the pool, not the
        loop, and are kept.

        The sessions move to ``_stopping`` rather than out of reach, so the exit backstop still
        sees them until their kill completes. Whatever an earlier, interrupted stop left there is
        returned too, so this takeover finishes it.

        :param loop: the loop taking the pool over
        :ptype loop: asyncio.AbstractEventLoop
        :return: the sessions the previous loop held, and any whose earlier stop never completed
        :rtype: list[PooledCliSession]
        """
        self._stopping.update(self._all)
        stranded = list(self._stopping)
        took_over = self._served_a_loop
        self._served_a_loop = True
        self._all.clear()
        self._idle.clear()
        self._live.clear()
        self._total = 0
        self._condition = asyncio.Condition()
        self._reaper = None
        # A spare's task lived on the closed loop and ran no further than the loop did; its CLI,
        # if it got that far, is registered in ``_all`` and stopped with the rest.
        self._spares = set()
        if took_over:
            _logger.info(
                "The event loop the Claude CLI pool served has closed; the pool now serves the next caller's loop",
                extra={"extra_data": {"stranded_sessions": len(stranded), "loop": repr(loop)}},
            )
        return stranded

    async def _abandon(self, stranded: list[PooledCliSession]) -> None:
        """Stop the CLIs a closed event loop left behind, without touching anything that loop owned.

        :param stranded: the sessions :meth:`_rebind_to` took out of the books
        :ptype stranded: list[PooledCliSession]
        """
        outcomes = await asyncio.gather(*(self._abandon_one(session) for session in stranded), return_exceptions=True)
        stopped = sum(1 for outcome in outcomes if outcome is True)
        _logger.info(
            "Stopped the Claude CLIs a closed event loop left behind",
            extra={"extra_data": {"stopped": stopped, "stranded": len(stranded), "pids": [s.pid for s in stranded]}},
        )

    async def _abandon_one(self, session: PooledCliSession) -> bool:
        """Stop one stranded session, and drop it from ``_stopping`` only once that has succeeded.

        A failure is logged and swallowed here, not raised: one CLI that could not be stopped must
        neither fail the call that found it nor stop the others. It stays in ``_stopping``, so the
        exit backstop and the next takeover or close still try it.

        :param session: the session to stop
        :ptype session: PooledCliSession
        :return: whether its CLI is now stopped
        :rtype: bool
        """
        stopped = False
        try:
            await session.abandon(grace_seconds=self._kill_grace)
            stopped = True
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- per-session isolation: logged with its pid and kept tracked for the next attempt
            _logger.warning(
                "Could not stop a Claude CLI a closed event loop left behind; it stays tracked for the next attempt",
                extra={"extra_data": {"pid": session.pid, "error": repr(exc)}},
            )
        if stopped:
            self._stopping.discard(session)
        return stopped

    def _note_prepare_failure(self, exc: ClaudeCliSessionError) -> None:
        """Count failures of the SDK surface itself, and stop pooling when they repeat.

        A private attribute or control request that moved in an SDK release fails every prepare.
        Without this every call would start a pooled CLI, fail, dispose it and then start a one-off
        CLI -- twice the start cost, logged only as a fallback.

        :param exc: the prepare failure
        :ptype exc: ClaudeCliSessionError
        """
        if not getattr(exc, "structural", False):
            return
        self._structural_failures += 1
        if self._structural_failures >= _STRUCTURAL_FAILURE_LIMIT and not self._broken:
            self._broken = True
            _logger.warning(
                "The Claude CLI pool turned itself off: the Claude Agent SDK surface it relies on is not there",
                extra={"extra_data": {"failures": self._structural_failures, "error": str(exc)}},
            )

    async def aclose(self) -> None:
        """Stop every session this pool holds.

        A host usually closes the pool from its shutdown hook, which can run on a loop other than
        the one the pool served -- after that loop has closed. The close therefore takes the pool
        over exactly as a checkout would, so it never waits on the closed loop's condition, and
        the closed loop's sessions are killed through :meth:`PooledCliSession.abandon` rather
        than disposed, which would await clients whose reader tasks died with their loop. Each
        session stays visible to the exit backstop until its stop completes, and a stop that an
        earlier takeover never finished is finished here.

        A close from a loop other than the served one while that loop is still OPEN is refused
        before anything is touched, as a checkout from it is: the condition, the idle sessions'
        reader tasks and the reaper all belong to the open loop, which may be using them.

        :raises ClaudeCliPoolExhausted: when another event loop that is still open serves this pool
        """
        running = asyncio.get_running_loop()
        owner, _ = self._take_over_if_unserved(running)
        if owner is not running:
            raise ClaudeCliPoolExhausted(
                "the Claude CLI pool serves another event loop that is still open; close it from that loop"
            )
        # A takeover just now put the closed loop's sessions here; an interrupted earlier stop may
        # have left others. Taken before this close adds the sessions it disposes itself.
        stranded = list(self._stopping)
        async with self._condition:
            self._closing = True
            sessions = list(self._all)
            self._stopping.update(sessions)
            self._all.clear()
            self._idle.clear()
            self._live.clear()
            self._total = 0
            self._condition.notify_all()
        spares = list(self._spares)
        for spare in spares:
            spare.cancel()
        # NOSILENT: each spare was cancelled one line above; a spare that had already failed logged
        # its own failure. Neither is this close's to raise.
        await asyncio.gather(*spares, return_exceptions=True)
        if self._reaper is not None:
            self._reaper.cancel()
            # NOSILENT: we cancelled the reaper ourselves one line above; its CancelledError is the
            # expected result of that, not a failure.
            with suppress(asyncio.CancelledError):
                await self._reaper
            self._reaper = None
        for session in sessions:
            await session.dispose(grace_seconds=self._kill_grace)
            self._stopping.discard(session)
        if stranded:
            await self._abandon(stranded)
        if sessions:
            _logger.info("Closed the Claude CLI pool", extra={"extra_data": {"disposed": len(sessions)}})

    async def _acquire(self, key: str, options: Any, agent: str | None) -> PooledCliSession:
        """Take an idle session that defines the call's agent, or start one, or report exhaustion.

        :param key: the launch key
        :ptype key: str
        :param options: launch options for a session that has to start
        :ptype options: Any
        :param agent: the agent holding the call's system prompt, ``None`` for a call with none
        :ptype agent: str | None
        :return: a session owned exclusively by this caller
        :rtype: PooledCliSession
        :raises ClaudeCliPoolExhausted: when the pool is closed or every slot stays busy
        :raises ClaudeCliSessionError: when a session cannot be started
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._checkout_timeout
        async with self._condition:
            while True:
                if self._closing:
                    raise ClaudeCliPoolExhausted("the Claude CLI pool is shutting down")
                taken = self._take_idle(key, agent)
                if taken is not None:
                    return taken
                if self._total < self._max_sessions and self._live.get(key, 0) < self._per_key:
                    self._live[key] = self._live.get(key, 0) + 1
                    self._total += 1
                    break
                victim = self._longest_idle_elsewhere(key) if self._live.get(key, 0) < self._per_key else None
                if victim is None:
                    # No slot is free, and an idle session of this key that lacks the call's agent is
                    # capacity this call cannot use: it makes room, as another key's idle session
                    # would. Found measuring: at the process cap, sessions started before a new
                    # prompt appeared sat idle while every call with it ran on a CLI of its own.
                    victim = self._idle_without(key, agent)
                if victim is not None:
                    # Found live: four distinct launch keys each left one IDLE session holding a
                    # slot, and the fifth key's call ran on a CLI of its own "because every session
                    # is busy" -- while none of them was. An idle session is capacity nobody is
                    # using; the longest-idle one gives up its slot to a call that needs one.
                    victim_key, victim_session = victim
                    self._condition.release()
                    try:
                        _logger.info(
                            "Evicting an idle Claude CLI to make room for another launch",
                            extra={"extra_data": {"evicted_key": victim_key[:12], "for_key": key[:12]}},
                        )
                        # Shielded: a Stop landing during the victim's disconnect would otherwise skip
                        # its kill and its slot release, and every later call would find the pool
                        # "busy" with a session that no longer exists.
                        await asyncio.shield(self._dispose(victim_key, victim_session))
                    finally:
                        # Shielded too: a second cancellation here would leave the enclosing
                        # ``async with`` releasing a lock this task no longer holds.
                        await _reacquire(self._condition)
                    continue
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise ClaudeCliPoolExhausted(
                        f"every Claude CLI session is busy ({self._total} live, cap {self._max_sessions})"
                    )
                try:
                    await asyncio.wait_for(self._condition.wait(), remaining)
                except TimeoutError as exc:
                    raise ClaudeCliPoolExhausted(
                        f"every Claude CLI session is busy ({self._total} live, cap {self._max_sessions})"
                    ) from exc

        # Starting a CLI is seconds of work, so it happens outside the lock with the slot already
        # reserved. A ``finally`` releases it, not an ``except``: a turn cancelled while its CLI
        # starts reaches no ``except`` clause, and a slot lost here is lost for the process's life.
        started = False
        try:
            session = await self._start(self._launch_options(key, options), key=key)
            started = True
        finally:
            if not started:
                await asyncio.shield(self._forget(key))

        try:
            await asyncio.shield(self._register(session))
        except BaseException:
            # The borrower was cancelled between starting its CLI and taking it: nobody will return
            # this session, so stop it and free its slot rather than leave both stranded.
            await asyncio.shield(self._dispose(key, session))
            raise
        self._ensure_reaper()
        _logger.info(
            "Started a pooled Claude CLI",
            extra={
                "extra_data": {
                    "key": key[:12],
                    "pid": session.pid,
                    "live": self._total,
                    "cap": self._max_sessions,
                    "fields": launch_fingerprint(options),
                }
            },
        )
        return session

    async def _register(self, session: PooledCliSession) -> None:
        """Add a started session to the live set.

        :param session: the session
        :ptype session: PooledCliSession
        """
        async with self._condition:
            self._all.add(session)

    def _learn_prompt(self, key: str, options: Any) -> str | None:
        """Record the call's system prompt under its key; the agent that holds it.

        :param key: the launch key
        :ptype key: str
        :param options: the call's launch options
        :ptype options: Any
        :return: the prompt's agent name, ``None`` when the call has no switched prompt
        :rtype: str | None
        """
        switchable, prompt = _switched_prompt(options)
        if not switchable or prompt is None:
            return None
        name = agent_name(prompt)
        known = self._prompts.setdefault(key, {})
        known.pop(name, None)
        known[name] = prompt
        while len(known) > _MAX_AGENTS_PER_KEY:
            known.pop(next(iter(known)))
        return name

    def _launch_options(self, key: str, options: Any) -> Any:
        """The options a CLI for ``key`` starts with: no system prompt, every known prompt an agent.

        :param key: the launch key
        :ptype key: str
        :param options: the call's launch options
        :ptype options: Any
        :return: a new options object; the call's own is not mutated
        :rtype: Any
        """
        if not _switched_prompt(options)[0]:
            return options
        return dataclasses.replace(options, system_prompt=None, agents=dict(self._prompts.get(key, {})))

    def _take_idle(self, key: str, agent: str | None) -> PooledCliSession | None:
        """Take the most recently idle session of ``key`` that defines ``agent``. Call under the lock.

        :param key: the launch key
        :ptype key: str
        :param agent: the agent the call needs, ``None`` for none
        :ptype agent: str | None
        :return: the session, or ``None`` when no idle one can serve the call
        :rtype: PooledCliSession | None
        """
        waiting = self._idle.get(key)
        if not waiting:
            return None
        for index in range(len(waiting) - 1, -1, -1):
            candidate = waiting[index].session
            if candidate.closed:
                del waiting[index]
                continue
            if agent is None or agent in candidate.agents:
                del waiting[index]
                return candidate
        return None

    def _idle_without(self, key: str, agent: str | None) -> tuple[str, PooledCliSession] | None:
        """Take the longest-idle session of ``key`` out of the idle set, one that lacks ``agent``.

        :param key: the launch key
        :ptype key: str
        :param agent: the agent the call needs
        :ptype agent: str | None
        :return: ``(key, the session)``, or ``None`` when there is none
        :rtype: tuple[str, PooledCliSession] | None
        """
        waiting = self._idle.get(key)
        if agent is None or not waiting:
            return None
        for index, idle in enumerate(waiting):
            if agent not in idle.session.agents:
                del waiting[index]
                return key, idle.session
        return None

    def _longest_idle_elsewhere(self, key: str) -> tuple[str, PooledCliSession] | None:
        """Take the longest-idle session under any OTHER key out of the idle set. Call under the lock.

        :param key: the key a slot is wanted for
        :ptype key: str
        :return: ``(its key, the session)``, or ``None`` when no other key has one idle
        :rtype: tuple[str, PooledCliSession] | None
        """
        oldest: tuple[float, str] | None = None
        for other, waiting in self._idle.items():
            if other == key or not waiting:
                continue
            if oldest is None or waiting[0].since < oldest[0]:
                oldest = (waiting[0].since, other)
        if oldest is None:
            return None
        return oldest[1], self._idle[oldest[1]].popleft().session

    async def _return(
        self, key: str, session: PooledCliSession, *, clean: bool, options: Any, first_message_uuid: str | None
    ) -> None:
        """Rewind and re-pool a session, or stop it and start a spare in its place.

        :param key: the launch key
        :ptype key: str
        :param session: the borrowed session
        :ptype session: PooledCliSession
        :param clean: whether the call finished without any exception
        :ptype clean: bool
        :param options: the launch options the session started with, for a spare
        :ptype options: Any
        :param first_message_uuid: the uuid of the call's first message, ``None`` when it sent none
        :ptype first_message_uuid: str | None
        """
        keep = clean and not session.closed and not self._closing
        if keep:
            try:
                await session.release_tools(timeout=self._reset_timeout)
                if first_message_uuid is not None:
                    await session.rewind(first_message_uuid, timeout=self._reset_timeout)
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- whatever a returning session raises, it is stopped and its slot freed, and a call that already succeeded must not fail here
                _logger.warning(
                    "A pooled Claude CLI could not be reset; stopping it and starting a spare",
                    extra={"extra_data": {"key": key[:12], "error": str(exc)}},
                )
                keep = False
        if keep:
            async with self._condition:
                if not self._closing:
                    self._idle.setdefault(key, deque()).append(_Idle(session, time.monotonic()))
                    self._condition.notify()
                    return
        await self._dispose(key, session)
        if clean and not self._closing and not self._broken:
            self._start_spare(key, options)

    def _start_spare(self, key: str, options: Any) -> None:
        """Start a spare CLI for ``key`` in the background.

        Spawned in a FRESH context, like the reaper: the call that finished is not the spare's, and
        its request id would otherwise ride on every line the spare logs.

        :param key: the launch key
        :ptype key: str
        :param options: launch options; copied, because a start adds its own marker to them
        :ptype options: Any
        """
        task = asyncio.create_task(
            self._spare(key, self._launch_options(key, copy.copy(options))), context=contextvars.Context()
        )
        self._spares.add(task)
        task.add_done_callback(self._spares.discard)

    async def _spare(self, key: str, options: Any) -> None:
        """Start one spare and put it in the idle set, when the caps have room and none is waiting.

        One spare per key is enough to take the next call off the start; more would hold slots other
        keys may need. A spare that cannot start is logged and dropped -- the next call starts its
        own CLI as it would have anyway. A close while it starts disposes of it.

        :param key: the launch key
        :ptype key: str
        :param options: launch options for the spare
        :ptype options: Any
        """
        async with self._condition:
            if (
                self._closing
                or self._idle.get(key)
                or self._total >= self._max_sessions
                or self._live.get(key, 0) >= self._per_key
            ):
                return
            self._live[key] = self._live.get(key, 0) + 1
            self._total += 1
        session: PooledCliSession | None = None
        try:
            session = await self._start(options, key=key)
        except ClaudeCliSessionError as exc:
            _logger.warning(
                "Could not start a spare Claude CLI; the next call starts its own",
                extra={"extra_data": {"key": key[:12], "error": str(exc)}},
            )
        finally:
            if session is None:
                await asyncio.shield(self._forget(key))
        if session is None:
            return
        async with self._condition:
            parked = not self._closing
            self._all.add(session)
            if parked:
                self._idle.setdefault(key, deque()).append(_Idle(session, time.monotonic()))
                self._condition.notify()
        if not parked:
            await self._dispose(key, session)
            return
        self._ensure_reaper()
        _logger.info(
            "Started a spare Claude CLI",
            extra={"extra_data": {"key": key[:12], "pid": session.pid, "live": self._total, "cap": self._max_sessions}},
        )

    async def _forget(self, key: str, session: PooledCliSession | None = None) -> None:
        """Release a slot and wake anyone waiting for one.

        :param key: the launch key
        :ptype key: str
        :param session: the session to drop from the live set, when there is one
        :ptype session: PooledCliSession | None
        """
        async with self._condition:
            if session is not None:
                self._all.discard(session)
            live = self._live.get(key, 0)
            if live > 0:
                self._live[key] = live - 1
                self._total = max(0, self._total - 1)
            if self._live.get(key) == 0:
                self._live.pop(key, None)
                self._idle.pop(key, None)
            self._condition.notify()

    async def _dispose(self, key: str, session: PooledCliSession) -> None:
        """Stop one session and release its slot once the process is actually gone.

        :param key: the launch key
        :ptype key: str
        :param session: the session to stop
        :ptype session: PooledCliSession
        """
        await session.dispose(grace_seconds=self._kill_grace)
        await self._forget(key, session)
        _logger.info("Stopped a pooled Claude CLI", extra={"extra_data": {"pid": session.pid, "live": self._total}})

    def _ensure_reaper(self) -> None:
        """Start the idle-eviction task if it is not already running.

        Spawned in a FRESH context: the first session is started inside some caller's request, and
        a task created there copies that request's context variables -- every reaper log line for
        the life of the process would carry a request id that ended long ago.
        """
        if self._idle_ttl <= 0 or (self._reaper is not None and not self._reaper.done()):
            return
        self._reaper = asyncio.create_task(self._reap_forever(), context=contextvars.Context())

    async def _reap_forever(self) -> None:
        """Dispose of sessions that have sat idle longer than the TTL."""
        interval = max(1.0, min(self._idle_ttl / 2.0, 30.0))
        while not self._closing:
            await asyncio.sleep(interval)
            try:
                await self._reap_once()
            except (
                Exception
            ) as exc:  # prawduct:allow prawduct/broad-except -- a background supervisor loop must outlive one bad pass
                _logger.warning("The Claude CLI idle reaper failed a pass", extra={"extra_data": {"error": str(exc)}})

    async def _reap_once(self) -> None:
        """One idle-eviction pass."""
        cutoff = time.monotonic() - self._idle_ttl
        expired: list[tuple[str, PooledCliSession]] = []
        async with self._condition:
            for key, waiting in self._idle.items():
                while waiting and waiting[0].since <= cutoff:
                    expired.append((key, waiting.popleft().session))
        for key, session in expired:
            _logger.info("Evicting an idle Claude CLI", extra={"extra_data": {"key": key[:12], "pid": session.pid}})
            await self._dispose(key, session)


# ---------------------------------------------------------------------------
# The process-wide pool
# ---------------------------------------------------------------------------

_pool_settings: dict[str, Any] = {}
_pool_disabled = False

#: The process-wide pool, built on first use. A sync ``invoke`` runs its own event loop on the
#: caller's thread, so a consumer calling models from several threads reaches the accessor from
#: several threads at once; built through ``BuildOnce`` so exactly one pool exists -- a second
#: would run twice the CLIs the limits allow and be orphaned. It serves the event loop that first
#: checks a session out of it until that loop closes (:meth:`ClaudeCliPool.checkout`).
_pool: BuildOnce[str, ClaudeCliPool] = BuildOnce()

#: The key of a ``BuildOnce`` that holds a single value: the process's pool, a pool's loop.
_ONLY = "only"


def configure_claude_cli_pool(*, enabled: bool = True, **settings: Any) -> None:
    """Set the process-wide pool's limits before first use, or turn pooling off.

    :param enabled: ``False`` runs every call on its own CLI, as before pooling existed
    :ptype enabled: bool
    :param settings: :class:`ClaudeCliPool` constructor arguments
    :ptype settings: Any
    """
    global _pool_settings, _pool_disabled
    _pool_disabled = not enabled
    _pool_settings = dict(settings)


def claude_cli_pool() -> ClaudeCliPool | None:
    """This process's pool, built on first use; ``None`` when the host turned pooling off.

    :return: the pool, or ``None``
    :rtype: ClaudeCliPool | None
    """
    if _pool_disabled:
        return None
    return _pool.get(_ONLY, _build_pool)


def _build_pool() -> ClaudeCliPool:
    """Build the process's pool with the configured limits, and make sure its CLIs die with it.

    :return: the new pool
    :rtype: ClaudeCliPool
    """
    pool = ClaudeCliPool(**_pool_settings)
    atexit.register(_kill_remaining_at_exit, pool)
    return pool


async def close_claude_cli_pool() -> None:
    """Stop every CLI this process holds. A host calls this on a clean shutdown.

    The pool is let go only once its close has succeeded: a close refused because another open
    loop serves the pool leaves it in place and serving, rather than dropping the one reference
    through which it could still be closed. While the close runs, a call that reaches the pool is
    told it is shutting down and runs on a CLI of its own.

    :raises ClaudeCliPoolExhausted: when another event loop that is still open serves the pool
    """
    pool = _pool.peek(_ONLY)
    if pool is None:
        return
    await pool.aclose()
    if _pool.peek(_ONLY) is pool:
        _pool.pop(_ONLY)


def _kill_remaining_at_exit(pool: ClaudeCliPool) -> None:
    """Last resort at interpreter exit for a host that never closed its pool.

    :param pool: the pool whose CLIs must not outlive the process
    :ptype pool: ClaudeCliPool
    """
    for pid in pool.known_pids():
        kill_process_tree(pid, grace_seconds=0.5)
