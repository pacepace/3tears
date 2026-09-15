"""Reuse Claude Code CLI subprocesses instead of starting one per model call.

A Claude subscription credential (``sk-ant-oat…``) has no HTTP API: it is spent by driving the
bundled Claude Code CLI, and ``langchain-claude-code`` starts a fresh subprocess for every call.
Measured on a production deployment: ~2.4 s to start and ~2.7 s per call, against ~600 ms for the
same model over HTTP. This module keeps CLIs alive, hands each out to one caller at a time, and
clears it between callers. Design, with the evidence for every protocol claim:
``docs/claude-cli-session-pool-design.md``.

What can change on a live CLI and what cannot decides the shape:

- the **system prompt** is a launch flag and never changes, so it is part of the key;
- the **model** changes per call with ``set_model``;
- the **bound tools** (an in-process MCP server) change per call with the CLI's
  ``mcp_set_servers`` control request -- ``reconnect_mcp_server`` refuses SDK servers;
- the **conversation** is dropped with the CLI's local ``/clear``, which is not billed.

Four things keep CLIs from piling up, because each of the others misses a case:

- every session's pid is tracked, and disposal kills the process and its descendants (see
  :func:`kill_process_tree` for why not the process group);
- the host closes the pool on a clean stop (:func:`close_claude_cli_pool`);
- a startup sweep kills CLIs a crashed previous process left behind
  (:func:`sweep_orphaned_claude_clis`), which is the shutdown a crash skips;
- an idle TTL and hard caps bound the live count while the process runs.

The pool is per process. A host running N worker processes has a ceiling of N times its cap.

Standard library and lazy SDK imports only, so importing this module costs nothing on a host that
never spends a subscription.
"""

from __future__ import annotations

import asyncio
import atexit
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

from threetears.observe import get_logger

__all__ = [
    "POOL_MARKER_ENV",
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
    """A pooled CLI session failed to start, prepare, or clear."""


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

#: Options applied per checkout rather than at launch, so they are not part of the key.
_PER_CHECKOUT_FIELDS = frozenset({"model", "mcp_servers"})

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
    return sorted(n for n in names if n not in _PER_CHECKOUT_FIELDS)


def poolable(options: Any) -> bool:
    """Whether a call with these options may run on a shared, reused CLI.

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


def _advertises_clear(server_info: dict[str, Any] | None) -> bool:
    """Whether the connected CLI advertises the local ``clear`` command.

    A CLI that does not advertise it gets single-use sessions instead: slower, never leaky.

    :param server_info: the initialize handshake result
    :ptype server_info: dict[str, Any] | None
    :return: whether ``clear`` is available
    :rtype: bool
    """
    commands = (server_info or {}).get("commands")
    if not isinstance(commands, list) or not commands:
        return False
    for command in commands:
        if isinstance(command, str) and command.lstrip("/") == "clear":
            return True
        if isinstance(command, dict) and str(command.get("name", "")).lstrip("/") == "clear":
            return True
    return False


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
    pid = getattr(getattr(getattr(client, "_transport", None), "_process", None), "pid", None)
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
    created when the CLI connected -- inside whichever caller STARTED the session. On a reused
    session every later borrower's tool calls therefore ran with the first borrower's context
    variables: an ``interrupt()`` was captured into the first caller's list and the graph never
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


class PooledCliSession:
    """One live CLI subprocess, lent to exactly one caller at a time."""

    def __init__(self, client: Any, *, key: str, pid: int | None, marker: str, reusable: bool) -> None:
        self.client = client
        self.key = key
        self.pid = pid
        self.marker = marker
        self.reusable = reusable
        self.closed = False
        self._model: str | None = None
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
        client = sdk.ClaudeSDKClient(options=options)
        try:
            await client.connect()
            server_info = await client.get_server_info()
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
        reusable = _advertises_clear(server_info)
        if not reusable:
            _logger.warning("The Claude CLI does not advertise the clear command; its sessions will be single-use")
        session = cls(client, key=key, pid=_discover_pid(client, marker), marker=marker, reusable=reusable)
        session._model = getattr(options, "model", None)
        return session

    async def prepare(
        self, *, model: str | None, tool_server: Any | None, call_context: contextvars.Context | None = None
    ) -> None:
        """Point this session at the caller's model and the caller's tools.

        The tool server is replaced on EVERY checkout, never reused: its handlers close over the
        caller's own tool objects, which belong to that caller's conversation. A stale server
        would run one conversation's tools on another's behalf.

        :param model: the model the call wants, or ``None`` to keep the current one
        :ptype model: str | None
        :param tool_server: the call's in-process MCP server instance, or ``None`` for no tools
        :ptype tool_server: Any | None
        :param call_context: the borrower's context, which every tool call on this checkout runs in
        :ptype call_context: contextvars.Context | None
        :raises ClaudeCliSessionError: when either change is refused
        """
        if self.closed:
            raise ClaudeCliSessionError("this Claude CLI session has been stopped")
        try:
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

        Bounded by the same short leash as ``/clear``: it runs after a call has already finished,
        so a hung CLI must cost that caller seconds, not the control request's default 30.

        :param timeout: seconds before the release is abandoned
        :ptype timeout: float
        :raises ClaudeCliSessionError: when the CLI refuses or times out; the pool disposes the session
        """
        if self.closed:
            raise ClaudeCliSessionError("this Claude CLI session has been stopped")
        try:
            query = self.client._query  # noqa: SLF001
            query.sdk_mcp_servers.pop(TOOL_SERVER_NAME, None)
            await query._send_control_request({"subtype": "mcp_set_servers", "servers": {}}, timeout=timeout)  # noqa: SLF001
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- any failure means the session cannot be trusted idle; the pool disposes it
            raise ClaudeCliSessionError(f"could not release the Claude CLI's tools: {exc}") from exc

    async def clear(self, *, timeout: float) -> None:
        """Drop the session's conversation with the CLI's local ``/clear``.

        :param timeout: seconds before the clear is abandoned
        :ptype timeout: float
        :raises ClaudeCliSessionError: when the clear fails; the pool disposes the session
        """
        import claude_agent_sdk as sdk  # noqa: PLC0415

        if self.closed:
            raise ClaudeCliSessionError("this Claude CLI session has been stopped")
        if not self.reusable:
            raise ClaudeCliSessionError("this CLI cannot clear its context")
        try:
            async with asyncio.timeout(timeout):
                await self.client.query("/clear")
                async for message in self.client.receive_response():
                    if isinstance(message, sdk.ResultMessage):
                        break
        except TimeoutError as exc:
            raise ClaudeCliSessionError(f"the Claude CLI did not clear within {timeout}s") from exc
        except Exception as exc:  # prawduct:allow prawduct/broad-except -- the SDK surfaces a CLI that died mid-clear as a bare Exception from receive_messages; every failure must dispose the session, never leak its slot
            raise ClaudeCliSessionError(f"the Claude CLI failed to clear: {exc}") from exc

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
        if self.pid is not None and _alive(self.pid):
            if self.start_ticks is not None and _process_start_ticks(self.pid) != self.start_ticks:
                # The CLI exited and its pid now belongs to something else: signal nothing.
                return
            await asyncio.to_thread(kill_process_tree, self.pid, grace_seconds=grace_seconds)


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
    """Bounded pools of reusable Claude CLI sessions, one pool per launch key."""

    def __init__(
        self,
        *,
        max_sessions: int = 8,
        per_key: int = 2,
        idle_ttl_seconds: float = 300.0,
        checkout_timeout_seconds: float = 2.0,
        clear_timeout_seconds: float = 5.0,
        kill_grace_seconds: float = 2.0,
        session_factory: Callable[..., Awaitable[PooledCliSession]] | None = None,
    ) -> None:
        if max_sessions < 1 or per_key < 1:
            raise ValueError("max_sessions and per_key must both be at least 1")
        self._max_sessions = max_sessions
        self._per_key = per_key
        self._idle_ttl = idle_ttl_seconds
        self._checkout_timeout = checkout_timeout_seconds
        self._clear_timeout = clear_timeout_seconds
        self._kill_grace = kill_grace_seconds
        self._condition = asyncio.Condition()
        self._idle: dict[str, deque[_Idle]] = {}
        self._live: dict[str, int] = {}
        self._all: set[PooledCliSession] = set()
        self._total = 0
        self._closing = False
        self._reaper: asyncio.Task[None] | None = None
        #: How a session is started. Injected by tests; a real host never passes it.
        self._start = session_factory or PooledCliSession.start
        self._structural_failures = 0
        self._broken = False

    @property
    def live_count(self) -> int:
        """How many CLI sessions this pool currently holds."""
        return self._total

    def known_pids(self) -> list[int]:
        """Every live session's pid, for a synchronous last-resort kill at interpreter exit.

        :return: the pids
        :rtype: list[int]
        """
        return [s.pid for s in self._all if s.pid is not None and not s.closed]

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
        ``ResultMessage``. A call that leaves normally is cleared and re-pooled; one that leaves by
        any exception -- a failure, a timeout, a cancellation, a consumer that stopped reading --
        disposes of the session, because an abandoned stream is the one way a later caller could
        read an earlier caller's answer.

        :param options: the launch options for a session this call could use
        :ptype options: Any
        :param token: the credential, for the key
        :ptype token: str | None
        :param tool_server: the call's in-process MCP server instance, or ``None``
        :ptype tool_server: Any | None
        :param call_context: the borrower's context; its tool calls run in a copy of it
        :ptype call_context: contextvars.Context | None
        :return: the connected ``ClaudeSDKClient``
        :rtype: AsyncIterator[Any]
        :raises ClaudeCliPoolExhausted: when no session frees up in time
        :raises ClaudeCliSessionError: when a session cannot be started or prepared
        """
        if self._broken:
            raise ClaudeCliPoolExhausted("pooling is off: the Claude Agent SDK's surface failed repeatedly")
        if not poolable(options):
            raise ClaudeCliPoolExhausted("this call carries callables or a resumed session and cannot share a CLI")
        key = launch_key(options, token)
        session = await self._acquire(key, options)
        clean = False
        try:
            try:
                await session.prepare(
                    model=getattr(options, "model", None), tool_server=tool_server, call_context=call_context
                )
            except ClaudeCliSessionError as exc:
                self._note_prepare_failure(exc)
                raise
            self._structural_failures = 0
            yield session.client
            clean = True
        finally:
            await asyncio.shield(self._return(key, session, clean=clean))

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
        """Stop every session this pool holds."""
        async with self._condition:
            self._closing = True
            sessions = list(self._all)
            self._all.clear()
            self._idle.clear()
            self._live.clear()
            self._total = 0
            self._condition.notify_all()
        if self._reaper is not None:
            self._reaper.cancel()
            # NOSILENT: we cancelled the reaper ourselves one line above; its CancelledError is the
            # expected result of that, not a failure.
            with suppress(asyncio.CancelledError):
                await self._reaper
            self._reaper = None
        for session in sessions:
            await session.dispose(grace_seconds=self._kill_grace)
        if sessions:
            _logger.info("Closed the Claude CLI pool", extra={"extra_data": {"disposed": len(sessions)}})

    async def _acquire(self, key: str, options: Any) -> PooledCliSession:
        """Take an idle session, or start one, or report exhaustion.

        :param key: the launch key
        :ptype key: str
        :param options: launch options for a session that has to start
        :ptype options: Any
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
                waiting = self._idle.get(key)
                while waiting:
                    candidate = waiting.pop().session
                    if not candidate.closed:
                        return candidate
                if self._total < self._max_sessions and self._live.get(key, 0) < self._per_key:
                    self._live[key] = self._live.get(key, 0) + 1
                    self._total += 1
                    break
                victim = self._longest_idle_elsewhere(key) if self._live.get(key, 0) < self._per_key else None
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
            session = await self._start(options, key=key)
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

    async def _return(self, key: str, session: PooledCliSession, *, clean: bool) -> None:
        """Clear and re-pool a session, or dispose of it.

        :param key: the launch key
        :ptype key: str
        :param session: the borrowed session
        :ptype session: PooledCliSession
        :param clean: whether the call finished without any exception
        :ptype clean: bool
        """
        keep = clean and session.reusable and not session.closed and not self._closing
        if keep:
            try:
                await session.release_tools(timeout=self._clear_timeout)
                await session.clear(timeout=self._clear_timeout)
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- whatever a returning session raises, it must be disposed and its slot freed, and a call that already succeeded must not fail here
                _logger.warning(
                    "A pooled Claude CLI would not clear its context; disposing of it",
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

_pool: ClaudeCliPool | None = None
_pool_settings: dict[str, Any] = {}
_pool_disabled = False


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
    global _pool
    if _pool_disabled:
        return None
    if _pool is None:
        _pool = ClaudeCliPool(**_pool_settings)
        atexit.register(_kill_remaining_at_exit, _pool)
    return _pool


async def close_claude_cli_pool() -> None:
    """Stop every CLI this process holds. A host calls this on a clean shutdown."""
    global _pool
    if _pool is None:
        return
    pool, _pool = _pool, None
    await pool.aclose()


def _kill_remaining_at_exit(pool: ClaudeCliPool) -> None:
    """Last resort at interpreter exit for a host that never closed its pool.

    :param pool: the pool whose CLIs must not outlive the process
    :ptype pool: ClaudeCliPool
    """
    for pid in pool.known_pids():
        kill_process_tree(pid, grace_seconds=0.5)
