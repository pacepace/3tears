"""no-bespoke-reuse enforcement guard (gu-task-06, GU-06-07 — 3tears half).

the Grand Unification epic has exactly ONE new component (the OpenAPI parser,
gu-task-23); everything else reuses an existing primitive. this guard is the durable
encoding of the epic reuse-map: it fails if any class under a 3tears package hand-rolls
one of the three patterns the platform already solves:

(a) a buffer + periodic-flush loop — reuse ``ConversationWriteBuffer``
    (``threetears.conversations.buffer``);
(b) a raw, long-lived ``httpx.AsyncClient`` / ``httpx.Client`` held as a field outside
    the sanctioned traced HTTP-client wrapper (``threetears.core.http_client``);
    **the norm, restated (D19).** The rule is not "no package may touch httpx". It is
    that raw HTTP client ownership belongs in a *dedicated transport module* which
    carries the full set of obligations -- configurable timeout, bounded retry with
    backoff, per-attempt accounting, SSRF guards, streamed reads under a byte cap --
    and that every SERVICE reaches an upstream through one of those rather than
    opening its own. Adding a path to ``_SANCTIONED_HTTPX_SITES`` is therefore a
    WIDENING of the norm rather than an exemption from it, and it is legitimate on
    exactly one condition: the module is a transport, it is single-purpose, and it
    discharges the obligations in full. A service, a tool, or a client-holding
    helper never qualifies, whatever its docstring says.
    That distinction is why ``3tears-search``'s ``standalone`` module is listed below
    and files no entry in ``_no_bespoke_reuse_exemptions.txt``: hosts without
    ``threetears.core`` need one sanctioned place to open a socket, and the
    alternative to giving them one is a bespoke ``httpx.get`` per call site --
    which is the thing this guard exists to prevent, arrived at by refusing to
    say where the sanctioned place is.
(c) a KV / counter that stores an ``SQLiteBackend`` and exposes cache-style verbs
    without subclassing ``BaseCollection`` — reuse the Collection primitive.

check (c) DELEGATES to the shared, already-proven ``find_wrapper_classes`` walker in
``threetears.enforcement.cache`` (reuse, not a re-rolled walker — the same rule this
guard enforces). checks (a) and (b) are small AST walkers local to this module, scoped
to STORED fields (a long-lived bespoke buffer / client), not one-shot ephemeral usage.

mode via ``REUSE_ENFORCEMENT_MODE`` (default ``strict``), mirroring
``CACHE_ENFORCEMENT_MODE``. exemptions live in ``_no_bespoke_reuse_exemptions.txt`` and
require a specific ``# rationale:`` line. the sanctioned reuse targets are allowlisted in
the walkers themselves — they are the components bespoke code should reuse.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from threetears.enforcement.cache import find_wrapper_classes
from threetears.enforcement.common import (
    MODE_REPORT,
    MODE_STRICT,
    Violation,
    apply_exemptions,
    discover_src_roots,
    emit_report,
    find_local_src_roots,
    iter_python_files,
    parse_exemptions_with_rationale,
    parse_python_file,
    relative_posix_path,
    resolve_mode,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXEMPTIONS_PATH = _REPO_ROOT / "tests" / "enforcement" / "_no_bespoke_reuse_exemptions.txt"
_MODE_ENV_VAR = "REUSE_ENFORCEMENT_MODE"

_FLUSH_LOOP_CATEGORY = "reuse.bespoke_flush_loop"
_HTTPX_CLIENT_CATEGORY = "reuse.bespoke_httpx_client"

# the sanctioned reuse targets — the components bespoke code is meant to reuse, not
# exceptions to the rule. paths are repo-relative posix.
_SANCTIONED_FLUSH_SITES: frozenset[str] = frozenset(
    {
        # the platform's periodic write-behind buffer primitive. bespoke buffer+flush
        # loops must reuse THIS rather than re-rolling a timer loop over an accumulator.
        "packages/conversations/src/threetears/conversations/buffer.py",
        # 3tears-evals' job-lease renewal. Not a write-behind buffer: each tick re-reads every held job record and
        # extends its lease with a conditional write, and what the read finds (a cancel asked, a record ended
        # elsewhere, a lapsed lease) decides whether the job keeps running. Batching or deferring those writes, which
        # is what the write buffer does, would let a lease lapse while its job still runs.
        "packages/evals/src/threetears/evals/kernel/job_leases.py",
    }
)
_SANCTIONED_HTTPX_SITES: frozenset[str] = frozenset(
    {
        # the sanctioned traced HTTP-client wrapper — the ONE place a long-lived
        # httpx.AsyncClient is owned + wrapped with tracing/retry.
        "packages/core/src/threetears/core/http_client.py",
        # the mcp package's single-purpose HTTP transport wrapper (its own traced
        # client module); the anti-pattern is a SERVICE holding a bespoke client, not
        # a dedicated transport module.
        "packages/mcp/src/threetears/mcp/http_client.py",
        # 3tears-search's bare-httpx SearchTransport implementation, behind its
        # ``[standalone]`` extra (D19). The leaf must not import core (SR-L7), and its
        # embedded consumers -- samsung, anything on a Pi -- have no traced wrapper to
        # inject, so this is the ONE module in the leaf permitted to open a client. It
        # earns the listing by discharging the full obligation set rather than a subset:
        # configurable timeout with a per-call override (SR-G1, SR-G2), bounded retry
        # with backoff where 5xx and connect/timeout retry and no 4xx does (SR-G4),
        # per-attempt accounting surfaced on the response so budget can follow the bill
        # (D4), private-address refusal plus a host allowlist plus a non-following
        # redirect policy (D21, SR-K3, SR-N3), and streamed reads under a byte cap
        # rather than an unbounded read (SR-G5). Every other module in that package
        # reaches an upstream through the injected ``SearchTransport`` protocol, which
        # this walker correctly never flags.
        "packages/search/src/threetears/search/standalone.py",
        # 3tears-models' transport to the self-hosted image servers (ComfyUI, SwarmUI) that
        # ``ComfyUIClient`` and ``SwarmUIClient`` drive. The models package must not import core
        # (its dependencies are media-contracts, observe and tool-schema), so core's
        # ``TracedHttpClient`` is out of reach; this is the ONE module in models' image providers
        # permitted to open a long-lived client, and both clients reach their server through its
        # ``ImageServerTransport`` rather than holding one. It discharges the full obligation set:
        # a configurable timeout with a per-call override; bounded retry with doubling backoff
        # (``RetryPolicy``, via observe's ``retry_bounded``) where a read repeats on connect,
        # timeout, dropped connection or 5xx and a request that starts work (queue a graph, draw a
        # picture, upload) repeats only when it never left (``NOT_SENT``), so nothing is drawn
        # twice; per-attempt accounting (each attempt logged with method, path, attempt number,
        # outcome and time, and an error log when a request fails for good); an origin guard in
        # place of a private-address refusal -- a caller names a path that is joined to the
        # operator-configured base URL, a full or scheme-relative URL is refused, a request whose
        # origin differs from the base's is refused, and redirects are not followed (private and
        # loopback addresses are allowed on purpose: these servers are GPU boxes on the LAN by
        # design and their address never comes from request data); and streamed reads under a
        # configurable byte cap, a declared ``Content-Length`` over it refused before reading
        # (``ResponseTooLargeError``).
        "packages/models/src/threetears/models/providers/image/transport.py",
    }
)

# check-(c) reuse knobs, mirroring the cache-enforcement defaults exactly so the two
# guards agree on what counts as a cache-verb wrapper (the proven-green set deliberately
# excludes ``remove`` — the sanctioned ``WriteBuffer`` flush primitive exposes it).
_BASE_COLLECTION_NAMES: frozenset[str] = frozenset({"BaseCollection"})
_CACHE_METHOD_NAMES: frozenset[str] = frozenset({"get", "put", "set", "delete", "upsert"})


# ---------------------------------------------------------------------------
# check (a) — bespoke buffer + periodic-flush loop
# ---------------------------------------------------------------------------


def _is_sleep_call(func: ast.expr) -> bool:
    """true iff ``func`` names a ``*.sleep`` call (``asyncio.sleep`` / ``time.sleep``).

    :param func: the ``ast.Call.func`` node under inspection
    :ptype func: ast.expr
    :return: whether the call target is a ``.sleep`` attribute access
    :rtype: bool
    """
    return isinstance(func, ast.Attribute) and func.attr == "sleep"


def _class_has_periodic_loop(node: ast.ClassDef) -> bool:
    """true iff the class contains a ``while`` loop whose body calls ``*.sleep``.

    :param node: class definition to inspect
    :ptype node: ast.ClassDef
    :return: whether a periodic (sleep-driven) loop is present
    :rtype: bool
    """
    result = False
    for sub in ast.walk(node):
        if isinstance(sub, ast.While):
            for inner in ast.walk(sub):
                if isinstance(inner, ast.Call) and _is_sleep_call(inner.func):
                    result = True
                    break
        if result:
            break
    return result


def _self_attr(value: ast.expr) -> str | None:
    """return the attribute name if ``value`` is ``self.<attr>``, else ``None``.

    :param value: expression to test
    :ptype value: ast.expr
    :return: the attribute name, or ``None`` when ``value`` is not ``self.<attr>``
    :rtype: str | None
    """
    result: str | None = None
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "self":
        result = value.attr
    return result


def _class_accumulation_buffer(node: ast.ClassDef) -> str | None:
    """return the name of a ``self`` buffer the class accumulates into, else ``None``.

    matches ``self.<buf>.append(...)`` / ``.appendleft(...)`` and subscript stores
    ``self.<buf>[key] = ...`` — the two long-lived-accumulator shapes.

    :param node: class definition to inspect
    :ptype node: ast.ClassDef
    :return: the buffer attribute name, or ``None`` when the class does not accumulate
    :rtype: str | None
    """
    result: str | None = None
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in {"append", "appendleft"}
        ):
            buffer_name = _self_attr(sub.func.value)
            if buffer_name is not None:
                result = buffer_name
                break
        if isinstance(sub, ast.Assign):
            hit = False
            for target in sub.targets:
                if isinstance(target, ast.Subscript):
                    buffer_name = _self_attr(target.value)
                    if buffer_name is not None:
                        result = buffer_name
                        hit = True
                        break
            if hit:
                break
    return result


def find_bespoke_flush_loops(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
    allowed_sites: frozenset[str],
) -> list[Violation]:
    """flag classes that hand-roll a buffer + periodic-flush loop.

    fingerprint: the class BOTH accumulates into a ``self`` buffer AND drives a
    ``while`` loop with a ``*.sleep`` tick. that pair is the write-behind buffer the
    platform already ships as ``ConversationWriteBuffer``; bespoke re-rolls must reuse
    it. sanctioned reuse targets in ``allowed_sites`` are skipped.

    :param scan_roots: src roots to scan for violations
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root for relative-path rendering
    :ptype repo_root: Path
    :param allowed_sites: repo-relative posix paths of sanctioned primitives to skip
    :ptype allowed_sites: frozenset[str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in scan_roots:
        for source in iter_python_files(root):
            if relative_posix_path(source, repo_root) in allowed_sites:
                continue
            tree = parse_python_file(source)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                buffer_name = _class_accumulation_buffer(node)
                if buffer_name is None:
                    continue
                if not _class_has_periodic_loop(node):
                    continue
                violations.append(
                    Violation(
                        category=_FLUSH_LOOP_CATEGORY,
                        file=source,
                        line=node.lineno,
                        symbol=node.name,
                        reason=(
                            f"class '{node.name}' hand-rolls a buffer ('self.{buffer_name}') "
                            "plus a periodic sleep loop; reuse "
                            "threetears.conversations.buffer.ConversationWriteBuffer instead"
                        ),
                    )
                )
    return violations


# ---------------------------------------------------------------------------
# check (b) — bespoke long-lived httpx client held as a field
# ---------------------------------------------------------------------------


def _names_httpx_client(value: ast.expr) -> bool:
    """true iff ``value`` names ``httpx.AsyncClient`` / ``httpx.Client`` (or bare).

    :param value: expression to test (a call target or an annotation)
    :ptype value: ast.expr
    :return: whether the expression references an httpx client type
    :rtype: bool
    """
    result = False
    if isinstance(value, ast.Attribute) and value.attr in {"AsyncClient", "Client"}:
        result = isinstance(value.value, ast.Name) and value.value.id == "httpx"
    elif isinstance(value, ast.Name) and value.id in {"AsyncClient", "Client"}:
        result = True
    return result


#: context-manager entry calls whose argument is the object kept: ``stack.enter_async_context(c)``
_ENTER_CONTEXT_NAMES: frozenset[str] = frozenset({"enter_async_context", "enter_context"})
#: typing wrappers whose arguments are the types a field may hold
_OPTIONAL_WRAPPERS: frozenset[str] = frozenset({"Optional", "Union"})


def _constructs_httpx_client(value: ast.expr) -> bool:
    """true iff ``value`` builds an httpx client, directly or through a context-manager entry.

    matches ``httpx.AsyncClient(...)``, ``await stack.enter_async_context(httpx.AsyncClient(...))``
    and ``stack.enter_context(httpx.Client(...))`` -- each keeps the client it builds. a client
    handed to anything else (a wrapper's constructor) is not this check's business.

    :param value: the stored expression
    :ptype value: ast.expr
    :return: whether the expression yields a raw httpx client
    :rtype: bool
    """
    if isinstance(value, ast.Await):
        value = value.value
    result = False
    if isinstance(value, ast.Call):
        if _names_httpx_client(value.func):
            result = True
        elif isinstance(value.func, ast.Attribute) and value.func.attr in _ENTER_CONTEXT_NAMES and value.args:
            result = _constructs_httpx_client(value.args[0])
    return result


def _annotation_holds_httpx_client(annotation: ast.expr) -> bool:
    """true iff an annotation admits an httpx client: bare, ``| None``, ``Optional``/``Union``, or quoted.

    :param annotation: the annotation
    :ptype annotation: ast.expr
    :return: whether a field so annotated may hold an httpx client
    :rtype: bool
    """
    result = False
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            parsed = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            parsed = None
        result = parsed is not None and _annotation_holds_httpx_client(parsed)
    elif isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        result = _annotation_holds_httpx_client(annotation.left) or _annotation_holds_httpx_client(annotation.right)
    elif isinstance(annotation, ast.Subscript):
        wrapper = annotation.value
        name = wrapper.attr if isinstance(wrapper, ast.Attribute) else getattr(wrapper, "id", None)
        if name in _OPTIONAL_WRAPPERS:
            members = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
            result = any(_annotation_holds_httpx_client(member) for member in members)
    else:
        result = _names_httpx_client(annotation)
    return result


def _class_stores_httpx_client(node: ast.ClassDef) -> bool:
    """true iff the class stores an httpx client as a field (constructed or annotated).

    matches, on ``self``: ``self.<x> = httpx.AsyncClient(...)``, the same built inside a
    context-manager entry (``self.<x> = await stack.enter_async_context(httpx.AsyncClient(...))``),
    and an annotated store whose annotation admits a client (``self.<x>: httpx.AsyncClient | None``,
    ``Optional[...]``, ``Union[...]``, quoted). in the class body: a field declared with such an
    annotation. one-shot ephemeral ``async with httpx.AsyncClient() as c`` usage is intentionally
    NOT matched.

    :param node: class definition to inspect
    :ptype node: ast.ClassDef
    :return: whether the class holds an httpx client field
    :rtype: bool
    """
    result = any(
        isinstance(stmt, ast.AnnAssign) and _annotation_holds_httpx_client(stmt.annotation) for stmt in node.body
    )
    for sub in ast.walk(node):
        if result:
            break
        if isinstance(sub, ast.Assign):
            stores_self = any(_self_attr(target) is not None for target in sub.targets)
            result = stores_self and _constructs_httpx_client(sub.value)
        elif isinstance(sub, ast.AnnAssign) and _self_attr(sub.target) is not None:
            result = _annotation_holds_httpx_client(sub.annotation) or (
                sub.value is not None and _constructs_httpx_client(sub.value)
            )
    return result


def find_bespoke_httpx_clients(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
    allowed_sites: frozenset[str],
) -> list[Violation]:
    """flag classes that hold a raw, long-lived httpx client field.

    the sanctioned traced HTTP-client wrapper modules in ``allowed_sites`` are skipped;
    every other class holding an ``httpx.AsyncClient`` / ``httpx.Client`` field is a
    bespoke client that should route through the traced wrapper.

    :param scan_roots: src roots to scan for violations
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root for relative-path rendering
    :ptype repo_root: Path
    :param allowed_sites: repo-relative posix paths of sanctioned wrapper modules to skip
    :ptype allowed_sites: frozenset[str]
    :return: violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in scan_roots:
        for source in iter_python_files(root):
            if relative_posix_path(source, repo_root) in allowed_sites:
                continue
            tree = parse_python_file(source)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                if not _class_stores_httpx_client(node):
                    continue
                violations.append(
                    Violation(
                        category=_HTTPX_CLIENT_CATEGORY,
                        file=source,
                        line=node.lineno,
                        symbol=node.name,
                        reason=(
                            f"class '{node.name}' holds a raw httpx client field; route through "
                            "the traced wrapper (threetears.core.http_client) instead"
                        ),
                    )
                )
    return violations


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _assert_clean(violations: list[Violation], domain: str) -> None:
    """apply exemptions + mode and fail in strict mode with a rendered report.

    :param violations: raw walker output
    :ptype violations: list[Violation]
    :param domain: report domain label
    :ptype domain: str
    :return: nothing
    :rtype: None
    :raises pytest.fail.Exception: in strict mode with surviving violations
    """
    exemptions = parse_exemptions_with_rationale(_EXEMPTIONS_PATH)
    filtered = apply_exemptions(violations, exemptions, _REPO_ROOT)
    mode = resolve_mode(_MODE_ENV_VAR, default=MODE_STRICT)
    report = emit_report(filtered, (_REPO_ROOT,), exemptions, mode, _REPO_ROOT, domain=domain)
    print(report, file=sys.stderr)
    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(f"{domain} found {len(filtered)} violation(s):\n{report}")


class TestNoBespokeReuse:
    """the three reuse-map checks over the 3tears package tree."""

    def test_no_bespoke_flush_loops(self) -> None:
        """no class hand-rolls a buffer + periodic-flush loop."""
        scan_roots = find_local_src_roots(_REPO_ROOT)
        violations = find_bespoke_flush_loops(scan_roots, _REPO_ROOT, _SANCTIONED_FLUSH_SITES)
        _assert_clean(violations, _FLUSH_LOOP_CATEGORY)

    def test_no_bespoke_httpx_clients(self) -> None:
        """no class holds a raw long-lived httpx client field."""
        scan_roots = find_local_src_roots(_REPO_ROOT)
        violations = find_bespoke_httpx_clients(scan_roots, _REPO_ROOT, _SANCTIONED_HTTPX_SITES)
        _assert_clean(violations, _HTTPX_CLIENT_CATEGORY)

    def test_no_bespoke_cache_wrappers(self) -> None:
        """no class wraps SQLiteBackend + cache verbs without subclassing BaseCollection."""
        scan_roots = find_local_src_roots(_REPO_ROOT)
        inheritance_roots = discover_src_roots(_REPO_ROOT)
        violations = find_wrapper_classes(
            scan_roots,
            _REPO_ROOT,
            inheritance_roots,
            _BASE_COLLECTION_NAMES,
            _CACHE_METHOD_NAMES,
        )
        _assert_clean(violations, "reuse.bespoke_cache_wrapper")


class TestWalkersFlagPlantedViolations:
    """each walker must fire on a planted violation (self-test of the guard)."""

    def test_flush_loop_walker_flags_planted(self, tmp_path: Path) -> None:
        source = (
            "import asyncio\n\n\n"
            "class BespokeBuffer:\n"
            "    def __init__(self) -> None:\n"
            "        self._pending: list[str] = []\n\n"
            "    def enqueue(self, item: str) -> None:\n"
            "        self._pending.append(item)\n\n"
            "    async def _loop(self) -> None:\n"
            "        while True:\n"
            "            await asyncio.sleep(5)\n"
            "            self._pending.clear()\n"
        )
        planted = tmp_path / "bespoke_buffer.py"
        planted.write_text(source, encoding="utf-8")
        violations = find_bespoke_flush_loops((tmp_path,), tmp_path, frozenset())
        assert [v.symbol for v in violations] == ["BespokeBuffer"]

    def test_httpx_walker_flags_planted(self, tmp_path: Path) -> None:
        source = (
            "import httpx\n\n\n"
            "class BespokeApiClient:\n"
            "    def __init__(self) -> None:\n"
            "        self._client = httpx.AsyncClient(timeout=30)\n"
        )
        planted = tmp_path / "bespoke_client.py"
        planted.write_text(source, encoding="utf-8")
        violations = find_bespoke_httpx_clients((tmp_path,), tmp_path, frozenset())
        assert [v.symbol for v in violations] == ["BespokeApiClient"]

    def test_httpx_walker_flags_a_client_entered_on_an_exit_stack(self, tmp_path: Path) -> None:
        # the shape ComfyUIClient / SwarmUIClient used before 0.71.0's image transport: the client
        # is built inside ``enter_async_context`` and the field is typed ``AsyncClient | None``
        source = (
            "import contextlib\n"
            "import httpx\n\n\n"
            "class EnteredClient:\n"
            "    async def __aenter__(self) -> 'EnteredClient':\n"
            "        stack = contextlib.AsyncExitStack()\n"
            "        self._http = await stack.enter_async_context(httpx.AsyncClient(timeout=5))\n"
            "        return self\n"
        )
        planted = tmp_path / "entered_client.py"
        planted.write_text(source, encoding="utf-8")
        violations = find_bespoke_httpx_clients((tmp_path,), tmp_path, frozenset())
        assert [v.symbol for v in violations] == ["EnteredClient"]

    @pytest.mark.parametrize(
        "annotation",
        [
            "httpx.AsyncClient | None",
            "None | httpx.Client",
            "Optional[httpx.AsyncClient]",
            "typing.Optional[httpx.AsyncClient]",
            "Union[httpx.AsyncClient, None]",
            "'httpx.AsyncClient | None'",
        ],
    )
    def test_httpx_walker_flags_an_optional_client_field(self, tmp_path: Path, annotation: str) -> None:
        source = (
            "import typing\n"
            "from typing import Optional, Union\n"
            "import httpx\n\n\n"
            "class OptionalClient:\n"
            "    def __init__(self) -> None:\n"
            f"        self._http: {annotation} = None\n"
        )
        planted = tmp_path / "optional_client.py"
        planted.write_text(source, encoding="utf-8")
        violations = find_bespoke_httpx_clients((tmp_path,), tmp_path, frozenset())
        assert [v.symbol for v in violations] == ["OptionalClient"]

    def test_httpx_walker_flags_a_class_level_client_annotation(self, tmp_path: Path) -> None:
        source = "import httpx\n\n\nclass DeclaredClient:\n    http: httpx.AsyncClient | None = None\n"
        planted = tmp_path / "declared_client.py"
        planted.write_text(source, encoding="utf-8")
        violations = find_bespoke_httpx_clients((tmp_path,), tmp_path, frozenset())
        assert [v.symbol for v in violations] == ["DeclaredClient"]

    def test_httpx_walker_leaves_one_shot_and_unrelated_fields_alone(self, tmp_path: Path) -> None:
        source = (
            "import httpx\n\n\n"
            "class OneShot:\n"
            "    def __init__(self, transport: httpx.AsyncBaseTransport | None) -> None:\n"
            "        self._transport: httpx.AsyncBaseTransport | None = transport\n"
            "        self._timeout: httpx.Timeout | None = None\n\n"
            "    async def fetch(self) -> bytes:\n"
            "        async with httpx.AsyncClient() as client:\n"
            "            return (await client.get('http://x/')).content\n"
        )
        planted = tmp_path / "one_shot.py"
        planted.write_text(source, encoding="utf-8")
        assert find_bespoke_httpx_clients((tmp_path,), tmp_path, frozenset()) == []

    def test_wrapper_walker_flags_planted(self, tmp_path: Path) -> None:
        source = (
            "from threetears.core.cache.sqlite import SQLiteBackend\n\n\n"
            "class BespokeCounter:\n"
            "    def __init__(self, backend: SQLiteBackend) -> None:\n"
            "        self._backend = backend\n\n"
            "    async def get(self, key: str) -> int:\n"
            "        return 0\n\n"
            "    async def put(self, key: str, value: int) -> None:\n"
            "        return None\n"
        )
        planted = tmp_path / "bespoke_counter.py"
        planted.write_text(source, encoding="utf-8")
        violations = find_wrapper_classes(
            (tmp_path,),
            tmp_path,
            (tmp_path,),
            _BASE_COLLECTION_NAMES,
            _CACHE_METHOD_NAMES,
        )
        assert [v.symbol for v in violations] == ["BespokeCounter"]
