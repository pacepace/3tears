"""3tears documented-schema priming middleware for ``langchain.agents.create_agent``.

The framework-aligned successor to the old ``schema_priming_node`` (a hand-rolled
graph node that ran before the agent node, removed with ``nodes.py``). Re-expressed
onto :meth:`langchain.agents.middleware.AgentMiddleware.awrap_model_call`: before a
model call, when a schema-priming integration is injected on
``config["configurable"]["schema_priming_integration"]`` and the agent is
datasource-backed (>=1 datasource id resolves), the cardinal honesty framing
(:data:`_HONESTY_PREAMBLE`) plus a bounded ``# Documented schema`` block (read BY
PRIMARY KEY from the injected integration's hot-L1 digest collection) are FOLDED into
``request.system_message``. The agent thus starts each cold thread already knowing the
documented tables + columns of its datasource(s) and does NOT re-run the live
``datasource.*.schema`` introspection tool on every turn.

Why the fold and not an extra ``SystemMessage`` in the message list: ``create_agent``
assembles the model prompt as ``[request.system_message, *request.messages]`` with NO
consolidation, and LangChain's Anthropic binding then RAISES on multiple
non-consecutive system messages (``ValueError("Received multiple non-consecutive
system messages")``). Injecting the primed block as a second ``SystemMessage`` (the
pattern an ``abefore_model`` hook would use) triggers exactly that crash against the
real gateway. Folding the block into the single ``request.system_message`` -- the
proven :class:`~threetears.langgraph.middleware_context.ContextMergeMiddleware`
pattern -- keeps one system message and sidesteps the failure.

The honesty rule ships whenever the agent is datasource-backed -- independent of
whether a digest documents tables -- because a read against an
introduced-but-undocumented datasource still returns an ``imperatives`` block and
still needs the rule that makes the model consume it. The documented-schema block
is merely appended when a digest documents tables; a fault reading the digest drops
the block but still ships the honesty rule (a SEPARATE soft-fail).

The block is a BOUNDED digest, not the whole documented schema, so which tables it
keeps is a decision and is made by a rule (:func:`_priority_key`): hazard notes first,
then the most-documented tables, tie-broken by qualified name. It used to be whatever
order the digest yielded, which meant the same arbitrary handful survived every turn no
matter what was asked; a live 35-table datasource logged 32 tables dropped, 145 times in
one run. The budget itself scales with the documented-table count between a floor and a
ceiling rather than sitting at a constant. Deciding WHICH table answers a given question
is NOT this block's job: that binding rides on the datasource read tool's definition,
the channel the model reads at the moment it picks a table.

The documented-schema block (honesty rule excluded) is persisted onto the
``metadata`` state channel via a returned
:class:`~langchain.agents.middleware.types.ExtendedModelResponse` carrying a
:class:`~langgraph.types.Command` state update, so a downstream node can read the
block TEXT after the model node folds the injected prompt away. The ``metadata``
channel carries a :func:`threetears.langgraph.merge_metadata` reducer (declared on
:class:`SchemaPrimingState`) so this injector's key composes with the sibling memory
/ knowledge injectors rather than clobbering them; an update to a channel the state
schema does not declare would be silently dropped.

The integration is read OPAQUELY off ``config["configurable"]`` as a duck-type
(exactly as the offload middleware reads its offloader): no concrete
``SchemaPrimingIntegration`` import, so the framework stays uncoupled from the
host's datasource wiring. Opt-in by construction: with no integration on
``config["configurable"]`` the seam is a pass-through.

Async is the real path -- the integration's ``datasource_ids`` / ``get_digest`` are
``async def``. The synchronous :meth:`~SchemaPrimingMiddleware.wrap_model_call` mirror
cannot drive them, so it degrades to a pass-through (no priming this turn) and warns
when an integration was in fact configured so the misconfiguration is visible.
"""

from __future__ import annotations

from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import AgentState, ExtendedModelResponse
from langchain_core.messages import SystemMessage
from langgraph.config import get_config
from langgraph.types import Command
from threetears.langgraph.state import merge_metadata
from threetears.observe import get_logger

__all__ = ["SchemaPrimingMiddleware", "SchemaPrimingState"]

log = get_logger(__name__)

#: Token allowance the block budget is derived from, per DOCUMENTED table. A fixed
#: budget cannot serve both a 3-table datasource and a 35-table one: on a live 35-table
#: corpus the old constant 1500 dropped 32 tables on EVERY turn. Measured on that
#: corpus, a documented table renders at roughly 430 tokens (description + documented
#: columns), so 500 keeps a normally-documented corpus whole with headroom while still
#: pricing each table rather than handing out an open budget.
_SCHEMA_PRIMING_TOKENS_PER_TABLE = 500

#: Floor on the derived budget: the previous fixed budget. A datasource documenting one
#: or two tables still gets a block worth reading, and the derived budget can never
#: shrink below what the block was already allowed to spend.
_SCHEMA_PRIMING_TOKEN_FLOOR = 1500

#: Ceiling on the derived budget. This is where the block stops being priming and starts
#: being a dump: the 35-table corpus above renders whole at roughly 15k tokens, which is
#: an accepted per-turn spend for a corpus that size, and past it the marginal table buys
#: less than the context it costs. It is also no longer the routing channel -- WHICH
#: table to query is decided from the read tool's definition, read at the moment the
#: model picks a table, so this block only has to be a bounded digest.
_SCHEMA_PRIMING_TOKEN_CEILING = 16000

#: Chars-per-token heuristic; keeps the budget accounting tokenizer-free + reproducible.
_CHARS_PER_TOKEN = 4

#: Tokens reserved for the one-line truncation footer so the rendered BLOCK total
#: (preamble + tables + footer) stays within budget, not just the table body. The
#: footer is ~1 short sentence; 30 tokens is a safe ceiling.
_FOOTER_TOKEN_RESERVE = 30

#: The cardinal honesty framing. Homed here (the framework middleware) rather than
#: each agent's system prompt so it ships ONCE to every datasource-backed agent. It is
#: the prompt-side POINTER to the mechanism: the load-bearing instruction lives in the
#: data -- the ``imperatives`` block the hub datasource tools render into each result,
#: re-read every turn. This rule only makes the model CONSUME that block; it never
#: carries the gap content itself. It ships whenever a datasource resolves, NOT only
#: when a documented digest resolves, because ``imperatives`` are emitted by ANY
#: datasource read (including reads against an introduced-but-undocumented datasource).
_HONESTY_PREAMBLE = (
    "# Ground every answer in tool results\n"
    "Every fact in your answer comes from a tool call you made this turn. "
    "If a tool did not return a value, say so. Never fill it in from training or memory.\n"
    "Every number you report comes from a tool too. If SQL can compute it, compute it in "
    "SQL. That includes any total, count, average, percent, or difference. If SQL cannot "
    "compute it, use the calculator tool. Never do arithmetic in your head.\n"
    "A tool result may carry an `imperatives` block. Follow each instruction in it. "
    "Do not report the named field from memory.\n"
    "Do not add a caveat the data does not carry. A date does not tell you the data is "
    "stale or fresh. A timestamp does not tell you it is final or complete. "
    "Call the data fresh, final, or complete only when a field in the result says so.\n"
    "# Rank only the rows that qualify\n"
    "When a question asks for a ranked set in one direction (the largest gain, the "
    "biggest decline, the items that moved toward something), keep only the rows that "
    "move that way, then rank them and take the top N. Never pad the list with a row "
    "that moved the other way to reach the requested count. If fewer than N rows "
    "qualify, return the smaller set and say how many qualified. Before you answer, "
    "read the returned rows and confirm each one matches the direction asked.\n\n"
)

#: Standing guidance prepended to the documented-schema block. Homed here rather than
#: each agent's system prompt so every datasource-backed agent gets it. It tells the
#: model to TRUST the digest and reach for the live schema tool ONLY for what the
#: digest does not cover -- the behavior that turns the expensive introspection tool
#: from a per-cold-thread default into a rare fallback.
_BLOCK_PREAMBLE = (
    "# Documented schema\n"
    "The documented schema for your datasources is below. "
    "It gives each table's purpose and its documented columns. "
    "Use it when you write SQL. "
    "Call the schema-inspection tool only for a table or column that is not listed here. "
    "Do not look up what is already below.\n\n"
)


class SchemaPrimingState(AgentState[Any]):
    """Agent state extended with the ``documented_schema_block`` metadata channel.

    Extends the base :class:`~langchain.agents.middleware.types.AgentState` with a
    ``metadata`` dict channel so :class:`SchemaPrimingMiddleware` can stash the
    rendered documented-schema block for a downstream consumer. The channel carries
    the :func:`threetears.langgraph.merge_metadata` reducer so this injector's key
    composes with the sibling memory / knowledge injectors (each writes disjoint
    top-level keys) rather than the last writer clobbering the others; LangGraph
    silently drops an update to a channel the state schema does not declare, so the
    channel must be declared here for the stash to persist.

    :ivar metadata: per-run metadata bag; the middleware writes the rendered
        documented-schema block under ``documented_schema_block``.
    """

    metadata: NotRequired[Annotated[dict[str, Any], merge_metadata]]


def _configurable() -> dict[str, Any]:
    """Return the active run's ``configurable`` dict, or empty when unavailable.

    Reads the :class:`~langchain_core.runnables.RunnableConfig` for the active run
    via :func:`langgraph.config.get_config` -- the canonical way a ``wrap_model_call``
    hook reaches ``configurable`` (the ``Runtime`` on a ``ModelRequest`` does not carry
    ``config`` for the model-call seam). Outside a runnable context ``get_config``
    raises ``RuntimeError``; with no config there is no injected integration, so this
    soft-fails to an empty dict and the seam becomes a pass-through rather than
    crashing.

    :return: the run's ``configurable`` mapping, or an empty dict.
    :rtype: dict[str, Any]
    """
    try:
        config = get_config()
    except RuntimeError:
        # SDS-04: no runnable config means no injected integration, so this whole
        # middleware becomes a pass-through and the turn silently loses schema priming.
        log.warning(
            "no runnable config available: schema priming is skipped for this turn",
        )
        return {}
    return config.get("configurable") or {}


def _fold_into_system(
    system_message: SystemMessage | None,
    block: str,
) -> SystemMessage:
    """Return a single ``SystemMessage`` carrying the base prompt plus the block.

    Preserves the base system prompt and appends the primed block after a blank line.
    When the base message has structured (list) content -- e.g. a prior middleware
    already annotated it -- the block is appended as an extra text part so existing
    parts (and any ``cache_control`` on them) are left intact. When there is no base
    system message, the block becomes the whole system message. Mirrors
    :func:`threetears.langgraph.middleware_context._fold_context_into_system`.

    :param system_message: the request's current system message, or ``None``.
    :ptype system_message: SystemMessage | None
    :param block: the non-empty primed block to fold in.
    :ptype block: str
    :return: the merged system message.
    :rtype: SystemMessage
    """
    if system_message is None:
        return SystemMessage(content=block)
    base_content = system_message.content
    if isinstance(base_content, list):
        merged_parts = [*base_content, {"type": "text", "text": block}]
        return system_message.model_copy(update={"content": merged_parts})
    base_text = base_content if isinstance(base_content, str) else str(base_content)
    return system_message.model_copy(update={"content": f"{base_text}\n\n{block}"})


def _estimate_tokens(text: str) -> int:
    """Estimate token count via the conventional chars-per-token heuristic.

    Tokenizer-free + reproducible so the budget is a consistent soft cost guard.

    :param text: text to estimate.
    :ptype text: str
    :return: estimated token count (>= 1 for non-empty text).
    :rtype: int
    """
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _render_table(table: dict[str, Any]) -> str:
    """Render one documented table: ``schema.table`` + description + columns.

    :param table: one projection entry (``{schema, table, description, columns}``,
        optionally ``unloaded_columns``).
    :ptype table: dict[str, Any]
    :return: rendered table text.
    :rtype: str
    """
    schema = table.get("schema", "")
    name = table.get("table", "")
    qualified = f"{schema}.{name}" if schema else name
    lines = [f"### {qualified}"]
    description = table.get("description")
    if description:
        lines.append(str(description))
    for column in table.get("columns", []) or []:
        col_name = column.get("name", "")
        col_type = column.get("type")
        col_desc = column.get("description")
        suffix = f" ({col_type})" if col_type else ""
        detail = f" -- {col_desc}" if col_desc else ""
        lines.append(f"- {col_name}{suffix}{detail}")
    # datasource-honesty coverage overlay: columns that are all-zero across the whole
    # table (unloaded). a 0 read from one is missing data, not a measured zero -- the
    # agent must be told BEFORE it queries (a bare SUM hides the all-zero pattern from
    # result-time gaps).
    unloaded = table.get("unloaded_columns") or []
    if unloaded:
        lines.append(
            "- NOT LOADED. These columns are all zero across the whole table. "
            "A 0 here means the data was never loaded. It is not a measured zero. "
            "Report these columns as unavailable. Never report them as 0. "
            "Columns: " + ", ".join(unloaded)
        )
    return "\n".join(lines)


def _scaled_budget(table_count: int) -> int:
    """Derive the block token budget from the documented-table count.

    Linear in the documented tables, clamped to
    ``[_SCHEMA_PRIMING_TOKEN_FLOOR, _SCHEMA_PRIMING_TOKEN_CEILING]``. A constant budget
    is wrong at both ends: it overspends on a datasource documenting two tables and
    silently drops most of a wide one (the 32-of-35 truncation this replaces). The
    clamp keeps the spend bounded no matter how wide the corpus grows.

    :param table_count: number of documented tables across every digest.
    :ptype table_count: int
    :return: token budget for the rendered block.
    :rtype: int
    """
    return min(
        max(table_count * _SCHEMA_PRIMING_TOKENS_PER_TABLE, _SCHEMA_PRIMING_TOKEN_FLOOR),
        _SCHEMA_PRIMING_TOKEN_CEILING,
    )


def _documentation_weight(table: dict[str, Any]) -> int:
    """Estimate the prose in one table that live introspection could NOT recover.

    Names and types come back free from a schema-inspection call; the description and
    the per-column descriptions do not exist anywhere but the digest. So the prose is
    exactly what is LOST when a table is dropped from the block, which makes it the
    honest measure of what priming buys per table.

    :param table: one projection entry.
    :ptype table: dict[str, Any]
    :return: estimated tokens of documentation prose on the table.
    :rtype: int
    """
    weight = _estimate_tokens(str(table.get("description") or ""))
    for column in table.get("columns", []) or []:
        weight += _estimate_tokens(str(column.get("description") or ""))
    return weight


def _priority_key(table: dict[str, Any]) -> tuple[int, int, str]:
    """Rank one table for block inclusion; lower sorts first (survives truncation).

    THE selection rule. It exists because the previous code took the digest's incidental
    order and stopped at the budget, so the same arbitrary first tables survived every
    turn regardless of the question and the rest were invisible forever. The order here
    is by what the model cannot get any other way, most first:

    1. Hazard notes (``unloaded_columns``, or ``caveats`` should the projection ever
       carry them). An all-zero unloaded column reads as a measured 0, and nothing in
       the live catalog contradicts it; a table whose documentation is the ONLY thing
       standing between the agent and a confidently wrong number is never the one to
       drop.
    2. Documentation weight (:func:`_documentation_weight`). Tables carrying real prose
       teach the most per token and are the least inferable; a bare row that documents
       little is the cheapest thing to lose to the schema-inspection tool.
    3. Qualified name, ascending. A total order, so the block is byte-identical across
       runs and independent of the order the digest happened to yield.

    Deliberately NOT query-aware: routing (which table answers THIS question) is decided
    from the read tool's definition, the channel the model reads at the moment it picks
    a table. This block is background, so it is ranked once and rendered the same way
    every turn, which also keeps it prefix-stable for prompt caching.

    :param table: one projection entry.
    :ptype table: dict[str, Any]
    :return: sort key (hazard, weight, qualified name) with the first two negated so
        that "more" sorts first.
    :rtype: tuple[int, int, str]
    """
    hazard = 1 if (table.get("unloaded_columns") or table.get("caveats")) else 0
    schema = table.get("schema", "")
    name = table.get("table", "")
    qualified = f"{schema}.{name}" if schema else str(name)
    return (-hazard, -_documentation_weight(table), qualified)


def _render_schema_block(digests: list[Any], *, budget: int | None = None) -> str:
    """Render the bounded ``# Documented schema`` block from the digests.

    Flattens the documented tables across every datasource's digest, orders them by
    :func:`_priority_key`, renders in that order and trims to the token budget -- always
    keeping at least one table so a single large table still primes. Truncation drops
    the lowest-priority TAIL (never a hole in the middle), so the dropped set is
    statable: it is the least-documented end of the corpus. When the tail is dropped a
    one-line footer names the count and the rule so truncation is never silent (logging
    contract). Returns an empty string when no documented table exists.

    :param digests: digest entities, each exposing a ``tables`` projection.
    :ptype digests: list[Any]
    :param budget: documented-schema token budget; ``None`` derives it from the
        documented-table count via :func:`_scaled_budget`.
    :ptype budget: int | None
    :return: rendered block (preamble + tables [+ footer]), or empty string.
    :rtype: str
    """
    all_tables: list[dict[str, Any]] = []
    for entity in digests:
        tables = getattr(entity, "tables", None) or []
        all_tables.extend(tables)
    if not all_tables:
        # SDS-04: this is what a missing digest materialization looks like from the
        # prompt side -- the agent ships the honesty preamble and looks normally
        # primed while carrying no schema at all.
        log.warning(
            "documented-schema priming has no tables to render (digests=%d): the "
            "agent is primed with NO schema; has the digest been materialized?",
            len(digests),
        )
        return ""

    effective_budget = _scaled_budget(len(all_tables)) if budget is None else budget
    # rank BEFORE spending, so what survives is decided by the rule and not by the order
    # the digest yielded. sorted() is stable, but the key is a total order, so the result
    # does not depend on that.
    ranked = sorted(all_tables, key=_priority_key)

    rendered: list[str] = []
    # seed the budget with the fixed block overhead so the BLOCK total (preamble +
    # tables [+ footer]) stays within budget, not just the table body. reserve a
    # footer allowance too, since dropping the tail appends a one-line footer.
    spent = _estimate_tokens(_BLOCK_PREAMBLE) + _FOOTER_TOKEN_RESERVE
    for table in ranked:
        text = _render_table(table)
        cost = _estimate_tokens(text)
        # keep at least one table even if it alone exceeds budget.
        if rendered and spent + cost > effective_budget:
            break
        rendered.append(text)
        spent += cost

    dropped = len(ranked) - len(rendered)
    body = "\n\n".join(rendered)
    if dropped > 0:
        # SDS-04: the footer tells the MODEL; this line tells the operator. a wide
        # datasource primes a fraction of its documented tables and the agent then
        # writes SQL against the un-primed remainder. name the RULE too: a table missing
        # from the block is then readable as a stated decision rather than a lookup that
        # quietly failed.
        log.warning(
            "documented-schema priming truncated: %d of %d table(s) dropped "
            "(budget=%d tokens); the block keeps tables with hazard notes first, then "
            "the most-documented, tie-broken by name, and drops that ranking's tail; "
            "the agent is primed with a subset of its schema",
            dropped,
            len(ranked),
            effective_budget,
        )
        body += (
            f"\n\n_{dropped} more documented table(s) not shown here. This block keeps "
            "the tables carrying the most documentation; the rest are documented too. "
            "Use the datasource schema-inspection tool to inspect them._"
        )
    return _BLOCK_PREAMBLE + body


async def _read_schema_block(integration: Any, datasource_ids: list[Any], budget: int | None) -> str:
    """Read each datasource's digest and render the documented-schema block.

    A SEPARATE soft-fail from the honesty rule: a fault reading any digest yields an
    empty block (the honesty rule still ships) rather than crashing the turn or
    suppressing the honesty framing.

    :param integration: the injected schema-priming integration (opaque duck-type
        exposing ``async get_digest(datasource_id)``).
    :ptype integration: Any
    :param datasource_ids: the agent's resolved datasource ids.
    :ptype datasource_ids: list[Any]
    :param budget: documented-schema token budget; ``None`` derives it from the
        documented-table count.
    :ptype budget: int | None
    :return: rendered ``# Documented schema`` block, or empty string when no digest
        documents tables / a digest read faults.
    :rtype: str
    """
    block = ""
    try:
        digests: list[Any] = []
        for datasource_id in datasource_ids:
            entity = await integration.get_digest(datasource_id)
            if entity is not None:
                digests.append(entity)
        block = _render_schema_block(digests, budget=budget)
    except Exception as exc:  # prawduct:allow prawduct/broad-except -- digest read is best-effort; a fault drops the block but the honesty rule still ships
        log.warning(
            "schema priming digest read failed (soft-fail; honesty rule still shipped): %s",
            type(exc).__name__,
        )
    return block


class SchemaPrimingMiddleware(AgentMiddleware[SchemaPrimingState, Any, Any]):
    """Fold the honesty framing + documented-schema digest into the system prompt.

    The ``create_agent`` successor to the ``schema_priming_node``. On every model
    call, when a schema-priming integration is injected on
    ``config["configurable"]["schema_priming_integration"]`` and the agent is
    datasource-backed, the honesty rule plus the bounded documented-schema block are
    FOLDED into ``request.system_message`` (yielding exactly one system message), and
    the block is persisted onto ``metadata["documented_schema_block"]`` via a returned
    ``Command`` state update. Opt-in: no integration (or a non-datasource agent) on
    ``config["configurable"]`` -> the seam is a pass-through.
    """

    name = "SchemaPrimingMiddleware"

    state_schema: type[SchemaPrimingState] = SchemaPrimingState

    def __init__(self, *, token_budget: int | None = None) -> None:
        """Configure the documented-schema block token budget.

        Defaults to ``None``: the budget is DERIVED from how many tables the digests
        document (:func:`_scaled_budget`), because one constant cannot fit both a
        two-table datasource and a thirty-five-table one. An explicit value pins it.

        :param token_budget: fixed token budget for the injected documented-schema
            block, or ``None`` to scale it with the documented-table count; tables past
            the budget are dropped by :func:`_priority_key` order with a one-line footer.
        :ptype token_budget: int | None
        :return: nothing.
        :rtype: None
        """
        super().__init__()
        self.token_budget = token_budget

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        """Prime the honesty framing + documented-schema digest, fold it into system.

        Reads the schema-priming integration off ``config["configurable"]`` and
        resolves the agent's datasource ids (memoized on the integration). When the
        agent is datasource-backed (>=1 id) it folds a block carrying the honesty rule
        -- which ships REGARDLESS of whether a digest resolves -- plus the bounded
        documented-schema block when a digest documents tables into
        ``request.system_message``, runs the call, and returns an
        :class:`ExtendedModelResponse` whose ``Command`` persists the block (honesty
        rule excluded) onto ``metadata``. Passes the call through un-primed for a
        non-datasource agent or a missing integration.

        :param request: the model request (carries ``system_message`` + ``messages``
            + ``state``).
        :ptype request: ModelRequest
        :param handler: the next handler in the middleware chain.
        :ptype handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
        :return: the model response, wrapped with a metadata ``Command`` when the
            block was primed.
        :rtype: ModelResponse[Any] | ExtendedModelResponse[Any]
        """
        integration = _configurable().get("schema_priming_integration")
        if integration is None:
            passthrough: ModelResponse[Any] = await handler(request)
            return passthrough
        # datasource_ids() soft-fails internally to [] (never raises): a
        # non-datasource agent, or one whose datasource has not resolved yet, yields
        # [] and no injection.
        datasource_ids = await integration.datasource_ids()
        if not datasource_ids:
            passthrough = await handler(request)
            return passthrough
        schema_block = await _read_schema_block(integration, datasource_ids, self.token_budget)
        # the honesty rule ships whenever a datasource resolves, so the folded block is
        # always non-empty even when no digest documented tables (schema_block == "").
        block = _HONESTY_PREAMBLE + schema_block
        merged = _fold_into_system(request.system_message, block)
        response: ModelResponse[Any] = await handler(request.override(system_message=merged))
        # stash ONLY the documented-schema block (not the honesty rule) so a downstream
        # node can read it after the model node folds the injected prompt away. empty
        # string when no digest documented tables.
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"metadata": {"documented_schema_block": schema_block}}),
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse[Any]:
        """Sync mirror: cannot drive the async integration, so pass the call through.

        The integration's ``datasource_ids`` / ``get_digest`` are ``async def``; a
        synchronous ``wrap_model_call`` path cannot await them. Rather than fail the
        run, priming is skipped this turn (the request passes through un-primed) and a
        warning is logged when an integration was in fact configured so the
        misconfiguration is visible.

        :param request: the model request (passed through unchanged).
        :ptype request: ModelRequest
        :param handler: the next handler in the middleware chain.
        :ptype handler: Callable[[ModelRequest], ModelResponse]
        :return: the model response from the un-primed request.
        :rtype: ModelResponse[Any]
        """
        if _configurable().get("schema_priming_integration") is not None:
            log.warning(
                "schema priming skipped: an integration is configured but the agent ran "
                "on the synchronous model path; the integration is async-only, so no "
                "documented-schema priming is injected this turn",
            )
        result: ModelResponse[Any] = handler(request)
        return result
