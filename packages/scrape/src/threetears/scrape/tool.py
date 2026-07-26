"""ScrapeTool -- exposes 3tears-scrape as a callable ``TearsTool`` (MCP tool).

**Design (2026-07-14, MCP exposure):** an ad-hoc, one-off scrape -- "fetch
this URL, extract these fields" -- with no pre-registered ``ScrapeTarget``,
matching the exact use case ``StaticTargetSource``/a single inline
``ScrapeTarget`` construction was already designed for -- a target does not
have to be persisted to be a valid target. Runs through the *same*
unmodified AI eval loop
(``eval_loop.run_eval_loop``/``run_eval_loop_multi_row``) every configured
target already uses -- no separate "MCP extraction path" to keep in sync.

Repeated calls against the same URL + field schema benefit from the eval
loop's own self-healing recipe reuse (zero further LLM calls once a recipe
wins) via a *deterministic* ``target_id`` derived from ``(url,
field_schema)`` when the caller doesn't supply one explicitly -- an LLM
caller shouldn't have to invent and remember target IDs itself for this to
work.

All real dependencies (collections, drivers, API key) are constructor-
injected, never resolved internally -- no env-var reads, no application
config or store lookups -- mirroring every other component in this package
(e.g. ``NodriverSidecarDriver``'s ``base_url``). A consuming application
registers this tool through its own thin wrapper, which is where resolving
real config/collections/drivers belongs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.observe import get_logger

from pydantic import SecretStr

from .circuit import FetchDecision, TargetCircuit
from threetears.core.egress import EgressDriver

from .robots import RobotsDecision, RobotsGate
from .session_state import usable_session_state
from .collections import ScrapeExtractionCollection, ScrapeRecipeCollection, decode_field_schema, decode_nav_steps
from .driver import NavStep, RenderedPage, ScrapeDriver
from .eval_loop import StrategyType, run_eval_loop, run_eval_loop_multi_row
from .extraction import FieldSchema
from .health import ScrapeTargetHealthCollection, clear_robots_block, record_robots_block

__all__ = ["ScrapeTool"]

log = get_logger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30.0

#: Distinguishes "the caller said nothing" from "the caller said no robots". A plain ``None``
#: default collapses those, and collapsing them is what made a documented on-by-default
#: setting off in every deployment.
_ROBOTS_DEFAULT: Any = object()


def _derive_target_id(url: str, field_schema: dict[str, Any]) -> str:
    """Deterministic recipe-reuse key for an ad-hoc scrape with no caller-supplied target_id.

    Hashes *url* plus the field schema's own field names (sorted -- dict
    iteration order isn't part of the schema's identity) so the exact same
    ``(url, field_schema)`` call always resolves to the same key, letting
    repeated calls reuse a winning recipe instead of re-running candidate
    generation every time. Field *types* aren't part of the hash input --
    two calls differing only in a field's declared type are rare enough
    (and a schema drift within the same fields is itself informative, not
    something to route around) that keying on names alone is the simpler,
    still-correct choice.
    """
    digest = hashlib.sha256(f"{url}|{','.join(sorted(field_schema))}".encode()).hexdigest()
    return f"adhoc_{digest[:16]}"


class ScrapeTool(TearsTool):
    """Ad-hoc "fetch this URL, extract these fields" tool, backed by 3tears-scrape.

    All state (collections, drivers, API key) is injected at construction --
    no internal env-var or application-config resolution, per this module's
    own docstring.
    """

    def __init__(
        self,
        *,
        recipe_collection: ScrapeRecipeCollection,
        extraction_collection: ScrapeExtractionCollection,
        drivers: dict[str, ScrapeDriver],
        api_key: str,
        health_collection: ScrapeTargetHealthCollection | None = None,
        circuit: TargetCircuit | None = None,
        session_state_key: SecretStr | None = None,
        robots: RobotsGate | None = _ROBOTS_DEFAULT,
        egress: EgressDriver | None = None,
        default_timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        :param recipe_collection: shared recipe store for the eval loop's self-healing reuse
        :ptype recipe_collection: ScrapeRecipeCollection
        :param extraction_collection: where each call's extraction result is persisted
        :ptype extraction_collection: ScrapeExtractionCollection
        :param health_collection: per-target fetch health. Supplying it opts this tool into
            telling a bot wall apart from a site redesign, so a blocked target keeps its
            recipe instead of burning it; omitted, every failure counts the same way it
            always has
        :ptype health_collection: ScrapeTargetHealthCollection | None
        :param circuit: per-target fetch circuit. Supplying it opts this tool into backing
            off a target that keeps coming back walled, so its fetch rate -- and with it the
            classification rate that only a fetch can incur -- decays instead of holding
            steady forever; omitted, every call fetches
        :ptype circuit: TargetCircuit | None
        :param session_state_key: operator master key that opens a stored human solve. Without
            it a sealed state on the health row is left sealed and the target is fetched as if
            no human had ever cleared it -- which is the safe direction, since the alternative
            would be a deployment silently not knowing whether its credentials were readable
        :ptype session_state_key: SecretStr | None
        :param robots: the ``robots.txt`` gate. **Omitted, a default gate is used and both
            behaviours are ON** -- a crawl delay is waited and a disallowed path is escalated
            to a human rather than fetched. Pass a configured :class:`RobotsGate` to change the
            policy or to share the scrape's egress and tracing with the robots request; pass
            ``None`` explicitly to consult no robots.txt at all, which is the pre-existing
            behaviour and now has to be asked for rather than being what everyone got
        :ptype robots: RobotsGate | None
        :param egress: which exit this tool's own requests leave by. Passed to the default
            robots gate so the robots.txt read shares the scrape's route rather than
            disclosing the container's address in front of it. Drivers take their own egress
            separately, because a driver may be shared between tools
        :ptype egress: EgressDriver | None
        :param drivers: ``driver_backend`` name -> ``ScrapeDriver`` instance
            (e.g. ``{"nodriver": ..., "camoufox": ..., "document": ...}``)
        :ptype drivers: dict[str, ScrapeDriver]
        :param api_key: OpenRouter API key for the eval loop's candidate-generation and judge calls
        :ptype api_key: str
        :param default_timeout: seconds to wait for a render when the caller doesn't specify one
        :ptype default_timeout: float
        """
        self._recipe_collection = recipe_collection
        self._extraction_collection = extraction_collection
        self._health_collection = health_collection
        self._circuit = circuit
        self._session_state_key = session_state_key
        # `_ROBOTS_DEFAULT` is a sentinel, not None: passing `robots=None` explicitly means
        # "consult nothing", and omitting the argument means "the default, which is on". Those
        # are different intentions and a plain `None` default cannot express both -- which is
        # how a documented on-by-default became off everywhere.
        self._egress = egress
        # The default gate inherits this tool's exit. A robots request on the container's own
        # route, in front of a proxied scrape, discloses the address the proxy exists to hide.
        self._robots = RobotsGate(egress=egress) if robots is _ROBOTS_DEFAULT else robots
        self._drivers = drivers
        self._api_key = api_key
        self._default_timeout = default_timeout

    def mcp_name(self) -> str:
        """Return the namespaced tool name.

        A generic ``3tears.scrape`` identity, since this class is a
        reusable 3tears component and not specific to any one application --
        a consuming wrapper that registers it is free to override this with
        its own namespaced name.
        """
        return "3tears.scrape"

    def mcp_version(self) -> str:
        """Return the tool version string."""
        return "1.0.0"

    def mcp_schema(self) -> MCPToolDefinition:
        """Return the MCP tool definition.

        **Deliberately narrower than the full backend/strategy set.**
        ``driver_backend`` offers only ``nodriver``/``camoufox``/``document``
        of the eight this package ships, and ``strategy_type`` only
        ``css``/``regex`` of ``eval_loop.StrategyType``'s four. The omissions
        are not oversights: each excluded option needs per-target
        configuration this flat, single-call input schema has nowhere to
        carry, and offering it would advertise a backend that fails at
        runtime. ``api`` and ``multi_document``'s JSON discovery mode need
        ``api_results_path``/``api_fragment_field``; ``multi_document``'s HTML
        mode needs ``link_selector``; ``multi_document``,
        ``network_capture``, and ``nodriver_download`` each need an inner
        driver injected at construction rather than named in a call;
        ``listing_detail`` needs a base URL plus a pacing policy for its
        per-row detail fetches. ``per_document`` and ``multi_row_vision``
        are only meaningful against pages those excluded drivers produce.
        The three backends and two strategies that remain are exactly the
        ones fully specified by a URL and a field schema -- which is the
        whole contract of an ad-hoc, no-pre-configuration call. A target
        needing more than that is a real ``ScrapeTarget``, seeded through
        ``target_source.bootstrap_targets()``, not an MCP one-off.
        """
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description=(
                "Scrape a URL (a web page, or a document -- PDF/DOCX/XLSX/TXT/Markdown/LaTeX -- "
                "when driver_backend='document') and extract structured fields via the real, "
                "AI-driven eval loop: LLM-proposed extraction strategy, structural validation, "
                "an LLM judge, and self-healing recipe reuse on repeated calls against the same "
                "URL and field schema -- the same system every pre-configured scrape target uses, "
                "not a separate one-shot extraction path."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "the page or document to scrape"},
                    "field_schema": {
                        "type": "object",
                        "description": (
                            "field name -> type name ('str' | 'int' | 'float' | 'bool') to extract, "
                            'e.g. {"employer": "str", "affected_count": "int"}'
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "multi_row": {
                        "type": "boolean",
                        "description": "true if the page/document lists many records (a table/listing) rather than one",
                        "default": False,
                    },
                    "driver_backend": {
                        "type": "string",
                        "enum": ["nodriver", "camoufox", "document"],
                        "description": "which ScrapeDriver renders the URL; 'document' for a PDF/DOCX/XLSX/TXT/Markdown/LaTeX file",
                        "default": "nodriver",
                    },
                    "wait_for": {
                        "type": "string",
                        "description": "CSS selector to wait for before considering the page rendered (ignored by driver_backend='document')",
                    },
                    "nav_steps": {
                        "type": "array",
                        "description": (
                            "ordered browser actions driving the page to its real content before "
                            "extraction (a search form, a second page in a listing) -- each item is "
                            '{"action": "click"|"fill"|"wait_for"|"wait_ms"|"scroll_into_view"|"scroll_page"|"evaluate", "selector": str, '
                            '"value": str, "ms": int} (fields other than action are optional per '
                            "action type). Ignored by driver_backend='document'."
                        ),
                        "items": {"type": "object"},
                    },
                    "target_id": {
                        "type": "string",
                        "description": (
                            "stable key for recipe reuse across repeated calls; derived from "
                            "(url, field_schema) when omitted, so repeated identical calls reuse "
                            "a winning recipe without the caller needing to invent one"
                        ),
                    },
                    "strategy_type": {
                        "type": "string",
                        "enum": ["css", "regex"],
                        "description": (
                            "'css' (default) for an HTML table -- CSS-selector candidates. 'regex' for "
                            "a page whose real content is prose/list text with no <table> structure at "
                            "all -- regex-pattern candidates (one (?P<field_name>...) named group per "
                            "field) matched against the page's plain text instead."
                        ),
                        "default": "css",
                    },
                },
                "required": ["url", "field_schema"],
            },
            timeout_seconds=self._default_timeout + 60.0,  # render timeout + eval loop's own LLM-call budget
        )

    async def _read_solved_state(self, target_id: str) -> dict[str, Any] | None:
        """A human's stored solve for *target_id*, or ``None`` when there is none to use.

        Never raises into a fetch. Every reason to decline -- no key configured, nothing
        stored, a passed expiry, a token that will not open -- means the same thing: fetch as
        if no human had cleared it. That is the safe direction, because the failure it avoids
        is not "no cookie" but a fetch that never happens over a credential problem.
        """
        if self._health_collection is None or self._session_state_key is None:
            return None
        try:
            row = await self._health_collection.get(target_id)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a health-store outage must degrade to fetching without a solve, exactly as the circuit degrades to fetching; losing the reuse is a cost, losing the fetch is an outage. Logged with its traceback below
            log.exception(
                "scrape tool: could not read the health row for target %s; fetching without a stored solve",
                target_id,
                extra={"extra_data": {"target_id": target_id}},
            )
            return None
        return usable_session_state(row, self._session_state_key)

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Fetch *url*, then extract *field_schema* via the real AI eval loop.

        :param kwargs: tool input; requires ``url``/``field_schema``, optional
            ``multi_row``/``driver_backend``/``wait_for``/``target_id``
        :ptype kwargs: Any
        :return: JSON-encoded ``{"target_id", "validation_status", "records"}``,
            or an error result
        :rtype: ToolResult
        """
        # Single-exit structure (an `error` sentinel gates each stage) rather than
        # early-returning per validation failure -- this function has several
        # independent, sequential things that can go wrong (bad input, an
        # unsupported backend, a transport failure), and this repo's own
        # single-return convention wants one exit point, not one per failure mode.
        error: str | None = None

        url = kwargs.get("url") or ""
        if not url:
            error = "url is required"

        schema: FieldSchema = {}
        raw_schema = kwargs.get("field_schema") or {}
        if error is None:
            try:
                schema = decode_field_schema(raw_schema)
            except ValueError as exc:
                error = str(exc)
            if error is None and not schema:
                error = "field_schema must declare at least one field"

        driver_backend = kwargs.get("driver_backend") or "nodriver"
        driver: ScrapeDriver | None = None
        if error is None:
            driver = self._drivers.get(driver_backend)
            if driver is None:
                error = f"unsupported driver_backend {driver_backend!r}; available: {sorted(self._drivers)}"

        nav_steps: list[NavStep] | None = None
        raw_nav_steps = kwargs.get("nav_steps")
        if error is None and raw_nav_steps:
            try:
                nav_steps = decode_nav_steps(raw_nav_steps)
            except TypeError as exc:
                error = f"invalid nav_steps: {exc}"

        strategy_type: StrategyType = kwargs.get("strategy_type") or "css"
        if error is None and strategy_type not in ("css", "regex"):
            error = f"unsupported strategy_type {strategy_type!r}; must be 'css' or 'regex'"

        multi_row = bool(kwargs.get("multi_row", False))
        wait_for = kwargs.get("wait_for") or None
        target_id = kwargs.get("target_id") or _derive_target_id(url, raw_schema)

        # Asked before the driver is touched, because a suppressed fetch is the entire point:
        # a target inside its backoff window must reach neither the candidate generator nor
        # the page classifier, and both of those live downstream of a page being fetched.
        # Consulted BEFORE the circuit, and both gates must be satisfied. They are different
        # kinds: a crawl delay is a FLOOR on politeness that applies to a target working
        # perfectly, and the circuit's window is a CEILING on cost that applies to one that is
        # not. Neither may be used to weaken the other -- in particular a circuit probe is not
        # exempt from the delay, or the politeness contract breaks exactly when a target is
        # already unhappy with us.
        robots_decision: RobotsDecision | None = None
        if error is None and self._robots is not None:
            robots_decision = await self._robots.check(url)
            if not robots_decision.allowed:
                # A disallowed path is not fetched unattended and not silently skipped: it is
                # reported as needing a person, through the same shape a bot wall takes, so a
                # queue can pick it up. The exclusion protocol governs automated agents; an
                # operator who opens a session and works it themselves is not one.
                log.info(
                    "scrape tool: %s is disallowed by robots.txt; escalating rather than fetching",
                    url,
                    extra={"extra_data": {"target_id": target_id, "url": url}},
                )
                if self._health_collection is not None:
                    # Recorded on the health row, not just returned. A decision that lives
                    # only in this ToolResult reaches no queue: `list_walled` answers from the
                    # row, so a target the scraper itself decided needs a human would never be
                    # findable by the platform whose job it is to send one.
                    await record_robots_block(
                        self._health_collection, target_id=target_id, reason=robots_decision.reason
                    )
                return ToolResult(
                    success=False,
                    error=f"needs a human: {robots_decision.reason}",
                    content=json.dumps({"target_id": target_id, "validation_status": "needs_human", "records": []}),
                    metadata={
                        "target_id": target_id,
                        "validation_status": "needs_human",
                        "record_count": 0,
                        "source_url": url,
                        "needs_human": True,
                        "reason": "robots_disallow",
                    },
                )
        if robots_decision is not None and robots_decision.allowed and self._health_collection is not None:
            # The file no longer disallows us, so this target stops needing a person for that
            # reason. A site that lifts its rule would otherwise sit in the queue forever, and
            # nobody working that queue would know why it was still there.
            await clear_robots_block(self._health_collection, target_id=target_id)

        decision: FetchDecision | None = None
        if error is None and self._circuit is not None:
            decision = await self._circuit.check(target_id)

        # The crawl delay is waited only once the circuit has ADMITTED the fetch. Waiting
        # before it would block a caller for up to the delay ceiling only to be told the fetch
        # was suppressed -- paying a politeness cost for a request that never happens. The
        # floor-vs-ceiling rule is satisfied either way: both gates are still honoured, and a
        # circuit probe still waits its delay.
        if (
            robots_decision is not None
            and robots_decision.wait_seconds > 0
            and (decision is None or decision.permitted)
        ):
            log.info(
                "scrape tool: waiting %.0fs before fetching %s, as its robots.txt asks",
                robots_decision.wait_seconds,
                url,
                extra={"extra_data": {"target_id": target_id}},
            )
            await asyncio.sleep(robots_decision.wait_seconds)

        # The human's solve, read once and passed to the driver. This is the step that makes
        # the stored solve a capability rather than plumbing: the columns, the sealing and the driver
        # parameter all existed and nothing carried a stored solve into an actual fetch, so a
        # person cleared the same challenge on every poll.
        #
        # Read AFTER the circuit decision, because a suppressed fetch has nothing to carry it
        # into, and opening a credential for a request that will not be made is work done to
        # no purpose on the hottest path this tool has.
        solved_state: dict[str, Any] | None = None
        if error is None and (decision is None or decision.permitted) and self._health_collection is not None:
            solved_state = await self._read_solved_state(target_id)

        page: RenderedPage | None = None
        if error is None and (decision is None or decision.permitted):
            assert driver is not None  # narrowed by `error is None` above
            # Nested so the outer handler covers the recovery handler too, not just the
            # render. `record_unreachable` clears the probe as its first act, but a
            # cancellation landing in the statements before that would otherwise escape
            # between the two handlers and strand it again -- a narrow window, and the third
            # one in this family, which is why the guard is placed to cover the block rather
            # than the call.
            try:
                try:
                    if self._robots is not None:
                        # The clock starts on the FETCH, not on the check: the circuit can
                        # suppress a fetch after robots was consulted, and a check that led
                        # nowhere must not consume the site's patience.
                        self._robots.note_fetched(url)
                    page = await driver.render(
                        url,
                        timeout=self._default_timeout,
                        wait_for=wait_for,
                        nav_steps=nav_steps,
                        session_state=solved_state,
                    )
                except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- any backend-specific driver error surfaces as a ToolResult, never crashes the tool call
                    log.warning(
                        "scrape tool: render failed",
                        extra={"extra_data": {"url": url, "driver_backend": driver_backend}},
                    )
                    error = f"fetch failed: {exc}"
                    if self._circuit is not None:
                        # A page that never arrived is a fetch failure, exactly like a wall,
                        # and a target that has become unreachable should back off rather than
                        # be retried at full rate. Only the wall stamps `last_blocked_at`.
                        await self._circuit.record_unreachable(target_id)
            except BaseException:
                # Two ways in, now that this guards the block rather than the render alone: a
                # cancelled poll, which the inner handler does not catch by design, and any
                # exception raised INSIDE that handler, most plausibly a store failure in
                # `record_unreachable`. Both leave an admitted probe unresolved -- neither
                # reports an outcome, and only an outcome clears the flag -- and this block
                # holds the longest await in the function, so it is where a cancellation most
                # often lands.
                #
                # Deliberately does not record a durable outcome: persisting one would back
                # the target off across every pod, and outlive the process that was cancelled,
                # for something the target did not do. Releasing the in-process probe does
                # cost that breaker a failure -- the protocol has no "never mind" -- but that
                # is seconds-scale, process-local, and dies with the process anyway.
                #
                # On the second path the `error` string the inner handler had already composed
                # is discarded by the re-raise, so a store failure during the report surfaces
                # as an exception rather than as the ToolResult the inner pragma promises.
                # That predates this guard -- such an exception escaped `execute` before too,
                # just without releasing the probe -- and is left alone rather than widened
                # into a behaviour change smuggled in under a probe-lifecycle fix.
                if self._circuit is not None:
                    self._circuit.release_probe(target_id)
                raise

        if error is not None:
            result = ToolResult(success=False, content="", error=error)
        elif decision is not None and not decision.permitted:
            log.info(
                "scrape tool: fetch of target %s suppressed by its circuit (%s)",
                target_id,
                decision.reason,
                extra={"extra_data": {"target_id": target_id, "circuit_state": decision.state.value}},
            )
            # Nothing is persisted for a suppressed poll. No observation was made, and an
            # extraction row per suppressed poll would write more rows the harder the
            # backoff worked -- the opposite of what backing off is for.
            result = ToolResult(
                success=False,
                # Deliberately does not say "behind a wall": the same circuit opens on repeated
                # transport failures, and telling a caller its target is being challenged when
                # the real problem is that the host stopped answering sends it looking in the
                # wrong place. `reason` carries which one it was.
                error=(
                    "backing off: this target's fetch circuit is open after repeated failed "
                    f"fetches, so it was not fetched. {decision.reason.capitalize()}. Retry in "
                    f"about {decision.retry_after_seconds:.0f}s; retrying sooner will not "
                    "fetch anything."
                ),
                # "backoff", not "blocked", for the same reason `error` above does not say
                # "behind a wall" -- and this is the half a machine reads. In this package
                # "blocked" means a bot wall stood where the content should be, which is a
                # fact about the target; the same circuit also opens on repeated transport
                # failures, and reporting those as "blocked" tells a consumer a host that
                # simply stopped answering is challenging it. No fetch happened here at all,
                # so the honest status is the circuit's own.
                content=json.dumps({"target_id": target_id, "validation_status": "backoff", "records": []}),
                metadata={
                    "target_id": target_id,
                    "validation_status": "backoff",
                    "record_count": 0,
                    "source_url": url,
                    "circuit_state": decision.state.value,
                    "retry_after_seconds": decision.retry_after_seconds,
                },
            )
        else:
            assert page is not None  # narrowed by `error is None` above
            eval_loop_fn = run_eval_loop_multi_row if multi_row else run_eval_loop
            try:
                extraction = await eval_loop_fn(
                    target_id,
                    page.html,
                    page.final_url,
                    schema,
                    recipe_collection=self._recipe_collection,
                    extraction_collection=self._extraction_collection,
                    health_collection=self._health_collection,
                    api_key=self._api_key,
                    strategy_type=strategy_type,
                    # The driver already knows the status; not passing it would leave the
                    # classifier guessing about evidence we are holding.
                    page_status=page.status,
                )
                records: list[dict[str, Any]] = extraction.structured_fields.get("records", [])
                content = json.dumps(
                    {"target_id": target_id, "validation_status": extraction.validation_status, "records": records},
                    default=str,
                )
                # `blocked` is not success -- no records were produced -- but it is also not
                # the same failure as the others, and a caller that cannot tell them apart
                # will retry a walled target forever and count it as a broken extraction.
                # The distinction is surfaced in `error` because that is the field a caller
                # actually reads on a failed ToolResult; `validation_status` was already in
                # metadata and was already being ignored.
                blocked = extraction.validation_status == "blocked"
                if self._circuit is not None:
                    # Every non-blocked outcome closes the circuit, including an extraction that
                    # failed: this circuit counts FETCHES, and a page we can plainly read is a
                    # fetch that worked. A recipe that keeps missing against a page we received
                    # is `ScrapeRecipe.consecutive_validation_failures`'s business, and backing
                    # off the fetch would only starve the regeneration that fixes it.
                    if blocked:
                        await self._circuit.record_blocked(target_id)
                    else:
                        await self._circuit.record_reachable(target_id)
            except BaseException:
                # The circuit permitted this fetch, and a permitted decision may have promoted
                # the in-process breaker to HALF_OPEN and marked its probe in flight. That flag
                # is cleared only by an outcome, and raising past here means no outcome is ever
                # reported -- so the breaker would hold the probe for the life of the process,
                # fast-failing every later check of this target before the durable row is even
                # read, and answering "retry in about 0s" forever. The durable side needs
                # nothing: its promotion already stamped `blocked_until` as the probe's own
                # reservation, which outlives the process that abandoned it.
                #
                # `BaseException`, not `Exception`, because a cancelled poll strands the probe
                # exactly as thoroughly as a failing one. Deliberately does not swallow: a
                # failing eval loop is not a fetch outcome and must not be recorded as one.
                if self._circuit is not None:
                    self._circuit.release_probe(target_id)
                raise
            result = ToolResult(
                success=extraction.validation_status == "validated",
                error=(
                    "blocked: a bot wall or human-verification page stood where the content "
                    "should be, so nothing was extracted. The stored extraction strategy is "
                    "not implicated and was left untouched; retrying immediately will hit the "
                    "same wall."
                )
                if blocked
                else None,
                content=content,
                metadata={
                    "target_id": target_id,
                    "validation_status": extraction.validation_status,
                    "record_count": len(records),
                    "source_url": page.final_url,
                },
            )
        return result
