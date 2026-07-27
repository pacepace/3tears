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
import inspect
import json
from enum import StrEnum
from typing import Any

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.observe import get_logger

from pydantic import SecretStr

from .circuit import FetchDecision, TargetCircuit
from threetears.core import fire_and_forget
from threetears.core.egress import EgressDriver, EgressRegistry

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
class _RobotsDefault:
    """Marks "the caller said nothing about robots", which `None` cannot.

    `robots=None` is the explicit opt-out, so the absent case needs a third value. A typed
    singleton rather than `Any = object()`: annotated `Any`, the parameter claimed a default of
    `RobotsGate | None` while holding neither, which turned off checking for the one
    distinction the sentinel exists to make -- and `project-preferences.md` requires a real
    type on a signature sentinel for exactly that reason.
    """

    __slots__ = ()


_ROBOTS_DEFAULT = _RobotsDefault()


class _Gate(StrEnum):
    """Which gate in :meth:`ScrapeTool.execute` refused, when one did.

    Recorded at the gate that decides rather than reconstructed afterwards. The tail of
    ``execute`` used to work out what had happened from four interdependent signals -- an error
    string, a robots decision, a circuit decision, and a boolean derived from two of them -- so
    predicting which branch would win meant holding all four at once, and adding a fifth gate
    meant finding every place that reasoning had been spelled out.

    That is the shape this module's own comments blame for a run of consecutive stranded-probe
    bugs. Consolidating the compensation into one ``except BaseException`` fixed those bugs; it
    did not touch the decision structure that produced them, so the conditions remained.

    ``None`` rather than a member for "nothing refused": the absence of a refusal is not itself
    a gate, and giving it a name invites code that checks for it by equality and then has to be
    updated when a real gate is added.
    """

    #: Missing or malformed tool input, an unknown driver backend, an unusable schema.
    INPUT = "input"
    #: ``robots.txt`` disallows this path. Escalates to a human rather than failing.
    ROBOTS = "robots"
    #: The target's durable fetch circuit is open, so no fetch was attempted.
    CIRCUIT = "circuit"
    #: The fetch was attempted and did not produce a page.
    RENDER = "render"


def _accepts_session_state(driver: ScrapeDriver) -> bool:
    """Whether *driver*'s ``render`` can be passed a human's stored solve.

    ``ScrapeDriver`` is published as a pluggable contract, so a driver written against an
    earlier release has no such parameter. That is a supported thing to encounter and a
    configuration fact rather than a fetch outcome, so the caller reports it instead of
    attempting the call and reading the wreckage.

    ``**kwargs`` counts as accepting: a driver that forwards everything can take it, and
    deciding otherwise would refuse to use a delegating wrapper that works.

    Returns ``True`` when the signature cannot be read at all. Some callables -- C extensions,
    exotic proxies -- have no introspectable signature, and a helper whose failure mode is
    "assume the incompatibility" would block a driver nobody has shown to be broken. Attempting
    the call is the honest fallback; a genuine mismatch then raises and is reported as a fetch
    failure, which is the behaviour that existed before this check.
    """
    try:
        params = inspect.signature(driver.render).parameters
    except TypeError, ValueError:
        return True
    if "session_state" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


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
        robots: RobotsGate | None | _RobotsDefault = _ROBOTS_DEFAULT,
        egress: EgressDriver | str | None = None,
        egress_registry: EgressRegistry | None = None,
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
        :ptype robots: RobotsGate | None | _RobotsDefault
        :param egress: which exit this tool's own requests leave by -- a constructed
            :class:`~threetears.core.egress.EgressDriver`, or the NAME of one to resolve
            through *egress_registry* so a deployment can write ``egress: "tor"`` in its own
            config. Passed to the default robots gate so the robots.txt read shares the
            scrape's route rather than disclosing the container's address in front of it.
            Drivers take their own egress separately, because a driver may be shared between
            tools
        :ptype egress: EgressDriver | str | None
        :param egress_registry: where a NAME is looked up; defaults to a registry carrying
            ``direct`` alone. An unknown name raises rather than falling back to the default
            route, because a deployment that asked for ``tor`` and silently got direct would
            look correct in every log line while being wrong about the one property it
            configured
        :ptype egress_registry: EgressRegistry | None
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
        # A NAME is accepted, not just a constructed driver, because that is the whole reason
        # `EgressRegistry` exists: configuration says `egress: "tor"` and the seam resolves it,
        # rather than every caller growing its own `if name == "tor"` branch in front of the
        # fetch. Without this the registry had no consumer at all, which made it a capability
        # that was true of an object nobody built.
        #
        # An unknown name RAISES here rather than falling back to the default route. A
        # deployment that asked for `tor` and silently got direct would be told nothing, would
        # look correct in every log line, and would be wrong about the single property it
        # configured this for.
        if isinstance(egress, str):
            egress = (egress_registry or EgressRegistry()).get(egress)
        self._egress = egress
        # The default gate inherits this tool's exit. A robots request on the container's own
        # route, in front of a proxied scrape, discloses the address the proxy exists to hide.
        self._robots = RobotsGate(egress=egress) if isinstance(robots, _RobotsDefault) else robots
        # Declared budget has to cover the wait, not just the render -- see `timeout_seconds`.
        self._max_robots_wait_seconds = self._robots.max_wait_seconds if self._robots is not None else 0.0
        # After `self._robots`, deliberately: the gate this tool ACTUALLY uses is what the
        # check is about, and a caller can supply its own with an egress of its choosing.
        self._warn_on_split_egress(drivers, self._robots)
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
            # Render budget + the eval loop's own LLM-call budget + the longest wait an
            # honoured `Crawl-delay` can impose. That third term is not padding: the robots
            # gate sleeps BEFORE the render, so a site politely asking for a long delay makes
            # the whole call outlast a budget derived from the render alone. Advertising the
            # smaller number told the executor to abandon the call mid-sleep, and that
            # abandonment is exactly what stranded the probe -- the declared timeout has to
            # cover everything `execute` can actually wait for, or the deadline manufactures
            # the cancellation.
            #
            # Read off the live policy rather than the module default, so a deployment that
            # raises its own ceiling does not silently reintroduce the gap.
            timeout_seconds=self._default_timeout + 60.0 + self._max_robots_wait_seconds,
        )

    @staticmethod
    def _warn_on_split_egress(drivers: dict[str, ScrapeDriver], robots: RobotsGate | None) -> None:
        """Say something when some of what this tool sends leaves by a configured exit and some does not.

        Safe configuration otherwise needs two independent wiring points for one security
        property, and getting only one right reproduces the disclosure: the target learns the
        real address from whichever request nobody was thinking about. Nothing about that is
        visible -- both halves work.

        **Both directions, because the untested one is the worse one.** Drivers proxied with an
        unproxied gate leaks the address on a ``robots.txt`` read. Gate proxied with an
        unproxied driver leaks it on the PAGE FETCH -- the request the exit was configured for.
        A check that only looked one way stayed silent on the second, which is the configuration
        a reader is most likely to believe is safe.

        **The gate, not the constructor argument.** A caller can build its own
        :class:`RobotsGate` with its own egress and pass it in, so ``ScrapeTool(egress=...)``
        describes what the DEFAULT gate would have been, not what this tool actually does. A
        check reading the argument called a correctly-proxied pair split, and a warning that
        fires on correct configuration is one readers learn to filter.

        Nothing is said when *robots* is ``None``: with the gate disabled there is no second
        request, so there is no split to have.

        A warning rather than a refusal, because a deployment may genuinely want it -- a robots
        read through a shared exit the target is not the audience for -- and refusing would be
        this library overriding a decision it cannot see the reasons for. But it should have to
        be a decision.
        """
        if robots is None:
            return

        # `egress` is declared on `ScrapeDriver` and on `RobotsGate` as a concrete property, so
        # the name is promised rather than invented here. That buys a documented name, not an
        # enforced one: `ScrapeDriver` is satisfied structurally as well as by inheritance, so a
        # consuming application's own driver -- or one written before this attribute existed --
        # is valid without it. Hence `getattr`: a constructor raising `AttributeError` would
        # turn a security warning into an outage.
        #
        # A wrapper driver that forwards nothing would report `None` for a genuinely proxied
        # inner driver and land on the wrong side of this comparison, so `NetworkCaptureDriver`
        # and `MultiDocumentDriver` delegate the property to what they wrap.
        unproxied = sorted(name for name, d in drivers.items() if getattr(d, "egress", None) is None)
        proxied = sorted(name for name in drivers if name not in unproxied)
        gate_proxied = getattr(robots, "egress", None) is not None

        if proxied and not gate_proxied:
            log.warning(
                "scrape tool: driver(s) %s leave by a configured exit but this tool's robots.txt "
                "reads do not, so they go out on the container's own address in front of every "
                "proxied fetch. Pass the same egress driver to ScrapeTool(egress=...) unless that "
                "is intended.",
                ", ".join(proxied),
                extra={"extra_data": {"proxied_drivers": proxied, "gate_proxied": False}},
            )
        elif unproxied and gate_proxied:
            log.warning(
                "scrape tool: this tool's robots.txt reads leave by a configured exit but driver(s) "
                "%s do not, so the page fetch itself goes out on the container's own address -- the "
                "request the exit was configured for. Give those drivers the same egress driver "
                "unless that is intended.",
                ", ".join(unproxied),
                extra={"extra_data": {"unproxied_drivers": unproxied, "gate_proxied": True}},
            )

    def _release_probe(self, target_id: str) -> None:
        """Give back an in-process probe that will now never report an outcome.

        One definition rather than the same two lines repeated in each guard. The compensation
        being scattered is what let a fourth stranding bug appear in a family the code's own
        comments already called "the third one": every new await near this path needed someone
        to remember the pattern, and the pattern lived in two places to copy from.

        Deliberately releases and persists NOTHING. A durable outcome here would back the
        target off across every pod and outlive the cancelled process, for something the
        target never did.
        """
        if self._circuit is not None:
            self._circuit.release_probe(target_id)

    async def _render_once(
        self,
        driver: ScrapeDriver,
        url: str,
        *,
        wait_for: str | None,
        nav_steps: list[NavStep] | None,
        solved_state: dict[str, Any] | None,
        target_id: str,
        driver_backend: str,
    ) -> tuple[RenderedPage | None, str | None, bool]:
        """Fetch the page once, returning ``(page, error, fetch_attempted)``.

        Exactly one of *page* and *error* is set. *fetch_attempted* says whether the driver was
        actually called, which the error alone cannot answer: a render that failed still spent
        the origin's turn and its crawl-delay, while a refusal issued before the call spent
        neither. The caller needs the difference to know whether to give the fleet turn back.

        Extracted so `execute` has ONE `except BaseException` over the whole permitted path
        rather than two adjacent ones. That shape produced four stranded-probe bugs in a row,
        each fixed as a symptom: with two guards and a boundary between them, every new `await`
        has to be placed against whichever guard its author happened to be reading. There is now
        one guard and one place the compensation lives.

        Returning the error rather than raising it keeps the driver contract this tool already
        had -- a backend-specific failure becomes a `ToolResult`, never a crashed tool call --
        while letting a failure INSIDE the recovery path (most plausibly the
        `record_unreachable` store write) propagate to the caller's guard, which is what
        releases the probe.

        :param driver: the backend to render with
        :ptype driver: ScrapeDriver
        :param solved_state: a human's stored solve, or ``None``
        :ptype solved_state: dict[str, Any] | None
        :return: the rendered page and no error, or no page and the error string, plus
            whether the driver was actually called
        :rtype: tuple[RenderedPage | None, str | None, bool]
        """
        # `session_state` is passed ONLY when there is one. `ScrapeDriver` is published as a
        # pluggable contract, so an out-of-tree driver written against 0.19.x has no such
        # parameter -- passing it unconditionally made every fetch through such a driver raise
        # `TypeError`, including the overwhelming majority carrying no stored solve at all.
        #
        # A solve that DOES exist and a driver that cannot take it is a real incompatibility, and
        # it is reported rather than rendered around: going ahead unauthenticated in silence
        # would send a person to solve a challenge they had already cleared.
        #
        # Detected by signature rather than by letting the call raise, because the two failures
        # need different answers and `TypeError` cannot tell them apart -- one raised inside a
        # driver's own code is a genuine fetch failure, and treating every `TypeError` as
        # configuration would swallow those. More to the point, the guard below reports any
        # exception to `record_unreachable`, so a static incompatibility -- identical on every
        # poll, and unfixable by waiting -- would trip the durable circuit and suppress the
        # target for hours. Backoff is an answer to a target that has gone away, not to a
        # deployment that wired two components that do not fit.
        extra: dict[str, Any] = {"session_state": solved_state} if solved_state else {}
        if extra and not _accepts_session_state(driver):
            log.warning(
                "scrape tool: a stored solve exists for this target but the driver cannot accept it",
                extra={"extra_data": {"url": url, "driver_backend": driver_backend}},
            )
            return (
                None,
                (
                    f"driver {driver_backend!r} does not accept session_state, so the stored solve "
                    f"for this target cannot be applied. Fetching without it would discard a "
                    f"person's work and re-present the challenge they already cleared."
                ),
                False,
            )

        if self._robots is not None:
            # The clock starts on the FETCH, not on the check: a check that led nowhere must not
            # consume the site's patience. BELOW the incompatibility guard for that reason --
            # that guard returns without calling `render`, so charging the origin there paces
            # every sibling target on it for a fetch that never happened, forever, since a
            # signature mismatch is identical on every poll.
            self._robots.note_fetched(url)

        try:
            page = await driver.render(
                url,
                timeout=self._default_timeout,
                wait_for=wait_for,
                nav_steps=nav_steps,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- any backend-specific driver error surfaces as a ToolResult, never crashes the tool call
            # `exc_info` and `error_type`, because this is the failure that opens the durable
            # circuit three lines below and suppresses the target for hours. Without them the
            # operator asking "why did my fleet back off" cannot tell a timeout from a proxy
            # refusal from a `TypeError` inside a driver -- and telling a dead exit apart from
            # genuinely broken targets is the reason `EgressDriver.health()` exists at all. The
            # cause reached only the returned string, which goes to the ToolResult, not the log.
            log.warning(
                "scrape tool: render failed",
                exc_info=True,
                extra={
                    "extra_data": {
                        "url": url,
                        "driver_backend": driver_backend,
                        "error_type": type(exc).__name__,
                    }
                },
            )
            if self._circuit is not None:
                # A page that never arrived is a fetch failure, exactly like a wall, and a
                # target that has become unreachable should back off rather than be retried at
                # full rate. Only the wall stamps `last_blocked_at`.
                await self._circuit.record_unreachable(target_id)
            return None, f"fetch failed: {exc}", True
        return page, None, True

    async def _clear_robots_block_if_any(self, target_id: str) -> None:
        """Take a target out of the human queue, but only if it was in it.

        Reads before writing because the read is cached three-tier and the write is not: the
        overwhelming majority of fetches are of targets no robots file has ever disallowed,
        and writing for all of them to correct the few is the wrong way round.
        """
        if self._health_collection is None:
            return
        try:
            row = await self._health_collection.get(target_id)
            if row is None or row.robots_blocked_at is None:
                return
            await clear_robots_block(self._health_collection, target_id=target_id)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- this is housekeeping after a fetch the caller has already paid for; a health-store failure must not turn a good page into a failed ToolResult, and the stale queue entry is visible and self-corrects. Logged with its traceback below
            log.exception(
                "scrape tool: could not clear the robots block for target %s; it may stay queued",
                target_id,
                extra={"extra_data": {"target_id": target_id}},
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
        #
        # ONE case departs, and it is the robots-disallow escalation below. The sentinel holds
        # an error STRING, and the exit it feeds builds `ToolResult(success=False, content="",
        # error=error)` -- no content, no metadata. The escalation needs both: it reports
        # `validation_status="needs_human"` in a JSON body and in metadata, so a caller can
        # route it to a queue rather than parse a message. Routing it through the sentinel
        # would mean encoding that structure into a string and decoding it at the exit.
        #
        # So the rule for whoever adds the next gate is not "never return early". It is: use
        # the sentinel if your failure is an error string, and return directly only if you are
        # producing a result shape the exit cannot build.
        error: str | None = None
        # WHICH gate refused, recorded where the refusal is decided. `error` says what to tell
        # the caller; this says who decided, and the tail branches on it rather than inferring
        # it from the combination of `error`, `robots_decision` and `decision`.
        declined_by: _Gate | None = None
        # The one gate whose outcome is a full result rather than an error string. Built where
        # the decision is made, returned by the tail like every other outcome.
        escalation: ToolResult | None = None

        url = kwargs.get("url") or ""
        if not url:
            error, declined_by = "url is required", _Gate.INPUT

        schema: FieldSchema = {}
        raw_schema = kwargs.get("field_schema") or {}
        if error is None:
            try:
                schema = decode_field_schema(raw_schema)
            except ValueError as exc:
                error, declined_by = str(exc), _Gate.INPUT
            if error is None and not schema:
                error, declined_by = "field_schema must declare at least one field", _Gate.INPUT

        driver_backend = kwargs.get("driver_backend") or "nodriver"
        driver: ScrapeDriver | None = None
        if error is None:
            driver = self._drivers.get(driver_backend)
            if driver is None:
                error = f"unsupported driver_backend {driver_backend!r}; available: {sorted(self._drivers)}"
                declined_by = _Gate.INPUT

        nav_steps: list[NavStep] | None = None
        raw_nav_steps = kwargs.get("nav_steps")
        if error is None and raw_nav_steps:
            try:
                nav_steps = decode_nav_steps(raw_nav_steps)
            except TypeError as exc:
                error, declined_by = f"invalid nav_steps: {exc}", _Gate.INPUT

        strategy_type: StrategyType = kwargs.get("strategy_type") or "css"
        if error is None and strategy_type not in ("css", "regex"):
            error = f"unsupported strategy_type {strategy_type!r}; must be 'css' or 'regex'"
            declined_by = _Gate.INPUT

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
        if declined_by is None and self._robots is not None:
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
                    try:
                        await record_robots_block(
                            self._health_collection, target_id=target_id, reason=robots_decision.reason
                        )
                    except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- this tool's contract is to return a ToolResult, never to raise; the escalation below is the answer whether or not the queue write lands, and dropping it here loses the queue entry rather than the decision. Logged with its traceback below
                        log.exception(
                            "scrape tool: could not record the robots block for %s; it is escalated to "
                            "the caller but will not appear in list_walled",
                            target_id,
                            extra={"extra_data": {"target_id": target_id, "url": url}},
                        )
                # Recorded and carried to the tail rather than returned from here. The
                # escalation needs a result shape the `error` sentinel cannot express, which
                # was the original reason it returned early -- but a gate that leaves by its
                # own exit is a gate the tail cannot reason about, and that was the finding.
                # Holding the built result is not the thing worth avoiding; encoding it into a
                # string and decoding it again would have been.
                declined_by = _Gate.ROBOTS
                escalation = ToolResult(
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
            #
            # Guarded, and the guard is the point: unconditional, this ran a durable write on
            # EVERY allowed fetch -- creating rows for targets that never had one, putting the
            # optimistic-lock fence on the hot path of every poll rather than on state change,
            # and doing it for the overwhelming majority of targets that have never been
            # blocked at all. Never raises, for the same reason `circuit.py` wraps its own
            # call: a housekeeping failure must not turn a good fetch into a failed one.
            await self._clear_robots_block_if_any(target_id)

        decision: FetchDecision | None = None
        if declined_by is None and self._circuit is not None:
            decision = await self._circuit.check(target_id)
            if not decision.permitted:
                declined_by = _Gate.CIRCUIT

        # From here to the render, an admitted probe is already outstanding. `check` above may
        # have claimed this target's one in-process probe slot, and only an outcome or an
        # explicit release ever returns it -- so a cancellation anywhere below strands it,
        # after which `release_probe`'s own docstring says the target is fast-failed with
        # `retry_after_seconds=0.0` for the life of the process.
        #
        # Not a theoretical window. This tool advertises a deadline of `default_timeout + 60`
        # while an honoured `Crawl-delay` is capped at 300s, so an executor cancelling inside
        # that sleep is the EXPECTED case, not a rare one.
        #
        # There is ONE guard over the whole permitted path -- the sleep, the credential read and
        # the render. There used to be two adjacent ones with a boundary between them, which is
        # how this family reached four members: each new await had to be placed against
        # whichever guard its author happened to be reading. Anything added below this line is
        # covered without anyone having to notice.
        # Now a READ of what the gates recorded, not a re-derivation of it. This predicate used
        # to be `error is None and (decision is None or decision.permitted)` -- two of the four
        # signals the tail also consulted, combined here and nowhere else, so a fifth gate meant
        # remembering to widen this expression as well as the tail's.
        fetch_will_happen = declined_by is None

        # Both bound before the guard, not inside it: the flow does guarantee `page` is set on
        # every path that later reads it, but that guarantee is three branches away from the
        # read, and an explicit `None` costs nothing to keep it out of the argument.
        solved_state: dict[str, Any] | None = None
        page: RenderedPage | None = None
        fleet_wait_claimed = False
        try:
            # The crawl delay is waited only once the circuit has ADMITTED the fetch. Waiting
            # before it would block a caller for up to the delay ceiling only to be told the fetch
            # was suppressed -- paying a politeness cost for a request that never happens. The
            # floor-vs-ceiling rule is satisfied either way: both gates are still honoured, and a
            # circuit probe still waits its delay.
            # The FLEET's turn is taken here rather than during `check`, and only once the
            # fetch is committed. `TokenBucket.claim` consumes: asking earlier charged this
            # origin's shared budget for polls that never fetched, so one walled target inside
            # its backoff delayed every sibling target on the same site. Same rule the local
            # clock already followed -- the site pays when we actually visit it.
            fleet_wait = 0.0
            if self._robots is not None and fetch_will_happen:
                # `claimed` comes from the gate rather than being inferred here. A granted turn
                # returns 0.0 seconds, indistinguishable from every path that never asked, and
                # a REFUSED turn returns a positive wait while consuming nothing -- so reading
                # the float alone gave a token back precisely when another pod was holding it.
                fleet_wait, fleet_wait_claimed = await self._robots.claim_fleet_turn(url)

            wait_seconds = max(robots_decision.wait_seconds if robots_decision is not None else 0.0, fleet_wait)
            if wait_seconds > 0 and fetch_will_happen:
                # Says which constraint is binding. "as its robots.txt asks" was wrong whenever
                # the fleet pacer was the longer of the two -- an operator reading it would go
                # looking at the site's file for a wait the site never asked for.
                because = (
                    "its robots.txt asks"
                    if wait_seconds > fleet_wait
                    else "the fleet pacer has not yet given this origin a turn"
                )
                log.info(
                    "scrape tool: waiting %.0fs before fetching %s, because %s",
                    wait_seconds,
                    url,
                    because,
                    extra={
                        "extra_data": {
                            "target_id": target_id,
                            "site_wait_seconds": robots_decision.wait_seconds if robots_decision is not None else 0.0,
                            "fleet_wait_seconds": fleet_wait,
                        }
                    },
                )
                await asyncio.sleep(wait_seconds)

            # The human's solve, read once and passed to the driver. This is the step that makes
            # the stored solve a capability rather than plumbing: the columns, the sealing and the driver
            # parameter all existed and nothing carried a stored solve into an actual fetch, so a
            # person cleared the same challenge on every poll.
            #
            # Read AFTER the circuit decision, because a suppressed fetch has nothing to carry it
            # into, and opening a credential for a request that will not be made is work done to
            # no purpose on the hottest path this tool has.
            solved_state = None
            if fetch_will_happen and self._health_collection is not None:
                solved_state = await self._read_solved_state(target_id)

            if fetch_will_happen:
                assert driver is not None  # narrowed by `error is None` above
                page, render_error, fetch_attempted = await self._render_once(
                    driver,
                    url,
                    wait_for=wait_for,
                    nav_steps=nav_steps,
                    solved_state=solved_state,
                    target_id=target_id,
                    driver_backend=driver_backend,
                )
                if render_error is not None:
                    error, declined_by = render_error, _Gate.RENDER
                if not fetch_attempted:
                    # The circuit may have admitted a probe before any of this ran, and only an
                    # outcome or an explicit release ever gives it back. This path reports no
                    # outcome -- deliberately, since a configuration error is not evidence about
                    # the target -- and returns normally, so the `except BaseException` guard
                    # below never sees it. Without this the breaker holds a probe forever and
                    # `release_probe`'s own docstring says the target is then fast-failed with
                    # `retry_after_seconds=0.0` for the life of the process, on every poll,
                    # because a signature mismatch recurs identically.
                    #
                    # The asymmetry is what gave it away: this block already gave the fleet turn
                    # back and left the probe. Compensating one resource and not its neighbour
                    # is the shape of every member of this bug family.
                    self._release_probe(target_id)
                if not fetch_attempted and fleet_wait_claimed and self._robots is not None:
                    # The turn was claimed for a fetch that then did not happen, so give it
                    # back. Conditioned on `fetch_attempted` rather than on `render_error`,
                    # because a render that FAILED still went out: it spent the origin's turn
                    # and is not owed a refund. Only a refusal issued before the call is.
                    #
                    # Fire-and-forget for the same reason the cancellation handler below uses
                    # it, and with the same worst case: if the refund never lands the bucket
                    # refills on its own.
                    fleet_wait_claimed = False
                    fire_and_forget(self._robots.refund_fleet_turn(url))
        except BaseException:
            # ONE home for the compensation, over the whole permitted path. This block holds
            # every await between the circuit admitting a fetch and the outcome being reported,
            # so it is where a cancellation lands -- and `_render_once` can also raise from its
            # own recovery path, most plausibly a store failure in `record_unreachable`, which
            # reaches here for the same reason.
            #
            # Deliberately records no durable outcome: persisting one would back the target off
            # across every pod, and outlive the process that was cancelled, for something the
            # target did not do. Releasing the in-process probe does cost that breaker a
            # failure -- the protocol has no "never mind" -- but that is seconds-scale,
            # process-local, and dies with the process anyway.
            self._release_probe(target_id)
            if fleet_wait_claimed and self._robots is not None:
                # Give the origin's shared turn back too. Fire-and-forget rather than awaited:
                # this handler runs during a cancellation more often than not, and an `await`
                # here would re-raise `CancelledError` before the refund ever reached the KV
                # store. `fire_and_forget` schedules it as its own task and holds a strong
                # reference, so it is not collected mid-flight; if the loop is going down with
                # us it does not complete, and the bucket refills on its own -- which is the
                # behaviour this replaces, so the worst case is no worse than before.
                fire_and_forget(self._robots.refund_fleet_turn(url))
            raise
        if declined_by is _Gate.ROBOTS:
            assert escalation is not None  # set with the marker, in the same block
            result = escalation
        elif declined_by is _Gate.INPUT or declined_by is _Gate.RENDER:
            assert error is not None  # both gates set it with the marker, in one statement
            result = ToolResult(success=False, content="", error=error)
        elif declined_by is _Gate.CIRCUIT:
            assert decision is not None  # only the circuit gate sets this marker
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
                    # `egress=` carries the exit the page was configured to come through, as
                    # REPORTED BY THE FETCHER rather than the circuit's constructor-time name --
                    # which describes how the circuit was wired and is wrong for any render that
                    # chose a different exit. Reported, not observed: it says which exit was
                    # asked for and confirmed by whoever performed the fetch, not that traffic
                    # left that way.
                    if blocked:
                        await self._circuit.record_blocked(target_id, egress=page.egress)
                    else:
                        await self._circuit.record_reachable(target_id, egress=page.egress)
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
                self._release_probe(target_id)
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
