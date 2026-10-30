"""Every refusal reaches the caller as a refusal: through the dispatcher a host's transport calls, never as an exception.

A host's transport hands each call to :meth:`MountedTool.call` and serves what comes back. An exception out of it is a
defect to the host (a 500 over MCP or REST), so a refusal of what the caller asked must come back as an
:class:`ActionOutcome` refusal with a 4xx status. Pinned here, every case driven through the dispatcher:

- **The backstop.** A lower layer's refusal (:class:`RefusedInput`, or any subclass) that no operation translated is
  refused ``validation_failed`` (422) with its fault and remedy, and one WARNING names the action and the function
  that raised it. Anything else a handler raises, a bare ``ValueError`` among them, still propagates.
- **Translated at the source.** ``campaign_record`` over an analysis frozen at a bundle shape this build no longer
  reads refuses 422 before any result is read and before any aspect's bundle is put back together; ``report_read``
  over one reads code-only, the shape checked before the evidence is assembled; a memo's page that refuses what it is
  laid out over is a 422. None of them logs the backstop's WARNING.
- **The planner's view.** ``analysis_estimate`` reads no run and no result, so the view's budget refusal cannot
  arise there; the planner's preparation, which can meet it, refuses 422.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import pytest
from pydantic import Field

import threetears.evals.analysis.report_input_build as report_input_build
from threetears.evals.actions import (
    Action,
    ActionCatalogue,
    Caller,
    MountedTool,
    Refusal,
    ToolSpec,
    eval_catalogue,
    standard_tools,
)
from threetears.evals.analysis import prepare_analysis_generation
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.lenses.export import ExportError
from threetears.evals.analysis.lenses.frontier import FrontierError
from threetears.evals.analysis.lenses.history import HistoryError
from threetears.evals.analysis.lenses.pivot import PivotError
from threetears.evals.analysis.plan_prompt import EVAL_REPORT_PLAN_DEFAULT
from threetears.evals.analysis.report.campaign.memo_blocks import NO_LONGER_READS
from threetears.evals.kernel import RefusedInput, ValidationFailedError
from threetears.evals.kernel.declaration import DesignRefused, UndeclarableAxisError
from threetears.evals.kernel.status_filter import StatusFilterError
from threetears.evals.ops import OpsHost
from threetears.evals.run.ceilings import CeilingRaisedError
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.factories import analysed_aspect, empty_bundle, make_analysis
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture

#: The bundle shape v0.75.0 and v0.76.0 froze analyses over, which this build no longer reads.
SHIPPED_SHAPE = 60

#: The catalogue's module, whose logger the backstop's WARNING is written to.
CATALOGUE_LOGGER = "threetears.evals.actions.catalogue"

#: Every refusal class of the package that is a lower layer's refusal rather than an engine error.
REFUSAL_CLASSES: tuple[type[RefusedInput], ...] = (
    RefusedInput,
    PivotError,
    HistoryError,
    FrontierError,
    ExportError,
    StatusFilterError,
    CeilingRaisedError,
    DesignRefused,
    UndeclarableAxisError,
)


class ReadCountingStore(InMemoryDocumentStore):
    """An in-memory store recording the document type of every read, in order, so a test sees what was read first."""

    def __init__(self) -> None:
        """Start with no reads recorded."""
        super().__init__()
        self.reads: list[str] = []

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        exclude: Sequence[str] = (),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        """Record the type, then read as the in-memory store does."""
        self.reads.append(doc_type)
        return super().by_doc_type(
            doc_type,
            scope_id,
            order_by=order_by,
            descending=descending,
            limit=limit,
            exclude=exclude,
            field_in=field_in,
            **field_eq,
        )

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Record the type, then read as the in-memory store does."""
        self.reads.append(doc_type)
        return super().get_many(doc_type, doc_ids, scope_id, exclude=exclude, keep=keep)

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        """Record the type, then read as the in-memory store does."""
        self.reads.append(doc_type)
        return super().iter_by_doc_type(doc_type, scope_id)

    def evidence_reads(self) -> list[str]:
        """The reads of runs and results: what assembling a campaign's evidence reads."""
        return [doc_type for doc_type in self.reads if doc_type in ("eval_run", "eval_result")]


class _Note(EvalBaseModel):
    """A host action's parameters."""

    note: str = Field(description="Anything; the action refuses whatever it is given.")


class _Never(EvalBaseModel):
    """A result the action never returns."""


def _raising_tool(raise_it: Callable[[], None]) -> MountedTool:
    """A read tool whose one action, ``note_refuse``, calls ``raise_it``."""

    async def handle(host: OpsHost, caller: Caller, params: _Note) -> _Never:
        raise_it()
        return _Never()

    action = Action(
        name="note_refuse",
        summary="Refuse a note.",
        workflow="Host",
        permission="read",
        params=_Note,
        result=_Never,
        handler=handle,
        render=lambda result: "accepted",
        example={"note": "hello"},
    )
    return ActionCatalogue([action]).mount(
        ToolSpec(name="host", description="A host tool.", permissions=frozenset({"read"}))
    )


def _backstop_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == CATALOGUE_LOGGER and record.levelno == logging.WARNING and "untranslated" in record.message
    ]


def _frozen_at(shape: int, campaign_id: str) -> dict[str, Any]:
    """An aspect bundle as an older build froze it: its shape, and a field that shape carried and this one dropped."""
    bundle = empty_bundle(campaign_id=campaign_id).to_dict()
    return {**bundle, "schema_version": shape, "measurement_window_disclosure": None}


def _store_analysis_frozen_at_the_shipped_shape(fixture: OpsFixture, *, readable_first: bool = False) -> str:
    """Store the campaign's report: an analysis whose (last) aspect was frozen at :data:`SHIPPED_SHAPE`."""
    campaign_id = fixture.campaign.id
    readable = empty_bundle(campaign_id=campaign_id)
    old = analysed_aspect(readable).model_copy(
        update={"label": "older", "bundle": _frozen_at(SHIPPED_SHAPE, campaign_id)}
    )
    aspects = [analysed_aspect(readable), old] if readable_first else [old]
    analysis = make_analysis(
        campaign_id=campaign_id,
        scope_id=TOYHOST_SCOPE,
        aspects=aspects,
        campaign_bundle=campaign_bundle(readable).frame(),
    )
    off_loop(fixture.host.eval_host.storage.save_analysis, analysis)
    return analysis.id


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


# =============================================================================
# The backstop
# =============================================================================


def _raise_from_a_lower_layer(refusal: type[RefusedInput]) -> None:
    """Raise ``refusal`` as a lower layer does, from a function of its own (the site the WARNING names)."""
    raise refusal("the axis 'colour' is not one this campaign sweeps", remedy="pivot on one of its levers")


@pytest.mark.parametrize("refusal", REFUSAL_CLASSES, ids=[cls.__name__ for cls in REFUSAL_CLASSES])
async def test_an_untranslated_refusal_is_refused_422_with_its_fault_and_remedy_and_one_warning_naming_its_site(
    refusal: type[RefusedInput], caplog: pytest.LogCaptureFixture
) -> None:
    tool = _raising_tool(lambda: _raise_from_a_lower_layer(refusal))
    caplog.set_level(logging.WARNING, logger=CATALOGUE_LOGGER)

    outcome = await tool.call({"action": "note_refuse", "note": "x"}, host=off_loop(ops_fixture).host, caller=CALLER)

    assert outcome.is_error and outcome.refusal == Refusal(kind="validation_failed", status=422)
    assert outcome.ending == "refused" and outcome.action == "note_refuse"
    assert outcome.text == (
        "refused: note_refuse: the axis 'colour' is not one this campaign sweeps\nremedy: pivot on one of its levers"
    )
    (warning,) = _backstop_warnings(caplog)
    assert "action note_refuse" in warning.message
    assert f"{__name__}:_raise_from_a_lower_layer" in warning.message, "the WARNING names the function that raised it"


async def test_an_untranslated_refusal_keeps_the_parameters_its_raiser_named_as_the_recovery() -> None:
    def raise_it() -> None:
        raise RefusedInput("no such note", remedy="name one", parameters=("note",), valid_values=("a", "b"))

    outcome = await _raising_tool(raise_it).call(
        {"action": "note_refuse", "note": "x"}, host=off_loop(ops_fixture).host, caller=CALLER
    )

    assert outcome.recovery is not None
    assert (outcome.recovery.parameters, outcome.recovery.valid_values) == (("note",), ("a", "b"))


def test_every_refusal_class_of_the_package_is_covered() -> None:
    def subclasses(cls: type[RefusedInput]) -> set[type[RefusedInput]]:
        found = {cls}
        for sub in cls.__subclasses__():
            found |= subclasses(sub)
        return found

    package = {cls for cls in subclasses(RefusedInput) if cls.__module__.startswith("threetears.")}
    assert package <= set(REFUSAL_CLASSES), f"a refusal class is not covered here: {package - set(REFUSAL_CLASSES)}"


async def test_a_bare_value_error_is_a_defect_and_propagates(caplog: pytest.LogCaptureFixture) -> None:
    def raise_it() -> None:
        raise ValueError("a bug")

    with pytest.raises(ValueError, match="a bug"):
        await _raising_tool(raise_it).call(
            {"action": "note_refuse", "note": "x"}, host=off_loop(ops_fixture).host, caller=CALLER
        )
    assert _backstop_warnings(caplog) == []


# =============================================================================
# Translated at the source, the stored shape read first
# =============================================================================


async def test_a_call_over_an_analysis_frozen_at_another_shape_is_refused_422_before_any_evidence_is_read(
    evals: MountedTool, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ReadCountingStore()
    fixture = off_loop(ops_fixture, store=store)
    analysis_id = _store_analysis_frozen_at_the_shipped_shape(fixture, readable_first=True)

    def never(*args: Any, **kwargs: Any) -> AnalysisContextBundle:
        raise AssertionError("an aspect's bundle was read before every aspect's shape was checked")

    monkeypatch.setattr(AnalysisContextBundle, "from_dict", never)
    store.reads.clear()
    caplog.set_level(logging.WARNING, logger=CATALOGUE_LOGGER)

    outcome = await evals.call(
        {"action": "campaign_record", "campaign_id": fixture.campaign.id, "arm": "1024tok", "analysis_id": analysis_id},
        host=fixture.host,
        caller=CALLER,
    )

    assert outcome.is_error and outcome.refusal == Refusal(kind="validation_failed", status=422), outcome.text
    assert f"shape {SHIPPED_SHAPE}" in outcome.text and "analysis_generate" in outcome.text
    assert store.evidence_reads() == [], "refused before any run or result was read"
    assert _backstop_warnings(caplog) == [], "translated by the operation, not by the backstop"


async def test_the_report_of_an_analysis_frozen_at_another_shape_reads_code_only_its_shape_checked_before_assembly(
    evals: MountedTool, caplog: pytest.LogCaptureFixture
) -> None:
    store = ReadCountingStore()
    fixture = off_loop(ops_fixture, store=store)
    analysis_id = _store_analysis_frozen_at_the_shipped_shape(fixture)
    store.reads.clear()
    caplog.set_level(logging.WARNING, logger=CATALOGUE_LOGGER)

    outcome = await evals.call(
        {"action": "report_read", "campaign_id": fixture.campaign.id, "format": "markdown"},
        host=fixture.host,
        caller=CALLER,
    )

    assert not outcome.is_error, outcome.text
    assert outcome.structured is not None
    assert (outcome.structured["basis"], outcome.structured["analysis_id"]) == ("code_only", analysis_id)
    needs = outcome.structured["needs_regeneration"]
    assert needs.startswith(f"{NO_LONGER_READS} (") and f"shape {SHIPPED_SHAPE}" in needs
    first_evidence = store.reads.index(store.evidence_reads()[0])
    assert "eval_analysis" in store.reads[:first_evidence], "the analysis's shape is read before the evidence is"
    assert _backstop_warnings(caplog) == []


async def test_a_memo_page_refusing_what_it_is_laid_out_over_is_a_422_refusal(
    evals: MountedTool, caplog: pytest.LogCaptureFixture
) -> None:
    fixture = off_loop(ops_fixture)
    # A bundle of this build's shape, so it is put back together, but read off another campaign: the page refuses it.
    other = empty_bundle(campaign_id="another-campaign")
    held = make_analysis(
        campaign_id=fixture.campaign.id,
        scope_id=TOYHOST_SCOPE,
        aspects=[analysed_aspect(other)],
        campaign_bundle=campaign_bundle(other).frame(),
    )
    off_loop(fixture.host.eval_host.storage.save_analysis, held)
    caplog.set_level(logging.WARNING, logger=CATALOGUE_LOGGER)

    outcome = await evals.call(
        {"action": "report_read", "campaign_id": fixture.campaign.id, "format": "json"},
        host=fixture.host,
        caller=CALLER,
    )

    assert outcome.is_error and outcome.refusal == Refusal(kind="validation_failed", status=422), outcome.text
    assert "another-campaign" in outcome.text and "\nremedy: " in outcome.text
    assert _backstop_warnings(caplog) == []


# =============================================================================
# The planner's view
# =============================================================================


async def test_the_generation_estimate_reads_no_run_and_no_result(evals: MountedTool) -> None:
    store = ReadCountingStore()
    fixture = off_loop(ops_fixture, store=store)
    store.reads.clear()

    outcome = await evals.call(
        {"action": "analysis_estimate", "campaign_id": fixture.campaign.id}, host=fixture.host, caller=CALLER
    )

    assert not outcome.is_error, outcome.text
    assert store.evidence_reads() == [], "no evidence is read, so the planner's view is never built here"


async def test_the_planners_preparation_over_its_view_budget_refuses_422(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = off_loop(ops_fixture)
    monkeypatch.setattr(report_input_build, "CAMPAIGN_MESSAGE_CHARS_PER_ASPECT", 10)

    async def prompt() -> str:
        return EVAL_REPORT_PLAN_DEFAULT

    with pytest.raises(ValidationFailedError) as refused:
        await prepare_analysis_generation(
            fixture.host.eval_host,
            fixture.campaign.id,
            TOYHOST_SCOPE,
            model=None,
            resolve_prompt=prompt,
            out_of_run_cap_usd=fixture.host.out_of_run_cap(),
        )
    assert refused.value.status_code == 422
    assert "after every trim, over its budget of 10" in refused.value.message
    assert refused.value.remedy is not None and "raise the planner's budget" in refused.value.remedy
