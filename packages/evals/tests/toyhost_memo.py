"""A fixtured analysis-gen completion over the toy campaign, and the coordinates a memo cites.

Harness support for the suites that generate over the toy host, not part of the toy host itself:
the toy host is example code that reaches the engine only through its public roots, and this
module reads the cell-reference helpers a test needs to address a memo's citations.

The generation path's one non-production part is the model, so a test that drives
:func:`~threetears.evals.analysis.generator.generate_analysis` over the toy bundle swaps in
:class:`FixturedClient`, which hands back one prepared completion and records what it was asked.
:func:`memo_payload` is that completion: a well-formed memo in the toy host's vocabulary and
nobody else's, written in the authored contract the generator is sent.

Cell coordinates are read off the assembled bundle rather than pasted: a cell ref is two content
hashes, and a literal would stop matching the moment either derivation moved under it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from packages.evals.tests.fixtures.toyhost.campaign import (
    TOYHOST_AXIS,
    TOYHOST_NARROW,
    TOYHOST_WIDE,
)
from threetears.evals.analysis.cells import cell_ref, variant_of_cell_ref
from threetears.evals.analysis import EVAL_ANALYSIS_GEN_DEFAULT, AnalysisContextBundle, CampaignBundle
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from threetears.evals.analysis.references import cell_aliases

__all__ = [
    "FIXTURED_CALL_CEILING_USD",
    "MODEL",
    "PROMPT",
    "PROMPT_ID",
    "FixturedClient",
    "FixturedCompletion",
    "alias_at",
    "arm_at",
    "cell_at",
    "memo_payload",
]

#: The prompt the generator is sent: the engine's default analysis-gen prompt.
PROMPT = EVAL_ANALYSIS_GEN_DEFAULT
#: The model the fixtured client reports itself bound to.
MODEL = "anthropic/claude-opus"
#: The registry key the analysis-gen prompt resolves under, supplied because the generator takes
#: provenance from its caller rather than defaulting to a guess.
PROMPT_ID = "eval_analysis_gen"


@dataclass
class FixturedCompletion:
    """The subset of an LLM result the generator reads, with a canned body."""

    content: str
    cost_usd: float | None = 0.02
    # ``None`` is a count the provider did not report, as ``CompletionResult`` allows.
    input_tokens: int | None = 1200
    output_tokens: int | None = 800
    model: str = MODEL
    stop_reason: str = "end_turn"
    temperature: float | None = None
    tool_calls: list[Any] = field(default_factory=list)
    reasoning_tokens: int | None = None


#: What every fixtured call is priced at, at most: above the completion's reported cost, well under any toy cap.
FIXTURED_CALL_CEILING_USD = 0.05


class FixturedClient:
    """A client that returns one prepared completion and records what it was asked."""

    def __init__(self, body: str) -> None:
        """Prepare the completion.

        :param body: The JSON the generator will parse.
        :ptype body: str
        """
        self.completion = FixturedCompletion(content=body)
        self.calls: list[dict[str, str]] = []
        #: The response format each call sent, in the order of :attr:`calls`.
        self.response_formats: list[Any] = []
        self.closed = 0
        #: What :meth:`price_ceiling` answers for every call; ``None`` is a client that cannot price.
        self.ceiling_usd: float | None = FIXTURED_CALL_CEILING_USD
        #: Every prompt pair priced, in order.
        self.priced: list[dict[str, str]] = []

    @property
    def model_name(self) -> str:
        """The model this client is bound to -- the port the service reads back."""
        return MODEL

    def price_ceiling(self, *, system: str, user: str, response_format: Any = None) -> float | None:
        """The fixtured ceiling of one call — what a generation is priced against its out-of-run cap by."""
        self.priced.append({"system": system, "user": user})
        return self.ceiling_usd

    async def generate(
        self, *, system: str, user: str, response_format: Any = None, tools: Any = None
    ) -> FixturedCompletion:
        """Record the call and hand back the prepared completion."""
        self.calls.append({"system": system, "user": user})
        self.response_formats.append(response_format)
        return self.completion

    async def aclose(self) -> None:
        """Release nothing: a fixtured client owns no transport. Counted, so a test can see it was released."""
        self.closed += 1

    async def __aenter__(self) -> FixturedClient:
        """Enter a scope that releases on exit, as the service enters every client it builds."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Release on the way out."""
        await self.aclose()


def cell_at(bundle: Any, level: int) -> str:
    """The ``cell_ref`` of the arm run at one chunk width, read off the assembled bundle.

    Joined through the arm's run, because the run summary is where the width is recorded and the
    cell is where the run was pooled.

    :param bundle: The assembled context, or a campaign bundle of one aspect.
    :ptype bundle: Any
    :param level: The chunk width, in tokens.
    :ptype level: int
    :return: The cell's ``<variant_key>:<apparatus_class_id>``.
    :rtype: str
    """
    bundle = bundle.aspects[0] if isinstance(bundle, CampaignBundle) else bundle
    (run_id,) = [summary.run_id for summary in bundle.run_summaries if summary.config[TOYHOST_AXIS] == str(level)]
    (cell,) = [cell for cell in bundle.cell_measures if run_id in cell.run_ids]
    return cell_ref(cell.variant_key, cell.apparatus_class_id)


def alias_at(bundle: Any, level: int) -> str:
    """The alias the writer names the cell at one chunk width by -- what an authored memo carries.

    :param bundle: The assembled context.
    :ptype bundle: Any
    :param level: The chunk width, in tokens.
    :ptype level: int
    :return: The cell's alias, which the generator stores as :func:`cell_at`'s ``cell_ref``.
    :rtype: str
    """
    aspect = bundle.aspects[0] if isinstance(bundle, CampaignBundle) else bundle
    (alias,) = [alias for alias, ref in cell_aliases(aspect.cell_measures).items() if ref == cell_at(aspect, level)]
    return alias


def arm_at(bundle: CampaignBundle, level: int) -> str:
    """The label the campaign bundle names the arm at one chunk width by -- what a recommendation carries.

    :param bundle: The campaign bundle, of one aspect.
    :ptype bundle: CampaignBundle
    :param level: The chunk width, in tokens.
    :ptype level: int
    :return: The arm's label.
    :rtype: str
    """
    variant = variant_of_cell_ref(cell_at(bundle, level))
    assert variant is not None
    return bundle.label_of(variant)


def memo_payload(bundle: CampaignBundle | AnalysisContextBundle) -> dict[str, Any]:
    """A well-formed memo in the toy host's vocabulary and nobody else's.

    One finding carrying the evidence, a summary of the one aspect and a recommendation to keep the narrow width,
    resting on that finding by position. Every reading names a cell and a measure and carries no number -- code
    fills it.

    :param bundle: The campaign bundle, of one aspect, or that aspect's bundle, read as a campaign of it. Its cells
        are the only ones a reading may name, and its arms the only ones the recommendation may.
    :ptype bundle: CampaignBundle | AnalysisContextBundle
    :return: The payload.
    :rtype: dict[str, Any]
    """
    if not isinstance(bundle, CampaignBundle):
        bundle = campaign_bundle(bundle)
    narrow, wide = alias_at(bundle, TOYHOST_NARROW), alias_at(bundle, TOYHOST_WIDE)
    return {
        "findings": [
            {
                "title": "widening the chunk slows every document down",
                "body": "The narrow width averaged about 920 ms per document and the wide one about 1420 ms.",
                "confidence": "high",
                "axes": [TOYHOST_AXIS],
                "evidence": [
                    {"cell": narrow, "measure_id": "total_ms", "reading": "measure"},
                    {"cell": wide, "measure_id": "total_ms", "reading": "measure"},
                ],
                "trials": [],
                "caveats": [
                    {
                        "kind": "sampling",
                        "text": "two documents at three repeats each — enough to see this gap, not enough to size a smaller one",
                    }
                ],
                "invalidates": [],
                "durable": "",
            }
        ],
        "recommendation": {
            "arm": arm_at(bundle, TOYHOST_NARROW),
            "reasons": ["The narrow chunk width is much faster on every document."],
            "confidence": "high",
            "rests_on": [0],
            "floors_overridden": [],
        },
        "aspect_summaries": [
            {
                "aspect": bundle.labels[0],
                "what_changed": "The wide chunk width slowed every document down.",
                "how_sure": "Sure of the direction; the gap is wide.",
                "means": "Both widths were measured over the same documents and the wider one is slower by a wide margin.",
            }
        ],
        "next": [
            {
                "title": "sweep the retriever width at the narrow chunk",
                "why": "the only other lever this campaign declared nothing about",
                "leverage": "high",
                "lever": "",
            }
        ],
    }
