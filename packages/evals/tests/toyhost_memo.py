"""A fixtured analysis-gen completion over the toy campaign, and the coordinates a memo cites.

Harness support for the suites that generate over the toy host, not part of the toy host itself:
the toy host is example code that reaches the engine only through its public roots, and the
cell-reference helpers a test needs to address a memo's citations are
:mod:`~packages.evals.tests.toyhost_cells`'s, imported here for the memo suites that bind them from this module.

The generation path's one non-production part is the model, so a test that drives
:func:`~threetears.evals.analysis.generator.generate_analysis` over the toy bundle swaps in
:class:`~packages.evals.tests.fixtured_client.FixturedClient`, which hands back one prepared completion and records
what it was asked.
:func:`memo_payload` is that completion: a well-formed memo in the toy host's vocabulary and
nobody else's, written in the authored contract the generator is sent.
"""

from __future__ import annotations

from typing import Any

from packages.evals.tests.fixtured_client import (
    FIXTURED_CALL_CEILING_USD,
    MODEL,
    FixturedClient,
    FixturedCompletion,
)
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_AXIS, TOYHOST_NARROW, TOYHOST_WIDE
from packages.evals.tests.toyhost_cells import alias_at, arm_at, cell_at
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis import EVAL_ANALYSIS_GEN_DEFAULT, AnalysisContextBundle, CampaignBundle
from packages.evals.tests.campaign_bundle_support import campaign_bundle

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
#: The registry key the analysis-gen prompt resolves under, supplied because the generator takes
#: provenance from its caller rather than defaulting to a guess.
PROMPT_ID = "eval_analysis_gen"


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
    # A choice's pick of an arm code shows ahead on no pair rests on cost or simplicity, so it is at low confidence.
    picked = {arm_at(bundle, TOYHOST_NARROW), variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW))}
    unseparated = bundle.kind == "choice" and not any(pair.ahead in picked for pair in bundle.readings.pairs)
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
        "recommendation": [
            {
                "arm": arm_at(bundle, TOYHOST_NARROW),
                "reasons": ["The narrow chunk width is much faster on every document."],
                "confidence": "low" if unseparated else "high",
                "rests_on": [0],
                "floors_overridden": [],
                "behind_on": [],
                # Every bar code read the narrow width missing, each addressed, as a recommendation over one must be.
                "bars_missed": [
                    {
                        "aspect": status.aspect,
                        "measure": status.measure,
                        "why": "The narrow width is recommended for its speed; this bar is the next run's work.",
                    }
                    for status in bundle.readings.missed_bars()
                    if status.arm_label == arm_at(bundle, TOYHOST_NARROW)
                ],
            }
        ],
        "aspect_summaries": [
            {
                "aspect": bundle.labels[0],
                "what_changed": "The wide chunk width slowed every document down.",
                "how_sure": "Sure of the direction; the gap is wide.",
                "means": "Both widths were measured over the same documents and the wider one is slower by a wide margin.",
            }
        ],
        "summary": "The narrow width is much faster on every document, so it is the width to run.",
        "next": [
            {
                "title": "sweep the retriever width at the narrow chunk",
                "why": "the only other lever this campaign declared nothing about",
                "leverage": "high",
                "lever": "",
            }
        ],
    }
