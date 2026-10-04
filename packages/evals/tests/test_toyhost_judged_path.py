"""The toy host's judged variant, end to end: its kind renders the evidence, the engine's judge scores it.

The toy host ships as reference code for a host adopting the engine, and its judged variant
(``fixtures/toyhost/judge.py``) is the shape a document subject takes: a kind that renders what its
judge reads, and a judge that scores only what it was shown. Its scripted judge grades the output
against the case material it finds in the prompt, so every score below is a function of the
evidence the kind rendered and the engine placed — a kind that stopped rendering, or an engine that
stopped forwarding, changes the scores rather than passing.
"""

from __future__ import annotations

from threetears.evals.contracts.models import JudgedArtifact
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.judge import (
    FAITHFULNESS_DIM,
    TOY_JUDGE_MODEL,
    TOY_JUDGE_PRICE_SOURCE,
    ScriptedJudgeClient,
    toyhost_judge_service,
    toyhost_judged_template,
)
from packages.evals.tests.fixtures.toyhost.kind import INVOICE_FIELDS, TOY_DOCUMENTS, TOY_SCRIPTS, render_invoice
from packages.evals.tests.fixtures.toyhost.run import execute_toyhost_run


async def test_every_judged_toy_cell_is_scored_on_what_its_kind_rendered_and_keeps_that_evidence():
    client = ScriptedJudgeClient()
    path = await execute_toyhost_run(
        host=toyhost_host(),
        template=toyhost_judged_template(),
        judge_service=toyhost_judge_service(client),
        judge_model=TOY_JUDGE_MODEL,
    )

    results = path.results
    assert results, "the judged drive produced no cells"
    misses = {(script.model, doc): len(fields) for script in TOY_SCRIPTS for doc, fields in script.misses.items()}
    for result in results:
        assert result.judge_error is None and not result.judge_cannot_tell
        assert result.transcript_score is None and result.outcome_score is None, "a document has no conversation axes"
        (score,) = result.rubric_scores
        trace = path.trace(result)
        assert trace is not None and trace.judged_artifact is JudgedArtifact.DOCUMENT
        assert trace.judge_evidence is not None
        document_id = trace.trace[0]["document_id"]
        document = next(doc for doc in TOY_DOCUMENTS if doc.document_id == document_id)
        assert trace.judge_evidence.case_material == render_invoice(document)
        # Scored from what the judge was shown: the share of fields the extraction reproduced.
        matched = len(INVOICE_FIELDS) - misses[result.model, document_id]
        assert (score.dim, score.score) == (FAITHFULNESS_DIM, 1 + round(4 * matched / len(INVOICE_FIELDS)))
        # The judge's dollars are qualified by the source its client reported, and by nothing the
        # engine supplied: the engine names no provider of its own.
        (judge_row,) = [row for row in result.usage if row.role == "judge"]
        assert judge_row.cost_usd is not None and judge_row.price_source == TOY_JUDGE_PRICE_SOURCE
    assert len(client.calls) == len(results), "one rubric call per cell, and no conversation axis"
