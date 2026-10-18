"""Where the campaign's question stands, and where the authored document lives.

The generator authors one document per campaign (:class:`~threetears.evals.kernel.authored.AuthoredCampaignAnalysis`)
and the stored analysis keeps it verbatim. These pin the two projection rules code still holds it to: the question is
answered by the one recommendation and one summary per aspect — never a list of answers keyed by question — and the
document as the only carrier of what was authored. That each aspect is summarised exactly once is pinned in
``test_memo_refusals.py``; the position links inside the document (``rests_on``, ``invalidates``) are the stored
model's own rule and are pinned beside it in ``test_analysis_models.py``.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.errors import SoundnessRefusal
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, memo_payload
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.schema.models import utc_now_iso
from packages.evals.tests.factories import make_analysis


class TestTheQuestionIsAnsweredByTheRecommendation:
    """A campaign asks one question, and the memo answers it with one recommendation and a summary per aspect."""

    @staticmethod
    async def _generate(question: str | None, **extra: object) -> EvalAnalysis:
        profile = toyhost_profile()
        aspect = toyhost_bundle(profile=profile)
        assert aspect.declared is not None, "the toy campaign declares a design; this check replaces its question"
        declared = None if question is None else aspect.declared.model_copy(update={"question": question})
        bundle = campaign_bundle(aspect.model_copy(update={"declared": declared}), question=question or "")
        payload = {**memo_payload(bundle), **extra}
        analysis, _ = await generate_analysis(
            bundle,
            prompt=PROMPT,
            model=MODEL,
            client=FixturedClient(json.dumps(payload)),
            prompt_id=PROMPT_ID,
            bundle_assembled_at=utc_now_iso(),
            profile=profile,
        )
        return analysis

    @pytest.mark.parametrize("question", ["Which chunk width?", "", None], ids=["asked", "asks-none", "no-design"])
    async def test_whatever_the_campaign_asks_the_memo_recommends_one_arm_and_summarises_each_aspect(
        self, question: str | None
    ) -> None:
        analysis = await self._generate(question)
        assert analysis.document.recommendation[0].arm
        assert [summary.aspect for summary in analysis.document.aspect_summaries] == [
            aspect.label for aspect in analysis.aspects
        ]
        assert analysis.design_snapshot.question == (question or "")

    async def test_a_list_of_answers_keyed_by_question_is_refused_as_off_contract(self) -> None:
        answers = [{"question_id": "q-1", "resolution": "answered", "answer": "Narrow.", "rests_on": [0]}]
        with pytest.raises(SoundnessRefusal, match="questions"):
            await self._generate("Which chunk width?", questions=answers)


class TestTheDocumentIsTheOnlyCarrier:
    """One home for what the generator authored, so nothing can hold a second copy."""

    def test_an_analysis_without_a_document_does_not_construct(self):
        """An analysis with nothing to read is unrepresentable."""
        fields = make_analysis().model_dump()
        del fields["document"]
        with pytest.raises(ValidationError, match="document"):
            EvalAnalysis(**fields)

    @pytest.mark.parametrize(
        "field", ["memo", "findings", "methodology", "methods_appendix", "bluf", "decisions", "questions", "bundle"]
    )
    def test_the_pre_document_fields_are_gone_from_the_model(self, field):
        """A ratchet, not a restatement — the removal is total only while nothing re-adds them."""
        assert field not in EvalAnalysis.model_fields

    def test_a_stored_top_level_copy_is_refused_rather_than_carried(self):
        """A document written with an old top-level field does not load at all.

        An old copy cannot return through storage and sit beside the document's own — the two-homes
        state the document exists to make unreachable — because a stored read refuses an unknown key.
        """
        raw = make_analysis().to_dict()
        raw["memo"] = {"bluf": {"title": "Something else entirely."}}
        with pytest.raises(ValidationError, match="memo"):
            EvalAnalysis.from_dict(raw)
