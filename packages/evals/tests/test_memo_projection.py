"""Where each declared question stands, and where the authored document lives.

The generator authors one document (:class:`~threetears.evals.kernel.authored.AuthoredAnalysis`)
and the stored analysis keeps it verbatim. These pin the two projection rules code still holds
it to: one answer per LIVE declared question, no more and no fewer, and the document as the only
carrier of what was authored. The position links inside the document (``rests_on``,
``invalidates``) are the stored model's own rule and are pinned beside it in
``test_analysis_models.py``.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.errors import SoundnessRefusal
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, memo_payload
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.kernel.authored import AuthoredAnalysis, Chart, Finding, QuestionAnswer
from threetears.evals.kernel.campaign import EvalAnalysis, GenerationProvenance
from threetears.evals.kernel.declaration import Question
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.kernel.surface import DecisionSurface


def _finding() -> Finding:
    """A schema-valid authored finding."""
    return Finding(
        title="rev2 holds tone at k=3.",
        body="",
        confidence="high",
        axes=["candidate_model"],
        evidence=[],
        chart=Chart(type="none", cells=[], measures=[], axis="", note="", caption=""),
        caveats=[],
        invalidates=[],
        durable="",
    )


def _answer(question_id: str = "q-1", resolution: str = "answered") -> QuestionAnswer:
    return QuestionAnswer(question_id=question_id, resolution=resolution, answer="The newer build.", rests_on=[0])


def _document(questions: list[QuestionAnswer] | None = None) -> AuthoredAnalysis:
    return AuthoredAnalysis(
        headline="Move to the newer build.",
        summary="",
        findings=[_finding()],
        decisions=[],
        questions=questions or [],
        next=[],
    )


def _generation() -> GenerationProvenance:
    return GenerationProvenance(
        prompt_id="eval_analysis_gen",
        prompt_version="v1",
        generator_model="anthropic/claude-opus",
        bundle_fingerprint="sha256:abc",
        generated_at="2026-08-22T00:00:00+00:00",
        token_cost=0.0,
        bundle_assembled_at="2026-01-01T00:00:00+00:00",
        repair_attempts=0,
        repaired_refusal=None,
        cell_model_version=1,
        user_message_digest="sha256:message",
    )


def _analysis(document: AuthoredAnalysis | None = None) -> EvalAnalysis:
    return EvalAnalysis(
        scope_id="uni-1",
        campaign_id="campaign-1",
        subject_id="ent-maple",
        subject_kind="agent",
        behavior="conversation",
        generation=_generation(),
        document=document or _document(),
        decision_surface=DecisionSurface(),
    )


class TestEachLiveQuestionIsAnsweredExactlyOnce:
    """'Exactly when', as a check rather than a convention — one answer per live question.

    Driven through :func:`~threetears.evals.analysis.generator.generate_analysis` over the toy host's
    campaign, its declaration's questions replaced per case: the check is the one a finished
    generation runs, and a refusal is the one it raises after its one repair. The toy memo's
    finding, decision and next step are kept, so a case differs from a clean generation only in its
    question answers.
    """

    @staticmethod
    async def _reject(questions: list[QuestionAnswer], questions_declared: list[Question] | None) -> None:
        """Generate over the toy campaign declaring ``questions_declared`` (``None``: no declaration at all)."""
        profile = toyhost_profile()
        bundle = toyhost_bundle(profile=profile)
        assert bundle.declared_design is not None, (
            "the toy campaign declares a design; this check replaces its questions"
        )
        design = (
            None
            if questions_declared is None
            else bundle.declared_design.model_copy(update={"questions": questions_declared})
        )
        bundle = bundle.model_copy(update={"declared_design": design})
        payload = memo_payload(bundle)
        payload["questions"] = [answer.model_dump(mode="json") for answer in questions]
        profile = toyhost_profile()
        await generate_analysis(
            bundle,
            prompt=PROMPT,
            model=MODEL,
            client=FixturedClient(json.dumps(payload)),
            prompt_id=PROMPT_ID,
            bundle_assembled_at=utc_now_iso(),
            profile=profile,
        )

    async def test_one_answer_per_live_question_passes(self):
        declared = [Question(id="q-1", text="Which model?"), Question(id="q-2", text="At what cost?")]
        await self._reject([_answer("q-2", "partial"), _answer("q-1")], declared)

    async def test_a_declared_question_with_no_answer_is_refused(self):
        declared = [Question(id="q-1", text="Which model?")]
        with pytest.raises(SoundnessRefusal, match=r"live declared question\(s\) \['q-1'\] have no answer"):
            await self._reject([], declared)

    async def test_an_answer_with_no_declared_question_is_refused(self):
        with pytest.raises(SoundnessRefusal, match="it declares none, so `questions` is empty"):
            await self._reject([_answer()], [])

    async def test_a_passive_campaign_answers_nothing(self):
        await self._reject([], [])

    async def test_a_campaign_with_no_declaration_at_all_is_passive_here(self):
        """No declaration is no declared question, which is the same obligation as declaring none."""
        await self._reject([], None)
        with pytest.raises(SoundnessRefusal, match="does not declare as live"):
            await self._reject([_answer()], None)

    async def test_a_question_answered_twice_is_refused(self):
        declared = [Question(id="q-1", text="Which model?")]
        with pytest.raises(SoundnessRefusal, match=r"question\(s\) \['q-1'\] are answered more than once"):
            await self._reject([_answer(), _answer(resolution="partial")], declared)

    async def test_an_answer_naming_an_undeclared_question_is_refused_and_names_the_live_ones(self):
        declared = [Question(id="q-1", text="Which model?")]
        with pytest.raises(SoundnessRefusal, match=r"answers \['q-ghost'\].*the live questions are \['q-1'\]"):
            await self._reject([_answer(), _answer("q-ghost")], declared)

    async def test_a_retired_question_stops_requiring_an_answer(self):
        declared = [Question(id="q-1", text="Which model?", retired_at="2026-08-01T00:00:00+00:00")]
        await self._reject([], declared)

    async def test_a_retired_question_may_not_be_answered_either(self):
        """Retired is not live, so an answer to it has nothing to attach to."""
        declared = [Question(id="q-1", text="Which model?", retired_at="2026-08-01T00:00:00+00:00")]
        with pytest.raises(SoundnessRefusal, match="does not declare as live"):
            await self._reject([_answer()], declared)


class TestTheDocumentIsTheOnlyCarrier:
    """One home for what the generator authored, so nothing can hold a second copy."""

    def test_an_analysis_without_a_document_does_not_construct(self):
        """An analysis with nothing to read is unrepresentable."""
        with pytest.raises(ValidationError, match="document"):
            EvalAnalysis(
                scope_id="uni-1",
                campaign_id="campaign-1",
                subject_id="ent-maple",
                subject_kind="agent",
                behavior="conversation",
                generation=_generation(),
                decision_surface=DecisionSurface(),
            )

    @pytest.mark.parametrize("field", ["memo", "findings", "methodology", "methods_appendix", "bluf", "decisions"])
    def test_the_pre_document_fields_are_gone_from_the_model(self, field):
        """A ratchet, not a restatement — the removal is total only while nothing re-adds them."""
        assert field not in EvalAnalysis.model_fields

    def test_a_stored_top_level_copy_is_refused_rather_than_carried(self):
        """A document written with an old top-level field does not load at all.

        An old copy cannot return through storage and sit beside the document's own — the two-homes
        state the document exists to make unreachable — because a stored read refuses an unknown key.
        """
        raw = _analysis().to_dict()
        raw["memo"] = {"bluf": {"headline": "Something else entirely."}}
        with pytest.raises(ValidationError, match="memo"):
            EvalAnalysis.from_dict(raw)
