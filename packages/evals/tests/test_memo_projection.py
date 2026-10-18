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
from threetears.evals.kernel.authored import AuthoredAnalysis, Finding, QuestionAnswer
from threetears.evals.kernel.campaign import EvalAnalysis, GenerationProvenance
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.kernel.surface import DecisionSurface
from packages.evals.tests.factories import CURRENT_BUNDLE_SHAPE, empty_bundle


def _finding() -> Finding:
    """A schema-valid authored finding."""
    return Finding(
        title="rev2 holds tone at k=3.",
        body="",
        confidence="high",
        axes=["candidate_model"],
        evidence=[],
        trials=[],
        caveats=[],
        invalidates=[],
        durable="",
    )


def _answer(question_id: str = "q-1", resolution: str = "answered") -> QuestionAnswer:
    return QuestionAnswer(question_id=question_id, resolution=resolution, answer="The newer build.", rests_on=[0])


def _document(questions: list[QuestionAnswer] | None = None) -> AuthoredAnalysis:
    return AuthoredAnalysis(
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
        bundle_schema_version=CURRENT_BUNDLE_SHAPE,
        host_declarations_digest="sha256:declarations",
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
        bundle=empty_bundle().to_dict(),
    )


class TestEachLiveQuestionIsAnsweredExactlyOnce:
    """'Exactly when', as a check rather than a convention — one answer to the aspect's question, keyed by its label.

    Driven through :func:`~threetears.evals.analysis.generator.generate_analysis` over the toy host's
    campaign, its declared question replaced per case: the check is the one a finished generation runs, and a
    refusal is the one it raises after its one repair. A bundle reads one aspect, and the campaign asks one
    question, so the one answer names the aspect's label. The toy memo's finding, decision and next step are kept,
    so a case differs from a clean generation only in its question answers.
    """

    @staticmethod
    async def _reject(questions: list[QuestionAnswer], question: str | None) -> str:
        """Generate over the toy campaign asking ``question`` (``None``: no declaration at all); the aspect's label."""
        profile = toyhost_profile()
        bundle = toyhost_bundle(profile=profile)
        assert bundle.declared is not None, "the toy campaign declares a design; this check replaces its question"
        declared = None if question is None else bundle.declared.model_copy(update={"question": question})
        bundle = bundle.model_copy(update={"declared": declared})
        payload = memo_payload(bundle)
        payload["questions"] = [answer.model_dump(mode="json") for answer in questions]
        await generate_analysis(
            bundle,
            prompt=PROMPT,
            model=MODEL,
            client=FixturedClient(json.dumps(payload)),
            prompt_id=PROMPT_ID,
            bundle_assembled_at=utc_now_iso(),
            profile=profile,
        )
        return bundle.aspect

    @staticmethod
    def _aspect() -> str:
        return toyhost_bundle(profile=toyhost_profile()).aspect

    async def test_one_answer_to_the_question_passes(self):
        await self._reject([_answer(self._aspect(), "partial")], "Which model?")

    async def test_a_declared_question_with_no_answer_is_refused(self):
        aspect = self._aspect()
        with pytest.raises(SoundnessRefusal, match=rf"live declared question\(s\) \['{aspect}'\] have no answer"):
            await self._reject([], "Which model?")

    async def test_an_answer_with_no_declared_question_is_refused(self):
        with pytest.raises(SoundnessRefusal, match="it declares none, so `questions` is empty"):
            await self._reject([_answer(self._aspect())], "")

    async def test_a_passive_campaign_answers_nothing(self):
        await self._reject([], "")

    async def test_a_campaign_with_no_declaration_at_all_is_passive_here(self):
        """No declaration is no declared question, which is the same obligation as declaring none."""
        await self._reject([], None)
        with pytest.raises(SoundnessRefusal, match="does not declare as live"):
            await self._reject([_answer(self._aspect())], None)

    async def test_a_question_answered_twice_is_refused(self):
        aspect = self._aspect()
        with pytest.raises(SoundnessRefusal, match=rf"question\(s\) \['{aspect}'\] are answered more than once"):
            await self._reject([_answer(aspect), _answer(aspect, resolution="partial")], "Which model?")

    async def test_an_answer_naming_another_question_is_refused_and_names_the_aspect(self):
        aspect = self._aspect()
        with pytest.raises(SoundnessRefusal, match=rf"answers \['q-ghost'\].*the live questions are \['{aspect}'\]"):
            await self._reject([_answer(aspect), _answer("q-ghost")], "Which model?")


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
                bundle=empty_bundle().to_dict(),
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
        raw["memo"] = {"bluf": {"title": "Something else entirely."}}
        with pytest.raises(ValidationError, match="memo"):
            EvalAnalysis.from_dict(raw)
