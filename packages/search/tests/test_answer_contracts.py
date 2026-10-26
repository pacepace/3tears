"""The grounded-answer contracts (answers-requirements-design.md §9 A, B, §10.1).

What a consumer relies on before any provider exists: the types refuse what
they do not know, round-trip, canonicalise by the request rules, register in
the one capabilities registry under their own kind, and make an uncited answer
impossible to pass off as a complete one.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.search.contracts import (
    PRICING_PER_WEIGHTED_UNIT,
    Anchor,
    AnswerCapabilities,
    AnswerProvider,
    AnswerRequest,
    Criterion,
    GroundedAnswer,
    ProviderCapabilities,
    SearchProvider,
    SearchRequest,
    Spend,
    get_capabilities,
    list_capabilities,
    register_capabilities,
)
from packages.search.tests.search_instances import (
    ANCHOR,
    ANSWER_CAPABILITIES,
    ANSWER_PROVENANCE,
    ANSWER_REQUEST,
    CITED,
    GROUNDED_ANSWER,
)

# --- every type refuses an unknown field ----------------------------------


@pytest.mark.parametrize(
    "instance",
    [ANSWER_REQUEST, ANCHOR, GROUNDED_ANSWER, ANSWER_CAPABILITIES],
    ids=lambda instance: type(instance).__name__,
)
def test_an_unknown_field_is_refused_at_the_border(
    instance: AnswerRequest | Anchor | GroundedAnswer | AnswerCapabilities,
) -> None:
    """``extra="forbid"``, as every contract type: a shape mismatch fails here, not later."""
    payload = instance.model_dump(mode="json")
    payload["messages"] = [{"role": "user", "content": "smuggled history"}]
    with pytest.raises(ValidationError, match="messages"):
        type(instance).model_validate(payload)


@pytest.mark.parametrize(
    "instance",
    [ANSWER_REQUEST, ANCHOR, GROUNDED_ANSWER, ANSWER_CAPABILITIES],
    ids=lambda instance: type(instance).__name__,
)
def test_every_answer_type_round_trips(
    instance: AnswerRequest | Anchor | GroundedAnswer | AnswerCapabilities,
) -> None:
    """JSON out and back is the same value, Decimal money and aware timestamps included (SR-L4)."""
    assert type(instance).model_validate_json(instance.model_dump_json()) == instance


# --- the request canonicalises by the request rules (AR-I1) ----------------


def test_a_question_alone_canonicalises_to_the_question() -> None:
    """Unset fields never appear, so a field added later shifts no key."""
    assert AnswerRequest(question="where do capybaras live?").canonical_form() == (
        '{"question":"where do capybaras live?"}'
    )


def test_budget_scope_tags_never_key_an_answer() -> None:
    """Per-run identity (SR-D2) would give every eval run its own digest."""
    bare = AnswerRequest(question="q")
    tagged = AnswerRequest(question="q", budget_scope_tags=("run:1", "persona:capy"))
    assert tagged.canonical_digest() == bare.canonical_digest()


def test_criteria_order_does_not_change_the_answer_key() -> None:
    """Two requests differing only in criteria order are one question."""
    language = Criterion.language("en")
    country = Criterion.namespaced("brave", "country", "us")
    forward = AnswerRequest(question="q", criteria=(language, country))
    backward = AnswerRequest(question="q", criteria=(country, language))
    assert forward.canonical_form() == backward.canonical_form()


def test_different_questions_and_criteria_diverge() -> None:
    """The form distinguishes what actually differs."""
    assert AnswerRequest(question="q").canonical_digest() != AnswerRequest(question="r").canonical_digest()
    assert (
        AnswerRequest(question="q", criteria=(Criterion.language("en"),)).canonical_digest()
        != AnswerRequest(question="q").canonical_digest()
    )


def test_an_answer_key_is_not_a_search_key() -> None:
    """A question and a query with the same words are different requests, under different field names."""
    assert AnswerRequest(question="capybara").canonical_form() != SearchRequest(query="capybara").canonical_form()


def test_a_request_is_one_question_never_a_message_list() -> None:
    """AR-B1: the caller composes one string; no history field exists to fill."""
    assert set(AnswerRequest.model_fields) == {"question", "criteria", "budget_scope_tags"}


# --- an answer accounts for its sources (AR-A1, AR-A3, AR-A4) --------------


def test_zero_sources_is_a_valid_answer_when_it_says_so() -> None:
    """An uncited answer is a success value a consumer must be able to receive."""
    answer = GroundedAnswer(
        text="Capybaras are rodents.",
        provenance=ANSWER_PROVENANCE,
        notices=("the provider returned no citations",),
    )
    assert answer.sources == ()
    assert answer.anchors == ()
    assert GroundedAnswer.model_validate_json(answer.model_dump_json()) == answer


def test_zero_sources_and_nothing_said_about_it_is_refused() -> None:
    """AG2: fluent, uncited and silent is the defect, so it cannot be constructed."""
    with pytest.raises(ValidationError, match="notice"):
        GroundedAnswer(text="Capybaras are rodents.", provenance=ANSWER_PROVENANCE)


def test_sources_are_candidates_whose_content_is_unread() -> None:
    """AD7: a cited page is a candidate, page-read later by the consumer's own path."""
    assert GROUNDED_ANSWER.sources == (CITED,)
    assert all(source.content is None for source in GROUNDED_ANSWER.sources)


def test_an_anchor_must_cite_a_source_the_answer_has() -> None:
    """An anchor naming no source is a citation to nothing."""
    with pytest.raises(ValidationError, match="cites source 1"):
        GroundedAnswer(
            text="Capybaras live near water.",
            sources=(CITED,),
            anchors=(Anchor(start=0, end=4, source_index=1),),
            provenance=ANSWER_PROVENANCE,
        )


def test_an_anchor_past_the_end_of_the_text_is_refused() -> None:
    """An offset past the text points at nothing a reader can see: wrong units, not an anchor."""
    text = "Schrödinger won in 1933."
    point = len(text)
    assert (
        GroundedAnswer(
            text=text,
            sources=(CITED,),
            anchors=(Anchor(start=point, end=point, source_index=0),),
            provenance=ANSWER_PROVENANCE,
        )
        .anchors[0]
        .end
        == point
    ), "a point at the very end of the text is the last claim's"
    with pytest.raises(ValidationError, match="past the answer text"):
        GroundedAnswer(
            text=text,
            sources=(CITED,),
            anchors=(Anchor(start=point, end=len(text.encode()), source_index=0),),
            provenance=ANSWER_PROVENANCE,
        )


def test_an_anchor_span_cannot_run_backwards_or_negative() -> None:
    """A half-open span that ends before it begins, or before zero, is no span."""
    with pytest.raises(ValidationError, match="before it starts"):
        Anchor(start=10, end=4, source_index=0)
    with pytest.raises(ValidationError):
        Anchor(start=-1, end=4, source_index=0)
    with pytest.raises(ValidationError):
        Anchor(start=0, end=4, source_index=-1)


def test_token_counts_are_optional_and_never_negative() -> None:
    """Typed when the provider reported them, absent when it did not."""
    answer = GroundedAnswer(text="t", sources=(CITED,), provenance=ANSWER_PROVENANCE)
    assert answer.prompt_tokens is None and answer.completion_tokens is None
    with pytest.raises(ValidationError):
        GroundedAnswer(text="t", sources=(CITED,), provenance=ANSWER_PROVENANCE, prompt_tokens=-1)


# --- capabilities: one registry, two kinds (AR-B2, §10.1) -----------------


def test_an_answer_declaration_says_what_kind_it_is() -> None:
    """The discriminator is a literal on both declarations, so neither can pose as the other."""
    assert ANSWER_CAPABILITIES.kind == "answer"
    assert ProviderCapabilities(provider="p").kind == "search"
    with pytest.raises(ValidationError):
        AnswerCapabilities.model_validate({"provider": "p", "kind": "search"})


def test_single_mode_is_the_v1_declaration() -> None:
    """AD6: a consumer branches on the declared modes before sending."""
    assert AnswerCapabilities(provider="p").modes == ("single",)


def test_one_provider_can_declare_both_kinds_without_one_replacing_the_other() -> None:
    """Brave sells search and answers under one name; the registry keys on kind as well."""
    search = ProviderCapabilities(provider="dual-kind-provider", pushdown_criteria=("language",))
    answer = AnswerCapabilities(provider="dual-kind-provider", citations_in_modes=("single",))
    register_capabilities(search)
    register_capabilities(answer)

    assert get_capabilities("dual-kind-provider") == search, "no kind still answers the search declaration"
    assert get_capabilities("dual-kind-provider", "search") == search
    assert get_capabilities("dual-kind-provider", "answer") == answer
    assert list_capabilities("answer")["dual-kind-provider"] == answer
    assert list_capabilities()["dual-kind-provider"] == search
    assert all(declared.kind == "search" for declared in list_capabilities().values())
    assert all(declared.kind == "answer" for declared in list_capabilities("answer").values())


def test_an_unregistered_answer_provider_answers_none() -> None:
    """'No adapter imported' is not 'the provider can do nothing'."""
    assert get_capabilities("no-such-answers-provider", "answer") is None


def test_an_answer_declaration_answers_every_vocabulary() -> None:
    """Pushdown, namespaced, unsatisfiable and unknown keys each get an answer; nothing is local."""
    assert ANSWER_CAPABILITIES.disposition_for("language") == "pushdown"
    assert ANSWER_CAPABILITIES.disposition_for("brave:country") == "pushdown"
    assert ANSWER_CAPABILITIES.disposition_for("time-range") == "unsatisfied"
    assert ANSWER_CAPABILITIES.disposition_for("searxng:engines") == "ignored-unknown"
    assert "local_criteria" not in AnswerCapabilities.model_fields


def test_a_contradictory_answer_declaration_is_refused() -> None:
    """A criterion has one disposition, and citations come only in a declared mode."""
    with pytest.raises(ValidationError, match="more than one"):
        AnswerCapabilities(provider="p", pushdown_criteria=("language",), unsatisfiable_criteria=("language",))
    with pytest.raises(ValidationError, match="research"):
        AnswerCapabilities(provider="p", citations_in_modes=("research",))


def test_the_metered_unit_is_qualified_the_way_a_search_provider_s_is() -> None:
    """One composition of ``<provider>:<unit>``, so spends from both kinds sum."""
    assert ANSWER_CAPABILITIES.pricing_model == PRICING_PER_WEIGHTED_UNIT
    assert ANSWER_CAPABILITIES.qualified_unit == "brave:searches"
    assert ProviderCapabilities(provider="brave", metered_unit="searches").qualified_unit == "brave:searches"
    assert AnswerCapabilities(provider="brave").qualified_unit is None


# --- the provider seam is structural, and separate from search -------------


class FakeAnswerProvider(AnswerProvider):  # parity-with: threetears.search.contracts.AnswerProvider
    """An answers provider with one canned answer, satisfying the seam by shape."""

    @property
    def provider(self) -> str:
        """The product name.

        :return: ``brave``
        :rtype: str
        """
        return "brave"

    @property
    def provider_instance(self) -> str:
        """The configured deployment.

        :return: ``brave-answers``
        :rtype: str
        """
        return "brave-answers"

    @property
    def capabilities(self) -> AnswerCapabilities:
        """The declaration.

        :return: the shared fixture's
        :rtype: AnswerCapabilities
        """
        return ANSWER_CAPABILITIES

    def estimate(self, request: AnswerRequest) -> Spend:
        """One call, unpriced.

        :param request: ignored
        :ptype request: AnswerRequest
        :return: the floor estimate
        :rtype: Spend
        """
        return Spend(calls=1)

    async def answer(self, request: AnswerRequest, *, timeout_seconds: float | None = None) -> GroundedAnswer:
        """Answer every question with the shared fixture.

        :param request: ignored
        :ptype request: AnswerRequest
        :param timeout_seconds: ignored
        :ptype timeout_seconds: float | None
        :return: the canned answer
        :rtype: GroundedAnswer
        """
        return GROUNDED_ANSWER


async def test_the_answer_seam_is_satisfiable_by_shape_and_is_not_the_search_seam() -> None:
    """An answers provider is its own protocol: it is not a search provider, and needs no base class."""
    provider = FakeAnswerProvider()
    assert isinstance(provider, AnswerProvider)
    assert not isinstance(provider, SearchProvider)
    answer = await provider.answer(ANSWER_REQUEST)
    assert answer.provenance.provider_instance == provider.provider_instance


def test_the_provider_seam_reserves_stream_and_does_not_declare_it() -> None:
    """AD12: v1 returns a whole answer; a streaming face is added later, by name."""
    assert not hasattr(AnswerProvider, "stream")
