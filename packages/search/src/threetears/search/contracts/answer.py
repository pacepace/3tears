"""Grounded answers -- a provider's cited synthesis, carried and never produced.

``search-requirements.md`` §2 (as amended 2026-10-22) keeps the rule that this
package summarises, judges and concludes nothing, and permits one thing beside
it: carrying a provider's synthesis when the provider delivers it with
citations -- the text as given, the citations as candidates, the spend as spend
(``answers-requirements-design.md`` AD2). These are the types that carry it.

The shape follows from one ruling (AD7): **the sources are candidates**
(:class:`~threetears.search.contracts.candidate.Candidate`), and only the
answer text is new. So everything the family already does with a candidate --
page reads, grounding, provenance, spend, the failure taxonomy -- applies to an
answer's sources unchanged, and a consumer grounding against them treats a
citation's snippet exactly as it treats a SERP snippet (AR-G1). The text is the
provider's prose and is never grounding text.

What deliberately is not here:

- **no answer-specific failure type** (AD9). A provider answering raises the
  :class:`~threetears.search.contracts.errors.SearchFailure` taxonomy, so a
  consumer's one mapping of it serves both;
- **no mode field** (AD6). v1 is single mode; :attr:`AnswerCapabilities.modes`
  declares ``("single",)`` so a consumer can branch before sending, and a
  ``mode`` field joins :class:`AnswerRequest` additively when a second mode is
  carried;
- **no streaming method on the provider** (AD12). Every named consumer is a
  tool and wants a whole answer; ``AnswerProvider.stream`` is the name reserved
  for a later face that streams an answer to a person, and is not declared.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Protocol, runtime_checkable

from pydantic import Field, model_validator

from threetears.search.contracts._base import ContractModel
from threetears.search.contracts._canonical import canonical_digest, canonicalize
from threetears.search.contracts.candidate import Candidate
from threetears.search.contracts.capabilities import qualify_unit
from threetears.search.contracts.criteria import Criterion, CriterionDisposition, Disposition
from threetears.search.contracts.provenance import Provenance
from threetears.search.contracts.spend import Spend

__all__ = [
    "Anchor",
    "AnswerCapabilities",
    "AnswerProvider",
    "AnswerRequest",
    "GroundedAnswer",
]


class AnswerRequest(ContractModel):
    """One question, as the caller composes it (AR-B1).

    A single string, never a message list: the providers this is shaped for
    take exactly one user message, and a ``messages`` field would invite a
    history they would ignore or reject. The question is untrusted user
    content (SR-K2, AR-K1) -- an adapter sends it as that one message and
    wraps it in nothing.

    Canonical serialization is exposed here as on
    :class:`~threetears.search.contracts.request.SearchRequest`, by the same
    rules (D26, SR-F1, AR-I1): explicitly-set semantic parameters only,
    criteria order irrelevant, and ``budget_scope_tags`` -- per-run identity
    (SR-D2) -- never participating.
    """

    CANONICAL_ORDER_INSENSITIVE: ClassVar[frozenset[str]] = frozenset({"criteria"})
    CANONICAL_EXCLUDED: ClassVar[frozenset[str]] = frozenset({"budget_scope_tags"})

    #: the question, composed by the caller. User content (D11).
    question: str
    #: constraints on the answer, in the search criteria vocabulary where it
    #: overlaps (language) and namespaced keys for the provider's own
    #: parameters (``brave:country``). One disposition is owed per criterion
    #: (SR-B2, SR-B3).
    criteria: tuple[Criterion, ...] = ()
    #: which budget scopes this call debits (SR-D2), as on ``SearchRequest``.
    budget_scope_tags: tuple[str, ...] = ()

    def canonical_form(self) -> str:
        """Render this request in canonical form (D26, SR-F1).

        :return: canonical JSON text
        :rtype: str
        """
        return canonicalize(self)

    def canonical_digest(self) -> str:
        """Digest this request's canonical form for equality keying.

        :return: hex SHA-256 of :meth:`canonical_form`
        :rtype: str
        """
        return canonical_digest(self)


class Anchor(ContractModel):
    """Where in the answer text one source is cited (AR-A1, AR-A4).

    **Provisional: the units of** ``start`` **and** ``end`` **are not yet
    fixed.** Whether a provider's offsets count characters or bytes, and
    whether they index the text before or after its citation tags are removed,
    is settled by the live capture in ``answers-requirements-design.md`` §12
    slice 0, and this field's meaning is finalised from that capture, not
    before. Until then the only promise is the shape: a half-open span
    ``[start, end)`` into :attr:`GroundedAnswer.text` naming one source.

    An anchor is never inferred by searching the text for the cited words
    (AR-A4): an answer whose provider gave no usable offsets carries no
    anchors and a notice saying so.
    """

    #: where the cited span begins. Units provisional -- see the class.
    start: int = Field(ge=0)
    #: where the cited span ends, exclusive. Units provisional -- see the class.
    end: int = Field(ge=0)
    #: index into :attr:`GroundedAnswer.sources` of the source cited here.
    source_index: int = Field(ge=0)

    @model_validator(mode="after")
    def _span_is_ordered(self) -> Anchor:
        """Refuse a span that ends before it begins.

        :return: the validated anchor
        :rtype: Anchor
        :raises ValueError: when ``end`` is before ``start``
        """
        if self.end < self.start:
            raise ValueError(f"an anchor's span ends at {self.end}, before it starts at {self.start}")
        return self


class GroundedAnswer(ContractModel):
    """One provider's cited answer, as the provider gave it (AR-A1).

    ``text`` is the provider's synthesis with every tag removed, never edited
    further. ``sources`` are the cited pages as candidates, in citation order,
    each with ``content=None`` so Extract reads the page before anything is
    grounded against it (AD7, AR-A2, AR-G1).

    **Zero sources is a success, never a silent one** (AR-A3, AG2). An answer
    that cites nothing is a value a consumer must be able to receive -- and a
    fluent, uncited paragraph that reads as complete is the defect P8 exists to
    prevent. So a sourceless answer must carry at least one notice: the
    contract refuses to construct one that says nothing about its emptiness.
    """

    #: the provider's answer text, with every tag removed and nothing else
    #: changed. Provider prose: never grounding text (AR-G1).
    text: str
    #: the cited sources, in citation order (AD7, AR-A2).
    sources: tuple[Candidate, ...] = ()
    #: where in ``text`` each source is cited; empty when the provider gave no
    #: usable offsets (AR-A4). Units provisional -- see :class:`Anchor`.
    anchors: tuple[Anchor, ...] = ()
    #: the answer's own origin (SR-A3): provider instance, the question as
    #: ``query``, retrieval time, egress, producer class. Each source carries
    #: its own as well.
    provenance: Provenance
    #: what the call consumed (SR-E1), read from the provider's own report
    #: (AD10, D4).
    spend: Spend = Field(default_factory=Spend)
    #: degradations and absences the reader must see -- no sources, no
    #: anchors, an unknown tag, a bill the provider did not state.
    notices: tuple[str, ...] = ()
    #: per-criterion answers (SR-B2, SR-B3).
    dispositions: tuple[CriterionDisposition, ...] = ()
    #: prompt tokens the provider reported, when it reported them. Typed on the
    #: answer rather than folded into spend (AD10).
    prompt_tokens: int | None = Field(default=None, ge=0)
    #: completion tokens the provider reported, when it reported them.
    completion_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _the_answer_accounts_for_its_sources(self) -> GroundedAnswer:
        """Refuse an uncited answer that says nothing, and an anchor citing nothing.

        :return: the validated answer
        :rtype: GroundedAnswer
        :raises ValueError: when there are no sources and no notice, or an
            anchor's ``source_index`` names no source
        """
        if not self.sources and not self.notices:
            raise ValueError(
                "an answer with no sources must carry a notice saying so (AR-A3): an uncited answer that "
                "reads as complete is the silent partial answer"
            )
        for anchor in self.anchors:
            if anchor.source_index >= len(self.sources):
                raise ValueError(
                    f"an anchor cites source {anchor.source_index}, but the answer has {len(self.sources)} source(s)"
                )
        return self


class AnswerCapabilities(ContractModel):
    """What one answers provider can express, declared before contact (AR-B2, SR-B4).

    Registered through the same registry as
    :class:`~threetears.search.contracts.capabilities.ProviderCapabilities`,
    discriminated by :attr:`kind`, so ``get_capabilities(provider, "answer")``
    is the one call a consumer makes.

    There is no ``local_criteria`` bucket: a synthesised paragraph cannot be
    filtered after it arrives, so a criterion is either pushed down to the
    provider or declared unsatisfiable.
    """

    #: product name (``brave``). Not an instance name.
    provider: str
    #: which kind of declaration this is (the registry's discriminator).
    kind: Literal["answer"] = "answer"
    #: the answer modes this provider carries. v1 carries single mode only
    #: (AD6); a consumer branches on this before sending.
    modes: tuple[str, ...] = ("single",)
    #: the modes in which the provider returns citations. A mode outside this
    #: tuple answers without sources, and the consumer can know so in advance.
    citations_in_modes: tuple[str, ...] = ()
    #: well-known criterion keys the provider's API expresses.
    pushdown_criteria: tuple[str, ...] = ()
    #: well-known criterion keys this provider cannot honour. Named, never
    #: silently dropped (SR-B3).
    unsatisfiable_criteria: tuple[str, ...] = ()
    #: ``<namespace>:<name>`` keys the adapter accepts for the provider's own
    #: parameters (``brave:country``).
    namespaced_parameters: tuple[str, ...] = ()
    #: how this provider charges, in the capabilities module's pricing
    #: vocabulary (``per-weighted-unit`` for a provider metering searches).
    pricing_model: str | None = None
    #: the bare name of the unit a weighted provider meters (``searches``).
    metered_unit: str | None = None
    #: the shortest timeout the provider says an answer needs, in seconds. A
    #: caller bounding a call below it is choosing to fail.
    minimum_timeout_seconds: float | None = Field(default=None, gt=0)

    @property
    def qualified_unit(self) -> str | None:
        """This provider's metered unit, qualified for :attr:`Spend.provider_unit`.

        :return: ``"<provider>:<metered_unit>"``, or ``None`` when no unit is metered
        :rtype: str | None
        """
        return qualify_unit(self.provider, self.metered_unit)

    @model_validator(mode="after")
    def _declarations_do_not_contradict(self) -> AnswerCapabilities:
        """Refuse a key under two dispositions, and citations in an undeclared mode.

        :return: the validated capabilities
        :rtype: AnswerCapabilities
        :raises ValueError: when a key is both pushed down and unsatisfiable,
            or a citing mode is not one of :attr:`modes`
        """
        clashes = sorted(set(self.pushdown_criteria) & set(self.unsatisfiable_criteria))
        if clashes:
            raise ValueError(
                f"provider {self.provider!r} declares {clashes} under more than one disposition; "
                f"a criterion has exactly one"
            )
        undeclared = sorted(set(self.citations_in_modes) - set(self.modes))
        if undeclared:
            raise ValueError(f"provider {self.provider!r} cites in modes {undeclared} it does not declare")
        return self

    def disposition_for(self, criterion_key: str) -> Disposition:
        """Answer how this provider would handle ``criterion_key`` (SR-B4).

        :param criterion_key: a well-known or ``<namespace>:<name>`` key
        :ptype criterion_key: str
        :return: the declared disposition; ``ignored-unknown`` for anything
            undeclared
        :rtype: Disposition
        """
        if criterion_key in self.pushdown_criteria or criterion_key in self.namespaced_parameters:
            return "pushdown"
        if criterion_key in self.unsatisfiable_criteria:
            return "unsatisfied"
        return "ignored-unknown"


@runtime_checkable
class AnswerProvider(Protocol):
    """One answers provider's API, behind one method (§10.1).

    What an implementation owes, as :class:`~threetears.search.contracts.provider.SearchProvider`
    owes it: capabilities before contact, one disposition per criterion, spend
    on every outcome, the :class:`~threetears.search.contracts.errors.SearchFailure`
    taxonomy and nothing else (AD9), and host-supplied configuration only
    (D21, SR-K1). And one obligation of its own: zero sources is a success
    that says so (AR-A3).
    """

    @property
    def provider(self) -> str:
        """Product name of the provider this implementation speaks to.

        :return: the product name -- the key its capabilities are registered under
        :rtype: str
        """
        ...

    @property
    def provider_instance(self) -> str:
        """Name of the configured deployment this implementation reaches.

        :return: the instance name, which pacing and provenance key on (D8, SR-A3)
        :rtype: str
        """
        ...

    @property
    def capabilities(self) -> AnswerCapabilities:
        """Declare what this provider can express, before anything is sent.

        :return: the provider's own declaration (SR-B4)
        :rtype: AnswerCapabilities
        """
        ...

    async def answer(self, request: AnswerRequest, *, timeout_seconds: float | None = None) -> GroundedAnswer:
        """Answer one question, with its sources.

        :param request: what the caller asked
        :ptype request: AnswerRequest
        :param timeout_seconds: bound for this call (SR-G2); None leaves the
            provider's configured bound in force (SR-G1)
        :ptype timeout_seconds: float | None
        :return: the whole answer -- text, sources, anchors, spend, notices
        :rtype: GroundedAnswer
        :raises threetears.search.contracts.errors.SearchFailure: one of the
            typed classes, carrying the spend consumed before the failure
            (SR-E3, AR-E1)
        """
        ...
