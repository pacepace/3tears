"""``Answer``: a candidate's answer, with what producing it spent.

An :class:`~threetears.evals.quick.contract.Eval`'s runner is an opaque callable, so the engine sees what it
returns and nothing of what it spent: a candidate that calls a paid model reports a ``cost_usd`` of 0 unless
it says otherwise. Returning an :class:`Answer` is how it says so. Its ``value`` is what the graders, the
classifier's label field and the judge grade, exactly as a plain return value would be; its spend becomes
the cell's ``candidate`` usage row, which the engine derives the result's ``cost_usd`` from as it does for
any kind (:attr:`~threetears.evals.kernel.CandidateTelemetry.usage`). A candidate that returns anything
else reports no spend, as before.

The spend fields carry the names a completion client's reply carries, so the usage ledger reads an
``Answer`` as it reads one: **missing is not zero** — a token count or a price left ``None`` is unreported,
and an unpriced answer makes its result's cost unknown rather than quietly smaller.

An ``Answer`` also says what the candidate read: its ``documents`` (:class:`Document`), the passages it was given or
the pages it fetched. They travel beside the answer, never in ``value``, so no grader or judge reads them as part of
it; a :class:`~threetears.evals.quick.claims.Claims` grader declaring ``grounded_in="run"`` reads them to tell a
claim the candidate's own sources support from one it made without. Missing is not zero here either: ``None`` says
nothing, and ``()`` says it read none.

An ``Answer`` may also carry the run's own record (``session``): what the candidate did beside its answer, such as
an order log or a play order. It is stored with the cell's claim evidence for a ``Claims`` grader's readings of your
own, and like ``documents`` it never joins ``value``, so no grader, judge or prompt reads it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals.kernel import CandidateTelemetry, RoleUsageLedger


@dataclass(frozen=True)
class Document:
    """A document claims are checked against: one a candidate read while answering, or one a case holds.

    :ivar text: Its text, as read: every quote a grader cites from it is checked against this, verbatim.
    :ivar title: What it is called; ``None`` names it by its address, else by its place in the list.
    :ivar url: Where it was read, when it has an address.
    :ivar id: Provisional. What the case calls it, kept as its stored id, so another case document can override it by
        name; ``None`` numbers it by its place (``case-1``). Ids are the case's to keep unique among its documents.
    :ivar overrides: Provisional. The case documents this one overrides, by id: where it and one of them bear on a
        claim with opposite stances, its own stance decides (house rules over a rulebook, an erratum over its text, a
        policy update over the old policy), so support quoted only from an overridden document that this one
        contradicts reads contradicted. Only a case's documents override; a candidate's own never do.
    """

    text: str
    title: str | None = None
    url: str | None = None
    id: str | None = None
    overrides: Sequence[str] = ()

    def __post_init__(self) -> None:
        """Refuse a document no claim could be checked against.

        :raises ValueError: Text that is not a string or is blank; a title, address or id that is blank; or
            overrides that are not a list of non-blank ids, name one id twice, or name the document's own.
        """
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(
                f"Document.text is the text a claim is checked against, a non-blank string, not {self.text!r}"
            )
        for name in ("title", "url"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Document.{name} is a non-blank string or None, not {value!r}")
        if self.id is not None and (not isinstance(self.id, str) or not self.id.strip()):
            raise ValueError(f"Document.id is a non-blank string or None, not {self.id!r}")
        overrides = self.overrides
        if isinstance(overrides, str | bytes) or not isinstance(overrides, Sequence):
            raise ValueError(f"Document.overrides is a list of the ids it overrides, not {overrides!r}")
        if not all(isinstance(other, str) and other.strip() for other in overrides):
            raise ValueError(
                f"Document.overrides names each document it overrides by a non-blank id, not {overrides!r}"
            )
        if len(set(overrides)) != len(overrides):
            raise ValueError(f"Document.overrides names an id more than once: {list(overrides)!r}")
        if self.id is not None and self.id in overrides:
            raise ValueError(f"Document {self.id!r} overrides itself")
        # Held as a tuple, so what the run stores is what was given.
        object.__setattr__(self, "overrides", tuple(overrides))


@dataclass(frozen=True)
class Answer:
    """What a candidate returns to report its own spend beside its answer.

    :ivar value: The answer itself: what is graded, and what the run stores as the cell's output.
    :ivar model: The model the call went to, which the usage row is attributed to; ``None`` when there is none.
    :ivar input_tokens: Prompt tokens; ``None`` when unreported.
    :ivar output_tokens: Completion tokens, any thinking included; ``None`` when unreported.
    :ivar cost_usd: What the call cost in US dollars, as you priced it; ``None`` when unpriced, which the
        engine records as unknown and never as $0.
    :ivar served_model: The model the provider's RESPONSE named as having answered (a completion's
        ``served_model``), never the one requested. ``None`` when the response named none: the run then records
        only the id asked for, and its summary says which model answered is unconfirmed, since a provider may
        answer an id with a model of its choosing.
    :ivar documents: Provisional, as :class:`Document` is: its type may change in any release. What the candidate
        read while answering: each passage it was given or page it fetched, a :class:`Document` or its text. ``()`` says it read none; ``None`` (the default) says nothing, which a
        ``Claims`` grader reading the candidate's own documents (``grounded_in="run"``) refuses, since no read
        could tell it from none.
    :ivar session: Provisional, as ``documents`` is. The run's own record beside the answer (an order log, a play
        order), a JSON object: stored as given with the cell's claim evidence (``ClaimEvidence.session``) for the
        readings of your own a ``Claims`` grader declares, and read by no grader, judge or prompt. ``None`` (the
        default): no record.
    """

    value: Any
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    served_model: str | None = None
    documents: Sequence[Document | str] | None = None
    session: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Refuse a spend no usage row can hold, so the candidate that built it fails rather than the run.

        :raises ValueError: A negative or non-integer token count, a negative or non-finite cost, or documents that are
            not a list of :class:`Document` values or non-blank strings, or a session record that is not a mapping.
        """
        if self.session is not None and not isinstance(self.session, Mapping):
            raise ValueError(
                f"Answer.session is the run's own record, a JSON object (a mapping), not {type(self.session).__name__}"
            )
        for name in ("input_tokens", "output_tokens"):
            count = getattr(self, name)
            if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
                raise ValueError(f"Answer.{name} is a token count, a non-negative int or None, not {count!r}")
        cost = self.cost_usd
        if cost is not None and (
            isinstance(cost, bool) or not isinstance(cost, int | float) or not math.isfinite(cost) or cost < 0
        ):
            raise ValueError(f"Answer.cost_usd is dollars, a finite non-negative number or None, not {cost!r}")
        if self.documents is not None:
            if isinstance(self.documents, str | bytes) or not isinstance(self.documents, Sequence):
                raise ValueError(
                    f"Answer.documents is a list of what the candidate read, not {type(self.documents).__name__}"
                )
            # Held as a tuple of Documents, so what the run stores is what was returned, and nothing can change it.
            read = tuple(_as_document(given) for given in self.documents)
            object.__setattr__(self, "documents", read)
            # Truth reads the case's evidence alike for every arm, so what one arm read never outranks it.
            if overriding := [document for document in read if document.overrides]:
                raise ValueError(
                    f"Answer.documents holds {len(overriding)} document(s) that override others; only a case's "
                    "documents override, since what a candidate read never decides truth: drop overrides="
                )


def _as_document(given: object) -> Document:
    """``given`` as a :class:`Document`: one as it is, a string as its text.

    :raises ValueError: Anything else, or a blank string.
    """
    if isinstance(given, Document):
        return given
    if isinstance(given, str):
        return Document(given)
    raise ValueError(f"Answer.documents holds Document values or their text, not {type(given).__name__}")


#: The fields an ``Answer`` reports a model call by: with every one ``None`` it reports none.
_SPEND_FIELDS = ("model", "served_model", "input_tokens", "output_tokens", "cost_usd")


def unwrap_answer(returned: Any) -> tuple[Any, CandidateTelemetry]:
    """What a candidate returned, as the answer to grade and the telemetry its spend lands on.

    :param returned: The candidate's return value: an :class:`Answer`, or a plain answer.
    :return: The answer's value and a telemetry carrying its ``candidate`` usage row; a plain answer as given,
        with telemetry that reports no usage. An ``Answer`` that reports no model, no served model, no token count
        and no cost (one returned only to carry its ``documents`` or its ``session``) reports no usage either: it says no call was made,
        so no unpriced call is counted and no model's identity is asked after. A runner that does call a model must
        name it (``model=``) for the call to count: left with every spend field unset, it reads as no call, as a
        plain return does, and its unpriced spend goes unsaid.
    """
    if not isinstance(returned, Answer):
        return returned, CandidateTelemetry()
    if all(getattr(returned, name) is None for name in _SPEND_FIELDS):
        return returned.value, CandidateTelemetry()
    ledger = RoleUsageLedger(role="candidate")
    ledger.add_llm_result(returned)
    return returned.value, CandidateTelemetry(usage=ledger.rows())


__all__ = ["Answer", "Document", "unwrap_answer"]
