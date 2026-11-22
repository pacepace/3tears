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
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals.kernel import CandidateTelemetry, RoleUsageLedger


@dataclass(frozen=True)
class Document:
    """A document claims are checked against: one a candidate read while answering, or one a case holds.

    :ivar text: Its text, as read: every quote a grader cites from it is checked against this, verbatim.
    :ivar title: What it is called; ``None`` names it by its address, else by its place in the list.
    :ivar url: Where it was read, when it has an address.
    """

    text: str
    title: str | None = None
    url: str | None = None

    def __post_init__(self) -> None:
        """Refuse a document no claim could be checked against.

        :raises ValueError: Text that is not a string or is blank, or a title or address that is blank.
        """
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(
                f"Document.text is the text a claim is checked against, a non-blank string, not {self.text!r}"
            )
        for name in ("title", "url"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Document.{name} is a non-blank string or None, not {value!r}")


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
    :ivar documents: What the candidate read while answering: each passage it was given or page it fetched, a
        :class:`Document` or its text. ``()`` says it read none; ``None`` (the default) says nothing, which a
        ``Claims`` grader reading the candidate's own documents (``grounded_in="run"``) refuses, since no read
        could tell it from none.
    """

    value: Any
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    served_model: str | None = None
    documents: Sequence[Document | str] | None = None

    def __post_init__(self) -> None:
        """Refuse a spend no usage row can hold, so the candidate that built it fails rather than the run.

        :raises ValueError: A negative or non-integer token count, a negative or non-finite cost, or documents that are
            not a list of :class:`Document` values or non-blank strings.
        """
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
            object.__setattr__(self, "documents", tuple(_as_document(given) for given in self.documents))


def _as_document(given: object) -> Document:
    """``given`` as a :class:`Document`: one as it is, a string as its text.

    :raises ValueError: Anything else, or a blank string.
    """
    if isinstance(given, Document):
        return given
    if isinstance(given, str):
        return Document(given)
    raise ValueError(f"Answer.documents holds Document values or their text, not {type(given).__name__}")


def unwrap_answer(returned: Any) -> tuple[Any, CandidateTelemetry]:
    """What a candidate returned, as the answer to grade and the telemetry its spend lands on.

    :param returned: The candidate's return value: an :class:`Answer`, or a plain answer.
    :return: The answer's value and a telemetry carrying its ``candidate`` usage row; a plain answer as given,
        with telemetry that reports no usage.
    """
    if not isinstance(returned, Answer):
        return returned, CandidateTelemetry()
    ledger = RoleUsageLedger(role="candidate")
    ledger.add_llm_result(returned)
    return returned.value, CandidateTelemetry(usage=ledger.rows())


__all__ = ["Answer", "Document", "unwrap_answer"]
