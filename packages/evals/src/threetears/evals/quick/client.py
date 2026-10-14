"""The one seam between the eval contract and a completion client: :class:`Model`, and what a runner's calls spent.

Everything the contract asks of a client goes through this module, and nothing else in :mod:`threetears.evals.quick`
names a client's type:

- **What a runner's calls spent.** Every runner call is made inside :func:`cell_spend`, a ledger for that one call
  that the callable kind opens around it; every :class:`Model` call made while it is open adds its completion
  (:func:`note_spend`). A runner that calls :class:`Model` therefore reports its spend with no ``Answer``, and the
  cell's ``candidate`` usage row is what those calls cost, unpriced where a completion was. The ledger is a context
  variable, so the cells of one run, each its own task, never share one, and a synchronous runner run in a worker
  thread writes to its own cell's.
- **The judge's client.** :func:`judge_client` turns ``Judge(model=...)`` into the client the judge calls and the
  model name the run records: a :class:`Model`, any client that names its model (``model_name``), or a model id.

**:class:`Model` here replays a recording and calls no provider.** The provider client
(``threetears.evals.providers.Model``: ``text``, ``json``, routing, pricing) lands separately; until it does, this
:class:`Model` has its surface over :class:`~threetears.evals.testing.RecordedCompletions`, so a program written
against it runs offline today, and a call it cannot replay is refused by name rather than sent. Reconciling the two
is this one file: the class below gives way to the provider's, which writes to :func:`note_spend`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ValidationError

from threetears.evals.kernel import CandidateTelemetry, RoleUsageLedger, strict_response_format
from threetears.evals.schema import CompletionClient, CompletionResult
from threetears.evals.testing import RecordedCompletions

#: The ledger of the runner call in progress in this context, or ``None`` outside one.
_SPEND: ContextVar[RoleUsageLedger | None] = ContextVar("threetears_evals_cell_spend", default=None)

#: The environment variable that asks for a recording to be made again rather than replayed.
RECORD_ENV = "RECORD"


@contextmanager
def cell_spend() -> Iterator[RoleUsageLedger]:
    """A ledger for one runner call: every :class:`Model` completion made while it is open is added to it.

    Yields:
        The ledger, the ``candidate`` role's, whose rows are the call's spend.
    """
    ledger = RoleUsageLedger(role="candidate")
    token = _SPEND.set(ledger)
    try:
        yield ledger
    finally:
        _SPEND.reset(token)


def note_spend(completion: CompletionResult) -> None:
    """Add one completion to the ledger of the runner call in progress; outside a runner call, nothing is kept.

    Args:
        completion: What the call returned: its model, tokens and cost, unpriced when ``cost_usd`` is ``None``.
    """
    ledger = _SPEND.get()
    if ledger is not None:
        ledger.add_llm_result(completion)


def spent(ledger: RoleUsageLedger, telemetry: CandidateTelemetry) -> CandidateTelemetry:
    """``telemetry`` with ``ledger``'s rows added: what the runner's :class:`Model` calls cost beside its ``Answer``.

    Args:
        ledger: The runner call's ledger (:func:`cell_spend`).
        telemetry: What the runner's return reported (an ``Answer``'s usage, or none).

    Returns:
        The telemetry carrying both; ``telemetry`` itself when no :class:`Model` call was made.
    """
    rows = ledger.rows()
    if not rows:
        return telemetry
    return telemetry.model_copy(update={"usage": [*telemetry.usage, *rows]})


class Model:
    """A model a runner or a judge calls: ``await model.text(prompt)``, ``await model.json(prompt, Schema)``.

    This build's :class:`Model` replays a recording made of the model and calls no provider: ``recording=`` is
    required, and a call the recording does not hold is refused naming the prompt, never sent. Each call's
    completion is added to the runner call's spend (:func:`cell_spend`), as recorded: a recorded price is what the
    call cost when it was made.

    Attributes:
        model_name: The model's id, as the recording holds it.
        recording: The recording replayed.
    """

    def __init__(self, id: str, *, recording: str | os.PathLike[str] | None = None) -> None:
        """Bind the model to its recording.

        Args:
            id: The model's id (``"deepseek/deepseek-v4-flash"``).
            recording: The recording of its calls (a :class:`~threetears.evals.testing.RecordedCompletions` file).

        Raises:
            ValueError: A blank id, no recording, a recording that is not there, or ``RECORD=1``, which asks for a
                live call this build cannot make.
        """
        if not isinstance(id, str) or not id.strip():
            raise ValueError(f"a Model is named by its model id, a non-blank string, not {id!r}")
        if recording is None:
            raise ValueError(
                f"Model({id!r}) has no recording=, and this build's Model calls no provider: it replays a recording "
                "of the model's calls. Pass recording='<file>.jsonl'"
            )
        path = Path(recording)
        if os.environ.get(RECORD_ENV) == "1":
            raise ValueError(
                f"RECORD=1 asks Model({id!r}) to record {path} live, and this build's Model calls no provider; "
                "unset RECORD to replay the recording"
            )
        if not path.is_file():
            raise ValueError(f"Model({id!r}) replays {path}, which is not there; record it, or name the file it is in")
        self.model_name = id
        self.recording = path
        self._client = RecordedCompletions.replay(path, settings=None, model=id)

    def __repr__(self) -> str:
        """The model, by id and recording."""
        return f"Model({self.model_name!r}, recording={str(self.recording)!r})"

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> CompletionResult:
        """One call, as the completion port makes it, its spend added to the runner call's.

        Args:
            system: The system prompt.
            user: The user prompt.
            response_format: The provider directive, or ``None``.

        Returns:
            The completion.
        """
        completion: CompletionResult = await self._client.generate(
            system=system, user=user, response_format=response_format
        )
        note_spend(completion)
        return completion

    async def text(self, user: str, *, system: str | None = None) -> str:
        """The model's reply to ``user``, as text.

        Args:
            user: The prompt.
            system: The system prompt; ``None`` sends an empty one.

        Returns:
            The reply's text.
        """
        completion = await self.generate(system=system or "", user=user)
        return completion.content

    async def json[T: BaseModel](self, user: str, schema: type[T], *, system: str | None = None) -> T:
        """The model's reply to ``user`` as ``schema``: sent as a strict ``json_schema``, validated on return.

        Args:
            user: The prompt.
            schema: The pydantic model the reply must be.
            system: The system prompt; ``None`` sends an empty one.

        Returns:
            The reply, validated.

        Raises:
            ValueError: The reply is not ``schema``; the message is pydantic's, so a miss says why, and nothing is
                filled in for it.
        """
        completion = await self.generate(system=system or "", user=user, response_format=json_directive(schema))
        try:
            return schema.model_validate_json(completion.content)
        except ValidationError as refused:
            raise ValueError(f"the reply is not a {schema.__name__}: {refused}") from refused

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """Nothing: a replay sends nothing and costs nothing.

        Args:
            system: Unread.
            user: Unread.
            response_format: Unread.

        Returns:
            ``0.0``.
        """
        return self._client.price_ceiling(system=system, user=user, response_format=response_format)

    async def aclose(self) -> None:
        """Release the recording."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Enter a scope that releases the model on exit."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Release the model."""
        await self.aclose()


def json_directive(schema: type[BaseModel]) -> dict[str, Any]:
    """The provider directive :meth:`Model.json` sends for ``schema``: the strict ``json_schema`` of it.

    Args:
        schema: The pydantic model the reply must be.

    Returns:
        The directive, which a recording of the call is keyed by.
    """
    return strict_response_format(schema.__name__, schema.model_json_schema())


def judge_client(model: str | CompletionClient) -> tuple[CompletionClient, str]:
    """The client a judge calls and the model the run records it as, from what ``Judge(model=...)`` was handed.

    Args:
        model: A :class:`Model`; any completion client that names its model (``model_name``); or a model id, which
            builds a :class:`Model`.

    Returns:
        The client, and its model's name.

    Raises:
        ValueError: A client that names no model, or a model id this build cannot call (:class:`Model`).
    """
    if isinstance(model, str):
        built = Model(model)
        return built, built.model_name
    name = getattr(model, "model_name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"Judge(model=...) takes a model id, a Model, or a client that names its model (model_name); "
            f"{type(model).__name__} names none, and the run records which model judged"
        )
    return model, name


__all__ = ["RECORD_ENV", "Model", "cell_spend", "json_directive", "judge_client", "note_spend", "spent"]
