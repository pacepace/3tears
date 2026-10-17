"""A completion client held to the run's cost cap, one call at a time, through one cell's sink.

A cell's paid calls are what a run's cost cap counts, once the cell's result is recorded; the cap is read
between cells and refuses no call inside one. :class:`CappedCompletion` reports a cell's spend as it goes,
for a completion client, written once: before each call it opens a reservation on the cell's sink
(:meth:`~threetears.evals.kernel.candidate_kind.CellSink.reserve`), and when the call returns it settles the
reservation to what the call cost.

The engine wraps its own cell-side clients in it — the simulated user's
(:func:`~threetears.evals.run.conversation.drive_conversation`) and the reporter kind's generator — and a
host's kind wraps its candidate's client the same way. :func:`call_ceiling` is the pricing it does,
for a caller that reserves a call it makes through something other than ``generate`` (a judge request,
whose service sends it).

**What a ceiling is comes from the client** (:meth:`~threetears.evals.schema.completion.PricedCompletion.price_ceiling`):
the engine knows neither a model's rates nor the output cap the client was built with. A client with no
``price_ceiling``, or one answering ``None``, is unpriced: its calls open a reservation with no ceiling.

**A client may refresh its prices before they are read.** ``price_ceiling`` is synchronous, so a client whose rates
come from somewhere that goes stale (the shipped client's catalogue) offers ``async refresh_prices()``, and the engine
awaits it through :func:`refresh_prices` before it prices a call: a client idle for hours is priced from fresh rates,
not refused for old ones. A client without one is priced from what it holds.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Mapping
from typing import Any, Final, Protocol, runtime_checkable

from threetears.evals.kernel.candidate_kind import CellPending, CellSink
from threetears.evals.kernel.usage_capture import CallUsage
from threetears.evals.schema.completion import CompletionGenerator

__all__ = ["PROMPT_FRAMING_TOKENS", "CappedCompletion", "call_ceiling", "prompt_token_bound", "refresh_prices"]

#: The tokens :func:`prompt_token_bound` allows for a prompt's framing beyond its text (role markers, a schema's
#: re-encoding).
PROMPT_FRAMING_TOKENS: Final = 2048


def _text_bytes(text: str) -> int:
    """The most tokens a tokenizer can read ``text`` as: one per UTF-8 byte, as written or after NFKC normalisation,
    whichever is more (a normalising tokenizer can expand a compatibility character: ``U+FDFA`` is 3 bytes, 33 after).
    """
    return max(
        len(text.encode("utf-8", errors="surrogatepass")),
        len(unicodedata.normalize("NFKC", text).encode("utf-8", errors="surrogatepass")),
    )


def prompt_token_bound(system: str, user: str, response_format: Mapping[str, Any] | None) -> int:  # Any: JSON
    """A bound on the prompt tokens a call sends, never below what the provider bills: one per UTF-8 byte.

    No token of a byte-level tokenizer is shorter than a byte, so one per byte of each text (and of the directive,
    written out as JSON either way), plus :data:`PROMPT_FRAMING_TOKENS` for the framing around them, is at or above
    any such tokenizer's count, however densely the text tokenizes. The shipped client prices a call's ceiling on it
    (:meth:`~threetears.evals.providers.Model.price_ceiling`), and the analysis generation logs it before its first
    call, so the size disclosed and the size priced are one figure.

    :param system: The system prompt the call sends.
    :ptype system: str
    :param user: The user message the call sends.
    :ptype user: str
    :param response_format: The provider directive the call sends, or ``None``.
    :ptype response_format: Mapping[str, Any] | None
    :return: The bound, in tokens.
    :rtype: int
    """
    bound = _text_bytes(system) + _text_bytes(user) + PROMPT_FRAMING_TOKENS
    if response_format is not None:
        written = json.dumps(response_format, ensure_ascii=False, default=str)
        bound += max(_text_bytes(written), len(json.dumps(response_format, default=str)))
    return bound


@runtime_checkable
class _PricesItsCalls(Protocol):
    """A client that can bound what one call costs before it is made."""

    def price_ceiling(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> float | None: ...  # pragma: no cover — protocol


async def refresh_prices(client: object) -> None:
    """Bring ``client``'s prices up to date before its calls are priced: await its ``refresh_prices()``, if it has one.

    :param client: The client whose calls are about to be priced.
    :ptype client: object
    """
    refresh = getattr(client, "refresh_prices", None)
    if callable(refresh):
        await refresh()


def call_ceiling(
    client: object,
    *,
    system: str,
    user: str,
    response_format: dict[str, Any] | None = None,
    attempts: int = 1,
) -> float | None:
    """The most ``attempts`` requests of one prompt pair on ``client`` can cost, or ``None`` when it cannot say.

    :param client: The client the call will be made on.
    :ptype client: object
    :param system: The system prompt the call sends.
    :ptype system: str
    :param user: The user message the call sends.
    :ptype user: str
    :param response_format: The provider directive the call sends, or ``None``.
    :ptype response_format: dict[str, Any] | None
    :param attempts: How many requests the call can become in the engine's own hands — a judge's parse retry
        (:data:`~threetears.evals.run.judge.JUDGE_CALL_ATTEMPTS`) re-sends the same pair. The client's
        own re-sends are already inside its ceiling.
    :ptype attempts: int
    :return: The ceiling in dollars; ``None`` for a client with no ``price_ceiling``, or one that answered ``None``.
    :rtype: float | None
    :raises ValueError: ``attempts`` is below 1, or the client answered an amount that is negative or not finite.
    """
    if attempts < 1:
        raise ValueError(f"a call is at least one request; got attempts={attempts}")
    if not isinstance(client, _PricesItsCalls):
        return None
    ceiling = client.price_ceiling(system=system, user=user, response_format=response_format)
    if ceiling is None:
        return None
    if isinstance(ceiling, bool) or not isinstance(ceiling, int | float) or not math.isfinite(ceiling) or ceiling < 0:
        raise ValueError(
            f"{type(client).__name__} priced a call at {ceiling!r}; a ceiling is a finite amount, 0 or more"
        )
    return float(ceiling) * attempts


class CappedCompletion:
    """One cell's view of a completion client: every call reserved against the run's cap, and settled to its cost.

    Per cell, since the sink is. It owns nothing — the client it forwards to is whoever built it — so it has no
    lifecycle of its own. What a call returns is handed back unchanged, so a caller reading its usage reads it
    as it would off the client.
    """

    def __init__(self, inner: CompletionGenerator, *, sink: CellSink, source: CellPending) -> None:
        """Bind the client, the cell's sink, and whose spend its calls are.

        :param inner: The client every call is forwarded to.
        :ptype inner: CompletionGenerator
        :param sink: The cell's sink, which holds each call against the cap.
        :ptype sink: CellSink
        :param source: Whose spend the calls are — ``simulator`` for the simulated user's, ``candidate`` for a
            candidate's own.
        :ptype source: CellPending
        """
        self._inner = inner
        self._sink = sink
        self._source: CellPending = source

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        """Open the call's reservation, make the call, and settle the reservation to what it cost.

        :param system: The system prompt.
        :ptype system: str
        :param user: The user message.
        :ptype user: str
        :param response_format: The provider directive, or ``None``.
        :ptype response_format: dict[str, Any] | None
        :return: The client's completion, unchanged.
        :rtype: Any
        """
        await refresh_prices(self._inner)
        held = self._sink.reserve(
            self._source, call_ceiling(self._inner, system=system, user=user, response_format=response_format)
        )
        try:
            result = await self._inner.generate(system=system, user=user, response_format=response_format)
        except BaseException:
            # A call that raised — or was cancelled — returned no cost to settle to; whatever it was billed is
            # unknown here, and the cell's own record reports what its caller learned of it.
            held.release()
            raise
        # The spend alone: the reservation is settled to what the call cost whatever else the completion holds, and a
        # cap reads nothing a candidate's call was sent with. The simulator's sent settings are read by its own caller,
        # after this has settled the call, so a refused ``sent_settings`` never leaves a paid call's ceiling held.
        held.settle(CallUsage.billed(result).cost_usd)
        return result
