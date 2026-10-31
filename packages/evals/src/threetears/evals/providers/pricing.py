"""Who prices a call: the :class:`Pricer` port a :class:`~threetears.evals.providers.Model` is handed, and the default.

The engine cannot know what a host pays: a special deal, a bring-your-own-key route, a subscription, a local model.
So pricing sits at the provider edge, as a port the host implements and injects (``Model(..., pricer=...)``). The
host owns the prices and the engine owns the arithmetic. A pricer has two halves:

* **Before a call**, :meth:`Pricer.rates_for` answers the per-token rates the call is billed at, as
  :class:`HostRates`: the :class:`ModelRates` and whether the host guarantees them. The engine computes the call's
  bound from them (:meth:`ModelRates.ceiling`), so no host writes that arithmetic. A bound built on rates the host
  does not guarantee (an OpenRouter route whose upstream may bill otherwise) still admits the call, and is disclosed
  as an estimate. ``None`` leaves the call to the catalogue route, both its bound and its price: OpenRouter's
  reported cost for an OpenRouter call, else OpenRouter's catalogue row, whose bound is always an estimate (a third
  party's listing, not the host's bill).
* **After a call**, :meth:`Pricer.price` prices the reply from its raw ``usage`` block as the wire returned it
  (OpenRouter's reported cost and bring-your-own-key upstream charge included, Anthropic's cache writes split by
  lifetime under ``cache_creation``) and the upstream OpenRouter served it from, as a :class:`CallPrice`. A call it
  cannot price is ``usd=None`` with the reason, never ``0``.

**The default is for standalone and development use.** :class:`AnthropicPublishedRates` prices a direct Anthropic call
from Anthropic's published per-token rates, which ship with the package, and leaves every other call to the
catalogue: a developer's ``Model("claude-opus-5-5", provider="anthropic", api_key=...)`` (the report writer's
command line, an example) is priced and bounded before its first call with no network read. A model is given it only
when it is handed neither a ``pricer`` nor a ``catalogue``. A call no pricer answers rates for is priced as it always
was: OpenRouter's own reported figure for an OpenRouter call, else OpenRouter's catalogue row. A host that hands its
models a catalogue (every :class:`~threetears.evals.providers.Clients` model is one) prices from that catalogue
exactly as before, and a host whose prices are its own hands every model its own pricer.

**The shipped rates.** One row per model Anthropic's API serves, each with its five standard rates in dollars per
million tokens: base input, a five-minute cache write, a one-hour cache write, a cache read (a hit or a refresh), and
output. Thinking tokens are output tokens. Claude Haiku 5.5 bills a prompt of more than 100,000 tokens at its second
row's rates, every token of the call included. They are Anthropic's first-party rates for the default global
inference (the client sends no ``inference_geo``, no ``speed`` and no batch), read from Anthropic's pricing page
(https://platform.claude.com/docs/en/about-claude/pricing) on :data:`_RATES_AS_OF`, the date every call they price
names in its ``price_source`` (``anthropic-published:<date>``). Retired models, which the API no longer serves, are
not listed.

**A ceiling is a true upper bound on the rates it is built from.** A prompt token is bounded at the dearest rate a
request the client builds can bill it at: the five-minute cache write, since the client marks five-minute breakpoints
alone (a tool turn's three, a completion's none) and never asks for the one-hour lifetime. A reply that reports
one-hour writes is still priced at that rate; one that writes to the cache without splitting the writes by lifetime
is priced at the five-minute rate, the only lifetime the client asked for.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Final, Protocol

from threetears.evals.providers.failures import Provider
from threetears.evals.providers.wire import usage_counts

__all__ = ["AnthropicPublishedRates", "CallPrice", "HostRates", "ModelRates", "Pricer"]

#: When :class:`AnthropicPublishedRates`' rates were read from Anthropic's pricing page, and the date every call they
#: price names in its ``price_source``.
_RATES_AS_OF: Final = "2026-10-28"

#: Where they were read.
_RATES_SOURCE: Final = "https://platform.claude.com/docs/en/about-claude/pricing"

#: The prompt size above which Claude Haiku 5.5 bills a call at its long-prompt rates ("for prompts over 100,000
#: tokens"), counted as the call's whole prompt: fresh input, cache reads and cache writes.
_LONG_PROMPT_TOKENS: Final = 100_000

#: Tokens in the million the rates are stated per.
_PER: Final = Decimal(1_000_000)

#: Why a reply that counted no tokens is unpriced, whoever prices it.
NOTHING_TO_PRICE: Final = "the reply reported no input or output token count, so there is nothing to price"

#: A date suffix on a model id (``claude-haiku-4-5-20251001``), which the table's ids leave off.
_DATE_SUFFIX: Final = re.compile(r"-\d{8}$")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRates:
    """What a call on one model is billed at, as a :class:`Pricer` answers it: five per-token rates, in dollars per
    million tokens, from which the engine bounds a call (:meth:`ceiling`) and a pricer may price one (:meth:`cost`).

    :ivar base_input: A prompt token neither read from nor written to the cache.
    :ivar cache_write_5m: A prompt token written to the five-minute cache.
    :ivar cache_write_1h: A prompt token written to the one-hour cache.
    :ivar cache_read: A prompt token read from the cache (a hit or a refresh).
    :ivar output: An output token, thinking included.
    :ivar long_prompt: The rates of a call whose prompt is more than ``long_prompt_tokens`` tokens, or ``None`` when
        the model has one set of rates whatever the prompt.
    :ivar long_prompt_tokens: The prompt size ``long_prompt`` applies above.
    """

    base_input: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal
    cache_read: Decimal
    output: Decimal
    long_prompt: ModelRates | None = None
    long_prompt_tokens: int = _LONG_PROMPT_TOKENS

    def tier_for(self, prompt_tokens: int) -> ModelRates:
        """The rates a call of ``prompt_tokens`` prompt tokens bills at: these, or ``long_prompt`` past its size.

        :param prompt_tokens: The call's whole prompt, cache reads and writes included (or a bound on it).
        :ptype prompt_tokens: int
        :return: The rates.
        :rtype: ModelRates
        """
        if self.long_prompt is not None and prompt_tokens > self.long_prompt_tokens:
            return self.long_prompt
        return self

    def ceiling(self, *, prompt_tokens: int, output_tokens: int) -> Decimal:
        """The most one call can be billed, in dollars: every prompt token at the dearest rate the client's requests
        can bill it at (the five-minute cache write), every output token at the output rate.

        :param prompt_tokens: A bound on the call's prompt tokens, which also says whether the long-prompt rates apply.
        :ptype prompt_tokens: int
        :param output_tokens: The call's output cap, thinking included.
        :ptype output_tokens: int
        :return: The ceiling.
        :rtype: Decimal
        """
        rates = self.tier_for(prompt_tokens)
        prompt_rate = max(rates.base_input, rates.cache_write_5m, rates.cache_read)
        return (prompt_tokens * prompt_rate + output_tokens * rates.output) / _PER

    def cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_write_1h_tokens: int | None = 0,
    ) -> tuple[Decimal | None, str | None]:
        """What a call cost, from what its reply reports: each share of the prompt at its own rate.

        :param input_tokens: Prompt tokens, cache reads and writes included.
        :ptype input_tokens: int
        :param output_tokens: Output tokens, thinking included.
        :ptype output_tokens: int
        :param cache_read_tokens: Prompt tokens read from the cache.
        :ptype cache_read_tokens: int
        :param cache_write_tokens: Prompt tokens written to the cache, both lifetimes.
        :ptype cache_write_tokens: int
        :param cache_write_1h_tokens: The one-hour share of ``cache_write_tokens``, or ``None`` when the reply does not
            split it: then every write is priced at the five-minute rate, the only lifetime the client asks for.
        :ptype cache_write_1h_tokens: int | None
        :return: ``(cost, None)`` in dollars, or ``(None, why)`` when the counts contradict each other.
        :rtype: tuple[Decimal | None, str | None]
        """
        cached = cache_read_tokens + cache_write_tokens
        fresh = input_tokens - cached
        if fresh < 0:
            return None, f"the reply counts {cached} cached prompt tokens within {input_tokens} prompt tokens in all"
        one_hour = cache_write_1h_tokens or 0
        if one_hour > cache_write_tokens:
            return None, (
                f"the reply counts {one_hour} one-hour cache writes within {cache_write_tokens} cache writes in all"
            )
        rates = self.tier_for(input_tokens)
        dollars = (
            fresh * rates.base_input
            + cache_read_tokens * rates.cache_read
            + (cache_write_tokens - one_hour) * rates.cache_write_5m
            + one_hour * rates.cache_write_1h
            + output_tokens * rates.output
        )
        return dollars / _PER, None


@dataclass(frozen=True, slots=True)
class HostRates:
    """The rates a :class:`Pricer` answers for a call before it is sent, and whether the host guarantees them.

    :ivar rates: The per-token rates the call is billed at, which the engine bounds it from.
    :ivar guaranteed: Whether the call cannot be billed above them. ``False`` for an estimate (an OpenRouter route
        whose upstream, or a bring-your-own-key route, may bill otherwise): the bound still admits the call, and a
        budget checked against it says it was checked against an estimate.
    """

    rates: ModelRates
    guaranteed: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class CallPrice:
    """What a call cost, as a :class:`Pricer` priced it after it returned.

    :ivar usd: The cost in dollars, or ``None`` when it could not be priced: unknown, never ``0``.
    :ivar source: Who priced it (the call's ``price_source``); ``None`` exactly when ``usd`` is.
    :ivar unpriced_reason: Why ``usd`` is ``None``, as a clause; ``None`` exactly when it is priced.
    :ivar stale_since: When the rates that priced it expired with every refresh since failing (UTC), so the cost is
        the last rates read; ``None`` for rates that were current, which is every rate a host holds in hand.
    """

    usd: float | None
    source: str | None
    unpriced_reason: str | None = None
    stale_since: datetime | None = None

    def __post_init__(self) -> None:
        """Refuse a price that contradicts itself.

        :raises ValueError: A cost with no source or with a reason, no cost and no reason, a source with no cost, or a
            cost that is not a finite amount of at least 0.
        """
        if self.usd is None:
            if not self.unpriced_reason:
                raise ValueError("an unpriced call says why (unpriced_reason): unknown is never written as 0")
            if self.source is not None:
                raise ValueError(f"an unpriced call names no price source, not {self.source!r}")
            return
        if isinstance(self.usd, bool) or not math.isfinite(self.usd) or self.usd < 0:
            raise ValueError(f"a call's cost is a finite number of dollars of at least 0, not {self.usd!r}")
        if not self.source:
            raise ValueError("a priced call names who priced it (source)")
        if self.unpriced_reason is not None:
            raise ValueError("a priced call has no unpriced_reason")


class Pricer(Protocol):
    """What a call is billed at, as the host knows it: the port a :class:`~threetears.evals.providers.Model` is handed
    (``pricer=``), or :class:`AnthropicPublishedRates` by default (the module docstring says when).

    Both halves are asked for every call, a completion and a tool turn alike: :meth:`rates_for` before it is sent and
    bounded, :meth:`price` after it returns, when :meth:`rates_for` answered rates. Neither reads anything over a
    network: a host's prices are in hand.

    The port covers every :class:`~threetears.evals.providers.Model` call, which is every client the engine ships.
    A host client that is not a ``Model`` prices its own calls and needs no pricer: its ``price_ceiling`` is its
    bound, and its completions' ``cost_usd`` and ``price_source`` are recorded as reported, a missing cost read as
    unpriced, never ``$0`` (:class:`~threetears.evals.schema.CompletionClient`).
    """

    def rates_for(self, model: str, *, provider: Provider) -> HostRates | None:
        """The rates a call on ``model`` sent to ``provider`` is billed at, or ``None`` to leave it to the default.

        :param model: The model id the call names.
        :ptype model: str
        :param provider: Where the call goes.
        :ptype provider: Provider
        :return: The rates and whether they are guaranteed, or ``None``: the catalogue route then bounds the call (an
            estimate) and prices it (OpenRouter's reported cost, else the catalogue row).
        :rtype: HostRates | None
        """
        ...

    def price(
        self, model: str, *, provider: Provider, usage: Mapping[str, Any], served_provider: str | None
    ) -> CallPrice:  # Any: a JSON object
        """What a call on ``model`` cost, from its reply's raw usage block.

        :param model: The model id the call named.
        :ptype model: str
        :param provider: Where it went.
        :ptype provider: Provider
        :param usage: The reply's ``usage`` object exactly as the wire returned it: OpenRouter's ``cost``, ``is_byok``
            and ``cost_details.upstream_inference_cost``; Anthropic's cache writes split by lifetime under
            ``cache_creation``. :func:`~threetears.evals.providers.wire.usage_counts` reads its token counts as the
            engine does.
        :ptype usage: Mapping[str, Any]
        :param served_provider: The upstream OpenRouter routed the call to, or ``None`` (any other route).
        :ptype served_provider: str | None
        :return: The price, or ``usd=None`` with the reason.
        :rtype: CallPrice
        """
        ...


#: What a call the default prices names as its ``price_source``: Anthropic's published rates, and when they were read.
_SOURCE: Final = f"anthropic-published:{_RATES_AS_OF}"


def _rates(
    base_input: str,
    cache_write_5m: str,
    cache_write_1h: str,
    cache_read: str,
    output: str,
    *,
    long_prompt: ModelRates | None = None,
) -> ModelRates:
    return ModelRates(
        base_input=Decimal(base_input),
        cache_write_5m=Decimal(cache_write_5m),
        cache_write_1h=Decimal(cache_write_1h),
        cache_read=Decimal(cache_read),
        output=Decimal(output),
        long_prompt=long_prompt,
    )


#: Fable 5.1 and Mythos 5.1: cache reads at 0.025x base input (the page's footnote 1).
_FABLE_5_1: Final = _rates("10", "12.50", "20", "0.25", "50")
#: Fable 5 and Mythos 5.
_FABLE_5: Final = _rates("10", "12.50", "20", "1", "50")
#: Opus 5 and the Opus 4.5 to 4.8 models.
_OPUS: Final = _rates("5", "6.25", "10", "0.50", "25")
#: Sonnet 4.5 and 4.6.
_SONNET_4: Final = _rates("3", "3.75", "6", "0.30", "15")

#: Anthropic's published rates per model id, and Haiku 5.5's long-prompt rates, read from :data:`_RATES_SOURCE` on
#: :data:`_RATES_AS_OF` (the module docstring says what each rate is).
_PUBLISHED: Final[Mapping[str, ModelRates]] = {
    "claude-fable-5-1": _FABLE_5_1,
    "claude-mythos-5-1": _FABLE_5_1,
    "claude-fable-5": _FABLE_5,
    "claude-mythos-5": _FABLE_5,
    # Opus 5.5 and Sonnet 5.5: cache reads at 0.05x base input (the page's footnote 2).
    "claude-opus-5-5": _rates("4", "5", "8", "0.20", "20"),
    "claude-opus-5": _OPUS,
    "claude-opus-4-8": _OPUS,
    "claude-opus-4-7": _OPUS,
    "claude-opus-4-6": _OPUS,
    "claude-opus-4-5": _OPUS,
    "claude-sonnet-5-5": _rates("2", "2.50", "4", "0.10", "10"),
    # Sonnet 5: its $2/$10 launch pricing is its standard price (the page's footnote 3).
    "claude-sonnet-5": _rates("2", "2.50", "4", "0.20", "10"),
    "claude-sonnet-4-6": _SONNET_4,
    "claude-sonnet-4-5": _SONNET_4,
    "claude-haiku-5-5": _rates(
        "0.10", "0.125", "0.20", "0.01", "0.50", long_prompt=_rates("0.50", "0.625", "1", "0.05", "2.50")
    ),
    "claude-haiku-4-5": _rates("1", "1.25", "2", "0.10", "5"),
}


def _plain(model_id: str) -> str:
    """A model id as the table keys it: lower case, dots as dashes, no date suffix."""
    return _DATE_SUFFIX.sub("", model_id.strip().lower().replace(".", "-"))


class AnthropicPublishedRates:
    """The default pricer: Anthropic's published rates for a direct Anthropic call, the catalogue for anything else.

    For standalone and development use (the module docstring); a host replaces it with its own :class:`Pricer`. Its
    rates are guaranteed: they are what Anthropic bills a direct call.
    """

    def rates_for(self, model: str, *, provider: Provider) -> HostRates | None:
        """Anthropic's published rates for ``model`` when it goes to Anthropic and the table lists it; else ``None``,
        which leaves the call to the catalogue.

        Ids are compared lower case, with dots as dashes and a date suffix left off, so ``claude-haiku-4-5-20251001``
        is ``claude-haiku-4-5``.

        :param model: The model id the call names.
        :ptype model: str
        :param provider: Where the call goes.
        :ptype provider: Provider
        :return: The rates, guaranteed, or ``None``.
        :rtype: HostRates | None
        """
        rates = _PUBLISHED.get(_plain(model)) if provider == "anthropic" else None
        return HostRates(rates, guaranteed=True) if rates is not None else None

    def price(
        self, model: str, *, provider: Provider, usage: Mapping[str, Any], served_provider: str | None
    ) -> CallPrice:  # Any: a JSON object
        """A direct Anthropic call on a listed model, priced share by share from its reply's usage block.

        :param model: The model id the call named.
        :ptype model: str
        :param provider: Where it went.
        :ptype provider: Provider
        :param usage: The reply's raw ``usage`` object.
        :ptype usage: Mapping[str, Any]
        :param served_provider: Unread: a direct call has no upstream.
        :ptype served_provider: str | None
        :return: The price, sourced ``anthropic-published:<date>``; unpriced for a model or route the rates do not
            list, a reply with no token count, or counts that contradict each other.
        :rtype: CallPrice
        """
        found = self.rates_for(model, provider=provider)
        if found is None:
            return CallPrice(usd=None, source=None, unpriced_reason=self.unlisted(model))
        counts = usage_counts(provider, usage)
        if counts.input_tokens is None or counts.output_tokens is None:
            return CallPrice(usd=None, source=None, unpriced_reason=NOTHING_TO_PRICE)
        dollars, why = found.rates.cost(
            input_tokens=counts.input_tokens,
            output_tokens=counts.output_tokens,
            cache_read_tokens=counts.cache_read_tokens,
            cache_write_tokens=counts.cache_write_tokens,
            cache_write_1h_tokens=counts.cache_write_1h_tokens,
        )
        if dollars is None:
            return CallPrice(usd=None, source=None, unpriced_reason=f"{model!r} cannot be priced: {why}")
        return CallPrice(usd=float(dollars), source=_SOURCE, unpriced_reason=None)

    def unlisted(self, model: str) -> str:
        """What to do about a direct Anthropic call on ``model``, which the shipped rates do not list.

        :param model: The model id.
        :ptype model: str
        :return: A sentence naming the shipped rates' source and date, the models they list, and the remedies.
        :rtype: str
        """
        return (
            f"Anthropic's published rates as 3tears-evals ships them (read {_RATES_AS_OF} from {_RATES_SOURCE}) do not "
            f"list {model!r}: use a model they list ({', '.join(_PUBLISHED)}), or hand the model a pricer with the "
            "host's own rates (Model(..., pricer=...))"
        )
