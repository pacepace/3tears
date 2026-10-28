"""Who prices a call: the :class:`Pricer` a :class:`~threetears.evals.providers.Model` is handed, and the default one.

The engine cannot know what a host pays: a special deal, a bring-your-own-key route, a subscription, a local model.
So pricing sits at the provider edge, as a dependency the host injects (``Model(..., pricer=...)``). A pricer answers
the per-token rates a call on a model is billed at (:class:`ModelRates`), or ``None`` to leave the call to OpenRouter's
catalogue, which prices it as it always has. The model prices the call, and bounds it before it is sent, from the
rates the pricer answered, and reads no catalogue for it.

**The default is for standalone and development use.** :class:`AnthropicPublishedRates` prices a direct Anthropic call
from Anthropic's published per-token rates, which ship with the package, and leaves every other call to the
catalogue: a developer's ``Model("claude-opus-5-5", provider="anthropic", api_key=...)`` (the report writer's
command line, an example) is priced and bounded before its first call with no network read. A model is given it only
when it is handed neither a ``pricer`` nor a ``catalogue``. A host that hands its models a catalogue (every
:class:`~threetears.evals.providers.Clients` model is one) prices from that catalogue exactly as before, and a host
whose prices are its own hands every model its own pricer.

**The shipped rates.** One row per model Anthropic's API serves, each with its five standard rates in dollars per
million tokens: base input, a five-minute cache write, a one-hour cache write, a cache read (a hit or a refresh), and
output. Thinking tokens are output tokens. Claude Haiku 5.5 bills a prompt of more than 100,000 tokens at its second
row's rates, every token of the call included. They are Anthropic's first-party rates for the default global
inference (the client sends no ``inference_geo``, no ``speed`` and no batch), read from Anthropic's pricing page
(https://platform.claude.com/docs/en/about-claude/pricing) on :data:`_RATES_AS_OF`, the date every call they price
names in its ``price_source`` (``anthropic-published:<date>``). Retired models, which the API no longer serves, are
not listed.

**A ceiling is a true upper bound.** A prompt token is bounded at the dearest rate a request the client builds can bill
it at: the five-minute cache write, since the client marks five-minute breakpoints alone (a tool turn's three, a
completion's none) and never asks for the one-hour lifetime. A reply that reports one-hour writes is still priced at
that rate; one that writes to the cache without splitting the writes by lifetime is priced at the five-minute rate,
the only lifetime the client asked for.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Final, Protocol

from threetears.evals.providers.failures import Provider

__all__ = ["AnthropicPublishedRates", "ModelRates", "Pricer"]

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

#: A date suffix on a model id (``claude-haiku-4-5-20251001``), which the table's ids leave off.
_DATE_SUFFIX: Final = re.compile(r"-\d{8}$")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRates:
    """What a call on one model is billed at, as a :class:`Pricer` answers it: five per-token rates, in dollars per
    million tokens, and the ``price_source`` a call priced from them names.

    :ivar base_input: A prompt token neither read from nor written to the cache.
    :ivar cache_write_5m: A prompt token written to the five-minute cache.
    :ivar cache_write_1h: A prompt token written to the one-hour cache.
    :ivar cache_read: A prompt token read from the cache (a hit or a refresh).
    :ivar output: An output token, thinking included.
    :ivar source: What a call priced from them names as its ``price_source``.
    :ivar long_prompt: The rates of a call whose prompt is more than ``long_prompt_tokens`` tokens, or ``None`` when
        the model has one set of rates whatever the prompt.
    :ivar long_prompt_tokens: The prompt size ``long_prompt`` applies above.
    """

    base_input: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal
    cache_read: Decimal
    output: Decimal
    source: str
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


class Pricer(Protocol):
    """What a call is billed at, as the host knows it: the dependency a :class:`~threetears.evals.providers.Model`
    is handed (``pricer=``), or :class:`AnthropicPublishedRates` by default (the module docstring says when)."""

    def rates_for(self, model: str, *, provider: Provider) -> ModelRates | None:
        """The rates a call on ``model`` sent to ``provider`` is billed at, or ``None`` to leave it to the catalogue.

        Asked before every call is priced or bounded, so it reads nothing over a network: a host's rates are in hand.

        :param model: The model id the call names.
        :ptype model: str
        :param provider: Where the call goes.
        :ptype provider: Provider
        :return: The rates, or ``None``.
        :rtype: ModelRates | None
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
        source=_SOURCE,
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

    For standalone and development use (the module docstring); a host replaces it with its own :class:`Pricer`.
    """

    def rates_for(self, model: str, *, provider: Provider) -> ModelRates | None:
        """Anthropic's published rates for ``model`` when it goes to Anthropic and the table lists it; else ``None``,
        which leaves the call to the catalogue.

        Ids are compared lower case, with dots as dashes and a date suffix left off, so ``claude-haiku-4-5-20251001``
        is ``claude-haiku-4-5``.

        :param model: The model id the call names.
        :ptype model: str
        :param provider: Where the call goes.
        :ptype provider: Provider
        :return: The rates, or ``None``.
        :rtype: ModelRates | None
        """
        if provider != "anthropic":
            return None
        return _PUBLISHED.get(_plain(model))

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
