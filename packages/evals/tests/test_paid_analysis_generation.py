"""The paid lane: one toy-host analysis written by a real model (#602).

Every other generator test hands ``generate_analysis`` a fixtured completion, so a defect in the writer
prompt, the response schema or the toy-host bundle reads the same as a working one until a real model
writes an analysis. This lane assembles the toy host's bundle, sends it with the seed prompt and the
authored contract to a real model, and asserts:

- the generation returns an analysis that round-trips through the stored contract;
- every finding's evidence resolved to a reading in the bundle, filled by code;
- no repair loop exhausted (a refused first output may buy its one repair; a second refusal fails);
- a deliberately broken contract (a required field removed) fails the lane, so the lane can fail.

**Opt-in, never in the default run.** Marked ``paid``, and skipped unless ``THREETEARS_EVALS_PAID=1`` is set
and ``ANTHROPIC_API_KEY`` is present: a missing key is a skip, not a failure. Run it with::

    THREETEARS_EVALS_PAID=1 ./scripts/test.sh evals -m paid -rs

``THREETEARS_EVALS_PAID_MODEL`` picks the writer (default :data:`DEFAULT_MODEL`; it must be in
:data:`PRICES`).

**The spend ceiling.** Every call is priced before it is sent, at its worst case (the prompt at three
characters a token, plus the full output cap at the output rate), through ``generate_analysis``'s own
``admit`` hook; a call that would take the lane's running total past :data:`SPEND_CEILING_USD` is refused
before it is sent, and the test fails rather than spend. Up to four calls run (two generations of at most
two calls), so the worst case is four ceilings; a typical run is two calls.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import Any

import pytest

from threetears.evals.analysis.errors import GenerationError
from threetears.evals.analysis.gen_prompt import EVAL_ANALYSIS_GEN_DEFAULT
from threetears.evals.analysis.generator import (
    DEFAULT_ANALYSIS_GEN_ANSWER_BUDGET_TOKENS,
    DEFAULT_ANALYSIS_GEN_REASONING_BUDGET_TOKENS,
    GenerationTally,
    generate_analysis,
)
from threetears.evals.contracts.campaign import EvalAnalysis
from threetears.evals.contracts.models import utc_now_iso
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: The writer when none is named.
DEFAULT_MODEL = "claude-opus-5-5"

#: Model id -> list price (input, output), USD per million tokens. Check it before trusting the ceiling.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

#: The most the whole lane may be priced at, worst case, before a call is refused unsent.
SPEND_CEILING_USD = 5.00

#: Each call's output cap: the generator's own default reasoning and answer budgets.
MAX_TOKENS = DEFAULT_ANALYSIS_GEN_REASONING_BUDGET_TOKENS + DEFAULT_ANALYSIS_GEN_ANSWER_BUDGET_TOKENS

#: Characters per token assumed when pricing a prompt before it is sent; low, so the price is high.
CHARS_PER_TOKEN = 3

_ENABLED = os.environ.get("THREETEARS_EVALS_PAID") == "1"
_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_MODEL = os.environ.get("THREETEARS_EVALS_PAID_MODEL", DEFAULT_MODEL)

pytestmark = [
    pytest.mark.paid,
    pytest.mark.skipif(not _ENABLED, reason="spends money: set THREETEARS_EVALS_PAID=1 to run the paid lane"),
    pytest.mark.skipif(not _KEY, reason="no ANTHROPIC_API_KEY: the paid lane has no provider to call"),
]


@dataclass(frozen=True)
class _Completion:
    """One real completion, in the fields the engine reads."""

    content: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None
    price_source: str | None
    model: str
    served_model: str | None
    stop_reason: str
    temperature: float | None


class _Claude:
    """The writer: Claude over the Anthropic SDK, streamed, priced at list price."""

    def __init__(self, model: str) -> None:
        import anthropic  # imported here so the default run never needs a client

        self.model_name = model
        self._client = anthropic.AsyncAnthropic()
        self._rates = PRICES[model]

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float:
        prompt_chars = len(system) + len(user) + len(repr(response_format or {}))
        input_rate, output_rate = self._rates
        return (prompt_chars / CHARS_PER_TOKEN * input_rate + MAX_TOKENS * output_rate) / 1e6

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Completion:
        config: dict[str, Any] = {}
        if (response_format or {}).get("type") == "json_schema":
            config["format"] = {"type": "json_schema", "schema": response_format["json_schema"]["schema"]}  # type: ignore[index]
        async with self._client.messages.stream(
            model=self.model_name,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            **({"output_config": config} if config else {}),
        ) as stream:
            response = await stream.get_final_message()
        usage = response.usage
        details = getattr(usage, "output_tokens_details", None)
        input_rate, output_rate = self._rates
        stopped = {"end_turn": "end_turn", "max_tokens": "max_tokens", "refusal": "content_filter"}
        return _Completion(
            content="".join(block.text for block in response.content if block.type == "text"),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=getattr(details, "thinking_tokens", None),
            cost_usd=(usage.input_tokens * input_rate + usage.output_tokens * output_rate) / 1e6,
            price_source="anthropic list price, from the paid lane's PRICES",
            model=self.model_name,
            served_model=response.model,
            stop_reason=stopped.get(response.stop_reason or "", "error"),
            temperature=None,
        )


@dataclass
class _Spend:
    """The lane's running worst-case total, shared by every call it admits."""

    priced_usd: float = 0.0
    refused: list[str] = field(default_factory=list)


_SPEND = _Spend()


def _admit(client: _Claude) -> Any:
    def admit(system: str, user: str, response_format: dict[str, Any] | None) -> None:
        price = client.price_ceiling(system=system, user=user, response_format=response_format)
        if _SPEND.priced_usd + price > SPEND_CEILING_USD:
            message = (
                f"refused before sending: this call is priced at up to ${price:.2f} and the lane has priced "
                f"${_SPEND.priced_usd:.2f} of its ${SPEND_CEILING_USD:.2f} ceiling"
            )
            _SPEND.refused.append(message)
            raise GenerationError(message)
        _SPEND.priced_usd += price

    return admit


class _BrokenContract:
    """Sends the contract with ``headline`` removed, so no compliant output can carry it."""

    def __init__(self, inner: _Claude) -> None:
        self._inner = inner
        self.model_name = inner.model_name

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Completion:
        broken = copy.deepcopy(response_format)
        assert broken is not None
        schema = broken["json_schema"]["schema"]
        schema["required"] = [name for name in schema["required"] if name != "headline"]
        del schema["properties"]["headline"]
        return await self._inner.generate(system=system, user=user, response_format=broken)


async def _generate(client: Any, tally: GenerationTally, admit: Any) -> EvalAnalysis:
    profile = toyhost_profile()
    analysis, _insights = await generate_analysis(
        toyhost_bundle(profile=profile),
        prompt=EVAL_ANALYSIS_GEN_DEFAULT,
        model=client.model_name,
        client=client,
        prompt_id="eval_analysis_gen",
        bundle_assembled_at=utc_now_iso(),
        tally=tally,
        admit=admit,
        profile=profile,
    )
    return analysis


async def test_a_real_writer_produces_a_valid_analysis_whose_evidence_resolves() -> None:
    client = _Claude(_MODEL)
    tally = GenerationTally()

    analysis = await _generate(client, tally, _admit(client))

    assert EvalAnalysis.model_validate(analysis.model_dump(mode="json")) == analysis, "it round-trips the contract"
    assert analysis.generation.repair_attempts <= 1, "no repair loop exhausted"
    cells = {entry.variant_key for entry in toyhost_bundle(profile=toyhost_profile()).variant_index}
    for finding in analysis.document.findings:
        for row in finding.evidence:
            assert row.cell_ref.split(":", 1)[0] in cells, f"evidence names a cell the bundle lacks: {row.cell_ref}"
            assert math.isfinite(row.value) and row.n >= 0, "every cited value was filled by code"
    assert analysis.generation.token_cost <= SPEND_CEILING_USD
    print(
        f"\npaid lane: {_MODEL} wrote {len(analysis.document.findings)} finding(s) in {tally.calls} call(s) for "
        f"${analysis.generation.token_cost:.4f} (repairs: {analysis.generation.repair_attempts})"
    )


async def test_a_broken_contract_fails_the_lane() -> None:
    """With ``headline`` gone from the contract, both calls are refused, so the lane cannot pass a broken prompt."""
    client = _Claude(_MODEL)
    tally = GenerationTally()

    with pytest.raises(GenerationError) as refused:
        await _generate(_BrokenContract(client), tally, _admit(client))

    assert not _SPEND.refused, f"the ceiling refused a call, which is not the failure under test: {_SPEND.refused}"
    assert tally.calls == 2 and len(tally.refusals) == 2, "the first output and its one repair were both refused"
    assert "headline" in tally.refusals[0] and "headline" in str(refused.value)
