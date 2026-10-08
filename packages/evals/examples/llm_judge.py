"""Grade open-ended answers with an LLM judge and a rubric, in one file.

Rung zero (``rung_zero.py``) grades with code. Some answers have no code that grades them: is this
reply helpful, and does it say only what its source supports? Here a model answers questions about a
short store policy, and a second model call, the judge, scores each answer on two rubric dimensions.
A cheap code scorer runs beside it.

Run it with ``python packages/evals/examples/llm_judge.py``.

- With ``ANTHROPIC_API_KEY`` set, the candidate and the judge are both Claude (``claude-haiku-5-5``),
  through the ``anthropic`` SDK (``pip install anthropic``).
- Without it, both are OFFLINE STAND-INS: a keyword matcher answers and a word-overlap script judges.
  The run goes through exactly the same engine path, so you can see what the output looks like, but
  its scores say nothing about any model. The output says which mode it ran in.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass

from threetears.evals.quick import EvalSummary, Judge, run_eval

MODEL = "claude-haiku-5-5"

# --- 1. What the candidate answers from, and the questions it is asked ---------------------------

POLICY = """\
Returns: unworn items can be returned within 30 days of delivery for a full refund.
Sale items can be exchanged but not refunded.
Refunds go back to the original payment method within 5 business days of the return arriving.
Shipping: orders over $50 ship free; other orders pay a flat $6.
We ship to the US and Canada only."""

# A case is any JSON object; the candidate, the scorers and the judge each receive it as given.
# The last question is not answered by the policy: a grounded answer says so rather than guessing.
CASES = [
    {"question": "How long do I have to return a jacket I haven't worn?"},
    {"question": "Can I get my money back for something I bought on sale?"},
    {"question": "How much is shipping on a $30 order?"},
    {"question": "Do you ship to Mexico?"},
    {"question": "Do you offer gift wrapping?"},
]

SYSTEM = "Answer the customer's question in one or two sentences, using only the store policy below. "
SYSTEM += "If the policy does not say, tell them you don't know.\n\n" + POLICY


# --- 2. A minimal adapter from the anthropic SDK to the engine's CompletionClient --------------
#
# The engine never names a provider: it calls whatever client you hand it through one small
# protocol, ``generate(*, system, user, response_format)``, and reads tokens, dollars and the
# model's name off the reply. Pricing is the client's job too, so this adapter prices each call
# from the usage the API reports. The rates are Anthropic's list prices for the model; check them
# against the current price list before you trust the dollars.

RATES_PER_MILLION = {"claude-haiku-5-5": (0.10, 0.50)}  # (input, output) USD per million tokens

# The engine's words for why a completion stopped; anything it does not know reads as an error.
STOP_REASONS = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "max_tokens": "max_tokens",
    "refusal": "content_filter",
}


@dataclass(frozen=True)
class Completion:
    """One reply, in the attribute names the engine reads (``CompletionResult``)."""

    content: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None  # None means "nobody priced this", which the engine keeps distinct from $0
    price_source: str | None
    model: str  # the model asked for
    served_model: str | None  # the model the response says answered
    stop_reason: str


class ClaudeClient:
    """``anthropic.AsyncAnthropic`` behind the engine's ``CompletionClient`` protocol."""

    def __init__(self, model: str) -> None:
        import anthropic  # imported here so the offline run never needs the package

        self.model = model
        self.spent_usd = 0.0  # every call's cost, the candidate's included, which the engine cannot see
        self._client = anthropic.AsyncAnthropic()

    async def generate(self, *, system: str, user: str, response_format: dict | None = None) -> Completion:
        # ``response_format`` is an OpenAI-style JSON directive; the judge's prompt already asks for
        # JSON, and the Messages API needs nothing more, so it is not sent.
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},  # short answers and rubric scores need little thinking
        )
        usage = response.usage
        details = usage.output_tokens_details
        input_rate, output_rate = RATES_PER_MILLION[self.model]
        cached_write = usage.cache_creation_input_tokens or 0  # billed at 1.25x input
        cached_read = usage.cache_read_input_tokens or 0  # billed at 0.1x input
        cost = (usage.input_tokens + 1.25 * cached_write + 0.1 * cached_read) * input_rate / 1e6
        cost += usage.output_tokens * output_rate / 1e6
        self.spent_usd += cost
        return Completion(
            content="".join(block.text for block in response.content if block.type == "text"),
            input_tokens=usage.input_tokens + cached_write + cached_read,
            output_tokens=usage.output_tokens,
            reasoning_tokens=details.thinking_tokens if details is not None else None,
            cost_usd=round(cost, 8),
            price_source="anthropic list price, from llm_judge.py",
            model=self.model,
            served_model=response.model,
            stop_reason=STOP_REASONS.get(response.stop_reason or "", "error"),
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> "ClaudeClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


# --- 3. The offline stand-ins, used only when there is no API key -------------------------------


async def offline_answer(case: dict) -> str:
    """Offline stand-in candidate: the policy line sharing the most words with the question."""
    asked = set(re.findall(r"[a-z]+", case["question"].lower())) - {"do", "you", "i", "a", "the", "can", "for", "on"}
    best = max(POLICY.splitlines(), key=lambda line: len(asked & set(re.findall(r"[a-z]+", line.lower()))))
    return best if asked & set(re.findall(r"[a-z]+", best.lower())) else "I don't know; the policy doesn't say."


class OfflineJudge:
    """Offline stand-in judge, speaking the same protocol as ClaudeClient. A script, not a model.

    It reads the two sections the engine's judge prompt carries (the case material and the output
    under review) and the dimension it is asked for, and scores by word overlap.
    """

    async def generate(self, *, system: str, user: str, response_format: dict | None = None) -> Completion:
        dim = re.search(r'single key "([^"]+)"', system).group(1)
        material, output = user.split("# Output under review\n", 1)
        words = set(re.findall(r"[a-z]+", output.lower()))
        if dim.endswith("grounded"):  # share of the answer's words that appear in the policy and question
            score = 1 + round(4 * len(words & set(re.findall(r"[a-z]+", material.lower()))) / max(len(words), 1))
        else:  # "helpful": an answer that commits to something beats one that does not
            score = 2 if "don't know" in output else 4
        reply = {"reasoning": "offline stand-in: word overlap, not a model", "criteria_scores": {dim: score}}
        return Completion(json.dumps(reply), None, None, None, 0.0, "offline stand-in", "offline", None, "end_turn")

    async def aclose(self) -> None:
        pass

    async def __aenter__(self) -> "OfflineJudge":
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass


# --- 4. The grades: one code scorer, and the judge's rubric -------------------------------------


def concise(case: dict, answer: str) -> bool:
    """Whether the answer kept to 40 words or fewer."""
    return len(answer.split()) <= 40


RUBRIC = {
    # Each entry is one judge call per answer, scored 1 (worst) to 5 (best). Write what a 5 looks like.
    "helpful": "The answer directly resolves the customer's question, or clearly says the policy does not cover it.",
    "grounded": "Every claim in the answer is stated in the store policy; nothing is invented or assumed.",
}


def judged_against(case: dict) -> str:
    """What the judge sees beside each answer: the policy the candidate had, and the question."""
    return f"Store policy:\n{POLICY}\n\nCustomer question: {case['question']}"


# --- 5. Run it ----------------------------------------------------------------------------------


async def main() -> EvalSummary:
    """Answer every question twice, grade each answer, and print what the run measured."""
    live = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if live:
        print(f"LIVE: the candidate and the judge are {MODEL}.")
        claude = ClaudeClient(MODEL)

        async def answer(case: dict) -> str:
            """Answer a customer's question from the store policy."""
            reply = await claude.generate(system=SYSTEM, user=case["question"])
            return reply.content

        candidate, judge_client, judge_model, arm = answer, claude, MODEL, MODEL
    else:
        print("OFFLINE: ANTHROPIC_API_KEY is not set, so the candidate and the judge are scripted stand-ins,")
        print("not models. The run below shows the shape of the output; its scores say nothing about Claude.")
        candidate, judge_client, judge_model, arm = offline_answer, OfflineJudge(), "offline-stand-in", "offline"

    judge = Judge(client=judge_client, model=judge_model, rubric=RUBRIC, case_material=judged_against)
    summary = await run_eval(CASES, candidate, [concise], judge=judge, scope_id="llm-judge", k=2, model=arm)
    print(summary.render())
    if live:
        # The judge's spend is in the summary; the candidate is your own code, so its calls are not.
        print(f"  all Claude spend, candidate included: ${claude.spent_usd:.6f}")
        await claude.aclose()
    return summary


if __name__ == "__main__":
    asyncio.run(main())
