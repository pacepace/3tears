"""Is each answer helpful, and does it say only what its source supports?

No code grades that, so a model does: a model answers questions about a store policy, and a second call, the
judge, scores each answer on a rubric, beside a code scorer. New here: ``Judge``, a completion client, its model
and a rubric, passed to ``run_eval`` as ``judge=``; ``intent=``, what the judge is told each case asks; and
``Answer``, the candidate's reply plus the tokens and dollars its call spent.
How far to trust it: ``docs/reading-reports.md``.

Run it with ``python packages/evals/examples/llm_judge.py``. With ``ANTHROPIC_API_KEY`` set, Claude answers and
judges (30 short calls, well under a cent); without it, a keyword matcher answers and a word-overlap script
judges, and their scores say nothing about Claude.
"""

import asyncio
import json
import os
from collections import namedtuple
import re
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

from threetears.evals.quick import Answer, EvalSummary, Judge, run_eval

MODEL = "claude-haiku-5-5"
RATES = (0.10, 0.50)  # Haiku 5.5's (input, output) list price, USD per million tokens, for prompts up to 100K

# -----------------------------------------------------------------------------
# 1. What the candidate answers from, and the questions it is asked.
# -----------------------------------------------------------------------------

POLICY = """\
Returns: unworn items can be returned within 30 days of delivery for a full refund.
Sale items can be exchanged but not refunded.
Refunds go back to the original payment method within 5 business days of the return arriving.
Shipping: orders over $50 ship free; other orders pay a flat $6.
We ship to the US and Canada only."""

CASES = [
    {"question": "How long do I have to return a jacket I haven't worn?"},
    {"question": "Can I get my money back for something I bought on sale?"},
    {"question": "How much is shipping on a $30 order?"},
    {"question": "Do you ship to Mexico?"},
    {"question": "Do you offer gift wrapping?"},  # the policy doesn't say: a grounded answer says so
]

SYSTEM = "Answer the customer's question in one or two sentences, using only the store policy below. "
SYSTEM += "If the policy does not say, tell them you don't know.\n\n" + POLICY

# -----------------------------------------------------------------------------
# 2. The grades: one code scorer, and the judge's rubric and intent.
# -----------------------------------------------------------------------------


# The engine reads this docstring's first line: it is the measure's description.
def concise(case: dict, answer: str) -> bool:
    """Whether the answer kept to 40 words or fewer."""
    return len(answer.split()) <= 40


RUBRIC = {  # one judge call per answer and dimension, scored 1 (worst) to 5 (best): write what a 5 looks like
    "helpful": "The answer directly resolves the customer's question, or clearly says the policy does not cover it.",
    "grounded": "Every claim in the answer is stated in the store policy; nothing is invented or assumed.",
}
INTENT = "Answer a customer's question from the store policy."  # what the judge is told each case asks


def judged_against(case: dict) -> str:  # what the judge reads beside each answer
    return f"Store policy:\n{POLICY}\n\nCustomer question: {case['question']}"


# -----------------------------------------------------------------------------
# 3. The live client: Claude behind the engine's ``CompletionClient`` protocol.
#
# The engine calls ``generate(system=, user=, response_format=)`` and reads a ``Completion``'s fields off the
# reply. It never names a provider, so pricing is the client's job; a cost of None is unpriced, never $0.
# -----------------------------------------------------------------------------

Completion = namedtuple(
    "Completion",
    "content input_tokens output_tokens reasoning_tokens cost_usd price_source model served_model stop_reason",
)


def claude_client() -> Any:
    """Claude as a completion client, shared by the candidate and the judge."""
    import anthropic  # imported here so the offline path does not need the package

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY

    async def generate(*, system: str, user: str, response_format: dict | None = None) -> Completion:
        # The judge's prompt already asks for JSON, so the OpenAI-style ``response_format`` is not sent.
        response = await client.messages.create(
            model=MODEL,
            max_tokens=2048,
            output_config={"effort": "low"},  # short answers and rubric scores need little thought
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        usage, details = response.usage, response.usage.output_tokens_details
        stopped = {"end_turn": "end_turn", "max_tokens": "max_tokens", "refusal": "content_filter"}
        return Completion(
            content="".join(block.text for block in response.content if block.type == "text"),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=details.thinking_tokens if details is not None else None,
            cost_usd=(usage.input_tokens * RATES[0] + usage.output_tokens * RATES[1]) / 1e6,
            price_source="anthropic list price, from llm_judge.py",
            model=MODEL,
            served_model=response.model,
            stop_reason=stopped.get(response.stop_reason or "", "error"),
        )

    return SimpleNamespace(generate=generate, aclose=client.close)


def claude_answerer(claude: Any) -> Callable[[dict], Awaitable[Answer]]:
    """The candidate: Claude answering from the policy, its spend returned beside the answer."""

    async def answer(case: dict) -> Answer:
        """Answer a customer's question from the store policy."""
        reply = await claude.generate(system=SYSTEM, user=case["question"])
        spent = {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens, "cost_usd": reply.cost_usd}
        return Answer(reply.content, model=MODEL, **spent)

    return answer


# -----------------------------------------------------------------------------
# 4. The OFFLINE stand-ins: scripts, not models.
# -----------------------------------------------------------------------------


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


async def offline_answer(case: dict) -> str:
    """Offline stand-in candidate: the policy line sharing the most words with the question."""
    asked = words(case["question"]) - {"do", "you", "i", "a", "the", "can", "for", "on"}
    best = max(POLICY.splitlines(), key=lambda line: len(asked & words(line)))
    return best if asked & words(best) else "I don't know; the policy doesn't say."


def offline_judge() -> Any:
    """A word-overlap script behind the same protocol, reading the judge prompt the engine sends."""

    async def generate(*, system: str, user: str, response_format: dict | None = None) -> Completion:
        dimension = re.search(r'single key "([^"]+)"', system).group(1)
        material, output = user.split("# Output under review\n", 1)
        if dimension.endswith("grounded"):  # the share of the answer's words found in the policy and question
            score = 1 + round(4 * len(words(output) & words(material)) / max(len(words(output)), 1))
        else:  # helpful: an answer that commits to something beats one that does not
            score = 2 if "don't know" in output else 4
        reply = {"reasoning": "offline stand-in: word overlap, not a model", "criteria_scores": {dimension: score}}
        return Completion(json.dumps(reply), None, None, None, 0.0, "offline stand-in", "offline", None, "end_turn")

    return SimpleNamespace(generate=generate)


# -----------------------------------------------------------------------------
# 5. Answer every question twice, grade each answer with the scorer and the judge, and print the summary.
# -----------------------------------------------------------------------------


async def main() -> EvalSummary:
    online = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if online:
        print(f"Running against Claude ({MODEL}).\n")
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with a keyword matcher answering and a script judging.\n")

    client = claude_client() if online else offline_judge()
    candidate, model = (claude_answerer(client), MODEL) if online else (offline_answer, "offline")
    # The judge: a client you own (the run never closes it), the model it calls, the rubric, and what it reads.
    judge = Judge(client=client, model=model, rubric=RUBRIC, case_material=judged_against)
    summary = await run_eval(
        CASES, candidate, [concise], judge=judge, intent=INTENT, scope_id="llm-judge", k=2, model=model
    )
    print(summary.render())  # each dimension's mean, then the judge's spend and, online, the candidate's
    if online:
        await client.aclose()
    return summary


if __name__ == "__main__":
    asyncio.run(main())
