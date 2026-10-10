"""Is each answer helpful, and does it say only what its source supports?

No code grades that, so a model does: a model answers questions about a store policy, and a judge, a second
model call, grades each answer on a rubric. New here: ``Judge``, passed to ``run_eval`` as ``judge=``: the client
it calls, that client's model, the rubric, and ``case_material``, what it reads beside each answer. Each dimension
here is pass/fail, so a fail is a miss and ``summary.misses()`` carries the judge's reason. Read them: offline,
the script judge fails an honest "I don't know". How far to trust a judge: ``docs/judges-and-calibration.md``.

Run it with ``python packages/evals/examples/llm_judge.py``. With ``ANTHROPIC_API_KEY`` set, Claude answers and
judges: 30 short calls (5 questions x 2 repeats, each answered once and judged on 2 dimensions), well under a
cent. Without it, a keyword matcher answers and a word-overlap script judges; they say nothing about Claude.
"""

import asyncio
import json
import re
from types import SimpleNamespace

from _live import Completion, claude, online
from threetears.evals.quick import EvalSummary, Judge, run_eval

MODEL = "claude-haiku-5-5"

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
# 2. The judge's rubric, and what it reads beside each answer.
# -----------------------------------------------------------------------------

RUBRIC = {  # one judge call per answer and dimension: write what a pass looks like
    "helpful": "The answer directly resolves the customer's question, or clearly says the policy does not cover it.",
    "grounded": "Every claim in the answer is stated in the store policy; nothing is invented or assumed.",
}


def judged_against(case: dict) -> str:
    return f"Store policy:\n{POLICY}\n\nCustomer question: {case['question']}"


# -----------------------------------------------------------------------------
# 3. The OFFLINE stand-ins: scripts, not models. Live, both are Claude (``_live.py``).
# -----------------------------------------------------------------------------


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


async def offline_answer(case: dict) -> str:
    """Answer a customer's question from the store policy."""  # a stand-in: the policy line sharing most words
    asked = words(case["question"]) - {"do", "you", "i", "a", "the", "can", "for", "on"}
    best = max(POLICY.splitlines(), key=lambda line: len(asked & words(line)))
    return best if asked & words(best) else "I don't know; the policy doesn't say."


async def offline_judge(*, system: str, user: str, response_format: dict | None = None) -> Completion:
    """A word-overlap script behind the client protocol, reading the judge prompt the engine sends."""
    dimension = re.search(r'single key "([^"]+)"', system).group(1)
    material, output = user.split("# Output under review\n", 1)
    said, source = words(output), words(material)
    if dimension.endswith("grounded"):  # passes when most of the answer's words are in the policy or question
        passed = len(said & source) >= 0.8 * len(said)
        why = f"{len(said & source)} of the answer's {len(said)} words are in the policy or question"
    else:  # helpful: passes when the answer shares a word with the question, or says the policy does not cover it
        passed = bool(said & words(material.split("Customer question:")[-1])) or "don't know" in output
        why = "it answers the question" if passed else "it shares no word with the question"
    verdict = "pass" if passed else "fail"
    reply = {"reasoning": f"offline stand-in, word overlap: {why}", "criteria_scores": {dimension: verdict}}
    return Completion(json.dumps(reply), None, None, None, 0.0, "offline", "offline-judge", None, "end_turn", None)


# -----------------------------------------------------------------------------
# 4. Answer every question twice, judge each answer, print the summary, then read every miss.
# -----------------------------------------------------------------------------


async def main() -> EvalSummary:
    if online():
        print(f"Running against Claude ({MODEL}).\n")
        client, model = claude(MODEL), MODEL  # one client, shared by the candidate and the judge

        async def candidate(case: dict) -> str:
            """Answer a customer's question from the store policy."""
            return (await client.generate(system=SYSTEM, user=case["question"])).content
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with a keyword matcher answering and a script judging.\n")
        client, model, candidate = SimpleNamespace(generate=offline_judge), "offline-judge", offline_answer

    # The judge: a client you own (the run never closes it), the model it calls, the rubric, and what it reads.
    judge = Judge(client=client, model=model, rubric=RUBRIC, scale="pass_fail", case_material=judged_against)
    # model= names the candidate's model, so the summary can warn when a model judges its own answers.
    summary = await run_eval(
        CASES, candidate, judge=judge, scope_id="llm-judge", k=2, model=MODEL if online() else None
    )
    print(summary.render())  # each dimension's pass rate, then what the judge spent

    print("\nWhat the judge failed, and why. A judge can be wrong too, so read them:")
    for miss in summary.misses():
        print(f"  {miss.input['question']!r}, repeat {miss.repeat}, answered {miss.answer!r}")
        print(f"    {'; '.join(miss.missed_because)}")
    if online():
        await client.aclose()
    return summary


if __name__ == "__main__":
    asyncio.run(main())
