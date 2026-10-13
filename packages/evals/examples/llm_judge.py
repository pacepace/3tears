"""Is each answer helpful, and does it say only what its source supports?

No code grades that, so a model does: a model answers questions about a store policy, and a judge, a second
model call, grades each answer on a rubric. New here: ``Judge``, passed to ``run_eval`` as ``judge=``: the client
it calls, that client's model, the rubric, and ``case_material``, what it reads beside each answer. Each dimension
here is pass/fail, so a fail is a miss and ``summary.misses()`` carries the judge's reason. Read them: a judge can
be wrong too. How far to trust a judge: ``docs/judges-and-calibration.md``.

Run it with ``python llm_judge.py`` (``README.md`` beside it says how). With ``ANTHROPIC_API_KEY`` set, Claude
answers and judges: 30 short calls (5 questions x 2 repeats, each answered once and judged on 2 dimensions), about a
cent. Without it, the example replays ``recordings/llm_judge.jsonl``, what Claude said when the run was recorded, so
it calls nothing and costs nothing. ``RECORD=1 python llm_judge.py``, with the key set, records it again.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from threetears.evals.analysis import EvalSummary
from threetears.evals.providers import models_client
from threetears.evals.quick import Judge, run_eval
from threetears.evals.run import JUDGE_REQUEST_SETTINGS
from threetears.evals.schema import ClientRequestSettings
from threetears.evals.testing import RecordedCompletions

MODEL = "claude-haiku-5-5"
#: How every answer is asked for: an output cap and low effort. No temperature: this model refuses one.
SETTINGS = ClientRequestSettings(max_tokens=1024, reasoning_effort="low")
#: How every judge call is asked for: as a judged run records it, its output cap and `high` effort.
JUDGE_SETTINGS = JUDGE_REQUEST_SETTINGS
RECORDING = Path(__file__).with_name("recordings") / "llm_judge.jsonl"
RERECORD = "RECORD=1 python llm_judge.py"

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
# 3. The clients: Claude when a key is set, else the recording of a run that called it.
# -----------------------------------------------------------------------------


def client(model: str, settings: ClientRequestSettings) -> Any:
    """``model``, asked as ``settings`` say: live with ``ANTHROPIC_API_KEY`` (recorded with ``RECORD=1``), else replayed."""
    key, recording = os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("RECORD") == "1"
    if not key:
        if recording:
            raise SystemExit("RECORD=1 records a live run, which needs ANTHROPIC_API_KEY")
        return RecordedCompletions.replay(RECORDING, settings=settings, model=model, rerecord=RERECORD)
    live = models_client(
        model,
        api_key=key,
        provider="anthropic",  # models' registry does not place this id on its own
        max_tokens=settings.max_tokens,
        reasoning_effort=settings.reasoning_effort,
        reasoning_max_tokens=settings.reasoning_max_tokens,
    )  # and no temperature, so the model's own sampling applies
    # Every client is built before the first call, so replace=True clears the old recording and nothing of this run.
    return RecordedCompletions.record(live, RECORDING, settings=settings, replace=True) if recording else live


def how_it_runs(what: str) -> str:
    """The first line printed: running live, recording, or replaying a recording made on its date."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        recording = f", recording it into recordings/{RECORDING.name}" if os.environ.get("RECORD") == "1" else ""
        return f"Running live against Claude ({what}){recording}."
    made = json.loads(RECORDING.read_text(encoding="utf-8").splitlines()[0])["recorded_at"][:10]
    return (
        f"ANTHROPIC_API_KEY is not set: replaying recordings/{RECORDING.name}, Claude ({what}) as recorded on {made}."
    )


# -----------------------------------------------------------------------------
# 4. Answer every question twice, judge each answer, print the summary, then read every miss.
# -----------------------------------------------------------------------------


async def main() -> EvalSummary:
    answering, judging = client(MODEL, SETTINGS), client(MODEL, JUDGE_SETTINGS)
    print(how_it_runs(MODEL) + "\n")

    async def candidate(case: dict) -> str:
        """Answer a customer's question from the store policy."""
        return (await answering.generate(system=SYSTEM, user=case["question"])).content

    # The judge: a client you own (the run never closes it), the model it calls, the rubric, and what it reads.
    judge = Judge(client=judging, model=MODEL, rubric=RUBRIC, scale="pass_fail", case_material=judged_against)
    try:
        # model= names the candidate's model, so the summary can warn when a model judges its own answers.
        summary = await run_eval(CASES, candidate, judge=judge, scope_id="llm-judge", k=2, model=MODEL)
    finally:
        await answering.aclose()
        await judging.aclose()
    print(summary.render())  # each dimension's pass rate, then what the judge spent

    print("\nWhat the judge failed, and why. A judge can be wrong too, so read them:")
    for miss in summary.misses():
        print(f"  {miss.input['question']!r}, repeat {miss.repeat}, answered {miss.answer!r}")
        print(f"    {'; '.join(miss.missed_because)}")
    return summary


if __name__ == "__main__":
    asyncio.run(main())
