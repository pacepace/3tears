"""What does the new prompt change, and is the difference real or noise?

Two versions of a support-ticket classifier, differing only in their prompt, run over the same tickets, and the
engine tests the new one against the old one, the control. New here: ``compare``, which runs each candidate as
one arm and reports every arm against the control: the difference (delta), its interval, the Holm-adjusted p,
and a verdict. Then read each arm per kind of case (``stratum_key=``, ``comparison.strata()``) and where the arms
disagree, through ``comparison.results(arm)``. The verdicts and what
each licenses: ``docs/reading-reports.md``.

The tickets are a file beside this one, ``tickets.jsonl``, not a list in the script, so the next prompt version,
and every one after it, runs against exactly the same cases. When production gets a ticket wrong, add it there
(``docs/designing-classifier-evals.md``, section 8, "Maintaining the set"). When the suite needs versions and
history, store it as a case set with ``mint_case_set`` or the ``case_set_mint`` action (``docs/concepts.md``,
"Case set").

Run it with ``python compare_two_prompts.py`` (``README.md`` beside it says how). With ``ANTHROPIC_API_KEY`` set it
calls Claude 48 times (12 tickets x 2 repeats x 2 prompts) for well under a cent; without it, it replays
``recordings/compare_two_prompts.jsonl``, what Claude answered when the run was recorded, and calls nothing.
``RECORD=1 python compare_two_prompts.py``, with the key set, records it again. A current model may get every
ticket right under both prompts, and then "not separated" is the right answer: these tickets cannot tell the
prompts apart, and harder ones are needed. It never means the prompts are equally good.
"""

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from threetears.evals.providers import models_client
from threetears.evals.quick import Comparison, compare
from threetears.evals.schema import ClientRequestSettings
from threetears.evals.testing import RecordedCompletions

MODEL = "claude-haiku-5-5"
#: How every call is asked for: an output cap and low effort. No temperature: this model refuses one.
SETTINGS = ClientRequestSettings(max_tokens=1024, reasoning_effort="low")
RECORDING = Path(__file__).with_name("recordings") / "compare_two_prompts.jsonl"
RERECORD = "RECORD=1 python compare_two_prompts.py"

# -----------------------------------------------------------------------------
# 1. The cases: each ticket, and the queue a person filed it under, kept in a file beside this script.
#
# The test set is data, not code: the next prompt version runs against exactly these cases, and when production
# gets a ticket wrong, it becomes a new line here (docs/designing-classifier-evals.md, section 8, "Maintaining the
# set"). Each line's "kind" marks the hard ones, the lookalikes that mention a second queue's words, and is the
# case's stratum (stratum_key="kind" below): the report reads each arm again per kind, and compare takes the key
# out of every case before an arm sees it, since a ticket that says it is a lookalike tells the model what to watch
# for. When the suite needs versions and history, store it as a case set (mint_case_set, or the case_set_mint
# action: docs/concepts.md, "Case set").
# -----------------------------------------------------------------------------

TICKETS = Path(__file__).with_name("tickets.jsonl")
CASES = [json.loads(line) for line in TICKETS.read_text().splitlines() if line.strip()]

# -----------------------------------------------------------------------------
# 2. The two prompts under comparison.
# -----------------------------------------------------------------------------

BASELINE_PROMPT = (
    "Classify the support ticket into one of: billing, bug, account, feature_request. Answer with the label only."
)

CANDIDATE_PROMPT = """Classify the support ticket into exactly one queue.

- billing: charges, refunds, invoices, plans and payments.
- bug: the product does something wrong or fails — errors, crashes, broken pages.
- account: access to and details of the customer's own account, when nothing is broken.
- feature_request: something the product does not do yet.

When a ticket touches two queues, choose the one whose team must act first: money taken
without the service delivered is billing; a login that fails because the product misbehaves is a bug.

Answer with the label only."""

# -----------------------------------------------------------------------------
# 3. The candidate: an async function from a case to a label, one per prompt, both asking the same client.
# -----------------------------------------------------------------------------

Candidate = Callable[[Mapping[str, Any]], Awaitable[str]]


def classifier(llm: Any, prompt: str) -> Candidate:
    """A candidate that asks ``llm`` to classify each ticket under ``prompt``."""

    async def classify(case: Mapping[str, Any]) -> str:
        return (await llm.generate(system=prompt, user=case["ticket"])).content.strip().lower()

    return classify


# -----------------------------------------------------------------------------
# 4. The client: Claude when a key is set, else the recording of a run that called it.
# -----------------------------------------------------------------------------


def client(model: str, settings: ClientRequestSettings) -> Any:
    """``model``, asked as ``settings`` say: live with ``ANTHROPIC_API_KEY`` (recorded with ``RECORD=1``), else replayed."""
    key, recording = os.environ.get("ANTHROPIC_API_KEY"), os.environ.get("RECORD") == "1"
    if not key:
        if recording:
            raise SystemExit("RECORD=1 records a live run, which needs ANTHROPIC_API_KEY")
        return RecordedCompletions.replay(RECORDING, settings=settings, rerecord=RERECORD)
    live = models_client(
        model,
        api_key=key,
        provider="anthropic",  # models' registry does not place this id on its own
        max_tokens=settings.max_tokens,
        reasoning_effort=settings.reasoning_effort,
    )  # and no temperature, so the model's own sampling applies
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
# 5. Run both prompts over every ticket twice, test the candidate against the baseline, print the verdict.
# -----------------------------------------------------------------------------


async def main() -> Comparison:
    llm = client(MODEL, SETTINGS)
    print(how_it_runs(MODEL) + "\n")

    try:
        comparison = await compare(
            CASES,
            {"baseline": classifier(llm, BASELINE_PROMPT), "candidate": classifier(llm, CANDIDATE_PROMPT)},  # by name
            expected=lambda case: case["queue"],  # a classifier eval: accuracy and a confusion matrix per arm
            stratum_key="kind",  # each case's kind: the report's By stratum table reads every arm per kind of case
            control="baseline",  # the arm every other arm is tested against
            name="ticket triage: baseline vs candidate prompt",  # titles the report
            k=2,  # repeats per case: a model's answer can change between calls
        )
    finally:
        await llm.aclose()

    # The verdict: each arm against the control on each reading, row["arm"] being the name you gave it.
    for row in comparison.contrasts():
        p = (
            "" if row["p_adjusted"] is None else f", Holm-adjusted p {row['p_adjusted']:.2g}"
        )  # none when nothing varied
        print(f"{row['arm']} vs {comparison.control} on {row['reading']}: {row['control_mean']:.2f} -> ", end="")
        print(f"{row['arm_mean']:.2f}, delta {row['delta']:+.2f}, interval {row['interval']}{p}: {row['verdict']}")

    # Where it changed: each arm's accuracy per kind of case. Twelve tickets make every kind too few to read
    # alone, which the report says; a real suite wants ten cases of each kind it acts on.
    print("\nAccuracy by kind of case:")
    for row in comparison.strata(reading="Accuracy"):
        print(f"  {row['arm']}: all {row['all']}; " + "; ".join(f"{k} {v}" for k, v in row["strata"].items()))

    # Where the arms disagree: each arm's answer to a ticket, on every repeat.
    print("\nWhere the arms disagree:")
    for index, case in enumerate(CASES):
        said = {arm: [r.answer for r in comparison.results(arm) if r.case == str(index)] for arm in comparison.arms}
        if len({str(answers) for answers in said.values()}) > 1:
            print(f"  {case['ticket']!r}, filed under {case['queue']} ({case['kind']}):")
            print("    " + "; ".join(f"{arm} said {', '.join(answers)}" for arm, answers in said.items()))
    print("\nThe full report: print(comparison.render()), or reports.py to write it to files.")
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
