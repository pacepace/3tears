"""What does changing the prompt do at each effort?

Two things vary at once, the prompt and the reasoning effort the model is asked for: four arms, one per
combination. New here: ``factors=``, which keys each arm by its level of each factor (``("medium", "v2", MODEL)``;
every run is at some model, so the model is a factor too, at its one level), and ``Comparison.against``. Every arm
is tested against one control, (low, v1), so the prompt's effect at medium effort is read against a second control
over the same runs, with nothing run again. Levers and arms: ``docs/concepts.md``.

Run it with ``python prompt_x_effort.py`` (``README.md`` beside it says how). With ``ANTHROPIC_API_KEY`` set it
calls Claude 112 times (14 tickets x 2 repeats x 4 arms) for about two US cents; without it, it replays
``recordings/prompt_x_effort.jsonl``, what Claude answered when the run was recorded, and calls nothing.
``RECORD=1 python prompt_x_effort.py``, with the key set, records it again. A current model may get every ticket
right, and then "not separated" is the right answer: these tickets cannot tell the arms apart, and harder ones are
needed.
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
#: Each effort's request settings, by its level: the same output cap, and one effort each. No temperature: this
#: model refuses one.
EFFORTS = {
    "low": ClientRequestSettings(max_tokens=2048, reasoning_effort="low"),
    "medium": ClientRequestSettings(max_tokens=2048, reasoning_effort="medium"),
}
RECORDING = Path(__file__).with_name("recordings") / "prompt_x_effort.jsonl"
RERECORD = "RECORD=1 python prompt_x_effort.py"

# -----------------------------------------------------------------------------
# 1. The cases: the tickets from compare_two_prompts.py, and two more hard ones.
# -----------------------------------------------------------------------------

CASES = [
    {"ticket": "I was charged twice for my March invoice.", "queue": "billing"},
    {"ticket": "Can I get a refund for the annual plan I bought yesterday?", "queue": "billing"},
    {"ticket": "The export button does nothing when I click it in Firefox.", "queue": "bug"},
    {"ticket": "The app crashes every time I upload a photo larger than 10 MB.", "queue": "bug"},
    {"ticket": "I can't log in; the reset-password email never arrives.", "queue": "account"},
    {"ticket": "Please change the email address on my account to my work one.", "queue": "account"},
    {"ticket": "It would be great if reports could be scheduled weekly.", "queue": "feature_request"},
    {"ticket": "Do you plan to add a dark mode?", "queue": "feature_request"},
    {"ticket": "My card was charged but the upgrade page shows an error and nothing changed.", "queue": "billing"},
    {"ticket": "Since the update, the login page loops back to itself after I sign in.", "queue": "bug"},
    {"ticket": "I was charged for a seat after removing that user from my account.", "queue": "billing"},
    {"ticket": "Could you add an option to pay by invoice?", "queue": "feature_request"},
    {"ticket": "Will you add a monthly plan for small teams?", "queue": "feature_request"},
    {"ticket": "I was charged again after my account was closed.", "queue": "billing"},
]

# -----------------------------------------------------------------------------
# 2. The two things varied: these two prompts, and two efforts (``EFFORTS``, above).
# -----------------------------------------------------------------------------

PROMPTS = {
    "v1": "Classify the support ticket into one of: billing, bug, account, feature_request. "
    "Answer with the label only.",
    "v2": "Classify the support ticket into one of: billing, bug, account, feature_request. Money taken without "
    "the service delivered is billing; a login the product breaks is a bug; anything it does not do yet is a "
    "feature_request. Answer with the label only.",
}

# -----------------------------------------------------------------------------
# 3. One candidate per combination: an async function from a case to a label.
# -----------------------------------------------------------------------------

Candidate = Callable[[Mapping[str, Any]], Awaitable[str]]


def classifier(llm: Any, prompt: str) -> Candidate:
    """A candidate that asks ``llm`` to classify each ticket under prompt ``prompt``."""

    async def classify(case: Mapping[str, Any]) -> str:
        return (await llm.generate(system=PROMPTS[prompt], user=case["ticket"])).content.strip().lower()

    return classify


# -----------------------------------------------------------------------------
# 4. The clients, one per effort: Claude when a key is set, else the recording of a run that called it.
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
# 5. Run all four combinations over every ticket twice, then read v2 against v1 at each effort.
# -----------------------------------------------------------------------------


async def main() -> Comparison:
    clients = {effort: client(MODEL, settings) for effort, settings in EFFORTS.items()}
    low, medium = EFFORTS
    print(how_it_runs(f"{MODEL} at {low} and at {medium} effort") + "\n")

    try:
        comparison = await compare(
            CASES,
            {(effort, prompt, MODEL): classifier(llm, prompt) for effort, llm in clients.items() for prompt in PROMPTS},
            # Each arm is keyed by its level of each factor, in this order. Every run is at some model, so the model
            # is a factor too, at one level: the arms share it.
            factors=("effort", "prompt", "model"),
            control=(low, "v1", MODEL),
            expected=lambda case: case["queue"],
            name="prompt x effort",
            scope_id="prompt-x-effort",
            k=2,
        )
    finally:
        for llm in clients.values():
            await llm.aclose()

    # Against (low, v1), v2 at low effort is the prompt's effect there. Its effect at medium effort needs v1 at
    # medium as the control: against() reads the same runs that way, running nothing again.
    for effort, read in ((low, comparison), (medium, await comparison.against((medium, "v1", MODEL)))):
        (row,) = [row for row in read.contrasts("accuracy") if row["arm"] == (effort, "v2", MODEL)]  # arms by key
        p = "" if row["p_adjusted"] is None else f", Holm-adjusted p {row['p_adjusted']:.2g}"  # none if nothing varied
        print(f"At {effort} effort, v2 vs v1 on {row['reading']}: {row['control_mean']:.2f} -> ", end="")
        print(f"{row['arm_mean']:.2f}, delta {row['delta']:+.2f}, interval {row['interval']}{p}: {row['verdict']}")
        for index, case in enumerate(CASES):  # where the two prompts disagree at this effort
            said = [[r.answer for r in comparison.results((effort, v, MODEL)) if r.case == str(index)] for v in PROMPTS]
            if said[0] != said[1]:
                print(f"  {case['ticket']!r} ({case['queue']}): v1 said {', '.join(said[0])}; v2 {', '.join(said[1])}")
    print("\nThe full report: print(comparison.render()), or reports.py to write it to files.")
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
