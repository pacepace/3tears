"""Can a model write a campaign's analysis from a frozen copy of the evidence?

A model writes the analysis from one input, the campaign's **analysis bundle**: every number code computed,
frozen into one JSON file with a sha256 fingerprint. The model never types a figure. It names a reading, code
fills the number in, and a reading the bundle lacks is refused and sent back for repair. New here:
``generate_analysis``. Only the cited readings are checked, not the prose around them, so a conclusion can still
overclaim: hold it against the code's verdict, printed first. The bundle: ``docs/concepts.md#analysis-bundle``.
Saving it and running a second prompt over the same fingerprint: ``docs/reading-reports.md``.

The writer's first draft is canned, not written: ``llm_analysis_first_draft.json`` beside this file cites
"precision", a reading this campaign never measured, so the check refuses it and sends it back with the refusal.
A model's own first draft may well pass, and then there is no refusal to show. The model writes the repair, from the
refusal and the bundle.

Run it with ``python llm_analysis.py`` (``README.md`` beside it says how); it writes ``./eval-analysis/bundle.json``.
The campaign's two classifiers are keyword rules, so the campaign always runs offline and free. With
``ANTHROPIC_API_KEY`` set, Claude writes the repair, one call for under a cent. Without it, the example replays
``recordings/llm_analysis.jsonl``, what Claude wrote when the run was recorded, and calls nothing. A recording
answers only the prompt it was made for, and the prompt is the bundle, which names this run's ids and times: so a
replay hands the writer the bundle the recording was made from, ``recordings/llm_analysis.bundle.json``, not this
run's. ``RECORD=1 python llm_analysis.py``, with the key set, records both again.
"""

import asyncio
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from threetears.evals.analysis import (
    EVAL_ANALYSIS_GEN_DEFAULT,
    AnalysisContextBundle,
    generate_analysis,
    inspect_campaign_bundle,
)
from threetears.evals.kernel import EvalAnalysis
from threetears.evals.providers import models_client
from threetears.evals.quick import compare
from threetears.evals.run import run_blocking
from threetears.evals.schema import ClientRequestSettings
from threetears.evals.testing import RecordedCompletions

MODEL = "claude-haiku-5-5"
#: How the writer is asked: room for a whole analysis, and low effort. No temperature: this model refuses one.
SETTINGS = ClientRequestSettings(max_tokens=12_000, reasoning_effort="low")
FIRST_DRAFT = Path(__file__).with_name("llm_analysis_first_draft.json")
RECORDING = Path(__file__).with_name("recordings") / "llm_analysis.jsonl"
#: The bundle the recording was made from: what a replay hands the writer.
RECORDED_BUNDLE = RECORDING.with_name("llm_analysis.bundle.json")
RERECORD = "RECORD=1 python llm_analysis.py"

# -----------------------------------------------------------------------------
# 1. The campaign: two keyword classifiers, plain rules that always run offline, over a dozen tickets.
# -----------------------------------------------------------------------------

TICKETS = [
    ("I was charged twice for my March invoice.", "billing"),
    ("Can I get a refund for the plan I bought yesterday?", "billing"),
    ("My card was charged but the upgrade page shows an error.", "billing"),
    ("I was charged for a seat after removing that user from my account.", "billing"),
    ("The export button does nothing in Firefox.", "bug"),
    ("The app crashes when I upload a large photo.", "bug"),
    ("Since the update, the login page loops back to itself.", "bug"),
    ("The reset-password email never arrives.", "account"),
    ("Please change the email address on my account.", "account"),
    ("It would be great if reports could be scheduled weekly.", "feature_request"),
    ("Do you plan to add a dark mode?", "feature_request"),
    ("Could you add an option to pay by invoice?", "feature_request"),
]
CASES = [{"ticket": ticket, "queue": queue} for ticket, queue in TICKETS]

# Each arm takes the first queue whose words appear; the candidate checks the ambiguous queues in a better order.
BASELINE_RULES = {"account": ("account", "email", "login"), "billing": ("charged", "refund", "invoice")}
CANDIDATE_RULES = {"feature_request": ("would be great", "plan to add", "could you add"), "billing": ("charged",)}
CANDIDATE_RULES |= {"bug": ("error", "crash", "nothing", "loops"), "account": ("account", "email", "password")}


def keyword_classifier(rules: dict[str, tuple[str, ...]]) -> Any:
    async def classify(case: dict) -> str:
        text = case["ticket"].lower()
        return next((queue for queue, words in rules.items() if any(word in text for word in words)), "bug")

    return classify


# -----------------------------------------------------------------------------
# 2. The writer: the canned first draft, then Claude when a key is set, else the recording of a run that called it.
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


def writer(llm: Any) -> Any:
    """A writer whose first draft is the canned one, and whose every later draft is ``llm``'s."""
    drafts = 0

    async def generate(*, system: str, user: str, response_format: dict | None = None) -> Any:
        nonlocal drafts
        drafts += 1
        if drafts > 1:
            return await llm.generate(system=system, user=user, response_format=response_format)
        # The canned draft, as a completion: nothing was called, so nothing was spent and no model served it.
        return SimpleNamespace(
            content=FIRST_DRAFT.read_text(encoding="utf-8"),
            input_tokens=None,
            output_tokens=None,
            reasoning_tokens=None,
            cost_usd=0.0,
            price_source="canned: no model was called",
            model="canned-first-draft",
            served_model=None,
            stop_reason="end_turn",
            temperature=None,
        )

    return SimpleNamespace(generate=generate)


# -----------------------------------------------------------------------------
# 3. Run the campaign, freeze its bundle to ``out_dir``, and have the writer write the analysis from a frozen one.
# -----------------------------------------------------------------------------


async def main(out_dir: Path = Path("eval-analysis")) -> EvalAnalysis:
    llm = client(MODEL, SETTINGS)
    print(how_it_runs(MODEL) + "\n")

    arms = {"baseline": keyword_classifier(BASELINE_RULES), "candidate": keyword_classifier(CANDIDATE_RULES)}
    comparison = await compare(
        CASES, arms, expected=lambda case: case["queue"], control="baseline", scope_id="llm-analysis", k=2
    )

    # The verdict code reached; nothing an analysis writes changes it.
    (row,) = comparison.contrasts("accuracy")
    print(
        f"Code's verdict: {row['arm']} vs {comparison.control} on {row['reading']}: delta {row['delta']:+.2f}, ", end=""
    )
    print(f"interval {row['interval']}, Holm-adjusted p {row['p_adjusted']:.2g}: {row['verdict']}\n")

    # Freeze: assemble the bundle once and save it. The fingerprint is a sha256 of its canonical JSON.
    # A store read, so it runs off the event loop.
    inspected = await run_blocking(
        comparison.host.blocking_executor,
        inspect_campaign_bundle,
        comparison.host,
        comparison.campaign_id,
        comparison.scope_id,
    )
    frozen = out_dir / "bundle.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen.write_text(inspected.bundle.to_json(indent=2), encoding="utf-8")
    print(f"Froze the evidence into {frozen}, fingerprint {inspected.bundle.fingerprint()}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        if not RECORDED_BUNDLE.is_file():
            raise SystemExit(
                f"{RECORDED_BUNDLE} is missing, and the recording answers only it; {RERECORD} records both"
            )
        frozen = RECORDED_BUNDLE  # this run's bundle is a prompt the recording never answered
        print(f"Replaying: the writer reads {RECORDED_BUNDLE.name}, the bundle the recording was made from.")
    elif os.environ.get("RECORD") == "1":
        shutil.copyfile(frozen, RECORDED_BUNDLE)  # the bundle a replay will hand the writer
    bundle = AnalysisContextBundle.from_json(frozen.read_text(encoding="utf-8"))  # the writer reads only the file

    try:
        analysis, _insights = await generate_analysis(
            bundle,  # the whole context: nothing else is read while the analysis is written
            prompt=EVAL_ANALYSIS_GEN_DEFAULT,
            model=MODEL,
            client=writer(llm),
            prompt_id="eval_analysis_gen",
            bundle_assembled_at=datetime.now(UTC).isoformat(),
            profile=comparison.host.profile,
        )
    finally:
        await llm.aclose()
    print(f"\n# {analysis.document.headline}")
    for finding in analysis.document.findings:
        print(f"- {finding.title.rstrip('.')}: {finding.body}")  # every figure in the body was filled in by code
    for decision in analysis.document.decisions:  # checked: adopting an arm needs a reading that separated
        print(f"  decision ({decision.disposition}): {decision.proposal}")
    made = analysis.generation  # provenance: the bundle it read, the prompt's version, the cost, any repair
    print(
        f"\nRead bundle {made.bundle_fingerprint[:12]} under prompt {made.prompt_version[:12]}, for ${made.token_cost:.4f}"
    )
    refused = (made.repaired_refusal or "nothing").split(". ")[0]
    print(f"Repairs: {made.repair_attempts}, after the check refused: {refused}")
    return analysis


if __name__ == "__main__":
    asyncio.run(main())
