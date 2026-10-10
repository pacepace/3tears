"""Can a model write a campaign's analysis from a frozen copy of the evidence?

A model writes the analysis from one input, the campaign's **analysis bundle**: every number code computed,
frozen into one JSON file with a sha256 fingerprint. The model never types a figure. It names a reading, code
fills the number in, and a reading the bundle lacks is refused and sent back for repair. New here:
``generate_analysis``. Only the cited readings are checked, not the prose around them, so a conclusion can still
overclaim: hold it against the code's verdict, printed first. The bundle: ``docs/concepts.md#analysis-bundle``.
Saving it and running a second prompt over the same fingerprint: ``docs/reading-reports.md``.

Run it with ``python packages/evals/examples/llm_analysis.py``; it writes ``./eval-analysis/bundle.json``.
The campaign's two classifiers are keyword rules, so the campaign always runs offline. With ``ANTHROPIC_API_KEY``
set, Claude writes the analysis, for about a cent. Without it, a scripted stand-in writes it, and its
first draft cites a reading the bundle lacks, to show the refusal.
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from _live import Completion, claude, online
from threetears.evals.analysis import EVAL_ANALYSIS_GEN_DEFAULT, generate_analysis, inspect_campaign_bundle
from threetears.evals.kernel import EvalAnalysis
from threetears.evals.quick import compare

MODEL = "claude-haiku-5-5"

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
# 2. The OFFLINE stand-in writer: a script, not a model. Live, the writer is Claude (``_live.py``).
# -----------------------------------------------------------------------------


def offline_writer() -> Any:
    """A scripted stand-in for the writer; its first draft cites "precision", which this campaign never measured."""
    drafts = 0

    async def generate(*, system: str, user: str, response_format: dict | None = None) -> Completion:
        nonlocal drafts
        drafts += 1
        evidence, _ = json.JSONDecoder().raw_decode(user, user.index("{"))  # the bundle, as the writer sees it
        tested = evidence["multiple_comparisons"]["families"][0]["comparisons"][0]
        arm, control = tested["contrast"]["cell"], tested["control"]["cell"]
        measure = "precision" if drafts == 1 else tested["name"]
        cite = {cell: "{{" + f"{cell}|{measure}|measure|mean" + "}}" for cell in (arm, control)}  # code fills these
        finding = {
            "title": f"{measure.capitalize()} by arm",
            "body": f"The tested arm reads {cite[arm]}, the control {cite[control]}.",
            "evidence": [{"cell": cell, "measure_id": measure, "reading": "measure"} for cell in cite],
            "chart": {"type": "none", "cells": [], "measures": [], "axis": "", "note": "", "caption": ""},
        } | {"confidence": "low", "axes": [], "caveats": [], "invalidates": [], "durable": ""}
        memo = {"headline": "Offline stand-in: a script wrote this, not a model", "summary": "- Shape only."}
        memo |= {"findings": [finding], "decisions": [], "questions": [], "next": []}
        return Completion(json.dumps(memo), None, None, None, 0.0, "offline", "offline-writer", None, "end_turn", None)

    return SimpleNamespace(generate=generate)


# -----------------------------------------------------------------------------
# 3. Run the campaign, freeze its bundle to ``out_dir``, and have the writer write the analysis from it.
# -----------------------------------------------------------------------------


async def main(out_dir: Path = Path("eval-analysis")) -> EvalAnalysis:
    if online():
        print(f"Running against Claude ({MODEL}).\n")
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with a scripted stand-in for the analysis writer.\n")

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
    bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, comparison.scope_id).bundle
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bundle.json").write_text(bundle.to_json(indent=2))
    print(f"Froze the evidence into {out_dir / 'bundle.json'}, fingerprint {bundle.fingerprint()}")

    analysis, _insights = await generate_analysis(
        bundle,  # the whole context: nothing else is read while the analysis is written
        prompt=EVAL_ANALYSIS_GEN_DEFAULT,
        model=MODEL if online() else "offline-writer",
        client=claude(MODEL, max_tokens=12_000) if online() else offline_writer(),
        prompt_id="eval_analysis_gen",
        bundle_assembled_at=datetime.now(UTC).isoformat(),
        profile=comparison.host.profile,
    )
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
