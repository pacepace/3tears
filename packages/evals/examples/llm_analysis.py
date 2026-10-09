"""Can a model write a campaign's analysis from a frozen copy of the evidence, so it can be checked and redone?

A model writes the analysis from one input, the campaign's **analysis bundle**: every number code computed,
frozen into one JSON file with a sha256 fingerprint. The model never types a figure: it names a reading, code
fills the number in, and a reading the bundle lacks is refused. Only the cited readings are checked, not the
prose around them, so a conclusion can still overclaim: compare each analysis's with the code's verdict, printed
first. New here: ``generate_analysis``, and a bundle saved, reloaded and written over again with another prompt.
The bundle: ``docs/concepts.md#analysis-bundle``.

Run it with ``python packages/evals/examples/llm_analysis.py``; it writes ``./eval-analysis/bundle.json``.
With ``ANTHROPIC_API_KEY`` set, Claude writes two analyses for about a cent; without it, a scripted stand-in
writes them, and they say nothing about Claude.
"""

import asyncio
import json
import os
from collections import namedtuple
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
from threetears.evals.contracts import EvalAnalysis
from threetears.evals.quick import compare

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 12_000
RATES = (0.10, 0.50)  # Haiku 5.5's (input, output) list price, USD per million tokens, for prompts up to 100K

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
# 2. The live writer: Claude, behind the one call the generator makes.
#
# ``generate(system=, user=, response_format=)`` returns a ``Completion``, the reply as ``llm_judge.py`` reads it.
# -----------------------------------------------------------------------------

Completion = namedtuple(
    "Completion",
    "content input_tokens output_tokens reasoning_tokens cost_usd price_source model served_model stop_reason",
)


def claude_writer() -> Any:
    """The analysis writer, as Claude."""
    import anthropic  # imported here so the offline path does not need the package

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY

    async def generate(*, system: str, user: str, response_format: dict | None = None) -> Completion:
        schema = (response_format or {})["json_schema"]["schema"]
        response = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            # The generator sends the analysis's shape as a JSON schema; Claude takes it as a structured output.
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        )
        usage = response.usage
        stopped = {"end_turn": "end_turn", "max_tokens": "max_tokens", "refusal": "content_filter"}
        return Completion(
            content="".join(block.text for block in response.content if block.type == "text"),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=None,
            cost_usd=(usage.input_tokens * RATES[0] + usage.output_tokens * RATES[1]) / 1e6,
            price_source="anthropic list price, from llm_analysis.py",
            model=MODEL,
            served_model=response.model,
            stop_reason=stopped.get(response.stop_reason or "", "error"),
        )

    return SimpleNamespace(generate=generate)


# -----------------------------------------------------------------------------
# 3. The OFFLINE stand-in writer: a script, not a model.
#
# Its first draft cites a reading the bundle lacks, so the generator's check refuses it and asks once more.
# -----------------------------------------------------------------------------


def offline_writer() -> Any:
    """A scripted stand-in for the writer, so the example runs with no API key."""
    drafts = 0

    async def generate(*, system: str, user: str, response_format: dict | None = None) -> Completion:
        nonlocal drafts
        drafts += 1
        evidence, _ = json.JSONDecoder().raw_decode(user, user.index("{"))  # the bundle, as the writer sees it
        tested = evidence["multiple_comparisons"]["families"][0]["comparisons"][0]
        arm, control = tested["contrast"]["cell"], tested["control"]["cell"]
        measure = "precision" if drafts == 1 else tested["name"]  # "precision" is not a measure of this campaign
        cite = {cell: "{{" + f"{cell}|{measure}|measure|mean" + "}}" for cell in (arm, control)}  # code fills these
        finding = {
            "title": f"The arms differ on {measure}",
            "body": f"The tested arm reads {cite[arm]}, the control {cite[control]}.",
            "evidence": [{"cell": cell, "measure_id": measure, "reading": "measure"} for cell in cite],
            "chart": {"type": "none", "cells": [], "measures": [], "axis": "", "note": "", "caption": ""},
        } | {"confidence": "low", "axes": [], "caveats": [], "invalidates": [], "durable": ""}
        memo = {"headline": "Offline stand-in: a script wrote this, not a model", "summary": "- Shape only."}
        memo |= {"findings": [finding], "decisions": [], "questions": [], "next": []}
        return Completion(json.dumps(memo), None, None, None, 0.0, "offline stand-in", "offline", None, "end_turn")

    return SimpleNamespace(generate=generate)


# -----------------------------------------------------------------------------
# 4. Run the campaign, freeze its bundle to ``out_dir``, write the analysis, then write it again from the file.
# -----------------------------------------------------------------------------


async def main(out_dir: Path = Path("eval-analysis")) -> list[EvalAnalysis]:
    online = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if online:
        print(f"Running against Claude ({MODEL}).\n")
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with a scripted stand-in for the analysis writer.\n")

    arms = {"baseline": keyword_classifier(BASELINE_RULES), "candidate": keyword_classifier(CANDIDATE_RULES)}
    comparison = await compare(
        CASES, arms, expected=lambda case: case["queue"], control="baseline", scope_id="llm-analysis", k=2
    )
    profile, assembled_at = comparison.host.profile, datetime.now(UTC).isoformat()

    # The verdict code reached; nothing an analysis writes changes it.
    for row in comparison.contrasts("accuracy"):
        arm, control = (row[key].removeprefix("model=") for key in ("contrast", "control"))
        p = "" if row["p_adjusted"] is None else f" (p={row['p_adjusted']:.2g})"  # none when nothing varied
        print(f"Code's verdict: {arm} vs {control} on {row['reading']}: {row['delta']:+.2g}{p}: {row['verdict']}\n")

    # Freeze: assemble the bundle once and save it. The fingerprint is a sha256 of its canonical JSON.
    bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, comparison.scope_id).bundle
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "bundle.json"
    bundle_path.write_text(bundle.to_json(indent=2))
    print(f"Froze the evidence into {bundle_path}, fingerprint {bundle.fingerprint()}")

    async def write(bundle: AnalysisContextBundle, prompt: str) -> EvalAnalysis:
        analysis, _insights = await generate_analysis(
            bundle,  # the whole context: nothing else is read while the analysis is written
            prompt=prompt,
            model=MODEL if online else "offline",
            client=claude_writer() if online else offline_writer(),
            prompt_id="eval_analysis_gen",
            bundle_assembled_at=assembled_at,
            profile=profile,
        )
        made = analysis.generation  # provenance: the bundle it read, the prompt's version, the cost, any repair
        print(f"\n# {analysis.document.headline}")
        for finding in analysis.document.findings:
            print(f"- {finding.title.rstrip('.')}: {finding.body}")  # every figure in the body was filled in by code
        print(f"  bundle {made.bundle_fingerprint[:12]}, prompt {made.prompt_version[:12]}, ${made.token_cost:.4f}")
        refused = (made.repaired_refusal or "nothing").split(". ")[0]
        print(f"  repairs {made.repair_attempts}, after the check refused: {refused}")
        return analysis

    first = await write(bundle, EVAL_ANALYSIS_GEN_DEFAULT)

    # The payoff: reload the file, check it is the same evidence, and run another prompt over it.
    reloaded = AnalysisContextBundle.from_json(bundle_path.read_text())
    same = reloaded.fingerprint() == bundle.fingerprint()
    print(f"\nReloaded {bundle_path}: fingerprint {'matches' if same else 'DOES NOT match'}")
    terser = EVAL_ANALYSIS_GEN_DEFAULT + "\n\nWrite for a support lead who has two minutes."
    second = await write(reloaded, terser)  # only the prompt moved
    return [first, second]


if __name__ == "__main__":
    asyncio.run(main())
