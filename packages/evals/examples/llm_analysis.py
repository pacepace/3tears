"""Can a model write a campaign's analysis from a frozen copy of the evidence, so it can be checked and redone?

``compare_two_prompts.py`` ended with a campaign and its code-only report. Here a model writes the
analysis, and the one new idea is the **analysis bundle**: every number code computed about the campaign,
frozen into one JSON file with a sha256 fingerprint. The model reads only the bundle and never types a
figure: it names a reading, code fills the number in, and a reading the bundle does not hold is refused.
Save the bundle, and regenerating over it with another prompt changes the prompt and nothing else.

Run it with ``python packages/evals/examples/llm_analysis.py``; it writes ``./eval-analysis/bundle.json``.
With ``ANTHROPIC_API_KEY`` set, Claude (``claude-haiku-5-5``) writes two analyses for about a cent; without
it, a scripted stand-in does, and says so. The bundle, field by field: ``docs/concepts.md#analysis-bundle``.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from threetears.evals.analysis import (
    EVAL_ANALYSIS_GEN_DEFAULT,
    AnalysisContextBundle,
    first_request,
    generate_analysis,
    inspect_campaign_bundle,
)
from threetears.evals.contracts import EvalAnalysis
from threetears.evals.quick import compare

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 12_000
RATES_PER_MILLION = (0.10, 0.50)  # Haiku 5.5's (input, output) list price, for prompts up to 100K tokens

# -----------------------------------------------------------------------------
# 1. The campaign: two keyword classifiers over a dozen support tickets.
#
# The eval is not the point here, so both arms are plain rules and always run offline.
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
# 2. The live analysis writer: Claude, behind the one call the engine's generator makes.
#
# ``generate(system=, user=, response_format=)`` returns a completion the generator reads by attribute.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Completion:
    content: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None
    price_source: str | None
    model: str
    served_model: str | None
    stop_reason: str


def call_cost_usd(input_tokens: int, output_tokens: int) -> float:
    input_rate, output_rate = RATES_PER_MILLION
    return (input_tokens * input_rate + output_tokens * output_rate) / 1e6


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
            # The engine sends the analysis's shape as a JSON schema; Claude takes it as a structured output.
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        )
        usage = response.usage
        stopped = {"end_turn": "end_turn", "max_tokens": "max_tokens", "refusal": "content_filter"}
        return Completion(
            content="".join(block.text for block in response.content if block.type == "text"),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=None,
            cost_usd=call_cost_usd(usage.input_tokens, usage.output_tokens),
            price_source="anthropic list price, from llm_analysis.py",
            model=MODEL,
            served_model=response.model,
            stop_reason=stopped.get(response.stop_reason or "", "error"),
        )

    return SimpleNamespace(generate=generate)


# -----------------------------------------------------------------------------
# 3. The offline stand-in writer: a script, not a model.
#
# Its first draft cites a reading the bundle does not hold, so the engine's check refuses it and asks once more.
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

        def figure(cell: str) -> str:
            return "{{" + f"{cell}|{measure}|measure|mean" + "}}"  # a figure reference; code writes the number

        finding = {
            "title": f"The arms differ on {measure}",
            "body": f"The tested arm reads {figure(arm)}, the control {figure(control)}.",
            "evidence": [{"cell": cell, "measure_id": measure, "reading": "measure"} for cell in (arm, control)],
            "chart": {"type": "none", "cells": [], "measures": [], "axis": "", "note": "", "caption": ""},
        } | {"confidence": "low", "axes": [], "caveats": [], "invalidates": [], "durable": ""}
        memo = {"headline": "Offline stand-in: a script wrote this, not a model", "summary": "- Shape only."}
        memo |= {"findings": [finding], "decisions": [], "questions": [], "next": []}
        return Completion(json.dumps(memo), None, None, None, 0.0, "offline stand-in", "offline", None, "end_turn")

    return SimpleNamespace(generate=generate)


# -----------------------------------------------------------------------------
# 4. Freeze the evidence, price the analysis, write it, then write it again from the saved file.
# -----------------------------------------------------------------------------


async def write_analysis(bundle: AnalysisContextBundle, prompt: str, profile: Any, assembled_at: str) -> EvalAnalysis:
    online = bool(os.environ.get("ANTHROPIC_API_KEY"))
    analysis, _insights = await generate_analysis(
        bundle,  # the whole context: nothing else is read while the analysis is written
        prompt=prompt,
        model=MODEL if online else "offline",
        client=claude_writer() if online else offline_writer(),
        prompt_id="eval_analysis_gen",
        bundle_assembled_at=assembled_at,
        profile=profile,
    )
    made = analysis.generation  # provenance: the bundle's fingerprint, the prompt's version, the cost
    print(f"\n# {analysis.document.headline}")
    for finding in analysis.document.findings:
        print(f"- {finding.title}: {finding.body}")
    print(f"  over bundle {made.bundle_fingerprint[:12]}, with prompt version {made.prompt_version[:12]}")
    refused = (made.repaired_refusal or "nothing").split(". ")[0]
    print(f"  cost ${made.token_cost:.4f}; repairs {made.repair_attempts}, after the check refused: {refused}")
    return analysis


async def main(out_dir: Path = Path("eval-analysis")) -> list[EvalAnalysis]:
    """Run the campaign, freeze its bundle to ``out_dir``, and write its analysis twice."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        print(f"Running against Claude ({MODEL}).")
    else:
        print("ANTHROPIC_API_KEY is not set: running OFFLINE, with a scripted stand-in for the analysis writer.")

    arms = {"baseline": keyword_classifier(BASELINE_RULES), "candidate": keyword_classifier(CANDIDATE_RULES)}
    comparison = await compare(
        CASES, arms, expected=lambda case: case["queue"], control="baseline", scope_id="llm-analysis", k=2
    )
    profile = comparison.host.profile

    # Freeze: assemble the bundle once and save it. The fingerprint is a sha256 of its canonical JSON.
    assembled_at = datetime.now(UTC).isoformat()  # kept on the analysis's provenance, not in the bundle
    bundle = inspect_campaign_bundle(comparison.host, comparison.campaign_id, comparison.scope_id).bundle
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "bundle.json"
    bundle_path.write_text(bundle.to_json(indent=2))
    print(f"Froze the evidence into {bundle_path}, fingerprint {bundle.fingerprint()}")

    # Price it before spending: the exact first request the generator will send, at Haiku's list price.
    system, user, _shape = first_request(bundle, EVAL_ANALYSIS_GEN_DEFAULT, profile)
    approx_input_tokens = (len(system) + len(user)) // 4
    estimate = call_cost_usd(approx_input_tokens, MAX_TOKENS)
    print(f"Estimate: about {approx_input_tokens:,} tokens in and {MAX_TOKENS:,} out at most, so about ${estimate:.4f}")
    print("  a call, and at most two calls: a refused draft buys exactly one repair.")

    first = await write_analysis(bundle, EVAL_ANALYSIS_GEN_DEFAULT, profile, assembled_at)

    # The payoff: reload the file, check it is the same evidence, and run another prompt over it.
    reloaded = AnalysisContextBundle.from_json(bundle_path.read_text())
    same = reloaded.fingerprint() == bundle.fingerprint()
    print(f"\nReloaded {bundle_path}: fingerprint {'matches' if same else 'DOES NOT match'}")
    terser = EVAL_ANALYSIS_GEN_DEFAULT + "\n\nWrite for a support lead who has two minutes."
    second = await write_analysis(reloaded, terser, profile, assembled_at)  # only the prompt moved
    return [first, second]


if __name__ == "__main__":
    asyncio.run(main())
