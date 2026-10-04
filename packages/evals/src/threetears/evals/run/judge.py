"""LLM judge utilities — slimmed for the eval rebuild.

This module holds the callable that code outside the rubric judge depends on:

- :func:`run_judge_llm` — shared LLM-judge execution path: call the LLM,
  parse a criteria-based response (``{"reasoning": ..., "criteria_scores":
  {name: 1..5, ...}}``), aggregate to a 0..1 normalized composite, log cost.
  Called from :mod:`threetears.evals.run.judge_service`, whose narrow single-dim rubric
  judge is built on it, and open to a host's own scorers.

**This is engine, and a host calling it is a host using the engine's public API**.
It is constructed with the injected
:class:`~threetears.evals.contracts.provider.CompletionClient` rather than building one, takes
prompts and criteria as plain data, returns eval's own
:class:`~threetears.evals.contracts.usage_capture.CallUsage`, and names no host concept. The
in-package caller is what forecloses the alternative rather than merely arguing
against it: moving this host-side would make an eval-core module import host code.
It is exported from the run package's public root, which is where a host imports it —
an engine has a public API, and there is nothing here to retire.

The Tier 2 / Tier 3 judging stack (``semantic_judge``, ``judge_cache``,
``judge_prompts.build_judge_*``) was deleted in the rebuild. The narrow
single-dim rubric/dual-score judge lives in
:mod:`threetears.evals.run.judge_service` and calls :func:`run_judge_llm` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from threetears.evals.contracts.models import PASS_FAIL_SCORES, SCALES, ClientRequestSettings
from threetears.evals.contracts.provider import (
    JSON_OBJECT_RESPONSE_FORMAT,
    ProviderFailureDescriber,
    describe_and_log_failure,
    describe_incomplete_completion,
    extract_json,
    sum_optional_tokens,
)
from threetears.evals.contracts.usage_capture import CallUsage
from threetears.observe import get_logger

log = get_logger(__name__)

#: Extra judge attempts when a FINISHED response cannot be parsed. The
#: parse step is robust (see :func:`threetears.evals.contracts.provider.extract_json`), but a
#: single malformed response would otherwise silently drop a whole rubric dimension
#: (score=None) from the run. Because the judge call is non-deterministic, a re-ask
#: almost always returns parseable JSON, so one retry recovers the dim rather than
#: losing it. Kept small — a persistently-unparseable judge is a real error, not a
#: transient blip. A response the provider reported CUT SHORT is not retried at all:
#: see :func:`run_judge_llm`.
_JUDGE_PARSE_RETRIES = 1

#: Calls one judge dimension can make: the first, plus its parse retries. What a caller sizing a
#: judge phase's wall clock multiplies by (:func:`threetears.evals.analysis.reporter_kind.reporter_cell_timeout_s`).
JUDGE_CALL_ATTEMPTS = _JUDGE_PARSE_RETRIES + 1

#: How many tokens the eval judge may spend on private reasoning before it answers.
#:
#: **Bounded, not disabled — the classifier's remedy is the wrong one here.** The
#: classifier asks for two words, reasoning buys it nothing, so it sends
#: ``reasoning: {enabled: false}``. A judge's reply is short too (the JSON contract in
#: ``judge_service`` asks for one paragraph and one integer), but the work behind the
#: integer is not. ``reporter.groundedness`` checks every claim in a memo against its
#: evidence bundle, and that is what reasoning is for, so taking reasoning away would
#: change the instrument, not just its cost. Disabling is also refused permanently by
#: models whose reasoning is mandatory (``openai/gpt-5-mini`` answers HTTP 400), which
#: would bar that whole class of models from judging. So reasoning stays on, with an
#: explicit budget, and the cap is derived from that budget below.
#:
#: **The value is a judgement, not a measurement.** A calibration run lost
#: ``reporter.groundedness`` and ``reporter.calibration`` on every case at the old flat
#: 4096-token cap, with nearly all of its completion tokens spent reasoning. That shows the
#: two longest-reasoning dims wanted more than 4096 in total. It does not show how much
#: more, because every one of those calls was stopped. Twice the cap they overran is a
#: bound, not a prediction. Re-derive it from a run's reported ``reasoning_tokens``
#: (a cut judge call now reports its reasoning share in its error) before moving it.
JUDGE_REASONING_BUDGET_TOKENS = 8192

#: Output room kept for the judge's visible answer: the JSON object with its one
#: reasoning paragraph and one integer score. That is a few hundred tokens, so this is
#: headroom, and unused output tokens are not billed.
JUDGE_ANSWER_BUDGET_TOKENS = 2048

#: The eval judge's output cap: the reasoning budget plus the answer budget, DERIVED
#: rather than chosen, so it always sits above the ceiling it wraps. The old flat 4096
#: sat below what a reasoning judge spent. The cap is the sum by construction, so moving either
#: part moves the cap with it.
#:
#: **What the reasoning bound does NOT guarantee.** It goes out as the router's
#: ``reasoning.max_tokens``, and each provider honours it in its own way.
#: Anthropic- and Gemini-style models take it as a budget. Effort-only models (OpenAI's
#: reasoning series) have it mapped to an effort level, which bounds nothing in tokens.
#: Models that do not reason ignore it, since a router's default routing may ignore
#: unknown parameters (``require_parameters`` is not set). The cap is therefore the
#: only hard stop. When a judge still reaches it, :func:`run_judge_llm` reports the cut
#: with its token counts and does not re-buy it.
#:
#: **What the bound changes.** Naming a budget turns reasoning ON for a model that
#: reasons only when asked. Such a judge now thinks before scoring, where it previously
#: did not. The rule is the same for every eval judge (subject runs, reporter runs, and
#: the per-dim ``JudgeConfig`` models alike), which is why these numbers live here and
#: not at any one call site. A host's client builder applies them to its
#: judge role.
JUDGE_MAX_TOKENS = JUDGE_REASONING_BUDGET_TOKENS + JUDGE_ANSWER_BUDGET_TOKENS

#: The judge role's request settings as ONE value: what the host's client builder applies to every
#: eval judge client, and what a run launched with a judge records as ``judge_request_settings``.
#: One object read by both, so the stamp cannot say something the requests did not do.
#:
#: **Why it is recorded at all.** Moving either number regrades every later run without changing
#: ``judge_model``: when these were introduced, every eval judge went from a flat 4096-token cap with no reasoning
#: parameter to this derived cap with an explicit reasoning budget. Runs launched before that carry
#: no stamp, and the ``judge_request_settings`` apparatus dimension reads them as unrecorded, so a
#: campaign pooling runs from both sides of that change is told its judge may have moved.
JUDGE_REQUEST_SETTINGS = ClientRequestSettings(
    max_tokens=JUDGE_MAX_TOKENS,
    reasoning_max_tokens=JUDGE_REASONING_BUDGET_TOKENS,
)


# ---------------------------------------------------------------------------
# Criteria-based parsing
# ---------------------------------------------------------------------------

#: The ordinal scale a 1–5 criterion is scored on, inclusive at both ends (pass/fail is answered in words).
JUDGE_SCORE_MIN, JUDGE_SCORE_MAX = SCALES["ordinal"].scores

#: The answer a judge gives in place of a score when the evidence does not let it score the
#: criterion at all. A protocol value, not a score and not a parse failure: the criterion is
#: left unscored and the trial is excluded from it. Only a caller that offers the answer
#: accepts it; for any other it is a reply that broke the protocol.
CANNOT_TELL = "cannot_tell"


def _ordinal_score(name: str, value: Any) -> int:
    """Read one criterion's score as the integer on the scale it was asked for, or refuse it.

    The score is the judge's protocol answer, so it is taken as given or not at all: a
    fraction, a number off the scale, a string or a boolean is a reply that broke the
    protocol, and the caller's parse-failure path handles it. Nothing is truncated, rounded
    or clamped into range — each of those would record a score the judge never gave.

    Args:
        name: The criterion, for the message.
        value: What the reply carried for it.

    Returns:
        The score.

    Raises:
        ValueError: The value is not an integer on the scale.
    """
    integral = isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if isinstance(value, bool) or not integral:
        raise ValueError(f"Score for criterion '{name}' is not an integer: {value!r}")
    score = int(value)
    if not JUDGE_SCORE_MIN <= score <= JUDGE_SCORE_MAX:
        raise ValueError(
            f"Score for criterion '{name}' is off the {JUDGE_SCORE_MIN}-{JUDGE_SCORE_MAX} scale: {value!r}"
        )
    return score


def _pass_fail_score(name: str, value: Any) -> int:
    """Read a pass/fail criterion's answer as its stored score, or refuse it.

    Only the two words the prompt offered are taken. A number is refused rather than read as a
    pass or a fail: the judge was asked a yes-or-no question, and a 4 is not an answer to it.

    Args:
        name: The criterion, for the message.
        value: What the reply carried for it.

    Returns:
        1 for ``"pass"``, 0 for ``"fail"``.

    Raises:
        ValueError: The value is not ``"pass"`` or ``"fail"``.
    """
    if not isinstance(value, str) or value not in PASS_FAIL_SCORES:
        raise ValueError(f"Criterion '{name}' is pass/fail, and the judge answered {value!r}")
    return PASS_FAIL_SCORES[value]


#: How each scale's answer is read into its stored score. Keyed like :data:`~threetears.evals.contracts.models.SCALES`,
#: so a scale this module has not learned is a KeyError, not another scale's reader.
SCALE_READERS: dict[str, Callable[[str, Any], int]] = {"ordinal": _ordinal_score, "pass_fail": _pass_fail_score}


def _parse_criteria_response(
    content: str,
    evaluation_criteria: list[dict[str, Any]],
    *,
    cannot_tell_offered: bool = False,
) -> dict[str, Any]:
    """Parse a judge's criteria-based JSON response.

    Expected format::

        {"reasoning": "...", "criteria_scores": {"name": 1-5, ...}}

    ``reasoning`` is part of the protocol, not decoration: a stored score and a recorded "can't
    tell" both carry it, so a reply without one as a string — absent, ``null``, a number, an
    object — is a reply that broke the protocol and is refused here, at the parse boundary, with
    every other malformed reply. Refused anywhere later it would arrive after the call's spend was
    folded, and either lose that spend with the score or fail the result outright.

    A criterion dict carrying ``"scale": "pass_fail"`` is answered ``"pass"`` or ``"fail"`` and
    returned as 1 or 0; one without a scale is ordinal. With ``cannot_tell_offered``, a criterion may carry :data:`CANNOT_TELL` in place of its
    score; it is returned under ``cannot_tell`` and left out of ``criteria_scores``.

    Raises:
        ValueError: If the response cannot be parsed, carries no string ``reasoning``, is missing
            a criterion, or scores one with anything but an answer on its scale (or the offered
            answer).
    """
    data = extract_json(content)

    if "criteria_scores" not in data:
        raise ValueError("Missing 'criteria_scores' in judge response")
    if "reasoning" not in data:
        raise ValueError("Missing 'reasoning' in judge response")
    reasoning = data["reasoning"]
    if not isinstance(reasoning, str):
        raise ValueError(f"Expected reasoning to be a string, got {type(reasoning).__name__}")

    scores = data["criteria_scores"]
    if not isinstance(scores, dict):
        raise ValueError(f"Expected criteria_scores to be a dict, got {type(scores).__name__}")

    expected_names = [c["name"] for c in evaluation_criteria]
    for name in expected_names:
        if name not in scores:
            raise ValueError(f"Missing criterion '{name}' in judge response")
    # Only the criteria asked for are read: a key the judge invented was never on the scale.
    cannot_tell = [name for name in expected_names if cannot_tell_offered and scores[name] == CANNOT_TELL]
    scales = {c["name"]: c.get("scale", "ordinal") for c in evaluation_criteria}
    ordinal = {
        name: SCALE_READERS[scales[name]](name, scores[name]) for name in expected_names if name not in cannot_tell
    }

    return {"criteria_scores": ordinal, "cannot_tell": cannot_tell, "reasoning": reasoning}


def _normalize_criteria_scores(scores: dict[str, int], scales: dict[str, str]) -> dict[str, float]:
    """Put each score on 0.0-1.0: ``(score - 1) / 4`` for 1–5, the 1 or 0 itself for pass/fail."""
    return {name: round(SCALES[scales[name]].normalized(score), 4) for name, score in scores.items()}


def _process_criteria_response(
    result: Any,
    evaluation_criteria: list[dict[str, Any]],
    *,
    cannot_tell_offered: bool = False,
) -> dict[str, Any] | None:
    """Process a criteria-based judge LLM response into a result dict."""
    try:
        parsed = _parse_criteria_response(result.content, evaluation_criteria, cannot_tell_offered=cannot_tell_offered)
    except ValueError as e:
        log.warning("Failed to parse criteria judge response: %s", e)
        return None

    criteria_scores = parsed["criteria_scores"]
    normalized = _normalize_criteria_scores(
        criteria_scores, {c["name"]: c.get("scale", "ordinal") for c in evaluation_criteria}
    )

    return {
        "criteria_scores": normalized,
        # The integers the judge gave, for a caller that stores the ordinal: reading them back
        # out of the normalised floats would round and clamp a value this module already holds.
        "criteria_ordinal_scores": dict(criteria_scores),
        # The criteria the judge answered it could not score, which carry no score at all.
        "criteria_cannot_tell": list(parsed["cannot_tell"]),
        "criteria_names": list(normalized.keys()),
        "reasoning": parsed["reasoning"],
    }


# ---------------------------------------------------------------------------
# Judge entry point
# ---------------------------------------------------------------------------


async def run_judge_llm(
    client: Any,
    system_prompt: str,
    user_prompt: str,
    criteria_dicts: list[dict[str, Any]],
    label: str,
    case_id: str,
    *,
    cannot_tell_offered: bool = False,
    failure_describer: ProviderFailureDescriber | None = None,
) -> dict[str, Any] | None:
    """Run an LLM judge call against a set of criteria and parse the response.

    Args:
        client: LLM client with an async ``generate(system=..., user=...)``
            method returning a :class:`~threetears.evals.contracts.provider.CompletionResult`:
            ``content``, ``input_tokens``, ``output_tokens``, ``cost_usd``,
            ``model``, ``served_model``, ``reasoning_tokens`` and the normalized ``stop_reason``.
        system_prompt: System prompt for the judge.
        user_prompt: User prompt containing the candidate output / context.
        criteria_dicts: List of criterion dicts with at least ``name`` +
            ``weight`` keys.
        label: Human label for log messages (e.g. ``"Judge"``).
        case_id: Test case ID for log context.
        cannot_tell_offered: Whether the prompt offered :data:`CANNOT_TELL`. When it did, a
            criterion answered that way is returned in ``criteria_cannot_tell`` rather than
            refused; when it did not, the answer is a reply that broke the protocol.
        failure_describer: The host's reading of what ``client`` raises. When given, a raised call's
            error is its description, and the result carries ``account_refused`` — set when the
            provider refused the call for the calling account, which no retry or other model
            changes. ``None`` — a host caller that acts on no account fault — keeps the raised text.

    Returns:
        Parsed result dict with ``criteria_scores`` (normalised to 0-1),
        ``criteria_ordinal_scores`` (the stored integers: 1-5, or 1/0 for pass/fail), ``criteria_names``,
        ``reasoning``, ``judge_served_model`` (see below), and ``judge_usage``; or
        ``{"error": "...", "response_preview": "..."}`` on failure. ``judge_usage`` is on every
        return that made a call: the cumulative spend of every attempt, parsed or not, whose
        ``cost_usd`` is ``None`` when any attempt reported no price — the dim's dollars are then
        unknown, never the priced attempts' partial sum. A reply the
        provider cut short fails with an error naming that cause and its token
        counts, after one call. A finished reply that will not parse — including one
        scoring a criterion with anything but an integer on the scale — fails with
        "Failed to parse" after the bounded retries. Each caller
        composites the per-dimension scores itself (``judge_service`` across its
        single-dim calls, a host's own scorer however it composites), so this path returns
        the raw normalized scores and no composite of its own.

        ``judge_served_model`` is the model the provider's response named for the attempt whose
        scores are returned — that attempt, not the last one to report a model, because it is the
        one that scored — and ``None`` when that response named none. It is read off
        ``served_model`` and never off ``model``: the host may fill ``model`` from the request,
        and a floating alias in the request would then stand where the concrete scorer belongs,
        making two runs judged by different models compare equal.
    """
    # Retry on parse-failure so a single malformed judge response can't
    # silently drop a rubric dimension. We retry on any parse failure of a FINISHED
    # reply (including a missing/omitted criterion key) — the judge is
    # non-deterministic, so a re-ask can recover an omitted dim, which is the whole
    # point. Two failures are NOT retried. The LLM call itself failing
    # (network/SDK) is a hard error. A reply the provider reported cut short is a
    # refusal of THIS request, and the identical request meets the identical cap
    # (or filter) and is billed again. Every attempt costs money, so each one's
    # spend is logged, and every return carries the cumulative spend as ``judge_usage``.
    last_preview = ""
    # The attempts' summed dollars, or ``None`` once any attempt reported no price: a sum over
    # the priced attempts alone would be read as what the dim cost, and it is not.
    total_cost: float | None = None
    # ``None`` until a provider reports a count: an attempt that reported none adds nothing,
    # and a run of attempts that all reported none is unmeasured, not a zero-token run.
    total_input: int | None = None
    total_output: int | None = None
    # Alongside the totals above, accumulate the honest per-role observation the R3 `judge`
    # RoleUsage row is built from: reasoning stays None until a provider reports a split.
    total_reasoning: int | None = None
    judge_model: str | None = None
    # Where the dollars came from, as the client said. One client serves every attempt, so the
    # last attempt to name a source names it for all of them — the same reading `judge_model` gets.
    judge_price_source: str | None = None
    attempts_made = 0
    max_attempts = JUDGE_CALL_ATTEMPTS

    def _cumulative_usage() -> CallUsage | None:
        """Snapshot the spend across every attempt made so far, parsed or not.

        ``None`` when no attempt completed — the very first call raised, so nothing was
        observed. Reporting zeros there would claim a zero-token measurement in a path
        whose whole point is that missing and zero are different facts.
        """
        if attempts_made == 0:
            return None
        return CallUsage(
            model=judge_model,
            input_tokens=total_input,
            output_tokens=total_output,
            reasoning_tokens=total_reasoning,
            cost_usd=total_cost,
            price_source=judge_price_source,
            # Every attempt was a real provider call that spent real tokens, so the usage
            # row must count them all — not just the one that finally parsed.
            calls=attempts_made,
        )

    for attempt in range(max_attempts):
        try:
            result = await client.generate(
                system=system_prompt, user=user_prompt, response_format=JSON_OBJECT_RESPONSE_FORMAT
            )
        except Exception as e:  # prawduct:ok-broad-except — LLM/network boundary; SDK can raise many types
            # Earlier attempts in this loop may already have spent tokens before this one
            # raised; report that spend rather than losing it to the error path.
            if failure_describer is None:
                log.exception("%s LLM call failed for case %s: %s", label, case_id, e)
                return {"error": f"{label} LLM call failed: {e}", "judge_usage": _cumulative_usage()}
            failure = describe_and_log_failure(
                failure_describer,
                e,
                logger=log,
                where=f"{label} (case {case_id})",
                message="%s LLM call failed for case %s",
                args=(label, case_id),
            )
            reason = "was refused for the calling account" if failure.account_refused else "failed"
            return {
                "error": f"{label} LLM call {reason}: {failure.description}",
                "account_refused": failure.account_refused,
                "judge_usage": _cumulative_usage(),
            }

        # Account for this attempt's spend regardless of parse outcome — a
        # failed-parse attempt still costs $.
        attempt_input = getattr(result, "input_tokens", None)
        attempt_output = getattr(result, "output_tokens", None)
        attempt_cost = getattr(result, "cost_usd", None)
        attempts_made += 1
        total_input = sum_optional_tokens(total_input, attempt_input)
        total_output = sum_optional_tokens(total_output, attempt_output)
        # The first attempt sets the total; an unpriced attempt makes it unknown for good.
        if attempts_made == 1:
            total_cost = attempt_cost
        elif total_cost is not None and attempt_cost is not None:
            total_cost += attempt_cost
        else:
            total_cost = None
        total_reasoning = sum_optional_tokens(total_reasoning, getattr(result, "reasoning_tokens", None))
        judge_model = getattr(result, "model", "") or judge_model
        judge_price_source = getattr(result, "price_source", None) or judge_price_source
        if attempt_input or attempt_output:
            cost_str = f" cost=${attempt_cost:.6f}" if attempt_cost is not None else ""
            log.info(
                "eval_judge: model=%s input=%s output=%s%s judge=%s case=%s attempt=%d/%d",
                getattr(result, "model", "unknown") or "unknown",
                attempt_input,
                attempt_output,
                cost_str,
                label,
                case_id,
                attempt + 1,
                max_attempts,
            )

        # Checked BEFORE parsing. ``extract_json`` salvages partial payloads, so a
        # cut-short reply could otherwise parse, or fail as "Failed to parse" and be
        # re-bought. Either way the provider has already said why, and that is the
        # reason to report.
        incomplete = describe_incomplete_completion(result)
        if incomplete is not None:
            log.warning(
                "%s response cut short for case %s (stop_reason=%s, attempt %d/%d); not retried",
                label,
                case_id,
                result.stop_reason,
                attempt + 1,
                max_attempts,
            )
            return {
                "error": f"{label} reply was cut short by the provider and was not retried: {incomplete}",
                "response_preview": (result.content or "")[:200],
                "judge_usage": _cumulative_usage(),
            }

        parsed = _process_criteria_response(result, criteria_dicts, cannot_tell_offered=cannot_tell_offered)
        if parsed is not None:
            # Thread CUMULATIVE spend (incl. any failed attempts) onto the result
            # so the price/performance verdict isn't understated by retries.
            parsed["judge_usage"] = _cumulative_usage()
            # The scorer is THIS attempt's responder, as the provider named it. Not ``model``,
            # which a host may fill from the request, and not ``judge_model`` above, which is the
            # last attempt to report anything and exists to attribute spend.
            parsed["judge_served_model"] = result.served_model or None
            return parsed

        last_preview = (getattr(result, "content", "") or "")[:200]
        log.warning("%s response parse failed for case %s (attempt %d/%d)", label, case_id, attempt + 1, max_attempts)

    return {
        "error": f"Failed to parse {label.lower()} response after {max_attempts} attempts",
        "response_preview": last_preview,
        # Exhausted attempts still spent tokens and dollars — a result the program paid for
        # and must account for, even though it produced no score.
        "judge_usage": _cumulative_usage(),
    }


__all__ = ["CANNOT_TELL", "run_judge_llm"]
