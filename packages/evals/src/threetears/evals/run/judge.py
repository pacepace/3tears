"""LLM judge utilities — slimmed for the eval rebuild.

This module holds the callable that code outside the rubric judge depends on:

- :func:`run_judge_llm` — shared LLM-judge execution path: call the LLM,
  parse a criteria-based response (``{"reasoning": ..., "criteria_scores":
  {name: "1".."5", ...}}``), aggregate to a 0..1 normalized composite, log cost.
  Called from :mod:`threetears.evals.run.judge_service`, whose narrow single-dim rubric
  judge is built on it, and open to a host's own scorers.

**This is engine, and a host calling it is a host using the engine's public API**.
It is constructed with the injected
:class:`~threetears.evals.schema.completion.CompletionClient` rather than building one, takes
prompts and criteria as plain data, returns eval's own
:class:`~threetears.evals.kernel.usage_capture.CallUsage`, and names no host concept. The
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

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from threetears.evals.schema.models import (
    MODEL_DEFAULT_TEMPERATURE,
    PASS_FAIL_SCORES,
    SCALES,
    ClientRequestSettings,
    JudgeTemperature,
    KnownReasoningEffort,
)
from threetears.evals.schema.completion import (
    CompletionProtocolDefect,
    CompletionResult,
    ProviderFailureDescriber,
    sent_settings_of,
)
from threetears.evals.kernel.provider import (
    describe_and_log_failure,
    describe_incomplete_completion,
    extract_json,
    sum_optional_tokens,
)
from threetears.evals.kernel.authored import strict_response_format
from threetears.evals.kernel.usage_capture import CallUsage
from threetears.evals.schema.logs import get_logger

log = get_logger(__name__)

#: Extra judge attempts when a FINISHED response cannot be parsed. The
#: parse step is robust (see :func:`threetears.evals.kernel.provider.extract_json`), but a
#: single malformed response would otherwise silently drop a whole rubric dimension
#: (score=None) from the run. Because the judge call is non-deterministic, a re-ask
#: almost always returns parseable JSON, so one retry recovers the dim rather than
#: losing it. Kept small — a persistently-unparseable judge is a real error, not a
#: transient blip. A response the provider reported CUT SHORT is not retried at all:
#: see :func:`run_judge_llm`.
_JUDGE_PARSE_RETRIES = 1

#: Calls one judge dimension can make: the first, plus its parse retries. Each is one request on the
#: host's client, so a caller sizing a judge phase's wall clock multiplies the host's own request ceiling
#: by this (:func:`threetears.evals.analysis.reporter_kind.judge_phase_ceiling_s`).
JUDGE_CALL_ATTEMPTS = _JUDGE_PARSE_RETRIES + 1

#: How hard the eval judge reasons before it answers, as the effort word its reasoning budget reads as.
#:
#: **Asked for, not disabled — the classifier's remedy is the wrong one here.** The
#: classifier asks for two words, reasoning buys it nothing, so it sends
#: ``reasoning: {enabled: false}``. A judge's reply is short too (the JSON contract in
#: ``judge_service`` asks for one paragraph and one score), but the work behind the
#: score is not. ``reporter.groundedness`` checks every claim in a memo against its
#: evidence bundle, and that is what reasoning is for, so taking reasoning away would
#: change the instrument, not just its cost. That holds for a pass/fail dimension as for a
#: 1-5 one: a two-word answer can stand on as much checking as a number. Disabling is also
#: refused permanently by models whose reasoning is mandatory (``openai/gpt-5-mini`` answers
#: HTTP 400), which would bar that whole class of models from judging.
#:
#: **Not what is sent.** The judge asks for its reasoning as a token budget
#: (:data:`JUDGE_REASONING_HEADROOM_TOKENS`); this is the word that budget's share of the cap
#: reads as (:data:`~threetears.evals.negotiation.BUDGET_SHARES`: 8192 of 10240 is 0.80, ``high``),
#: which is what a model that takes words only is sent in its place, and what the docs and the
#: analysis writer's settings call the judge's effort.
JUDGE_REASONING_EFFORT: KnownReasoningEffort = "high"

#: The judge's private reasoning, as a bound: the budget its request asks for, and the room its output cap leaves
#: for it. **Sent**, as ``reasoning_max_tokens``: a word bounds no tokens, and a model that reads an effort word as
#: licence to think on (the G2 GLM judge did) reasoned past the cap and was cut off with no answer written. A budget
#: is a real bound wherever a model takes one (Anthropic's ``thinking.budget_tokens``); a model that takes effort
#: words only is sent the word the budget's share of the cap reads as (:data:`JUDGE_REASONING_EFFORT`), byte for byte
#: what it was sent when the judge asked by the word, and there the cap stays the only hard stop. A reply cut off
#: anyway is excluded as ``truncated`` with the cap it was sent (:class:`~threetears.evals.schema.JudgeCutShort`),
#: never re-bought.
#:
#: **The value is a judgement, not a measurement.** A calibration run lost
#: ``reporter.groundedness`` and ``reporter.calibration`` on every case at the old flat
#: 4096-token cap, with nearly all of its completion tokens spent reasoning. That shows the
#: two longest-reasoning dims wanted more than 4096 in total. It does not show how much
#: more, because every one of those calls was stopped. Twice the cap they overran is a
#: bound, not a prediction. Re-derive it from a run's reported ``reasoning_tokens``
#: (a cut judge call reports its reasoning share, on the result's ``judge_cut_short``) before moving it.
JUDGE_REASONING_HEADROOM_TOKENS = 8192

#: Output room kept for the judge's visible answer: the reply schema the judge is held to
#: (:func:`judge_reply_format`), one reasoning paragraph and one score per single-dimension
#: call, whatever the scale. That is a few hundred tokens, so this is headroom, and unused
#: output tokens are not billed.
JUDGE_ANSWER_BUDGET_TOKENS = 2048

#: The eval judge's output cap: the reply schema's answer budget plus the declared reasoning headroom, DERIVED
#: rather than chosen, so it always leaves room to think and then answer. The old flat 4096 sat below what a
#: reasoning judge spent. The cap is the sum by construction, so moving either part moves the cap with it, and the
#: reasoning budget sits a whole answer below it.
#:
#: **The cap is the only hard stop on a model that takes no budget.** A router's default routing may also drop a
#: parameter its chosen provider does not support, so the judge's settings ask for strict routing
#: (``ClientRequestSettings.strict_output``), which a host's client applies as its router's ``require_parameters``;
#: a client that resends a strict call without enforcement reports ``strict_output=False`` in what it sent. When a
#: judge still reaches the cap, :func:`run_judge_llm` reports the cut with its stop reason, the settings the call was
#: sent with and its token counts, and does not re-buy it.
#:
#: **What the budget changes.** Asking for reasoning turns it ON for a model that reasons only
#: when asked. The rule is the same for every eval judge (subject runs, reporter runs, and the
#: per-dim ``JudgeConfig`` models alike), which is why these numbers live here and not at any one
#: call site. A host's client builder applies them to its judge role.
JUDGE_MAX_TOKENS = JUDGE_REASONING_HEADROOM_TOKENS + JUDGE_ANSWER_BUDGET_TOKENS

#: The judge role's request settings as ONE value: what the host's client builder applies to every
#: eval judge client, and what a run launched with a judge records as ``judge_asked_settings``.
#: One object read by both, so the stamp cannot say something the engine did not ask.
#:
#: **Why it is recorded at all.** Moving any of it regrades every later run without changing
#: ``judge_model``, so the stamp differs on either side of the move, and the ``judge_request_settings``
#: apparatus dimension tells a campaign pooling runs from both sides that its judge may have moved. A run
#: that carries no stamp reads as unrecorded. What each call was SENT with is the client's to report, on
#: each score: a client sends the closest setting a model accepts
#: (:func:`~threetears.evals.negotiation.closest_settings`), so a model that takes no budget is sent
#: :data:`JUDGE_REASONING_EFFORT` in the budget's place and reports that.
JUDGE_REQUEST_SETTINGS = ClientRequestSettings(
    max_tokens=JUDGE_MAX_TOKENS,
    reasoning_max_tokens=JUDGE_REASONING_HEADROOM_TOKENS,
    # The budget must bind, so only a provider honouring it may answer (#686).
    strict_output=True,
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

#: The answers an ordinal criterion is asked for, as the strict reply format enumerates them: each score on the scale
#: written as a string, ``"1"`` to ``"5"``, read back to its integer by the parse.
#:
#: **Strings, because a string enum is the one enum form every provider's strict mode enforces.** An integer enum is
#: not portable: ``google/gemini-3.5-flash-lite`` through OpenRouter, with ``require_parameters`` on, accepted a
#: strict schema scoring a criterion as ``{"type": "integer", "enum": [1, 2, 3, 4, 5]}`` and did not enforce it,
#: answering ``"criteria_scores": {}`` with the score in prose, so every ordinal call of a run failed its parse twice
#: and was paid for twice; Gemini's structured output supports string enums only. The projection every package
#: directive takes now refuses a non-string enum (:func:`~threetears.evals.kernel.authored.strict_schema`).
ORDINAL_ANSWERS: tuple[str, ...] = tuple(str(score) for score in range(JUDGE_SCORE_MIN, JUDGE_SCORE_MAX + 1))


def _ordinal_score(name: str, value: Any) -> int:
    """Read one criterion's score as the integer on the scale it was asked for, or refuse it.

    The score is asked for as a string (:data:`ORDINAL_ANSWERS`, the strict reply format's enum) and read back to its
    integer here; a bare integer is read too, since a provider that does not enforce the schema may still send one.
    Either is the judge's protocol answer, so it is taken as given or not at all: a fraction, a number off the scale,
    any other string or a boolean is a reply that broke the protocol, and the caller's parse-failure path handles it.
    Nothing is truncated, rounded, stripped or clamped into range — each of those would record a score the judge
    never gave.

    Args:
        name: The criterion, for the message.
        value: What the reply carried for it.

    Returns:
        The score.

    Raises:
        ValueError: The value is neither one of :data:`ORDINAL_ANSWERS` nor an integer on the scale.
    """
    if isinstance(value, str):
        if value not in ORDINAL_ANSWERS:
            raise ValueError(
                f"Score for criterion '{name}' is not one of the {JUDGE_SCORE_MIN}-{JUDGE_SCORE_MAX} scale's answers "
                f"{list(ORDINAL_ANSWERS)}: {value!r}"
            )
        return int(value)
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


#: How each scale's answer is read into its stored score. Keyed like :data:`~threetears.evals.schema.models.SCALES`,
#: so a scale this module has not learned is a KeyError, not another scale's reader.
SCALE_READERS: dict[str, Callable[[str, Any], int]] = {"ordinal": _ordinal_score, "pass_fail": _pass_fail_score}


def judge_reply_format(criteria: Sequence[Mapping[str, Any]], *, cannot_tell_offered: bool) -> dict[str, Any]:
    """The strict ``json_schema`` directive a judge call sends: its reply, exactly as :func:`run_judge_llm` reads it.

    ``{"reasoning": ..., "criteria_scores": {...}}`` with every criterion a key and nothing else, each answered
    on its scale (``ordinal`` when the criterion names none) or with :data:`CANNOT_TELL` when it was offered.
    Derived from the criteria the call scores, as the parse is, so a provider that enforces it cannot write a reply
    the parse refuses, and a judge's parse retry spends nothing on one. The reply is still read and refused as
    before, for a provider that cannot enforce a strict schema.

    **Every answer is a string, an ordinal score included.** A criterion's answers are ONE string enum: a pass/fail
    criterion's two words, an ordinal criterion's :data:`ORDINAL_ANSWERS` (``"1"`` to ``"5"``, read back to integers
    by the parse), and :data:`CANNOT_TELL` beside either when it was offered, so no ``anyOf`` is needed. A string enum
    is the one enum form every provider's strict mode enforces; an integer enum was sent and not enforced
    (:data:`ORDINAL_ANSWERS` says where).

    Args:
        criteria: The criteria, as :func:`run_judge_llm` takes them: each with its ``name`` and, optionally, its
            ``scale`` (a key of :data:`SCALE_READERS`).
        cannot_tell_offered: Whether the prompt offered :data:`CANNOT_TELL`.

    Returns:
        The directive.
    """
    answers = {
        str(criterion["name"]): _scale_answers(str(criterion.get("scale", "ordinal")), cannot_tell_offered)
        for criterion in criteria
    }
    # Through the one projection every package directive takes, so the judge's sends only the portable strict subset.
    return strict_response_format(
        "judge_reply",
        {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "criteria_scores": {"type": "object", "properties": answers},
            },
        },
    )


def _scale_answers(scale: str, cannot_tell_offered: bool) -> dict[str, Any]:
    """One criterion's answers as a strict schema admits them, as one string enum: those :data:`SCALE_READERS` reads in
    the form the schema asks for them, and the offered one."""
    offered = [CANNOT_TELL] if cannot_tell_offered else []
    if scale == "pass_fail":
        return {"type": "string", "enum": [*PASS_FAIL_SCORES, *offered]}
    if scale != "ordinal":
        raise KeyError(f"no reader for scale {scale!r}")
    return {"type": "string", "enum": [*ORDINAL_ANSWERS, *offered]}


def _parse_criteria_response(
    content: str,
    evaluation_criteria: list[dict[str, Any]],
    *,
    cannot_tell_offered: bool = False,
) -> dict[str, Any]:
    """Parse a judge's criteria-based JSON response.

    Expected format::

        {"reasoning": "...", "criteria_scores": {"name": "1"-"5", ...}}

    ``reasoning`` is part of the protocol, not decoration: a stored score and a recorded "can't
    tell" both carry it, so a reply without one as a string — absent, ``null``, a number, an
    object — is a reply that broke the protocol and is refused here, at the parse boundary, with
    every other malformed reply. Refused anywhere later it would arrive after the call's spend was
    folded, and either lose that spend with the score or fail the result outright.

    A criterion dict carrying ``"scale": "pass_fail"`` is answered ``"pass"`` or ``"fail"`` and
    returned as 1 or 0; one without a scale is ordinal, answered as one of :data:`ORDINAL_ANSWERS` (or a bare
    integer on the scale) and returned as its integer. With ``cannot_tell_offered``, a criterion may carry :data:`CANNOT_TELL` in place of its
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
            method returning a :class:`~threetears.evals.schema.completion.CompletionResult`:
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
        ``reasoning``, ``judge_served_model`` and ``judge_temperature`` (see below), and ``judge_usage``; or
        ``{"error": "...", "response_preview": "..."}`` on failure. ``judge_usage`` is on every
        return that made a call: the cumulative spend of every attempt, parsed or not, whose
        ``cost_usd`` is ``None`` when any attempt reported no price — the dim's dollars are then
        unknown, never the priced attempts' partial sum. A reply the
        provider cut short fails with an error naming that cause and its token
        counts, after one call. A completion that carries no ``temperature`` (its type does not meet the
        protocol, :func:`sent_temperature`) is not a failed reply but the host's defect: it raises
        :class:`~threetears.evals.schema.completion.CompletionProtocolDefect`, naming the conformance check, after
        one call. A finished reply that will not parse — including one
        scoring a criterion with anything but a score on the scale — fails with
        "Failed to parse" after the bounded retries. Each caller
        composites the per-dimension scores itself (``judge_service`` across its
        single-dim calls, a host's own scorer however it composites), so this path returns
        the raw normalized scores and no composite of its own.

        ``judge_temperature`` is what that same attempt was sent at (:func:`sent_temperature`), and
        ``judge_sent_settings`` the request settings it was sent with, as its completion reported them
        (:func:`~threetears.evals.schema.completion.sent_settings_of`; ``None`` = unverified). ``judge_usage`` names
        sent settings only when every attempt reported the same value.

        ``judge_served_model`` is the model the provider's response named for the attempt whose
        scores are returned — that attempt, not the last one to report a model, because it is the
        one that scored — and ``None`` when that response named none. It is read off
        ``served_model`` and never off ``model``: the host may fill ``model`` from the request,
        and a floating alias in the request would then stand where the concrete scorer belongs,
        making two runs judged by different models compare equal.

    Raises:
        CompletionProtocolDefect: The client's completion does not meet the protocol (:func:`sent_temperature`), or
            its optional ``sent_settings`` holds something other than request settings or ``None``, carrying the
            calls made and what they cost.
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
    # What each attempt's response named as having answered, ``None`` for one that named none. The
    # usage record names a served model only when every attempt named the same one: one record
    # folding attempts answered by different models cannot name either.
    attempt_served: set[str | None] = set()
    # What each attempt's request was sent with, as its completion reported it (``None`` = unverified). The usage
    # record names one value only when every attempt reported the same one: a fold of unlike or unreported sends
    # reads as unverified, never as one of them.
    attempt_sent: list[ClientRequestSettings | None] = []
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
            served_model=next(iter(attempt_served)) if len(attempt_served) == 1 else None,
            input_tokens=total_input,
            output_tokens=total_output,
            reasoning_tokens=total_reasoning,
            cost_usd=total_cost,
            price_source=judge_price_source,
            # Every attempt was a real provider call that spent real tokens, so the usage
            # row must count them all — not just the one that finally parsed.
            calls=attempts_made,
            sent_settings=_agreed_sent_settings(attempt_sent),
        )

    def _thread_spend(defect: CompletionProtocolDefect) -> None:
        """Thread the attempts' spend onto a protocol defect and log it, for the caller to re-raise.

        Args:
            defect: The defect a member read raised, given the calls made so far and what they cost.
        """
        spent = _cumulative_usage()
        if spent is not None:
            defect.calls = spent.calls
            defect.model = spent.model
            defect.served_model = spent.served_model
            defect.input_tokens = spent.input_tokens
            defect.output_tokens = spent.output_tokens
            defect.reasoning_tokens = spent.reasoning_tokens
            defect.cost_usd = spent.cost_usd
            defect.price_source = spent.price_source
        log.error(
            "%s reply for case %s cannot be recorded, after %d attempt(s) costing %s: the completion breaks the "
            "protocol",
            label,
            case_id,
            attempts_made,
            "unpriced" if total_cost is None else f"${total_cost:.6f}",
        )

    # Every attempt sends the reply's shape, derived from the criteria the parse reads it by.
    response_format = judge_reply_format(criteria_dicts, cannot_tell_offered=cannot_tell_offered)
    for attempt in range(max_attempts):
        try:
            result = await client.generate(system=system_prompt, user=user_prompt, response_format=response_format)
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
        attempt_served.add(getattr(result, "served_model", None) or None)
        # What this attempt's request carried, as its client reports it. Absent is unverified and recorded so; a value
        # of the wrong type is the host's defect, raised like a missing temperature, with the spend threaded on. Read
        # here, before the reply is judged usable, so a cut-short or unparsable attempt's spend row says it too.
        try:
            sent = sent_settings_of(result)
        except CompletionProtocolDefect as defect:
            attempt_sent.append(None)
            _thread_spend(defect)
            raise
        attempt_sent.append(sent)
        # Logged whenever the provider reported anything about the attempt's spend — a call with
        # unreported counts and a known cost still spent, and a truthiness test would drop it.
        if attempt_input is not None or attempt_output is not None or attempt_cost is not None:
            cost_str = f" cost=${attempt_cost:.6f}" if attempt_cost is not None else ""
            log.info(
                "eval_judge: model=%s input=%s output=%s%s judge=%s case=%s attempt=%d/%d",
                getattr(result, "model", "unknown") or "unknown",
                "unreported" if attempt_input is None else attempt_input,
                "unreported" if attempt_output is None else attempt_output,
                cost_str,
                label,
                case_id,
                attempt + 1,
                max_attempts,
            )

        # Checked BEFORE parsing. ``extract_json`` salvages partial payloads, so a
        # cut-short reply could otherwise parse, or fail as "Failed to parse" and be
        # re-bought. Either way the provider has already said why, and that is the
        # reason to report. Every cut-short reason takes this one branch: ``truncated`` (a cause
        # the provider did not name) is handled as ``max_tokens`` is, its parse not trusted, and
        # nothing here treats ``error`` apart from them.
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

        # The temperature is read before the reply is. A completion whose type carries none is the host's defect, not
        # a judging outcome: it is raised (CompletionProtocolDefect) and fails the run, since every call through this
        # client returns the same type. The attempts' spend goes up with it (``calls``, ``cost_usd``) and is logged.
        try:
            temperature = sent_temperature(result)
        except CompletionProtocolDefect as defect:
            _thread_spend(defect)
            raise

        parsed = _process_criteria_response(result, criteria_dicts, cannot_tell_offered=cannot_tell_offered)
        if parsed is not None:
            # Thread CUMULATIVE spend (incl. any failed attempts) onto the result
            # so the price/performance verdict isn't understated by retries.
            parsed["judge_usage"] = _cumulative_usage()
            # The scorer is THIS attempt's responder, as the provider named it. Not ``model``,
            # which a host may fill from the request, and not ``judge_model`` above, which is the
            # last attempt to report anything and exists to attribute spend.
            parsed["judge_served_model"] = result.served_model or None
            # And the temperature that attempt was sent at, as the client reports it — the rest of the
            # scorer's identity, since a model that refuses a temperature is sent none.
            parsed["judge_temperature"] = temperature
            # And the request settings that attempt was sent with, as the client reports them (None = unverified):
            # the rest of how the scorer was asked, which a client may have moved to the closest setting the model
            # accepts.
            parsed["judge_sent_settings"] = sent
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


def defect_usage(defect: CompletionProtocolDefect) -> CallUsage | None:
    """What the calls made before a completion's protocol defect was found spent, as one usage record.

    What a caller counts the billed calls by on its way out (:func:`run_judge_llm` sets them on the defect), so a run
    failed by the defect still sums what it paid for.

    Args:
        defect: The defect, as :func:`run_judge_llm` raised it.

    Returns:
        The usage, or ``None`` when no call was counted on it.
    """
    if defect.calls == 0:
        return None
    return CallUsage(
        model=defect.model,
        served_model=defect.served_model,
        input_tokens=defect.input_tokens,
        output_tokens=defect.output_tokens,
        reasoning_tokens=defect.reasoning_tokens,
        cost_usd=defect.cost_usd,
        price_source=defect.price_source,
        calls=defect.calls,
    )


def _agreed_sent_settings(attempts: Sequence[ClientRequestSettings | None]) -> ClientRequestSettings | None:
    """The one value every attempt's request was sent with, or ``None`` when they do not all report the same one.

    What a usage record folding the attempts can name: one record cannot name two values, and an attempt that reported
    none leaves the fold unverified, as one unpriced attempt leaves its dollars unknown.

    Args:
        attempts: What each attempt was sent with, in order (``None`` = that attempt's client did not say).

    Returns:
        The shared value, or ``None``.
    """
    if not attempts or attempts[0] is None:
        return None
    first = attempts[0]
    return first if all(sent == first for sent in attempts[1:]) else None


def sent_temperature(completion: CompletionResult) -> JudgeTemperature:
    """The temperature a judge completion was actually sent at, as its client reports it (#633).

    Read off :attr:`~threetears.evals.schema.completion.CompletionResult.temperature`, which the client
    sets to what its request carried — ``None`` when it sent none, as it must for a model that refuses a
    temperature. The engine asks every judge call for :data:`~threetears.evals.schema.models.DEFAULT_JUDGE_TEMPERATURE`
    unless a config says otherwise, and never assumes the request was honoured.

    **The attribute is the protocol's, not optional.** Read through ``hasattr``, a host whose completion type
    omitted it had every judge recorded at no temperature — unrecordable as a profile, and "unknown" in every
    judge comparison — with no error anywhere; so a completion without it is refused here, naming the check that
    holds a host's type to the protocol.

    Args:
        completion: The judge call's completion.

    Returns:
        The number sent; :data:`~threetears.evals.schema.models.MODEL_DEFAULT_TEMPERATURE` when the client sent none.

    Raises:
        CompletionProtocolDefect: The completion carries no ``temperature`` — its type does not meet the
            :class:`~threetears.evals.schema.completion.CompletionResult` protocol. The host's defect, which fails
            the run rather than one dimension (a ``TypeError``).
    """
    try:
        sent = completion.temperature
    except AttributeError as missing:
        raise CompletionProtocolDefect(
            completion,
            "temperature",
            "a judge is recorded at the temperature its call was sent at, so report it (None when the request "
            "carried none)",
        ) from missing
    return MODEL_DEFAULT_TEMPERATURE if sent is None else float(sent)


__all__ = ["CANNOT_TELL", "defect_usage", "run_judge_llm", "sent_temperature"]
