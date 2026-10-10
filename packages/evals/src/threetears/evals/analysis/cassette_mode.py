"""Cassette mode: the disclosure for runs that did not all record the same cassette mode.

:func:`cassette_mode_disclosure` names the span when pooled runs mixed recorded and replayed calls;
:data:`SUBSTITUTING_CASSETTE_MODE` is the one mode that substitutes third-party output.
"""

from __future__ import annotations

from collections.abc import Mapping


# The distinguishing clause of the cassette-span disclosure, split out for the same
# reason :data:`DISJOINT_WINDOWS_CLAUSE` is: a test pins the CLAUSE, so the sentence
# around it stays free to be rewritten for a reader.
CASSETTE_SPAN_CLAUSE = "did not all record the same cassette mode"

#: The one cassette mode that SUBSTITUTES third-party output. ``off`` and ``capture``
#: both run the third party live — capture additionally records what came back — so a
#: span across those two is a difference in recording, not in measurement. Naming the
#: substituting mode once keeps the disclosure's two branches from drifting apart.
SUBSTITUTING_CASSETTE_MODE = "replay"


def cassette_mode_disclosure(modes_by_run: Mapping[str, str]) -> str | None:
    """The sentence a comparison must carry when its runs recorded different cassette modes.

    **One vocabulary, two layers.** ``cassette_mode`` is already a declared apparatus
    confound in the host's sweepable declarations — "tool output was
    live for some of these runs and replayed for others, which changes both latency and
    content" — which is what ``bisect_runs`` and the analysis bundle's confound scan read,
    and the generated analysis already names it correctly. The badge derived from this
    sentence is therefore ``cassette_mode_differs``: the declared dimension name plus the
    suffix its sibling ``tool_config_differs`` already uses, so the reporting layer and the
    analysis layer name one thing once. What follows is the same claim said in the detail a
    comparison surface needs and a confound catalogue does not carry.

    Reports what the runs RECORDED and says so in those words, because nothing on a run
    corroborates the mode it recorded. Three candidate corroborators were checked and each
    is the same claim one level down or weaker than it: :attr:`~threetears.evals.schema.models.EvalRun.cassette_corpus_id` is
    set exactly when the run claims replay, and
    :func:`~threetears.evals.kernel.usage_capture.count_substituted_deliveries` counts seeded case
    findings alongside replayed cassettes, so a non-zero count does not evidence replay
    and a zero one cannot separate "did not replay" from "replayed a template with no
    async delivery". So this asserts only the record and names the ambiguity, the choice
    the run-detail badge already makes.

    **Two spans, one badge.** Only ``replay`` substitutes: an arm that replayed re-served
    a recording rather than measuring the third party, which confounds its quality — a
    replayed arm is served only the asks its capture made (any other ask stops the cell), so
    it is measured on the questions the capturing candidate chose. **Its cost is confounded at
    one of the two seams, not both**, which is why the sentence names them separately: a
    replayed background DELIVERY spent no inner-agent dollars and its
    production-replicating cost is withheld, while a replayed tool ACTION (a search, say) saved
    provider credits that never entered the per-role rows on a live arm either, so its cost
    is reported and is the same figure the live arm would report. Saying "a replayed arm's
    cost is withheld" flatly was wrong for the second seam and told a reader to expect a
    blank where a number correctly appears — see
    :func:`~threetears.evals.kernel.usage_capture.production_replicating_cost`. A span that stays
    within ``off``/``capture`` is a difference in what was *recorded* while both arms ran
    live. Both are disclosed, because both are a difference in a condition that should have
    been holding still; the branch decides only what the sentence says the difference costs.

    Descriptive, never a verdict, on the rule :func:`measurement_window_disclosure`
    follows: no comparison is refused, no number is adjusted and no severity is assigned.
    Capture-versus-replay is a legitimate thing to run.

    Args:
        modes_by_run: One recorded ``cassette_mode`` per run being compared, keyed by run id.

    Returns:
        The disclosure, rendered verbatim by every surface, or ``None`` when fewer than
        two runs were supplied or every run recorded the same mode. Two arms that both
        replayed are uniform and disclose nothing: the substitution is on both sides, so
        the delta between them is honest.
    """
    if len(modes_by_run) < 2:
        return None
    recorded = set(modes_by_run.values())
    if len(recorded) < 2:
        return None
    detail = "; ".join(f"{run_id} recorded {mode}" for run_id, mode in sorted(modes_by_run.items()))
    if SUBSTITUTING_CASSETTE_MODE in recorded:
        cost = (
            "A replayed arm did not measure the third party — it re-served a recording. Where it "
            "replayed a background DELIVERY its production-replicating cost is withheld as unknown "
            "rather than reported smaller; where it only replayed a tool ACTION the cost is "
            "still reported, and is not understated, because that tool's spend never reaches the "
            "per-role rows on a live arm either — what replay saved there is search credits, not "
            "dollars. Its QUALITY is confounded either way: a replayed arm is served only the asks its "
            "capture made, so it is measured on the questions the capturing candidate chose to ask."
        )
    else:
        cost = (
            "Both 'off' and 'capture' run the third party live — capture additionally records what "
            "came back — so this is a difference in what was recorded rather than in what was measured."
        )
    return f"These runs {CASSETTE_SPAN_CLAUSE} ({detail}). {cost}"


__all__ = [
    "cassette_mode_disclosure",
    "CASSETTE_SPAN_CLAUSE",
    "SUBSTITUTING_CASSETTE_MODE",
]
