"""The analysis engine's generation-contract exceptions.

This module is their one import path: callers (the eval service, the REST and MCP layers, tests)
name them from here without importing the generator.
:mod:`threetears.evals.analysis.generator` owns the generation round-trip and its structural
validators and raises them; :mod:`threetears.evals.analysis.references`, which resolves the model's
references against the decision surface on the generator's behalf, raises
:class:`UnresolvableReference`.
"""

from __future__ import annotations


class GenerationError(RuntimeError):
    """The generator's output could not be turned into a valid analysis.

    Raised for a generation-contract violation the caller should surface (and may
    retry), never silently absorb. The bare class is now narrow: it means the call
    itself did not finish, or never happened. Everything a finished call can get
    wrong raises the :class:`SoundnessRefusal` subclass instead, which is what buys
    the repair round-trip.
    """


class SoundnessRefusal(GenerationError):
    """A finished generator call whose OUTPUT was refused — the repairable half.

    The distinction from a bare :class:`GenerationError` is what bounds the repair
    round-trip in :func:`~threetears.evals.analysis.generator.generate_analysis`, so it is a
    contract and not a label.

    **The boundary is "did the call finish?", not "which step complained".** A
    ``SoundnessRefusal`` means the provider ran to completion and returned something the
    engine will not store: output that is not JSON, a payload missing a required field, an
    insight element missing its statement, or a whole well-formed analysis that then broke a
    structural rule — a reference naming a reading the decision surface does not hold, a
    confidence stated as a number rather than a tier, a citation naming a run the bundle never
    supplied.
    Every such refusal names what was wrong, and usually the correct alternative, so feeding
    it back is telling a non-deterministic component precisely what to change. That is
    retry-with-feedback.

    A bare ``GenerationError`` means there was no output to correct: the provider cut the
    call short (an output-cap truncation or a content filter), or nothing was ever sent —
    because the resolved prompt predates the memo contract, because the bundle describes
    no arm at all, so nothing the memo said could name which arm it meant, or because the host
    does not allow the requested writer model. Those
    must NOT be repaired here, and the reason is the truncation case specifically — regenerating a generation that hit
    the output cap spends a second full charge to hit the same cap
    (:func:`~threetears.evals.contracts.provider.describe_incomplete_completion` says so in the message).
    A repair that fired there would be the "fallback tier accumulating untraceable
    behavior" the project rule warns about, dressed as a retry.

    **The nothing-was-sent cases are preconditions on what the CALLER holds, and that is why
    they belong on this side.** A repair round-trip corrects what a generator claimed; neither a
    prompt that never asks for a memo, nor a bundle whose every arm carries a key and no levels, nor
    a writer model the host's ``analysis_writer_models`` does not list (#644) is something the
    generator said, so there is no claim to correct and a second call reaches the identical place at
    a second full charge. Each is decidable for free before the first call, and each is refused there
    (:func:`~threetears.evals.analysis.generator.refuse_an_unlisted_writer` is the third, argued
    here: the request names a writer the host refuses, so no call is made and none could succeed).
    Adding another such case means arguing it here.

    **The boundary used to sit one step further out, and drawing it by exception class made
    it inconsistent.** Schema-validation and parse failures were excluded alongside
    truncation, on the reasoning that "there is no analysis to correct, so there is nothing
    to feed back". That reasoning does not survive contact with what is actually fed back:
    the refusal text for a field failure is the pydantic message naming the missing key, and
    for a parse failure it is the decoder's line and column — both instructions a generator
    can act on, unlike a cap it cannot see and cannot raise. Because those two steps run
    FIRST in the chain, output failing basic field shape died before it could reach any of
    the validators that would have repaired it, so a generation could be billed and
    discarded that way. The truncation exclusion, which rests on the second call provably
    repeating the first, is what remains.

    **The truncation exclusion is enforced, not assumed.** This docstring once asserted that
    "a generation that hit the output cap arrives as a parse failure" — nothing checked it.
    ``extract_json`` carries three salvage fallbacks (a fenced-block strip, a ``raw_decode``
    prefix read, and a first-brace-to-last-brace slice), so a cut-short payload that any of
    them recovers reaches the soundness chain looking complete. It is then a strong candidate
    for any ``_reject_*`` validator — and a refusal there would buy exactly the second full
    generation a truncation must never buy.
    :func:`~threetears.evals.analysis.generator._reject_incomplete_generation` therefore runs
    before anything parses, so no cut-short result can reach a ``_reject_*`` validator at
    all. That ordering is now the only thing holding the boundary, which is why it is a
    chokepoint rather than a check at each call site.

    **The residual, stated rather than papered over.** That chokepoint reads exactly one
    signal — ``result.stop_reason`` against ``INCOMPLETE_STOP_REASONS`` — so it is only as
    good as the host's normalisation, and
    :class:`~threetears.evals.contracts.completion.CompletionResult` documents the hole in its own words:
    an implementation that passes a raw provider string through (OpenAI's ``length``, say)
    reads as finished here. Widening the repairable class made that residual more expensive,
    not less: such a result used to fail terminally at one billed call, because a parse
    failure was terminal, and now buys a second full generation into the identical cap —
    the outcome this class exists to prevent. The fix belongs at the port, where the mapping
    is a host's contract to satisfy, not in a second stop_reason check here that would be a
    fallback tier for a host that already broke the contract.
    """


class UnresolvableReference(SoundnessRefusal):
    """A reference in the model's output names a reading the decision surface does not hold.

    A :class:`SoundnessRefusal` because the model can correct it: the message names the cell and
    measure it asked for and what that cell does hold, which is exactly what the repair round
    feeds back. Refusing rather than dropping the reference is the point — a figure that could not
    be resolved must never reach a reader as a figure, and a silently dropped evidence row misstates
    what the finding rested on.
    """


__all__ = ["GenerationError", "SoundnessRefusal", "UnresolvableReference"]
