"""What the toy host sweeps — scalars and bytes, each chosen for the engine property it exercises.

| Declaration | What it exercises |
|---|---|
| ``chunk_tokens`` | a numeric lever with real spacing, naming the covariate it acts on (``acts_on``) |
| ``retriever_top_k`` | the same, at a second arity |
| ``extraction_schema`` | content-addressing an opaque blob |
| ``ocr_engine_version`` | the ``unknown`` state, by design: a value a host cannot always record |
| ``reviewer_pool`` | a confound no engine could guess, pinned into a role the ENGINE never declared |
| ``grader_version`` | a NON-MODEL grader nominated into the engine's own judge role |
| ``batch_label`` | a label that identifies and never confounds |
| ``retrieval_overrides`` / ``resolved_retrieval_config`` | an open family and the surface it is merged into, one change reaching two levers |

**Readers take a run.** The engine's reader signature is ``(EvalRun, results)`` — the engine's own
carrier, not a host type — so a host reads its vocabulary out of the engine-owned ``host_payload``
slot it wrote at launch. One carrier, any vocabulary, and an engine that cannot tell which it is
holding.

The numeric levers register without a ``scale``. A level built through ``SweepableValue`` carries
one; a *declaration* does not, because a ``Sweepable`` says what an input IS and a
``SweepableValue`` says what one level of it was.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from threetears.evals.contracts.host import SHARED_CORE, RolePins, Sweepable, SweepableRegistry, SweepableValue

if TYPE_CHECKING:
    from threetears.evals.contracts import EvalResult, EvalRun

#: Where the toy host keeps its own vocabulary on the shared carrier: ``EvalRun.host_payload``,
#: the engine-owned opaque slot. The engine stores it verbatim and never reads it, so a host's data
#: needs no shape the engine knows.
_TOYHOST_NAMESPACE = "toyhost"


def _reader(key: str, *, default: Any = None) -> Any:
    """Build a reader for one toy-host key.

    Args:
        key: The toy-host vocabulary name.
        default: What an observation that never recorded this key reads as.

    Returns:
        A reader the engine calls and never inspects.
    """

    def read(run: EvalRun, _results: Sequence[EvalResult]) -> Any:
        return (run.host_payload or {}).get(_TOYHOST_NAMESPACE, {}).get(key, default)

    return read


TOYHOST_SWEEPABLES: tuple[Sweepable, ...] = (
    # The subject's one component, carried into the variant key as itself: a component no lever carries
    # is refused when the variant identity is derived, since the key hashes only the lever map.
    Sweepable(
        name="extraction_prompt",
        role="lever",
        read=lambda run, _results: (
            prompt.content_hash if (prompt := run.subject_snapshot.components.get("extraction_prompt")) else None
        ),
        reader_prose="the extraction prompt the subject carried, by content",
    ),
    # A lever that names its MECHANISM: a wider chunk is supposed to carry more context into extraction,
    # so the engine's own covariate `context_tokens_in` should move across its levels. A sweep where it
    # did not is a knob that never took effect, and the bundle reports it as `inert` rather than letting
    # "chunk width does not matter" stand as a finding. Any registered measure or covariate will do; a
    # name neither the engine nor this host declares is refused when the profile is built.
    Sweepable(
        name="chunk_tokens",
        role="lever",
        read=_reader("chunk_tokens"),
        reader_prose="how many tokens each document chunk carried into extraction",
        acts_on="context_tokens_in",
    ),
    Sweepable(
        name="retriever_top_k",
        role="lever",
        read=_reader("retriever_top_k"),
        reader_prose="how many candidate chunks the retriever returned per field",
    ),
    Sweepable(
        name="extraction_schema",
        role="lever",
        read=_reader("extraction_schema"),
        reader_prose="the field schema the extractor was asked to fill, by content",
    ),
    # Deliberately UNRECORDED on part of the corpus: `unknown` as a state a host designs for, not
    # a field that predates a stamp. A host whose config epoch is a clock rather than a value has
    # this permanently — two observations at different epochs may or may not have differed.
    Sweepable(
        name="ocr_engine_version",
        role="apparatus",
        read=_reader("ocr_engine_version"),
        reader_prose="which OCR build produced the text the extractor read",
        confounds=(
            "a different OCR build transcribed the page, so the extractor was reading different text — a field it "
            "missed may never have been legible rather than having been extracted badly"
        ),
        indeterminate_when_blank=True,
    ),
    Sweepable(
        name="reviewer_pool",
        role="apparatus",
        read=_reader("reviewer_pool"),
        reader_prose="which pool of human reviewers adjudicated the sampled documents",
        confounds=(
            "a different reviewer pool adjudicated these documents, and pools disagree on borderline fields — a "
            "measured accuracy gain can be a change in who was grading"
        ),
    ),
    # The grader, and it is code. Nominated into the ENGINE's ``judge`` role below, which is the
    # assertion: "who graded the work" is a role every product has, and a model is only one
    # answer to it. The core's ``judge_model`` holds a model id and its documented
    # ``None`` means the judge is UNRECOVERABLE, so there is nothing legal a code grader could put
    # there — its kind contract leaves that dimension unseated and names its grader here instead.
    #
    # ``indeterminate_when_blank`` because an observation that never recorded a grader version did
    # not grade with version "" — the absence is an absence, and two such observations did not
    # thereby grade alike.
    Sweepable(
        name="grader_version",
        role="apparatus",
        read=_reader("grader_version"),
        reader_prose="the version of the extraction grader that scored the fields against the adjudicated key",
        confounds=(
            "a different grader version scored the extractions — the comparison rule, the numeric tolerance or the "
            "adjudicated key itself can have moved, so a measured accuracy change can be a change in what counts as "
            "correct rather than in what was extracted"
        ),
        indeterminate_when_blank=True,
    ),
    Sweepable(
        name="batch_label",
        role="label",
        read=_reader("batch_label"),
        reader_prose="the operator's name for the batch these documents were drawn from",
    ),
)

#: The toy host's pinned roles — both halves of what a second product needs from the role seam.
#:
#: **``judge`` carries no subject phrase** because the engine already introduced it; this entry
#: NOMINATES a host dimension into a role the core declared, and re-wording what "judge" means
#: would be an edit rather than an extension. The disclosure then groups ``grader_version`` under
#: the same "judged the same way" heading the core's ``judge_model`` gets, which is the whole
#: point: an operator reading a grader change reads it as a grader change and not as an unnamed
#: apparatus confound.
#:
#: **``adjudicator`` is a role the engine has never heard of**, and it supplies its own phrase.
#: The ground truth a grader scores against is decided by people here, and who they were is a seat
#: in the rig exactly as the judge is — but it is not a judge, and folding it into that heading
#: would print one sentence that is false about whichever of the two did not move.
TOYHOST_ROLES: tuple[RolePins, ...] = (
    RolePins(name="judge", pins=("grader_version",)),
    RolePins(
        name="adjudicator",
        pins=("reviewer_pool",),
        subject_phrase="adjudicated by the same reviewer pool",
    ),
)

#: The toy host's own registry. Built on the same shared core every host extends, which is the whole
#: assertion: two hosts, one core, no engine surface that can tell them apart. The extractor's overlay
#: levers are not here: the profile derives them from the kind's contract on ``kinds``.
TOYHOST_SWEEPABLE_REGISTRY: SweepableRegistry = SHARED_CORE.extend(TOYHOST_SWEEPABLES, roles=TOYHOST_ROLES)


# ---------------------------------------------------------------------------
# Retrieval tuning — an open family and the surface it is merged into.
#
# The toy host's retrieval stage has tuning knobs an operator may overlay per batch, and the batch
# records the RESOLVED retrieval configuration those overlays were merged into. Both are levers:
# the knob is what a campaign swept, the resolved configuration is what would ship. So one
# overlaid knob moves two levers, which is the shape a lens must read as one change.
#
# Kept off TOYHOST_SWEEPABLE_REGISTRY and registered as its own extension, because every toy-host
# suite that does not sweep retrieval would otherwise carry one more variant coordinate for a lever
# it never sets. The profile opts in (``toyhost_profile(tunable_retrieval=True)``).

#: The retrieval knobs an operator may overlay. The host's keyspace, which is why membership is
#: asked of this set rather than inferred from a name's shape.
TOY_RETRIEVAL_KEYS: frozenset[str] = frozenset({"rerank_depth", "dedupe_threshold"})

#: Prefix a retrieval knob's member name carries — its lever name is ``retrieval.<key>``.
RETRIEVAL_MEMBER_PREFIX = "retrieval."

#: The resolved retrieval configuration's lever name — the surface the family writes into.
RESOLVED_RETRIEVAL_CONFIG = "resolved_retrieval_config"


def _payload(run: EvalRun) -> dict[str, Any]:
    """The toy host's own keys for this batch."""
    payload: dict[str, Any] = (run.host_payload or {}).get(_TOYHOST_NAMESPACE, {})
    return payload


def _retrieval_members(run: EvalRun, _results: Sequence[EvalResult]) -> dict[str, Any]:
    """One member per retrieval knob this batch's launch overlaid.

    Args:
        run: The batch.
        _results: Unused — overlays are declared on the batch.

    Returns:
        ``{retrieval.<key>: value}``; empty for a batch that overlaid nothing.
    """
    return {
        f"{RETRIEVAL_MEMBER_PREFIX}{key}": value
        for key, value in (_payload(run).get("retrieval_overrides") or {}).items()
    }


def _is_retrieval_member(name: str) -> bool:
    """Whether ``name`` is a retrieval knob this host lets a campaign sweep."""
    return name.startswith(RETRIEVAL_MEMBER_PREFIX) and name.removeprefix(RETRIEVAL_MEMBER_PREFIX) in TOY_RETRIEVAL_KEYS


def _resolved_retrieval(run: EvalRun, _results: Sequence[EvalResult]) -> str | None:
    """The resolved retrieval configuration's content hash — the level two batches agree on.

    Args:
        run: The batch.
        _results: Unused.

    Returns:
        The hash, or ``None`` for a batch that recorded no resolved configuration.
    """
    config = _payload(run).get("retrieval_config")
    return None if config is None else SweepableValue.of(config, display="retrieval config").content_hash


def _retrieval_residual(run: EvalRun, _results: Sequence[EvalResult], removed: frozenset[str]) -> dict[str, Any] | None:
    """The resolved retrieval configuration with the named knobs taken back out.

    Args:
        run: The batch.
        _results: Unused.
        removed: ``retrieval.<key>`` member names.

    Returns:
        The configuration minus those keys, or ``None`` for a batch that recorded none.
    """
    config = _payload(run).get("retrieval_config")
    if config is None:
        return None
    return {key: value for key, value in config.items() if f"{RETRIEVAL_MEMBER_PREFIX}{key}" not in removed}


TOYHOST_RETRIEVAL_TUNING: tuple[Sweepable, ...] = (
    # Indeterminate when blank: a batch with no resolved configuration is one that never recorded
    # it, and two such batches did not thereby retrieve alike.
    Sweepable(
        name=RESOLVED_RETRIEVAL_CONFIG,
        role="lever",
        read=_resolved_retrieval,
        reader_prose="the retrieval configuration the batch actually ran, after any overlaid knob",
        indeterminate_when_blank=True,
    ),
    Sweepable(
        name="retrieval_overrides",
        role="lever",
        read=_retrieval_members,
        owns_member=_is_retrieval_member,
        reader_prose="the retrieval knobs the launch overlaid, one lever per knob",
        open_family="an operator may overlay any retrieval knob, so the members are whatever a batch reached for",
        resolves_into=RESOLVED_RETRIEVAL_CONFIG,
        read_residual=_retrieval_residual,
    ),
)

#: The toy host's registry with retrieval tuning registered.
TOYHOST_TUNABLE_SWEEPABLE_REGISTRY: SweepableRegistry = TOYHOST_SWEEPABLE_REGISTRY.extend(TOYHOST_RETRIEVAL_TUNING)


__all__ = [
    "RESOLVED_RETRIEVAL_CONFIG",
    "RETRIEVAL_MEMBER_PREFIX",
    "TOYHOST_RETRIEVAL_TUNING",
    "TOYHOST_ROLES",
    "TOYHOST_SWEEPABLES",
    "TOYHOST_SWEEPABLE_REGISTRY",
    "TOYHOST_TUNABLE_SWEEPABLE_REGISTRY",
    "TOY_RETRIEVAL_KEYS",
]
