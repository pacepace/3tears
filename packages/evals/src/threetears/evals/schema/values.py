"""One level of one swept input — the half of the vocabulary that crosses storage.

Its sibling :mod:`~threetears.evals.kernel.host.sweepables` declares what an input IS: a runtime
registration carrying a live reader, never stored. This module declares what one LEVEL of one was:
a Pydantic model where every field is a persisted-format commitment. The two were one file, and
the file's own docstring described only the first half — which matters because the persisted half
is the one whose changes need format review, and it was the one nobody's eye was drawn to.

**Identity and rendering, together and not separable.** R9 requires identity to be
content-addressed — a value joins across observations by what it contains, never by a name the
engine would have to resolve against a store it cannot see. R8 separately requires legibility: an
eval system whose inputs are opaque hashes degrades to "component 3 moved" and cannot write a
readable analysis. A hash with no rendering and a rendering with no hash are each half of what
every consumer downstream needs, so they are one model rather than two.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.schema.hashing import bytes_digest, canonical_digest


class NominalScale(BaseModel):
    """Unordered categories. Two levels are different, and neither is larger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["nominal"] = "nominal"


class OrdinalScale(BaseModel):
    """Ordered but unspaced — ``small`` / ``medium`` / ``large``.

    ``rank`` orders the levels and says nothing about the distance between them, which is the
    whole reason this is not :class:`IntervalScale`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ordinal"] = "ordinal"
    rank: int = Field(description="Position in the host's ordering. Comparable within one axis only.")


class IntervalScale(BaseModel):
    """A real number with real spacing — ``0.4``, ``15``, ``2000ms``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["interval"] = "interval"
    value: float = Field(description="The number itself, in ``unit``.")
    unit: str | None = Field(
        default=None, description="The unit, when the number has one. ``None`` for a dimensionless value."
    )


#: What kind of axis a swept value sits on. Three, not two, and **not a bare ordinal rank**:
#: rank without spacing draws a false chart. Sweep a threshold at 0.1 / 0.4 / 0.85 and an ordinal
#: renders those three equally spaced, putting the knee in the wrong place — wrong precisely in
#: the case that motivates having the field at all.
Scale = Annotated[NominalScale | OrdinalScale | IntervalScale, Field(discriminator="kind")]

#: What the "not a run of this kind" level is addressed by, ahead of the kind's name. A NUL byte never
#: begins canonical JSON, so this level's hash can equal no value a field holds — ``None`` included.
_NOT_THIS_KIND = b"\x00not a run of kind "

#: How the "not a run of this kind" level reads, around the kind's name.
_NOT_THIS_KIND_DISPLAY = ("(not a ", " run)")


class SweepableValue(BaseModel):
    """One level of one swept input: what it *is*, what to call it, and what kind of axis it is on.

    R9 requires identity to be **content-addressed** — a value joins across observations by what
    it contains, never by a name the engine would have to resolve against a store it cannot see.
    R8 separately requires **legibility**: an eval system whose inputs are opaque hashes degrades
    to "component 3 moved" and cannot write a readable analysis. This model is where the two meet,
    and it is deliberately not two models: a hash with no rendering and a rendering with no hash
    are each half of what every consumer downstream needs, and splitting them invites one to be
    passed without the other.

    Unlike :class:`Sweepable` — a runtime registration carrying a live callable — this **crosses
    storage**, so it is a Pydantic model and every field here is a persisted-format commitment.

    ``raw`` is for a human debugging a key that came out wrong, and is the one field nothing may
    compute from: it is absent whenever the underlying value is large or is content a snapshot
    must not retain, and a consumer reading it as "the value" would work until the first time it
    was dropped.
    """

    # Strict like every stored eval type, and like the scale models above: a field this build does not
    # declare is refused on read as on construction. Frozen, so it declares its own config.
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    content_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description="sha256 over the value's content. THE identity — what joins this level across observations (R9).",
    )
    display: str = Field(
        min_length=1,
        description=(
            "What a reader is shown. Required and never derived from ``content_hash``, because a hash has no "
            "natural rendering; for an interval value the constructors derive it from the number and unit."
        ),
    )
    scale: Scale = Field(description="What kind of axis this level sits on — see :data:`Scale`.")
    raw: Any | None = Field(
        default=None,
        description="The underlying value, for debugging only. Absent whenever retaining it would be wrong or large.",
    )

    @classmethod
    def of(
        cls,
        content: Any,
        *,
        display: str | None = None,
        scale: Scale | None = None,
        keep_raw: bool = False,
    ) -> SweepableValue:
        """Build a value by content-addressing a JSON-safe structure.

        Args:
            content: The value's content. Hashed canonically, so key order and whitespace do not
                move the identity.
            display: What to show a reader. Required unless ``scale`` is an interval, where it is
                derived from the number and unit.
            scale: The axis kind. Defaults to nominal, which is the only honest default for a
                value the caller said nothing about.
            keep_raw: Retain ``content`` on the model. Off by default: most callers are
                snapshotting, and a snapshot that keeps its inputs scales retention with what the
                candidate produced.

        Returns:
            The value.

        Raises:
            ValueError: When ``display`` is absent and the scale cannot derive one.
        """
        return cls._build(canonical_digest(content), display=display, scale=scale, raw=content if keep_raw else None)

    @classmethod
    def of_bytes(
        cls,
        payload: bytes,
        *,
        display: str | None = None,
        scale: Scale | None = None,
    ) -> SweepableValue:
        """Build a value by content-addressing an opaque blob.

        The case R9 names that :meth:`of` cannot serve: a host hands over bytes with no structure
        to canonicalise, and running them through JSON would address the *encoding* rather than
        the blob. No registry is consulted and none is needed — content-addressing is exactly the
        property that makes a value supplyable by a host whose store the engine cannot see.

        Args:
            payload: The raw bytes.
            display: What to show a reader. Defaults to the digest's short prefix, which is a poor
                label and an honest one — bytes carry no name.
            scale: The axis kind. Defaults to nominal.

        Returns:
            The value. ``raw`` is always absent: bytes are the case where retaining the content is
            most likely to be exactly what a snapshot must not do.
        """
        digest = bytes_digest(payload)
        return cls._build(digest, display=display or f"sha256:{digest[:12]}", scale=scale, raw=None)

    @classmethod
    def not_this_kind(cls, kind: str) -> SweepableValue:
        """The level a lever of ``kind`` sits at on a run of another kind — the lever does not apply to it.

        Every such run shares it, so the lever splits nothing among them, and it hashes apart from every
        value a field can hold, ``None`` included, so a pivot over the lever never joins a run of another
        kind with a run of ``kind`` whose field is ``None``. Build this level, rather than a value of your
        own displayed alike, wherever a host's reader answers for a run its lever does not apply to: it is
        what :attr:`not_of_kind` recognises, and a report never names an arm by a lever that does not apply
        to it.

        Args:
            kind: The kind whose lever this is, as a template's ``candidate_kind`` spells it.

        Returns:
            The level.
        """
        opening, closing = _NOT_THIS_KIND_DISPLAY
        return cls.of_bytes(_NOT_THIS_KIND + kind.encode(), display=f"{opening}{kind}{closing}")

    @property
    def not_of_kind(self) -> str | None:
        """The kind this level says its run is not, when it is :meth:`not_this_kind`'s level; else ``None``.

        Read off the content hash, which is the level's identity: the display only proposes the kind, and
        the hash must be the one :meth:`not_this_kind` mints for it. So a value of a host's own that merely
        reads ``(not a … run)`` is a value like any other, and is shown like one.

        Returns:
            The kind's name, or ``None``.
        """
        opening, closing = _NOT_THIS_KIND_DISPLAY
        if not (self.display.startswith(opening) and self.display.endswith(closing)):
            return None
        kind = self.display[len(opening) : -len(closing)]
        return kind if self.content_hash == bytes_digest(_NOT_THIS_KIND + kind.encode()) else None

    @classmethod
    def _build(cls, digest: str, *, display: str | None, scale: Scale | None, raw: Any | None) -> SweepableValue:
        """Assemble a value once its digest is known, deriving the display an interval implies.

        Args:
            digest: The content hash.
            display: The caller's label, or ``None``.
            scale: The caller's scale, or ``None`` for nominal.
            raw: The underlying value to retain, or ``None``.

        Returns:
            The value.

        Raises:
            ValueError: When no display was given and the scale cannot derive one.
        """
        resolved_scale: Scale = scale if scale is not None else NominalScale()
        if display is None:
            if not isinstance(resolved_scale, IntervalScale):
                raise ValueError(
                    "a nominal or ordinal SweepableValue needs a display: its content hash has no natural rendering, "
                    "and an analysis that falls back to one reads as 'component 3 moved'"
                )
            display = f"{resolved_scale.value:g}{resolved_scale.unit or ''}"
        return cls(content_hash=digest, display=display, scale=resolved_scale, raw=raw)


class ProductionFooting(BaseModel):
    """Which inputs one run held away from the subject's production configuration, read off the host's declarations.

    The disclosure a production-replicating cost travels with (#571). That cost sums what the
    production roles spent **under whatever this run set**, so it is what production would spend only
    for a run that set nothing: a cheaper swept model understates it, a candidate stripped of its
    learned state can overstate it, and the error has no reliable sign. A caveat printed on every run
    alike is one a reader stops reading, so this says which inputs THIS run moved.

    Built by :meth:`~threetears.evals.kernel.host.sweepables.SweepableRegistry.production_footing`
    from the host's declarations alone, so a lever a host declares reaches every cost surface without
    any of them being edited. Three buckets, never two: an input whose departure nothing can decide is
    ``unchecked``, not held, so "moved nothing" is claimed only when every input was checked.

    Crosses storage on a bundle's run summary; a summary assembled before it carries ``None``, which
    reads as "nobody checked", never as "nothing moved".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    moved: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Input name -> the level this run set it at, for every input it held away from the subject's own "
            "production configuration: a lever the launch set, or an apparatus input the host declares moves the "
            "candidate off its production footing (stripping its learned state, say)."
        ),
    )
    unchecked: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Input name -> the level this run carried, for every lever whose departure from production could not be "
            "decided: it carried a value and neither its declaration nor the run says whether that value is the "
            "subject's own. Not evidence that it moved, and not evidence that it held."
        ),
    )
    held: list[str] = Field(
        default_factory=list,
        description="Inputs checked and found at the subject's own production setting, sorted.",
    )

    @property
    def moved_nothing(self) -> bool:
        """True only when every input was checked and none moved — the one reading that needs no caveat."""
        return not self.moved and not self.unchecked

    def sentence(self) -> str:
        """The disclosure in words, for a surface that prints a production-replicating cost.

        Returns:
            One sentence naming what moved and what could not be checked, or saying nothing moved.
        """
        if self.moved_nothing:
            return (
                f"this run moved none of the {len(self.held)} input(s) checked off the subject's own production "
                "configuration, so its production-replicating cost was measured at that configuration"
            )
        parts: list[str] = []
        if self.moved:
            levels = ", ".join(f"{name}={level}" for name, level in sorted(self.moved.items()))
            parts.append(f"this run set {levels} in place of the subject's own production configuration")
        if self.unchecked:
            levels = ", ".join(f"{name}={level}" for name, level in sorted(self.unchecked.items()))
            parts.append(f"nothing records whether {levels} is the subject's own production setting")
        return (
            "; ".join(parts)
            + " — so its production-replicating cost is what was spent under these settings, not necessarily what "
            "production spends"
        )


class PooledProductionFooting(BaseModel):
    """The production footings of the runs a POOLED production-replicating cost was drawn from (#571).

    A frontier point's cost axis and an analysis arm's cost each mean one cost over several runs, and each run
    may have set different inputs away from the subject's production configuration — or one run may have been
    read and another not. So the pooled figure carries every run's own :class:`ProductionFooting`, keyed by run,
    and says where they disagree rather than merging them into one claim none of the runs made.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    runs: dict[str, ProductionFooting | None] = Field(
        default_factory=dict,
        description=(
            "Run id -> that run's production footing, for every run the pooled cost was drawn from. None where "
            "nobody checked the run (it was read with its host payload elided), which is never 'nothing moved'."
        ),
    )

    @property
    def moved_nothing(self) -> bool:
        """True only when every pooled run was checked and moved nothing."""
        return bool(self.runs) and all(footing is not None and footing.moved_nothing for footing in self.runs.values())

    def sentence(self) -> str:
        """The pooled disclosure in words, grouping runs that ran at one footing.

        Returns:
            One sentence per footing the pooled runs ran at, naming its runs where they differ, and the runs
            nobody checked.
        """
        groups: dict[str, tuple[ProductionFooting, list[str]]] = {}
        unread: list[str] = []
        for run_id, footing in sorted(self.runs.items()):
            if footing is None:
                unread.append(run_id)
                continue
            key = canonical_digest(footing.model_dump(mode="json"))
            groups.setdefault(key, (footing, []))[1].append(run_id)
        parts: list[str] = []
        if len(groups) == 1 and not unread:
            ((footing, run_ids),) = groups.values()
            parts.append(f"every one of the {len(run_ids)} run(s) pooled here: " + footing.sentence())
        else:
            if len(groups) > 1:
                parts.append(f"the runs pooled here ran at {len(groups)} different production footings")
            parts += [f"run(s) {', '.join(run_ids)}: {footing.sentence()}" for footing, run_ids in groups.values()]
            if unread:
                parts.append(f"nobody checked run(s) {', '.join(unread)}, so whether they moved anything is unknown")
        return "; ".join(parts)


__all__ = [
    "IntervalScale",
    "NominalScale",
    "OrdinalScale",
    "PooledProductionFooting",
    "ProductionFooting",
    "Scale",
    "SweepableValue",
]
