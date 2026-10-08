"""One level of one swept input — the half of the vocabulary that crosses storage.

Its sibling :mod:`~threetears.evals.contracts.host.sweepables` declares what an input IS: a runtime
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

from threetears.evals.contracts.hashing import bytes_digest, canonical_digest


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


__all__ = ["IntervalScale", "NominalScale", "OrdinalScale", "Scale", "SweepableValue"]
