"""Who was measured, in a shape that names no host concept.

An eval run has a **subject** — the thing under test whose identity decides what may be pooled
with what. One host's subject is a conversational agent; another's is a retrieval configuration,
a scrape recipe, or a model release. The engine once read the subject through a host-shaped
snapshot field, a required field carrying the first host's prose fields and catalogue, with a
live round-trip contract against that host's running component. Every consumer that wanted
"which subject was this" reached through that host's concept to get it.

Two rules shape what replaced it.

**The snapshot holds identity, never content.** :attr:`SubjectSnapshot.components` maps a
host-registered name to a :class:`~threetears.evals.contracts.host.sweepables.SweepableValue`, which carries a
content *hash*. A snapshot that stored the component text instead would grow with whatever the
candidate produced — carried memory, accumulated notes — and an eval run's retention must be
bounded by its matrix, never by what it observed. So the subject's prose is addressable from here
and not readable from here, and that is the point rather than a limitation.

**Two maps, because a subject has two halves and they belong to different keys.**
:attr:`~SubjectSnapshot.components` is what the subject **is** — what a campaign sweeps, and what
the variant key is computed over. :attr:`~SubjectSnapshot.state` is what the subject **carried
in**, which no campaign chose, and it is what the measurement-context key is computed over. They
are deliberately symmetric — host-named, content-addressed, never interpreted here, holding hashes
rather than content — and the split is the one every host already draws: a run that hashed
accumulated state into the variant would mint a new variant every time the subject remembered
something.

**The key and the label are separate, and both are required.** A host may legitimately pass the
same string into both — plenty of subjects are identified by their name. What it may not do is
pass *only* a name, because then a rename silently splits a population and nothing in the data
shows it happened. With two fields the mistake is visible as ``subject_id == subject_label``, and
the engine holds the whole population, so it can say when one key appears under two labels.

The rich object a host still needs — a host's runner may instantiate a live subject from it, and
its judge render it into the scoring prompt — does not live here and cannot. It rides on
``EvalRun.host_payload``, which the engine never reads.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from threetears.evals.contracts.host.sweepables import SweepableValue


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        The timestamp.
    """
    return datetime.now(UTC).isoformat()


class SubjectSnapshot(BaseModel):
    """The subject a set of observations was taken against, frozen at capture.

    Crosses storage, so every field is a persisted-format commitment.
    """

    # Strict like every stored eval type: a field this build does not declare is refused on read as
    # on construction. Frozen, so it declares its own config rather than inheriting the mutable base.
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    subject_id: str = Field(
        min_length=1,
        description=(
            "The pooling boundary. Required and non-empty: a blank subject key is the one coordinate that cannot be "
            "defaulted, because calling two unknowns 'the same subject' pools measurements that are not comparable. "
            "It is unrepresentable here rather than handled downstream."
        ),
    )
    subject_label: str = Field(
        min_length=1,
        description=(
            "What a reader is shown. Separate from the key even when a host has only one string for both — passing "
            "the name into both is allowed and leaves the tell in the data, whereas a single field would hide it."
        ),
    )
    components: dict[str, SweepableValue] = Field(
        default_factory=dict,
        description=(
            "Host-registered name → the level this subject carried, content-addressed. What the variant key is "
            "computed over. Holds hashes, never component text."
        ),
    )
    state: dict[str, SweepableValue] | None = Field(
        description=(
            "Host-registered name → the state this subject CARRIED IN, content-addressed. What the "
            "measurement-context key is computed over, and the symmetric sibling of `components`. Holds "
            "hashes, never the state itself. `{}` is a RECORDING — this subject carries nothing outside "
            "its variant components — while None is an ABSENCE: a host that wired no state reader. "
            "Required and nullable, with no default, "
            "because reading the recording as the absence pools a subject carrying a year of "
            "accumulated state with a stripped probe, and a default would let a writer skip the "
            "distinction rather than state it."
        ),
    )
    labels: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Host-registered name → a string that identifies without determining. Reportable in a diff, never part "
            "of any key, and never a confound."
        ),
    )
    captured_at: str = Field(default_factory=_now_iso, description="When this snapshot was taken, ISO-8601 UTC.")

    @field_validator("subject_id", "subject_label")
    @classmethod
    def _refuse_a_blank_identity(cls, value: str) -> str:
        """Refuse a key or label that is only whitespace.

        ``min_length`` alone admits ``"   "``, which is blank by every reading a reader or a pooling
        boundary makes, so the "unrepresentable here" promise above held only for hosts that stripped
        before constructing. Refused rather than stripped: a blank key is the coordinate that cannot
        be defaulted.

        Args:
            value: The submitted key or label.

        Returns:
            The value, unchanged.

        Raises:
            ValueError: The value is empty after stripping whitespace.
        """
        if not value.strip():
            raise ValueError("must not be blank or whitespace only")
        return value

    def component_hashes(self) -> dict[str, str]:
        """Return the component map reduced to its identities.

        A convenience for a caller that wants the identities and nothing else, so it is not
        tempted to reach past a component's hash to something else on it.

        **Not the variant key's input, and the difference matters.** The key is computed over the
        host's full per-observation lever map, of which a subject's components are one part; it
        takes ``SweepableValue`` objects and reduces them itself. A docstring claiming this method
        is "the one place the key predicate reads" would assert a property wider than the
        mechanism behind it — which is the shape ``.claude/rules/learnings/`` names — and would send a
        maintainer here to change what the key hashes.

        Returns:
            Component name → content hash, unordered.
        """
        return {name: value.content_hash for name, value in self.components.items()}


__all__ = ["SubjectSnapshot"]
