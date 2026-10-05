"""The seed walk — what a template's world seed may write, checked against the registry before anything is.

Every host that seeds through a :class:`~threetears.evals.contracts.host.world.WorldRegistry` asks the
same five questions of each value a seed names, and none of them needs anything but the registry
and the seed:

* does the subject attach the carrier the seed names it under;
* does the host declare the dimension at all;
* is the value under the carrier that supplies it — a write through the wrong carrier reaches nothing;
* can a run set it — a witnessed dimension has no seed handle, and a run claiming to have set one
  would record an instantiation that never happened;
* does the value conform to the dimension's schema — presence and type, never plausibility: a seed
  supplies every field the subject meets outside an eval, and whether a value is *sensible* is the
  template author's to answer for.

**One walk, so a seeding rule added here reaches every host.** These rules used to live in one
host's adapter and be copied by hand into the reference toy host, so a rule added to one copy —
schema conformance was the instance — had to be added to the other separately, and a host copied
from the toy would never have received the next one. A host keeps what is genuinely its own: how a
seed key addresses a dimension (its registry's ``address``), and what a refusal costs — the error
type, the termination, and the wording its operators read. That is why the walk raises one structured
:class:`SeedRefused` rather than a host's exception: the host translates it at its call site.

**There is no pass-over.** The walk once took namespaces and keys to skip — state a host's run wrote
from its own record rather than a template, the instance being a call ledger kept inside the world.
The ledger now lives beside the world (``CallLedger``), so nothing a run writes shares a namespace with
what a template seeds, and a skipped key was only ever a seed value the walk let through unchecked.
State a run writes itself is a dimension with no ``seed`` handle, refused here as ``unseedable`` like
any other.

**Every write is checked before any is made.** :func:`check_seed` returns the writes and makes
none, so a refusal leaves the world untouched rather than half-seeded; the caller applies exactly
what was checked, through the seed handle the walk names, which is the path the conformance kit
proves.

**The registry's answers come before the subject's.** Whether a carrier is attached is asked only
once every path has passed the registry, so a seed naming a dimension no host declares is refused
as that — a fix to the template — rather than as a carrier this subject happens not to hold.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Literal, NamedTuple

from threetears.evals.contracts.host.world import WorldDimension, WorldRegistry
from threetears.evals.contracts.host.world_schema import schema_violations

#: Why a seed was refused, one value per question the walk asks.
#:
#: ``malformed``      a namespace's value is not a mapping of seed key to value.
#: ``undeclared``     no dimension of this host's world is named by the path.
#: ``misplaced``      the dimension is declared, under a different carrier than the one the seed names.
#: ``unseedable``     the dimension is declared and has no seed handle — no run can set it.
#: ``nonconforming``  the value fails the dimension's schema.
#: ``unattached``     the seed is sound, and the subject does not attach the carrier it names.
SeedRefusalKind = Literal["malformed", "undeclared", "misplaced", "unseedable", "nonconforming", "unattached"]


class SeedWrite(NamedTuple):
    """One write a checked seed makes: through ``handle``, of ``value``, into dimension ``name``."""

    #: The seed namespace the value was written under — the carrier it reaches the subject through.
    namespace: str
    #: The dimension the value sets, as the registry declares it.
    name: str
    #: The dimension's seed handle, which is the only path a write takes.
    handle: str
    #: The value, exactly as the seed holds it.
    value: Any


class SeedRefused(ValueError):
    """A seed asked for a write the registry or the subject cannot make.

    A ``ValueError`` so a caller that only needs "refused, and why" can catch it as one, and
    structured so a host that words refusals for its own operators can: :attr:`kind` says which
    question failed, :attr:`name` the dimension, and :attr:`violations` the schema's sentences. The
    default message is host-neutral and names the path, which is enough for a host that has no
    wording of its own.

    Attributes:
        kind: Which question failed.
        namespace: The seed namespace the refused value sits under.
        name: The dimension the seed addressed, or None when the namespace itself is refused.
        declared: The registry's declaration for ``name``, or None when it has none.
        violations: The schema's sentences, each naming the path it fails at; empty unless ``nonconforming``.
        attached: The carriers the subject attaches, sorted; empty unless ``unattached``.
    """

    def __init__(
        self,
        kind: SeedRefusalKind,
        *,
        namespace: str,
        name: str | None = None,
        declared: WorldDimension | None = None,
        violations: tuple[str, ...] = (),
        attached: tuple[str, ...] = (),
    ) -> None:
        """Record the refusal and compose its default sentence.

        Args:
            kind: Which question failed.
            namespace: The seed namespace the refused value sits under.
            name: The dimension the seed addressed, or None when the namespace itself is refused.
            declared: The registry's declaration for ``name``, when it has one.
            violations: The schema's sentences, for a ``nonconforming`` refusal.
            attached: The carriers the subject does attach, for an ``unattached`` refusal.
        """
        self.kind = kind
        self.namespace = namespace
        self.name = name
        self.declared = declared
        self.violations = violations
        self.attached = attached
        super().__init__(self._sentence())

    def _sentence(self) -> str:
        """The host-neutral refusal, naming the path and what would have to change.

        Returns:
            The message.
        """
        if self.kind == "malformed":
            return f"the seed's {self.namespace!r} entry is not a mapping of dimension to value"
        if self.kind == "undeclared":
            return f"the seed names world dimension {self.name!r}, which this host does not declare"
        if self.kind == "misplaced":
            carrier = self.declared.carrier if self.declared is not None else None
            return (
                f"the seed puts {self.name!r} on carrier {self.namespace!r}, but it is supplied by {carrier!r} — "
                "a write through the wrong carrier reaches nothing"
            )
        if self.kind == "unseedable":
            return (
                f"the seed sets {self.name!r}, which no run controls — it is witnessed, and a run that claimed to "
                "set it would be recording an instantiation that never happened"
            )
        if self.kind == "nonconforming":
            return (
                f"the seed sets {self.name!r} to a value its schema refuses — a seed supplies every field the "
                f"subject sees outside an eval: {'; '.join(self.violations)}"
            )
        return (
            f"the seed names carrier {self.namespace!r}, which this subject does not attach "
            f"(attached: {', '.join(self.attached) or 'none'}) — nothing seeded through it would reach the subject"
        )


def check_seed(
    registry: WorldRegistry,
    namespaces: Mapping[str, Any],
    *,
    attached: Collection[str] | None = None,
) -> tuple[SeedWrite, ...]:
    """Walk a world seed against the registry: every write it would make, checked, or the first refusal.

    Args:
        registry: The world the seed writes into.
        namespaces: Carrier → (seed key → value), as a template's world seed holds them.
        attached: The carriers this subject attaches, or None where no subject exists yet — at
            authoring, the registry's questions are the only ones there are to ask.

    Returns:
        One :class:`SeedWrite` per seeded value, in the seed's order. Nothing has been written.

    Raises:
        SeedRefused: The first value the walk cannot write, naming its path. Every registry refusal
            is found before any ``unattached`` one.
        UnsupportedSchemaError: A dimension's schema is outside the honoured subset — a defect in the
            registration, not in the seed, so it is not translated into a refusal of the seed.
    """
    writes: list[SeedWrite] = []
    for namespace, seeded in namespaces.items():
        if not isinstance(seeded, Mapping):
            raise SeedRefused("malformed", namespace=namespace)
        for key, value in seeded.items():
            name = registry.address(namespace, key)
            declared = registry.get(name)
            if declared is None:
                raise SeedRefused("undeclared", namespace=namespace, name=name)
            if declared.carrier != namespace:
                raise SeedRefused("misplaced", namespace=namespace, name=name, declared=declared)
            if declared.seed is None:
                raise SeedRefused("unseedable", namespace=namespace, name=name, declared=declared)
            if violations := schema_violations(declared.schema, value, at=name):
                raise SeedRefused(
                    "nonconforming", namespace=namespace, name=name, declared=declared, violations=tuple(violations)
                )
            writes.append(SeedWrite(namespace, name, declared.seed, value))
    if attached is not None:
        for write in writes:
            if write.namespace not in attached:
                raise SeedRefused(
                    "unattached", namespace=write.namespace, name=write.name, attached=tuple(sorted(attached))
                )
    return tuple(writes)


__all__ = ["SeedRefusalKind", "SeedRefused", "SeedWrite", "check_seed"]
