"""What the toy host resolved for each of its levers, for one run.

The sweepables registry projects each input for the bisection to compare; the variant key needs
the typed levels it is digested from, every lever at once.
:data:`~threetears.evals.contracts.host.profile.VariantLeverReader` is that second question, and this
module is the toy host's answer to it.

**Why a host writes one.** The variant key is what pools observations into cells, and a lever the
key does not cover is one two observations can differ on and still pool. A host that declares
levers of its own resolves their levels here, out of what it stamped on the run at launch.

**What is not here is the engine's.** The extractor model (the shared core's ``model`` lever), the
candidate kind, and every field of the kind's overlay model are resolved by the engine for every run,
from the run itself and from the kind's contract on the profile — a reader returning one of them is
refused. This reader resolves only the levers the toy host declares beyond those.

**Two scale kinds, deliberately.** ``chunk_tokens`` and ``retriever_top_k`` resolve as
:class:`~threetears.evals.contracts.host.values.IntervalScale` levels, so the toy host is the fixture where a
level carries true spacing. ``extraction_schema`` resolves through
:meth:`~threetears.evals.contracts.host.values.SweepableValue.of_bytes`, the opaque-blob path: a
level known only by its content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from threetears.evals.contracts.host import IntervalScale, SweepableValue
from packages.evals.tests.fixtures.toyhost.sweepables import RESOLVED_RETRIEVAL_CONFIG

if TYPE_CHECKING:
    from threetears.evals.contracts import EvalRun

#: Where the toy host keeps its vocabulary on the shared carrier — the same engine-owned slot
#: ``packages.evals.tests.fixtures.toyhost.sweepables`` reads, named once here rather than imported, so the two
#: readers stay independently checkable.
_TOYHOST_NAMESPACE = "toyhost"

#: The level an observation that never recorded a field schema sits at.
#:
#: An explicit level rather than an absence: "this observation recorded no schema" is a real state
#: that several observations share, and dropping the coordinate would merge them with observations
#: that recorded one.
NO_EXTRACTION_SCHEMA = "(no field schema recorded)"


def _payload(run: EvalRun) -> dict[str, Any]:
    """The toy host's own keys for this observation's batch."""
    payload: dict[str, Any] = (run.host_payload or {}).get(_TOYHOST_NAMESPACE, {})
    return payload


def _interval(value: Any, *, unit: str | None, absent: str) -> SweepableValue:
    """Content-address a numeric level, keeping its spacing where it has one.

    Args:
        value: The recorded number, or ``None`` when the observation never recorded it.
        unit: The unit the number is in, used to render the level. ``None`` for a dimensionless
            count — an empty string there would be a unit that renders as nothing, which is a
            different claim from having none.
        absent: What to call the level an observation that recorded nothing sits at.

    Returns:
        The level. An unrecorded value is nominal, because there is no number to space it by.
    """
    if value is None:
        return SweepableValue.of(None, display=absent)
    return SweepableValue.of(value, scale=IntervalScale(value=float(value), unit=unit))


def variant_levers(run: EvalRun) -> dict[str, SweepableValue]:
    """Resolve a toy-host run's level of every lever the toy host itself registers.

    Args:
        run: The run whose observations the levels describe.

    Returns:
        Lever name → the level carried, covering every fixed ``lever`` in the toy host's own
        registry. Every value is content-addressed, so nothing here needs a store the engine cannot see.
    """
    payload = _payload(run)
    schema = payload.get("extraction_schema")
    prompt = run.subject_snapshot.components.get("extraction_prompt")
    return {
        # The subject's component, as itself — the only route a component has into the key.
        "extraction_prompt": prompt
        if prompt is not None
        else SweepableValue.of(None, display="(no extraction prompt)"),
        "chunk_tokens": _interval(payload.get("chunk_tokens"), unit="tok", absent="(no chunk size recorded)"),
        "retriever_top_k": _interval(payload.get("retriever_top_k"), unit=None, absent="(no retriever width recorded)"),
        "extraction_schema": (
            SweepableValue.of_bytes(str(schema).encode(), display=f"field schema {schema}")
            if schema is not None
            else SweepableValue.of(None, display=NO_EXTRACTION_SCHEMA)
        ),
    }


def tunable_variant_levers(run: EvalRun) -> dict[str, SweepableValue]:
    """:func:`variant_levers`, plus the resolved retrieval configuration's coordinate.

    The reader for the profile that registers retrieval tuning. The family's members carry no
    coordinate of their own — an open family has no per-member reader — so everything an overlaid
    knob does to identity arrives through the resolved configuration it was merged into.

    Args:
        run: The run whose observations the levels describe.

    Returns:
        Lever name → level, covering every fixed lever the tunable registry declares.
    """
    config = _payload(run).get("retrieval_config")
    return {
        **variant_levers(run),
        RESOLVED_RETRIEVAL_CONFIG: SweepableValue.of(
            config, display="retrieval config" if config is not None else "(no retrieval config recorded)"
        ),
    }


__all__ = ["NO_EXTRACTION_SCHEMA", "tunable_variant_levers", "variant_levers"]
