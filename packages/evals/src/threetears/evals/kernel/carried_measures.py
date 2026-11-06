"""The measure names one result carries a value under, read off the result model by reflection.

The per-result sets the declaration gate checks a bar against
(:func:`~threetears.evals.kernel.declaration.resolve_bar_name`) and a host profile checks a lever's mechanism
against (:meth:`~threetears.evals.kernel.host.profile.HostProfile`'s registration). Defined below both, since
:mod:`~threetears.evals.kernel.host` is imported by the declaration module and needs these names itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from typing import TYPE_CHECKING, get_args, get_origin

from pydantic import BaseModel

from threetears.evals.kernel.covariates import COVARIATE_KEYS
from threetears.evals.kernel.metrics import DERIVED_PER_RESULT_MEASURES, METRIC_DESCRIPTORS
from threetears.evals.schema.models import EvalResult

if TYPE_CHECKING:
    from threetears.evals.kernel.host.measures import MeasureRegistry

__all__ = ["mechanism_measure_names", "per_result_measure_names"]


#: The covariate keys a result can carry — every key ``derive_covariates`` writes. A covariate is a
#: per-result observation keyed by a described name, so a bar may name one.
_COVARIATE_MEASURES: frozenset[str] = COVARIATE_KEYS


def _carriers_of(annotation: object) -> Iterator[type[BaseModel]]:
    """Yield every pydantic model inside a field annotation, through ``Optional`` and collections."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
    for argument in get_args(annotation):
        yield from _carriers_of(argument)


@cache
def _carried_field_names() -> frozenset[str]:
    """Every name a result's own fields can carry a measure under — its scalars and its carriers' fields.

    The type-level mirror of the bundle's measure walk, which reads a result's top-level scalars and
    the scalar leaves of every sub-model it holds, single or listed. Derived from ``EvalResult`` by
    reflection so a carrier the result gains is counted with no edit here. Cached because the
    answer is a property of the model class and never of a run.
    """
    names: set[str] = set()
    for field_name, field in EvalResult.model_fields.items():
        carriers = list(_carriers_of(field.annotation))
        if not carriers:
            names.add(field_name)
        for carrier in carriers:
            names.update(carrier.model_fields)
    return frozenset(names)


def _host_measure_names(measures: MeasureRegistry) -> frozenset[str]:
    """The names only the host declares — each one a key its kinds may land on ``host_measures``.

    The one open door of the per-result set, and the gate is honest about what it cannot see
    through it: a host may declare a run-level aggregate beside its per-cell measures, and nothing
    in a descriptor tells the two apart, so every name the host declares is admitted. A bar on one
    its kind never emits is admitted here and adjudicated as carried by no result.

    :param measures: The host's measure registry.
    :ptype measures: MeasureRegistry
    """
    names = measures.names
    # A host catalogue may re-declare the engine's core, and the core wins that tie wherever a name
    # is described. A re-declared core name is carried where the core carries it or nowhere, so it
    # opens no door here.
    return frozenset(names) - METRIC_DESCRIPTORS.keys()


def per_result_measure_names(measures: MeasureRegistry) -> frozenset[str]:
    """Every described name a single result can carry a code-graded value under.

    Four sources, which are exactly the four the bundle's measure walk reads: the result's own
    fields, the measures it implies, its covariates, and the host's own measures. A core
    measure outside them — a run-summary statistic such as ``mean_total_ms`` or ``p95_total_ms``,
    a report-level accuracy — is described and never carried by any one result, so a bar on it is
    never read.

    :param measures: The host's measure registry.
    :ptype measures: MeasureRegistry
    """
    return _carried_field_names() | DERIVED_PER_RESULT_MEASURES | _COVARIATE_MEASURES | _host_measure_names(measures)


def _listed(annotation: object) -> bool:
    """Whether a field annotation holds MANY carriers (one observation per element), through ``Optional``.

    A map counts as a list: ``judge_cut_short`` holds one record per rubric dimension, so a result carries
    as many ``reasoning_tokens`` under it as it has cut-short dimensions, and no one of them is the result's.
    """
    if get_origin(annotation) in (list, tuple, dict):
        return True
    return any(_listed(argument) for argument in get_args(annotation) if argument is not type(None))


@cache
def _single_carried_field_names() -> frozenset[str]:
    """Every name a result's own fields carry ONCE — its scalars and the fields of its single sub-models.

    :func:`_carried_field_names` without the listed carriers: a usage row per role or a delivery per
    async call carries its fields once per element, so a result holds several values under such a name
    and no one of them is the result's. Derived by reflection, cached, for the same reasons.
    """
    names: set[str] = set()
    for field_name, field in EvalResult.model_fields.items():
        carriers = list(_carriers_of(field.annotation))
        if not carriers:
            names.add(field_name)
        elif not _listed(field.annotation):
            for carrier in carriers:
                names.update(carrier.model_fields)
    return frozenset(names)


def mechanism_measure_names(measures: MeasureRegistry) -> frozenset[str]:
    """Every described name one result carries a single value under — what a lever may declare it acts on.

    The per-result set :func:`per_result_measure_names` reads, narrowed to the values a result holds
    once: its own scalars and single sub-models, the measures it implies, its covariates and the host's
    own measures. A field of a listed carrier is left out — ``reasoning_tokens`` is recorded per usage
    row, one per role — because comparing a lever's levels needs one value per result, and pooling one
    role's count with another's describes no mechanism.

    :param measures: The host's measure registry.
    :ptype measures: MeasureRegistry
    """
    return (
        _single_carried_field_names()
        | DERIVED_PER_RESULT_MEASURES
        | _COVARIATE_MEASURES
        | _host_measure_names(measures)
    )
