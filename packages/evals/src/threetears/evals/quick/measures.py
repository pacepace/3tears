"""A measure declared on the function that computes it: :func:`measure` and the :class:`Measure` it returns.

A host declares each measure it reads (:class:`~threetears.evals.contracts.MetricDescriptor`) in its profile, and
computes it somewhere else — in its kind, from what the candidate did. Two places to keep in step. ``@measure(...)``
puts the declaration on the function: the returned :class:`Measure` is that function, still callable, carrying its
descriptor, so a host registers ``field_accuracy.descriptor`` and grades with ``field_accuracy(...)``. A
:class:`Measure` is a scorer too: :func:`~threetears.evals.quick.run_eval`, :func:`~threetears.evals.quick.compare`
and :func:`~threetears.evals.quick.callable_host` register its own descriptor rather than writing one for it.

Nothing here is read by the engine: it builds a contract object and hands it over, and a host that writes its
descriptors by hand is unaffected.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import update_wrapper
from typing import Any

from threetears.evals.contracts import MetricDescriptor


class Measure:
    """A function that computes a measure, carrying the measure's declaration.

    Called exactly as the function it wraps. :attr:`descriptor` is what a host registers
    (``MeasureRegistry([...,  field_accuracy.descriptor])``); built once, by :func:`measure`, from the function's
    own name and docstring and what the decorator declares, so the declaration lives on the code that computes it.
    """

    def __init__(self, compute: Callable[..., Any], descriptor: MetricDescriptor) -> None:
        """Pair the function with its declaration.

        Args:
            compute: The function computing the measure.
            descriptor: Its declaration.
        """
        update_wrapper(self, compute)
        # Keyed by the measure's name wherever a scorer is: the kind lands each score under its scorer's name.
        self.__name__ = descriptor.name
        self.compute = compute
        self.descriptor = descriptor

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the measure."""
        return self.compute(*args, **kwargs)

    def __repr__(self) -> str:
        """The measure, by name."""
        return f"Measure({self.descriptor.name!r})"


def measure(
    *,
    name: str | None = None,
    reader_name: str | None = None,
    description: str | None = None,
    family: str = "mechanical",
    transferability_class: str = "mechanical",
    attribution_scope: str = "end_to_end",
    data_type: str = "numeric",
    **declared: Any,
) -> Callable[[Callable[..., Any]], Measure]:
    """Declare the measure a function computes, on the function: ``@measure(higher_is_better=True, ...)``.

    The descriptor is built when the function is decorated, so a declaration the engine refuses (a guardrail on a
    merit axis, a range below zero on a nonnegative measure) is refused at import, beside the code it describes.

    Args:
        name: The measure's key on results; the function's ``__name__`` when ``None``.
        reader_name: What a reader calls it; the name in words, marked a score, when ``None``.
        description: One sentence an operator reads; the first line of the function's docstring when ``None``.
        family: Its measure family: ``mechanical`` (the default), another engine family, or one the host declares
            (:class:`~threetears.evals.contracts.MeasureFamily`).
        transferability_class: How far a reading of it carries; ``mechanical`` by default.
        attribution_scope: What it is a reading of; ``end_to_end`` by default.
        data_type: ``numeric`` by default.
        **declared: Every other :class:`~threetears.evals.contracts.MetricDescriptor` field by name:
            ``higher_is_better``, ``merit_axis``, ``value_range``, ``materiality_threshold``, ``unit``,
            ``population``, ``diagnostic``, ``guardrail``, ``reader_prose`` and the rest.

    Returns:
        The decorator, which returns the :class:`Measure`.

    Raises:
        ValueError: The function has no name to key the measure by, no docstring and no ``description``, or a
            declaration the descriptor refuses.
    """

    def declare(compute: Callable[..., Any]) -> Measure:
        key = name or getattr(compute, "__name__", None)
        if not key or key == "<lambda>":
            raise ValueError(f"{compute!r} has no name to key its measure by; pass name=")
        doc = inspect.getdoc(compute)
        said = description or (doc.splitlines()[0] if doc else None)
        if not said:
            raise ValueError(f"the measure {key} needs a description: give its function a docstring, or description=")
        words = " ".join(key.replace("_", " ").split())
        descriptor = MetricDescriptor.model_validate(
            {
                "name": key,
                "reader_name": reader_name or f"{words[:1].upper()}{words[1:]} score",
                "description": said,
                "family": family,
                "transferability_class": transferability_class,
                "attribution_scope": attribution_scope,
                "data_type": data_type,
                **declared,
            }
        )
        return Measure(compute, descriptor)

    return declare


__all__ = ["Measure", "measure"]
