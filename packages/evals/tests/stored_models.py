"""The stored eval models, derived from the model graph rather than listed.

A stored model is any :class:`~threetears.evals.contracts.base.EvalBaseModel` subclass defined in the
package that declares a ``doc_type`` — the field a document store discriminates on. Deriving the set
is the point: a list kept by hand fails by omission, and a model added without a row would then be
exempt from every rule the set is used to enforce, with nothing going red.
"""

from __future__ import annotations

import importlib
import pkgutil

import threetears.evals
from threetears.evals.contracts.base import EvalBaseModel

__all__ = ["doc_type_of", "stored_models"]


def stored_models() -> list[type[EvalBaseModel]]:
    """Every package model that declares a ``doc_type``, after importing all of ``threetears.evals``.

    Returns:
        The models, sorted by name so a parametrised test has stable ids.
    """
    for module in pkgutil.walk_packages(threetears.evals.__path__, prefix="threetears.evals."):
        importlib.import_module(module.name)
    seen: set[type[EvalBaseModel]] = set()
    stack: list[type[EvalBaseModel]] = [EvalBaseModel]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return sorted(
        (
            model
            for model in seen
            if model.__module__.startswith("threetears.evals.") and "doc_type" in model.model_fields
        ),
        key=lambda model: model.__name__,
    )


def doc_type_of(model: type[EvalBaseModel]) -> str:
    """The ``doc_type`` value a stored model writes.

    Args:
        model: A stored model.

    Returns:
        Its ``doc_type`` default — the discriminator every document of it carries.
    """
    default = model.model_fields["doc_type"].default
    assert isinstance(default, str), f"{model.__name__}.doc_type has no string default"
    return default
