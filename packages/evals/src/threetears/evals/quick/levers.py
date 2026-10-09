"""Levers a ``run_eval`` run states beside its model: the second factor of a factorial comparison.

A run's model is one coordinate of its variant; a comparison over two prompts AND two models needs a
second. The engine's way to carry one is a kind contract's overlay model
(:class:`~threetears.evals.contracts.host.KindContract`): every field is a lever named ``<kind>.<field>``,
validated at launch, frozen onto the run, resolved into its variant key by the engine and declarable as a
campaign's axis. This module builds that model for the callable kinds from nothing but the levers' names,
so a caller states ``levers={"prompt": "v2"}`` and the run is a variant of its own on a declared axis.

Each level is a non-blank string naming the level — ``"v2"``, not the prompt's text — so the variant is
keyed by the name the caller gave it, exactly as the model coordinate is keyed by the model's id. Every
lever a host declares is required on every run in it: a run that left one out would sit at no level of it.
"""

from __future__ import annotations

import keyword
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, create_model

from threetears.evals.contracts.host import CANDIDATE_MODEL_LEVER


class CallableLevers(BaseModel):
    """The base of every overlay model :func:`levers_model` builds: one named level per lever.

    A host of the caller's own may declare a callable kind's overlays only with a model built here, since
    the callable kind turns nothing else; :func:`~threetears.evals.quick.run_eval` refuses any other.
    """

    model_config = ConfigDict(protected_namespaces=())


def refuse_unusable_lever_names(names: Iterable[str]) -> tuple[str, ...]:
    """The lever names, refusing one no overlay field could carry or that would read as the model.

    Args:
        names: The levers' names, as a caller spells them.

    Returns:
        The names, in the order given.

    Raises:
        ValueError: A name that is not an identifier, is a keyword, starts with ``_``, is ``model`` or one
            of a model's own attributes, or is given twice.
    """
    if isinstance(names, str):
        raise ValueError("levers are a collection of names, not one string")
    given = tuple(names)
    reserved = set(dir(BaseModel)) | {CANDIDATE_MODEL_LEVER}
    if unusable := [
        repr(name)
        for name in given
        if not isinstance(name, str)
        or not name.isidentifier()
        or keyword.iskeyword(name)
        or name.startswith("_")
        or name in reserved
    ]:
        raise ValueError(
            f"a lever's name is a Python identifier that does not start with '_' and is not 'model' (the arm's "
            f"model is its own coordinate), and {', '.join(unusable)} is not one"
        )
    if repeated := sorted({name for name in given if given.count(name) > 1}):
        raise ValueError(f"levers named {', '.join(repeated)} more than once; each name is one lever")
    return given


def levers_model(names: Iterable[str]) -> type[CallableLevers]:
    """The overlay model declaring one required, non-blank string level per lever name.

    Args:
        names: The levers' names.

    Returns:
        The model, for a callable kind's :class:`~threetears.evals.contracts.host.KindContract`.

    Raises:
        ValueError: As :func:`refuse_unusable_lever_names`.
    """
    fields: dict[str, object] = {
        name: (str, Field(pattern=r"\S", description=f"the {name} level the arm ran at, as its caller named it"))
        for name in refuse_unusable_lever_names(names)
    }
    model: type[CallableLevers] = create_model("CallableLevers", __base__=CallableLevers, **fields)  # type: ignore[call-overload]
    return model


__all__ = ["CallableLevers", "levers_model", "refuse_unusable_lever_names"]
