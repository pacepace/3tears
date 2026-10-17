"""What a router's catalogue says of a model's request settings: only what no endpoint of it takes.

OpenRouter's models endpoint (``GET /api/v1/models``) lists, per model, the ``supported_parameters`` its endpoints
take. The list is a union: a parameter is listed when SOME endpoint serving the model takes it, never that ONE
endpoint takes every listed parameter together, so it cannot say a request carrying several of them will be placed
under ``require_parameters``, and it cannot rule a temperature out of a 404 "no endpoint" answer. What it can say is
the opposite: a parameter it does not list, no endpoint takes. :func:`facts_from_router_catalogue` reads exactly that,
as the facts :func:`~threetears.evals.negotiation.closest_settings` uses to not send such a parameter at all.
"""

from __future__ import annotations

from collections.abc import Collection

from threetears.evals.negotiation.settings import SettingFacts

__all__ = ["facts_from_router_catalogue"]


def facts_from_router_catalogue(supported_parameters: Collection[str] | None) -> SettingFacts:
    """The facts a router's catalogue row states of a model: each parameter it does not list, as not taken.

    Only the negative facts: a listed parameter is left unknown (sent as asked, for a refusal to teach otherwise),
    because a parameter SOME endpoint takes may still be one no endpoint takes beside the request's others. A
    temperature not listed is not sent; ``reasoning`` not listed, no effort word and no budget; ``structured_outputs``
    not listed, no strict schema (the request then goes unenforced). A caller passes these as ``declared`` to
    :func:`~threetears.evals.negotiation.negotiate`, under its own registry's facts (:meth:`SettingFacts.over`).

    Read the catalogue once, at a point the caller chooses (the shipped client reads it once per ``Catalogue``, at boot
    or at its first call), and keep it: facts that change mid-run because another client happened to fetch the catalogue
    make two calls of one run sent differently for no reason the run records.

    :param supported_parameters: The model's ``supported_parameters`` as the catalogue lists them, or ``None`` when the
        catalogue was not read, does not list the model, or lists it without them.
    :ptype supported_parameters: Collection[str] | None
    :return: The facts: every one unknown for ``None``.
    :rtype: SettingFacts
    """
    if supported_parameters is None:
        return SettingFacts()
    listed = frozenset(supported_parameters)
    return SettingFacts(
        accepts_temperature=None if "temperature" in listed else False,
        reasoning_efforts=None if "reasoning" in listed else (),
        reasoning_budget=None if "reasoning" in listed else False,
        structured_output=None if "structured_outputs" in listed else False,
    )
