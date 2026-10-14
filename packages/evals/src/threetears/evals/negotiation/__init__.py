"""Request negotiation: a client that never fails a request because a model lacks a setting.

An eval asks each role's client for request settings (an output cap, a temperature, a reasoning word or budget, an
enforced reply format), and the model a client calls may not take all of them: a model that takes no temperature, an
effort word it lacks, a budget where it takes words only, a router that finds no endpoint under
``require_parameters``. None of that may fail the request (owner ruling, 2026-10-14: "we can't have an eval system
that forbids models because of their temperature/etc capabilities"). This root is the one implementation of what a
client does instead, used by the shipped adapter (:mod:`threetears.evals.providers`) and by a host's own client
alike, so the next "this model can't be a judge" bug is fixed once:

* :func:`closest_settings` chooses what a request carries for what was asked, from what is known of the model
  (:class:`SettingFacts`), as :class:`WireSettings`; pure, and table-tested per family (:data:`ProviderFamily`).
* :func:`negotiate` sends through the caller's own ``send``, reads a failure through the caller's ``read_refusal``,
  and on a :class:`SettingRefusal` of a setting the request carried learns it (:class:`LearnedSettings`, shared as the
  caller chooses) and resends in a fixed order: the refused setting, a temperature first and keeping enforcement,
  and only then enforcement. It returns :class:`Negotiated`, whose ``sent_settings`` and ``temperature`` are what
  the served request carried, for the caller's completion.
* :func:`read_router_refusal`, :func:`read_anthropic_refusal` and :func:`read_openai_refusal` read a refusal off
  the status and JSON body each wire answers with, so a client of raw HTTP and an SDK wrapper read the same.

It imports nothing but the package's schema: no SDK, no HTTP client, no LangChain, so it is on the base install.
``docs/adopting-a-host.md`` ("A client of your own that never fails a request") is a worked client of a router
over raw HTTP, and :func:`~threetears.evals.testing.check_refusal_negotiation` the check a host runs over its own.

**This module is the package's public root** for negotiation. A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.negotiation.loop import LearnedSettings, Negotiated, negotiate
from threetears.evals.negotiation.readers import read_anthropic_refusal, read_openai_refusal, read_router_refusal
from threetears.evals.negotiation.refusals import RefusedSetting, SettingRefusal
from threetears.evals.negotiation.settings import (
    ANTHROPIC_MIN_THINKING_BUDGET,
    BUDGET_SHARES,
    EFFORT_LADDER,
    ProviderFamily,
    SettingFacts,
    WireSettings,
    anthropic_thinking,
    closest_settings,
)

__all__ = [
    "ANTHROPIC_MIN_THINKING_BUDGET",
    "BUDGET_SHARES",
    "EFFORT_LADDER",
    "LearnedSettings",
    "Negotiated",
    "ProviderFamily",
    "RefusedSetting",
    "SettingFacts",
    "SettingRefusal",
    "WireSettings",
    "anthropic_thinking",
    "closest_settings",
    "negotiate",
    "read_anthropic_refusal",
    "read_openai_refusal",
    "read_router_refusal",
]
