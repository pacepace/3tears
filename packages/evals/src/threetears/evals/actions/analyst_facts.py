"""The analyst's fact tools: what the frozen report input holds, each figure by its key.

The actions of the ``Investigate: facts`` workflow, each an
:class:`~threetears.evals.actions.catalogue.Action` over its operation in :mod:`threetears.evals.ops` and rendered by
:func:`~threetears.evals.actions.render.render_cited`. :func:`~threetears.evals.actions.analyst.analyst_actions`
gathers them onto the analyst's cut, and the analyst's and editor's toolsets are read off this workflow, so no file
lists their names.
"""

from __future__ import annotations

from threetears.evals.actions.catalogue import Action

#: The help index's group for these actions, which the toolsets are read off.
FACTS_WORKFLOW = "Investigate: facts"


def analyst_facts_actions() -> tuple[Action, ...]:
    """The ``Investigate: facts`` actions; none are built yet.

    :return: The actions, in the order help lists them.
    :rtype: tuple[Action, ...]
    """
    return ()


__all__ = ["FACTS_WORKFLOW", "analyst_facts_actions"]
