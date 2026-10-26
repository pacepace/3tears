"""The analyst's analyses: computations over the frozen report input that no single figure states.

The actions of the ``Investigate: analyses`` workflow, each an
:class:`~threetears.evals.actions.catalogue.Action` over its operation in :mod:`threetears.evals.ops` and rendered by
:func:`~threetears.evals.actions.render.render_cited`. :func:`~threetears.evals.actions.analyst.analyst_actions`
gathers them onto the analyst's cut, and the analyst's and editor's toolsets are read off this workflow, so no file
lists their names.
"""

from __future__ import annotations

from threetears.evals.actions.catalogue import Action

#: The help index's group for these actions, which the toolsets are read off.
ANALYSES_WORKFLOW = "Investigate: analyses"


def analyst_analyses_actions() -> tuple[Action, ...]:
    """The ``Investigate: analyses`` actions; none are built yet.

    :return: The actions, in the order help lists them.
    :rtype: tuple[Action, ...]
    """
    return ()


__all__ = ["ANALYSES_WORKFLOW", "analyst_analyses_actions"]
