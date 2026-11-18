"""The engine's own actions, read off the catalogue a host mounts.

A test asks the catalogue (:func:`~threetears.evals.actions.eval_catalogue`) for its actions and leaves out the
analyst's (:func:`~threetears.evals.actions.analyst.analyst_actions`): what is left is every action the engine contributes, in
the catalogue's order.
"""

from __future__ import annotations

from threetears.evals.actions import Action, eval_catalogue
from threetears.evals.actions.analyst import analyst_actions


def engine_actions_in_catalogue() -> tuple[Action, ...]:
    """Every action of the engine's catalogue that is not the analyst's, in the catalogue's order."""
    analyst = {action.name for action in analyst_actions()}
    return tuple(action for action in eval_catalogue().actions if action.name not in analyst)
