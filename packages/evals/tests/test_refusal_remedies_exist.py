"""Every action or operation a refusal names as its remedy exists (#689).

Two refusals told the operator to call ``campaign_set_control`` and ``campaign_update``. Neither is an
action: the catalogue has ``campaign_create`` (whose control parameter is ``control_from_run_id``) and
``campaign_archive``, and the Python calls are ``set_campaign_control`` and ``update_campaign``. A
user following the message hit an unknown action.

This walks every ``raise`` in the package's source, collects each action-shaped name its message
text mentions — a catalogue noun followed by a verb — and requires it to be an action in the
catalogue or a public callable the package exports.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import threetears.evals
from threetears.evals.actions.engine import engine_actions

_SRC = Path(threetears.evals.__file__).parent


def _catalogue() -> set[str]:
    return {action.name for action in engine_actions()}


def _action_shaped() -> re.Pattern[str]:
    """A catalogue noun, then a verb: the catalogue's own verbs, plus the write verbs a remedy reaches for.

    Derived from the catalogue so a new noun or verb there widens what is checked. Identifiers that
    merely start with a noun (``campaign_id``, ``run_eval``, ``job_manager``) are not remedies and do
    not match, because what follows the noun is not a verb.
    """
    names = _catalogue()
    nouns = {name.split("_", 1)[0] for name in names} | {"reporter_case", "reporter_cases"}
    verbs = {name.split("_", 1)[1] for name in names if "_" in name and not name.startswith("reporter_case")}
    verbs |= {"update", "edit", "patch", "set_[a-z_]+", "add_[a-z_]+", "remove_[a-z_]+"}
    noun = "|".join(sorted(map(re.escape, nouns), key=len, reverse=True))
    verb = "|".join(sorted(verbs, key=len, reverse=True))
    # The Python calls are spelled verb first (``set_campaign_control``, ``add_runs_to_campaign``).
    call_verbs = "create|update|set|add|remove|archive|unarchive|delete|designate"
    return re.compile(rf"\b(?:(?:{noun})_(?:{verb})|(?:{call_verbs})_(?:[a-z]+_)*?(?:{noun})(?:_[a-z]+)*)\b")


def _exported_callables() -> set[str]:
    """Every public callable a ``threetears.evals`` module lists in its ``__all__`` — the Python calls."""
    found: set[str] = set()
    for module_info in pkgutil.walk_packages(threetears.evals.__path__, "threetears.evals."):
        try:
            module = importlib.import_module(module_info.name)
        except ImportError:  # an optional extra this environment lacks names nothing a refusal cites
            continue
        for name in getattr(module, "__all__", ()):
            if callable(getattr(module, name, None)) and not inspect.isclass(getattr(module, name)):
                found.add(name)
    return found


def _named_in_refusals(pattern: re.Pattern[str]) -> dict[str, list[str]]:
    """Each action-shaped name any ``raise`` statement's message text mentions, with where."""
    named: dict[str, list[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            for sub in ast.walk(node.exc):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    for match in pattern.findall(sub.value):
                        named.setdefault(match, []).append(f"{path.relative_to(_SRC)}:{node.lineno}")
    return named


def test_the_pattern_catches_the_two_names_that_did_not_exist() -> None:
    """The walk would have caught the defect: both retired names are action-shaped, neither is a field."""
    pattern = _action_shaped()
    assert pattern.findall("call campaign_set_control, or campaign_update, or set_campaign_controls") == [
        "campaign_set_control",
        "campaign_update",
        "set_campaign_controls",
    ]
    assert pattern.findall("campaign_id run_eval job_manager analysis_reporter") == []


def test_every_remedy_a_refusal_names_exists() -> None:
    named = _named_in_refusals(_action_shaped())
    known = _catalogue() | _exported_callables()

    missing = {name: sites for name, sites in named.items() if name not in known}

    assert not missing, f"refusals name actions or calls that do not exist: {missing}"


def test_refusals_do_name_remedies() -> None:
    """Guards the walk itself: an empty collection would pass the check above vacuously."""
    named = _named_in_refusals(_action_shaped())

    assert {"set_campaign_control", "update_campaign", "add_runs_to_campaign"} <= set(named)
