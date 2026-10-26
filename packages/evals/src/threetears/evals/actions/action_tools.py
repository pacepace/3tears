"""Actions as tools of their own, for a loop that runs a model in process.

A transport mounts the catalogue as a few tools, each a set of actions selected by an ``action`` key
(:class:`~threetears.evals.actions.catalogue.MountedTool`). A loop that hands a model its tools in process — the
analyst's and the editor's (:func:`~threetears.evals.actions.analyst.analyst_toolset`,
:func:`~threetears.evals.actions.analyst.editor_toolset`) — presents each action as one tool instead: named for the
action, described by its help, its input the action's own parameters. :class:`ActionTool` is that tool, and
:func:`action_tools` cuts a set of them from a catalogue by name.

**The same call, cut another way.** An :class:`ActionTool` carries the mounted tool it was taken from, and a call is
that tool's call with the action named for it: the same validation, the same handler, the same result and the same
render. Only its refusals read differently (``standalone``, :mod:`~threetears.evals.actions.render`): there is no
``action`` key to show in an example and no ``help`` to call, and another action a remedy names is the tool of that
name. A call naming ``action`` or ``topic`` is refused: the tool is its action.

**What the loop consumes.** Each tool's :attr:`ActionTool.name`, :attr:`ActionTool.description` and
:meth:`ActionTool.input_schema` are primitives a provider's tool definition is built from, and a call's
:class:`~threetears.evals.actions.catalogue.ActionOutcome` is its text, its error flag and its structured result.

**Results read as their action renders them.** A result's text is the action's own render, which names no tool. The
analyst's actions render through :func:`~threetears.evals.actions.render.render_cited`, which names no other call;
an engine action whose render points to its next call (a page of results, a started job) still writes it as the
mounted tool's ``action='…'`` form.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals.actions import render
from threetears.evals.actions.catalogue import (
    DEFAULT_PREFIX,
    RESERVED_PARAMETERS,
    Action,
    ActionCatalogue,
    ActionOutcome,
    Caller,
    MountedTool,
    PermissionClass,
    Refusal,
    ToolSpec,
)
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.ops import OpsHost

#: The refusal of a call naming ``action`` or ``topic``: the call fails the tool's declared input, as a call naming a
#: parameter its action does not declare does.
_RESERVED_REFUSAL = Refusal.of(
    ValidationFailedError(
        "the call names a key its tool reserves", remedy="call the tool with its own parameters, and no action"
    )
)


def _untitled(schema: Any) -> Any:
    """A JSON schema with every ``title`` dropped: the model names it, and a tool's name already says what it is.

    A parameter may itself be named ``title``, so the names under ``properties`` are kept, and only the schemas they
    map to are walked.
    """
    if isinstance(schema, list):
        return [_untitled(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    kept: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if key == "properties" and isinstance(value, dict):
            kept[key] = {name: _untitled(property_schema) for name, property_schema in value.items()}
        else:
            kept[key] = _untitled(value)
    return kept


@dataclass(frozen=True)
class ActionTool:
    """One action as a tool of its own: its name, its help as a description, its parameters as its input.

    :ivar action: The action.
    :ivar mounted: The tool it was cut from (:func:`action_tools`), whose call carries it out: the same validation,
        refusals and render as a transport's.
    """

    action: Action
    mounted: MountedTool

    def __post_init__(self) -> None:
        """Refuse a mounted tool that does not carry the action.

        :raises ValueError: ``mounted`` has no action of this name, or another one of it.
        """
        if self.mounted.action(self.action.name) is not self.action:
            raise ValueError(f"tool {self.mounted.name!r} does not carry action {self.action.name!r}")

    @property
    def name(self) -> str:
        """The tool's name: its action's."""
        return self.action.name

    @property
    def description(self) -> str:
        """What the tool does: its action's summary, its detail, then its example's arguments."""
        parts = [self.action.summary]
        if self.action.detail:
            parts.append(self.action.detail)
        parts.append(f"Example arguments: {render.example_call(self.action, standalone=True)}")
        return "\n\n".join(parts)

    def input_schema(self) -> dict[str, Any]:
        """The tool's input: its action's parameters as a JSON schema, without titles, and nothing beyond them.

        :return: A JSON Schema object: the parameters, their defaults and which are required.
        :rtype: dict[str, Any]
        """
        schema: dict[str, Any] = _untitled(self.action.params.model_json_schema())
        return {**schema, "additionalProperties": False}

    async def call(self, arguments: Mapping[str, Any], *, host: OpsHost, caller: Caller) -> ActionOutcome:
        """Carry out one call of the action, as its mounted tool does, refusing a call that names ``action`` or ``topic``.

        :param arguments: The action's parameters, and nothing else.
        :ptype arguments: Mapping[str, Any]
        :param host: The host the action works in.
        :ptype host: OpsHost
        :param caller: Who is calling, and their scope.
        :ptype caller: Caller
        :return: The outcome: its text, its structured result, and why it was refused when it was.
        :rtype: ActionOutcome
        """
        reserved = sorted(RESERVED_PARAMETERS & set(arguments))
        if reserved:
            return ActionOutcome(
                render.refuse_undeclared(self.mounted, self.action, reserved, standalone=True),
                refusal=_RESERVED_REFUSAL,
                action=self.action.name,
            )
        return await self.mounted.call(
            {**arguments, "action": self.action.name}, host=host, caller=caller, standalone=True
        )


def action_tools(catalogue: ActionCatalogue, names: Sequence[str]) -> tuple[ActionTool, ...]:
    """The named actions, each as a tool of its own, cut from the catalogue as a transport mounts them.

    Each action is taken from a tool mounting its cut (:attr:`~threetears.evals.actions.catalogue.Action.cut`) and
    the classes of the named actions on that cut, so a parameter name means one thing across those actions as it does
    across a mounted tool (:meth:`~threetears.evals.actions.catalogue.ActionCatalogue.mount`).

    :param catalogue: The catalogue: :func:`~threetears.evals.actions.eval_catalogue`, with the host's actions.
    :ptype catalogue: ActionCatalogue
    :param names: The actions, by name: :func:`~threetears.evals.actions.analyst_toolset` or
        :func:`~threetears.evals.actions.editor_toolset`, or any others.
    :ptype names: Sequence[str]
    :return: The tools, in the order named.
    :rtype: tuple[ActionTool, ...]
    :raises ValueError: A name is no action of the catalogue or is named twice, or a parameter name means two things
        on one cut's tool.
    """
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        raise ValueError(f"actions named twice: {', '.join(repeated)}; name each tool once")
    unknown = [name for name in names if catalogue.get(name) is None]
    if unknown:
        raise ValueError(
            f"no action of the catalogue is named {', '.join(map(repr, unknown))}; its actions are "
            f"{', '.join(action.name for action in catalogue.actions)}"
        )
    actions = [action for name in names if (action := catalogue.get(name)) is not None]
    classes: dict[str | None, set[PermissionClass]] = {}
    for action in actions:
        classes.setdefault(action.cut, set()).add(action.permission)
    mounted = {
        cut: catalogue.mount(
            ToolSpec(
                name=cut or DEFAULT_PREFIX,
                description="The actions presented one tool each, by name.",
                permissions=frozenset(held),
                cut=cut,
            )
        )
        for cut, held in classes.items()
    }
    return tuple(ActionTool(action, mounted[action.cut]) for action in actions)


__all__ = ["ActionTool", "action_tools"]
