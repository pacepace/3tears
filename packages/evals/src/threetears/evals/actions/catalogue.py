"""The action catalogue: every eval action once — its name, help, parameters, result, permission class and rendering.

A transport mounts the catalogue; it never describes an action itself. An MCP adapter, a CLI or a REST
mapping reads the same :class:`Action` values, so the parameters an agent is shown, the ones it is held
to and the text it reads back cannot drift between surfaces.

**Tools are cut by permission class.** Every action is ``read``, ``spend``, ``write`` or
``destructive``, and a :class:`ToolSpec` names the classes one tool mounts. The tool boundary is the only
level at which MCP's read-only and destructive hints, a client's allow-list and a server's per-tool
grants work, so the classes are what a host splits on: :func:`standard_tools` gives ``evals`` (read,
spend, write) and ``evals_admin`` (destructive), and :func:`read_only_tools` a tool a host can mount for
an agent that may only look. The host supplies the prefix.

**One tool, many actions, flat parameters.** A mounted tool's input is ``action`` — an enum of its
actions plus ``help`` — and the union of its actions' parameters, flat. A parameter name means one
thing across the tool: two actions declaring it with different types or descriptions are refused at
mount. An action is held to its own parameters at call time: one it does not declare is refused,
naming the ones it accepts and an example, rather than ignored.

**Errors teach.** Every refusal a call can meet — no action, an unknown one, one this tool does not
mount, an undeclared parameter, a value that does not validate, a refusal from the engine — says what
was wrong and what would be right: the valid actions, the accepted parameters, an example call. Each
also carries a :class:`Refusal` — its kind and status — so a host's code branches on data, never on
the teaching text.

**Long work is a job, and no call waits on a model.** An action whose work outlives a call (``long_running``)
returns :class:`~threetears.evals.ops.JobsStarted`, and the caller polls with ``job_poll``. A ``spend`` action is
either long-running or makes no completion request during its call: pricing, the cap check and admission stay in
the call, and the work that spends runs as the job. Every ``spend`` action, the engine's and a host's, declares a
:data:`HappyPath` — the parameters that reach its work — and one without is refused when it is declared, so the jobs
conformance kit (:func:`~threetears.evals.testing.jobs_conformance_cases`) can show each one makes no request in its
call and some in its job.

**Destructive actions keep their confirm strings.** A ``destructive`` action must declare a required
``confirm`` parameter, which the operation checks echoes the id it destroys.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from pydantic import BaseModel, ValidationError

from threetears.evals.actions import render
from threetears.evals.kernel.errors import (
    AdmissionRefusedError,
    ConflictError,
    EvalServiceError,
    NotFoundError,
    ProviderRefusedError,
    StorageError,
    ValidationFailedError,
)
from threetears.evals.ops import JobsStarted, OpsHost
from threetears.evals.run import start_run

#: What an action does to the world, which decides the tools that may mount it.
#:
#: ``spend`` says an action may cost money; it is a label a tool cut splits on, not a promise that
#: the engine meters it. The engine's own spend actions are priced and capped in the call, before they spend
#: (``run_launch`` against each run's cost cap, which a launch may only lower; ``analysis_generate`` and the judge
#: actions against the host's out-of-run cap), and spend only in their job. :class:`Action` requires of a
#: host-contributed ``spend`` action a :data:`HappyPath`, so the jobs conformance kit can hold it to making no
#: completion request in its call; how far it meters what its job spends is the host's own handler's.
PermissionClass = Literal["read", "spend", "write", "destructive"]

#: The classes, in the order help lists them.
PERMISSION_CLASSES: tuple[PermissionClass, ...] = get_args(PermissionClass)

#: An action's name: ``noun_verb`` — lowercase words joined by underscores, at least two of them.
ACTION_NAME = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)+$")

#: A tool's name: a lowercase identifier.
TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

#: The action every mounted tool answers, generated from the catalogue rather than declared in it.
HELP_ACTION = "help"

#: Parameter names a tool reserves: ``action`` selects the action, ``topic`` is help's.
RESERVED_PARAMETERS: frozenset[str] = frozenset({"action", "topic"})

#: The default prefix a host's eval tools carry.
DEFAULT_PREFIX = "evals"


@dataclass(frozen=True, kw_only=True)
class Caller:
    """Who is calling, and the scope they act in — resolved by the host for every call.

    Attributes:
        scope_id: The scope every read and write of the call is in. The host decides it from the caller;
            no action takes it as a parameter, so an agent cannot reach another scope by naming it.
        identity: Who the caller is, as the host knows them — recorded where a write records an author.
    """

    scope_id: str
    identity: str

    def __post_init__(self) -> None:
        """Refuse a blank identity. The scope is opaque to the engine, which never reads its value.

        Raises:
            ValueError: The identity is blank.
        """
        if not self.identity.strip():
            raise ValueError("Caller.identity is blank; the host names who is calling")


#: Carries an action out: handed the host, the caller and the validated parameters, it returns the
#: action's result model.
ActionHandler = Callable[[OpsHost, Caller, Any], Awaitable[BaseModel]]


@dataclass(frozen=True, kw_only=True)
class ConformanceSample:
    """One launch a host can run end to end, which the engine's spend actions' happy paths build on.

    The jobs conformance kit needs each spend action to reach its work; the engine's own reach theirs through a
    run, a judged and finished one, and only the host knows a template its kinds launch. So the host names one: its
    runs must finish in a few seconds under the host the kit builds and be judged — through the host's completion
    clients — on at least one dimension, so a second judge and a temperature comparison have scores to read.

    Attributes:
        template_id: A template in the caller's scope whose kind the host launches.
        subject_id: The subject its runs measure.
        models: The candidate models, one arm each; at least one, since a sweep's arm names its model.
        second_judge_model: A judge model other than the run's own, for a second judge's pass and a drift check.
    """

    template_id: str
    subject_id: str
    models: tuple[str, ...]
    second_judge_model: str

    def __post_init__(self) -> None:
        """Refuse a sample naming no model.

        Raises:
            ValueError: ``models`` is empty.
        """
        if not self.models:
            raise ValueError("a conformance sample names at least one candidate model; a sweep's arm needs one")


@dataclass(frozen=True, kw_only=True)
class ConformanceStage:
    """What a :data:`HappyPath` is handed: the host the kit built, the caller, and the host's sample launch.

    Attributes:
        host: The host the action's handler will run in, its completion clients guarded by the kit.
        caller: Who the kit calls as.
        sample: The host's sample launch (:class:`ConformanceSample`).
    """

    host: OpsHost
    caller: Caller
    sample: ConformanceSample

    async def finished_run(self) -> str:
        """Launch the sample and wait for its runs to end, as a script would; the first run's id.

        The runs execute as jobs, so their judges' requests are a job's. Arranged before the call under test.

        Returns:
            The first run's id; it finished, and was judged when the host's judge answered.
        """
        launch = self.host.launch
        runs = await start_run(
            launch,
            template_id=self.sample.template_id,
            subject_id=self.sample.subject_id,
            models=list(self.sample.models),
            k_runs=1,
            scope_id=self.caller.scope_id,
        )
        await launch.job_manager.wait_for([run.id for run in runs])
        return runs[0].id


#: A spend action's happy path: arranges what its work needs in the stage's host — a finished run, a campaign —
#: and returns the parameters (without ``action``) of a call that reaches that work.
HappyPath = Callable[[ConformanceStage], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True, kw_only=True)
class Action:
    """One action: what it is called, what it takes and returns, who may call it, and how it reads back.

    Attributes:
        name: ``noun_verb`` (:data:`ACTION_NAME`), unique in the catalogue.
        summary: One line, shown in the help index.
        workflow: The help index's group: the step of the work this action belongs to.
        permission: Its class: ``read``, ``spend``, ``write`` or ``destructive``.
        params: Its parameters, as a flat pydantic model: every field described, none a nested model.
            A ``destructive`` action's model declares a required ``confirm``.
        result: The model its handler returns.
        handler: Carries it out.
        render: Its result as the text an agent reads.
        example: A call's parameters (without ``action``) that validate against ``params`` — shown in help
            and in every refusal of a bad call.
        long_running: Whether its work outlives the call; such an action returns
            :class:`~threetears.evals.ops.JobsStarted`, and no other may.
        detail: More help, shown on the action's own page.
        happy_path: For a ``spend`` action, required: what reaches its work, for the jobs conformance kit.
            ``None`` for every other class.
    """

    name: str
    summary: str
    workflow: str
    permission: PermissionClass
    params: type[BaseModel]
    result: type[BaseModel]
    handler: ActionHandler
    render: Callable[[Any], str]
    example: Mapping[str, Any]
    long_running: bool = False
    detail: str = ""
    happy_path: HappyPath | None = None

    def __post_init__(self) -> None:
        """Refuse an action no transport could mount honestly.

        Raises:
            ValueError: The name is not ``noun_verb``; the summary is blank or more than one line; the
                class is not one of :data:`PERMISSION_CLASSES`; a parameter is reserved, undescribed or a
                nested model; a destructive action declares no required ``confirm``; ``long_running``
                disagrees with whether the result is ``JobsStarted``; a spend action declares no
                ``happy_path``; or the example does not validate.
        """
        where = f"action {self.name!r}"
        if not ACTION_NAME.fullmatch(self.name):
            raise ValueError(f"{where}: an action is named noun_verb — lowercase words joined by underscores")
        # prose-canary: allow — Action.summary is help text a host's code declares, not text a model wrote
        if not self.summary.strip() or "\n" in self.summary.strip():
            raise ValueError(f"{where}: the summary is one non-blank line; longer help goes in `detail`")
        if self.permission not in PERMISSION_CLASSES:
            raise ValueError(f"{where}: class {self.permission!r} is none of {', '.join(PERMISSION_CLASSES)}")
        declared = self.params.model_fields
        reserved = sorted(RESERVED_PARAMETERS & declared.keys())
        if reserved:
            raise ValueError(f"{where}: {', '.join(reserved)} is reserved by every tool and cannot be a parameter")
        undescribed = sorted(name for name, info in declared.items() if not (info.description or "").strip())
        if undescribed:
            raise ValueError(f"{where}: every parameter is described; {', '.join(undescribed)} is not")
        if "$defs" in self.params.model_json_schema():
            raise ValueError(
                f"{where}: parameters are flat — a nested model has no place in one tool's flat parameter list"
            )
        if self.permission == "destructive" and not ("confirm" in declared and declared["confirm"].is_required()):
            raise ValueError(f"{where}: a destructive action declares a required `confirm`, echoing what it destroys")
        if self.long_running != issubclass(self.result, JobsStarted):
            raise ValueError(
                f"{where}: long work returns JobsStarted for the caller to poll, and only long work does; "
                f"long_running={self.long_running} with result {self.result.__name__}"
            )
        if self.permission == "spend" and self.happy_path is None:
            raise ValueError(
                f"{where}: a spend action declares a happy_path — the parameters that reach its work — so the jobs "
                "conformance kit can show its call makes no completion request and its job makes some"
            )
        try:
            self.params.model_validate(dict(self.example))
        except ValidationError as invalid:
            raise ValueError(f"{where}: its example is not a valid call: {invalid}") from invalid


@dataclass(frozen=True, kw_only=True)
class ToolSpec:
    """One tool a transport mounts: its name, its short description and the classes of action it carries.

    Attributes:
        name: The tool's name (:data:`TOOL_NAME`).
        description: What the tool is for, in a sentence or two; the tool adds how to get help.
        permissions: The classes of action it mounts.
    """

    name: str
    description: str
    permissions: frozenset[PermissionClass]

    def __post_init__(self) -> None:
        """Refuse a malformed name, a blank description or a set of classes that is empty or unknown.

        Raises:
            ValueError: As described.
        """
        if not TOOL_NAME.fullmatch(self.name):
            raise ValueError(f"tool name {self.name!r} is not a lowercase identifier")
        if not self.description.strip():
            raise ValueError(f"tool {self.name!r}: the description is blank")
        unknown = sorted(set(self.permissions) - set(PERMISSION_CLASSES))
        if not self.permissions or unknown:
            raise ValueError(
                f"tool {self.name!r} mounts classes {sorted(self.permissions)}; it mounts one or more of "
                f"{', '.join(PERMISSION_CLASSES)}"
            )


def standard_tools(prefix: str = DEFAULT_PREFIX) -> tuple[ToolSpec, ToolSpec]:
    """The two tools a host mounts by default: ``<prefix>`` and ``<prefix>_admin``.

    Args:
        prefix: The host's prefix for its eval tools.

    Returns:
        ``<prefix>`` — read, spend and write — and ``<prefix>_admin`` — the destructive actions.
    """
    return (
        ToolSpec(
            name=prefix,
            description="Run, read and analyse evals: launch runs, poll their jobs, and read campaigns and reports.",
            permissions=frozenset({"read", "spend", "write"}),
        ),
        ToolSpec(
            name=f"{prefix}_admin",
            description="Irreversibly delete eval records. Every action needs `confirm` echoing what it destroys.",
            permissions=frozenset({"destructive"}),
        ),
    )


def read_only_tools(prefix: str = DEFAULT_PREFIX) -> tuple[ToolSpec]:
    """One tool carrying only the ``read`` actions, for an agent that may look but not act.

    Args:
        prefix: The host's prefix for its eval tools.

    Returns:
        ``<prefix>``, read actions only.
    """
    return (
        ToolSpec(
            name=prefix,
            description="Read evals: templates, runs, jobs, campaigns and reports. Nothing here launches or changes.",
            permissions=frozenset({"read"}),
        ),
    )


#: What refused a call: the public engine error it was, by kind. One kind per
#: :class:`~threetears.evals.kernel.EvalServiceError` class in :mod:`threetears.evals.kernel.errors` (``JobFenced``,
#: which no caller is told, is none) — ``refused`` for :class:`~threetears.evals.kernel.EvalServiceError`
#: itself, raised with a status of its own — so the set is closed: an error class a host derives maps to
#: the kind of its nearest public ancestor, and adding a kind means adding a public error class.
RefusalKind = Literal[
    "refused",
    "not_found",
    "conflict",
    "validation_failed",
    "storage_failed",
    "provider_refused",
    "admission_refused",
]

#: Every refusal kind.
REFUSAL_KINDS: tuple[RefusalKind, ...] = get_args(RefusalKind)

#: The kind each public error class refuses as. ``tests/test_action_refusals.py`` holds every
#: :class:`~threetears.evals.kernel.EvalServiceError` class :mod:`threetears.evals.kernel.errors` exports to a kind of
#: its own, together covering :data:`REFUSAL_KINDS`.
_KIND_OF_ERROR: Mapping[type[EvalServiceError], RefusalKind] = {
    EvalServiceError: "refused",
    NotFoundError: "not_found",
    ConflictError: "conflict",
    ValidationFailedError: "validation_failed",
    StorageError: "storage_failed",
    ProviderRefusedError: "provider_refused",
    AdmissionRefusedError: "admission_refused",
}


@dataclass(frozen=True, kw_only=True)
class Refusal:
    """Why a call was refused, as data: the kind of error and the status it carries.

    The outcome's text is the refusal an agent reads; this is the same refusal for a host's code — a
    REST mapping's status, a metric's label, an audit row — which would otherwise parse that text, or
    lose the difference between a run that does not exist and a launch the process has no room for.

    Attributes:
        kind: Which public error refused it (:data:`RefusalKind`).
        status: The HTTP-style status the error carries (``status_code``), so every surface reports one
            refusal with one status without owning a table of its own.
    """

    kind: RefusalKind
    status: int

    def __post_init__(self) -> None:
        """Refuse a kind outside the closed set.

        Raises:
            ValueError: ``kind`` is not one of :data:`REFUSAL_KINDS`.
        """
        if self.kind not in REFUSAL_KINDS:
            raise ValueError(f"refusal kind {self.kind!r} is none of {', '.join(REFUSAL_KINDS)}")

    @classmethod
    def of(cls, error: EvalServiceError) -> Refusal:
        """The refusal an engine error makes: the kind of its nearest public class, and its own status.

        Args:
            error: The error a handler raised.

        Returns:
            The refusal. A host's own subclass of a public error takes that error's kind, and keeps the
            status it was raised with.
        """
        kind = next(_KIND_OF_ERROR[base] for base in type(error).__mro__ if base in _KIND_OF_ERROR)
        return cls(kind=kind, status=error.status_code)


#: The refusal of a call the catalogue itself turns away — no action, one the tool lacks, an undeclared
#: parameter, a value that does not validate, a help topic that is no action. Each is the call failing
#: the tool's own declared input (``action``'s enum, the action's parameters), which is what
#: :class:`~threetears.evals.kernel.ValidationFailedError` means; derived from that class rather than
#: spelled here, so the two cannot disagree on the status.
_CALL_REFUSAL = Refusal.of(
    ValidationFailedError(
        "the call does not validate against the tool's input", remedy="call it as action='help' describes"
    )
)


@dataclass(frozen=True)
class ActionOutcome:
    """What one call came to: the text an agent reads, the typed result as data, and why it was refused.

    Attributes:
        text: The rendering — the result's, a help page's, or a refusal that teaches.
        structured: The result model's JSON form; ``None`` for help and for a refusal.
        refusal: Why the call was refused, as data; ``None`` for a call that was not.
        action: The action called, when the call named one the tool has.
    """

    text: str
    structured: dict[str, Any] | None = None
    refusal: Refusal | None = None
    action: str | None = None

    def __post_init__(self) -> None:
        """Refuse an outcome that is both a result and a refusal.

        Raises:
            ValueError: ``structured`` and ``refusal`` are both set.
        """
        if self.structured is not None and self.refusal is not None:
            raise ValueError("an outcome carries a result or a refusal, never both")

    @property
    def is_error(self) -> bool:
        """Whether the call was refused."""
        return self.refusal is not None


@dataclass(frozen=True)
class ToolHints:
    """The MCP behaviour hints a mounted tool's actions earn.

    Attributes:
        read_only: Every action is ``read``.
        destructive: Some action is ``destructive``.
        open_world: Some action spends on a provider outside the host.
    """

    read_only: bool
    destructive: bool
    open_world: bool


def _property_meaning(schema: Mapping[str, Any]) -> dict[str, Any]:
    """A parameter's schema with what may differ between actions removed: its title, its default, and whether it may be null.

    One action may require a parameter another takes optionally — a template id a launch must name and a campaign
    may — and the optional one's schema admits ``null`` for "not given". What the value IS does not differ, so a
    nullable schema means what its one non-null branch means.
    """
    meaning = {key: value for key, value in schema.items() if key not in {"title", "default"}}
    branches = meaning.get("anyOf")
    if isinstance(branches, list) and len(branches) == 2 and {"type": "null"} in branches:
        (value,) = [branch for branch in branches if branch != {"type": "null"}]
        meaning = {**{key: v for key, v in meaning.items() if key != "anyOf"}, **value}
    return meaning


@dataclass(frozen=True)
class MountedTool:
    """A tool cut from the catalogue: the actions its classes admit, and how a call to it is carried out.

    Built by :meth:`ActionCatalogue.mount`, which checks the tool's flat parameters mean one thing each.

    Attributes:
        spec: The tool.
        actions: Its actions, in catalogue order.
        catalogue: The catalogue it was cut from, so a refusal can say where an action it lacks lives.
        properties: Its actions' parameters, flat, by name — each one's schema without title or default.
    """

    spec: ToolSpec
    actions: tuple[Action, ...]
    catalogue: ActionCatalogue
    properties: dict[str, dict[str, Any]] = field(repr=False)

    @property
    def name(self) -> str:
        """The tool's name."""
        return self.spec.name

    @property
    def description(self) -> str:
        """The tool's short description, and how to get help from it."""
        return (
            f"{self.spec.description} Call action='help' for the actions grouped by workflow, and "
            "action='help', topic='<action>' for one action's parameters and an example."
        )

    @property
    def hints(self) -> ToolHints:
        """The behaviour hints this tool's actions earn."""
        classes = {action.permission for action in self.actions}
        return ToolHints(
            read_only=classes == {"read"}, destructive="destructive" in classes, open_world="spend" in classes
        )

    def action(self, name: str) -> Action | None:
        """This tool's action of that name, or ``None``."""
        return next((action for action in self.actions if action.name == name), None)

    def input_schema(self) -> dict[str, Any]:
        """The tool's input: ``action`` and ``topic``, then every action's parameters, flat and described.

        Returns:
            A JSON Schema object. Nothing beyond ``action`` is required at the tool: each action holds
            the call to its own required parameters.
        """
        names = [HELP_ACTION, *(action.name for action in self.actions)]
        properties: dict[str, Any] = {
            "action": {"type": "string", "enum": names, "description": "The action to run; 'help' lists them."},
            "topic": {"type": "string", "description": "help only: the action to explain, by name."},
        }
        properties.update(self.properties)
        return {"type": "object", "properties": properties, "required": ["action"], "additionalProperties": False}

    def help_index(self) -> str:
        """The help index: this tool's actions grouped by workflow, each with its class and summary."""
        return render.render_help_index(self)

    def help_page(self, name: str) -> str:
        """One action's help page: its parameters, what it returns and an example call.

        Args:
            name: The action.

        Returns:
            The page.

        Raises:
            KeyError: This tool has no such action.
        """
        action = self.action(name)
        if action is None:
            raise KeyError(name)
        return render.render_help_page(self, action)

    async def call(self, arguments: Mapping[str, Any], *, host: OpsHost, caller: Caller) -> ActionOutcome:
        """Carry out one call: select the action, hold it to its parameters, run it, render what it returns.

        A refusal the engine raises (:class:`~threetears.evals.kernel.EvalServiceError`) is a refused
        call, rendered with its reason and carrying its kind and status (:meth:`Refusal.of`); a call the
        catalogue turns away is ``validation_failed`` (422). Anything else the handler raises propagates: it is a defect, and
        the transport's error path is where a defect belongs.

        Args:
            arguments: The call's arguments, ``action`` among them.
            host: The host the action works in.
            caller: Who is calling, and their scope.

        Returns:
            The outcome.
        """
        given = dict(arguments)
        name = given.pop("action", None)
        if not isinstance(name, str) or not name:
            return ActionOutcome(render.refuse_no_action(self), refusal=_CALL_REFUSAL)
        if name == HELP_ACTION:
            return self._help(given)
        action = self.action(name)
        if action is None:
            return ActionOutcome(render.refuse_unknown_action(self, name), refusal=_CALL_REFUSAL)
        undeclared = sorted(set(given) - set(action.params.model_fields))
        if undeclared:
            return ActionOutcome(render.refuse_undeclared(self, action, undeclared), refusal=_CALL_REFUSAL, action=name)
        try:
            params = action.params.model_validate(given)
        except ValidationError as invalid:
            return ActionOutcome(render.refuse_invalid(self, action, invalid), refusal=_CALL_REFUSAL, action=name)
        try:
            result = await action.handler(host, caller, params)
        except EvalServiceError as refused:
            return ActionOutcome(
                render.refuse_by_engine(self, action, refused), refusal=Refusal.of(refused), action=name
            )
        return ActionOutcome(action.render(result), structured=result.model_dump(mode="json"), action=name)

    def _help(self, given: Mapping[str, Any]) -> ActionOutcome:
        undeclared = sorted(set(given) - {"topic"})
        if undeclared:
            return ActionOutcome(
                render.refuse_help_parameters(self, undeclared), refusal=_CALL_REFUSAL, action=HELP_ACTION
            )
        topic = given.get("topic")
        if topic is None:
            return ActionOutcome(self.help_index(), action=HELP_ACTION)
        if not isinstance(topic, str) or self.action(topic) is None:
            return ActionOutcome(
                render.refuse_unknown_action(self, str(topic)), refusal=_CALL_REFUSAL, action=HELP_ACTION
            )
        return ActionOutcome(self.help_page(topic), action=HELP_ACTION)


class ActionCatalogue:
    """Every action a host offers: the engine's, and any the host contributes.

    Immutable: :meth:`extended` returns a new catalogue with a host's actions added, refusing a name the
    catalogue already has, so a host cannot shadow an engine action by accident.
    """

    def __init__(self, actions: Iterable[Action] = ()) -> None:
        """Hold the actions, refusing a duplicate name.

        No action can be named ``help``: it is one word, and an action's name is ``noun_verb``.

        Args:
            actions: The actions, in the order help lists them within a workflow.

        Raises:
            ValueError: Two actions share a name.
        """
        held: dict[str, Action] = {}
        for action in actions:
            if action.name in held:
                raise ValueError(
                    f"two actions are named {action.name!r}; an action a host contributes takes a name the "
                    "catalogue does not already have"
                )
            held[action.name] = action
        self._actions = held

    @property
    def actions(self) -> tuple[Action, ...]:
        """Every action, in catalogue order."""
        return tuple(self._actions.values())

    def get(self, name: str) -> Action | None:
        """The action of that name, or ``None``."""
        return self._actions.get(name)

    def extended(self, actions: Iterable[Action]) -> ActionCatalogue:
        """This catalogue with a host's actions added after the engine's.

        Args:
            actions: The host's actions.

        Returns:
            The new catalogue.

        Raises:
            ValueError: A host action's name is already in the catalogue, or two of them share one.
        """
        return ActionCatalogue([*self._actions.values(), *actions])

    def mount(self, spec: ToolSpec) -> MountedTool:
        """Cut one tool from the catalogue: the actions whose class the tool mounts.

        Args:
            spec: The tool.

        Returns:
            The mounted tool.

        Raises:
            ValueError: No action has a class the tool mounts, or two of its actions declare one
                parameter name with different meanings (type or description).
        """
        actions = tuple(action for action in self._actions.values() if action.permission in spec.permissions)
        if not actions:
            raise ValueError(f"tool {spec.name!r} mounts {sorted(spec.permissions)} and no action has those classes")
        properties: dict[str, dict[str, Any]] = {}
        declared_by: dict[str, str] = {}
        for action in actions:
            for name, schema in action.params.model_json_schema().get("properties", {}).items():
                meaning = _property_meaning(schema)
                if name in properties and properties[name] != meaning:
                    raise ValueError(
                        f"tool {spec.name!r}: parameter {name!r} means two things — {declared_by[name]} declares "
                        f"{json.dumps(properties[name], sort_keys=True)} and {action.name} declares "
                        f"{json.dumps(meaning, sort_keys=True)}. One name, one meaning across a tool: rename one."
                    )
                properties.setdefault(name, meaning)
                declared_by.setdefault(name, action.name)
        return MountedTool(spec, actions, self, properties)

    def mount_all(self, specs: Sequence[ToolSpec]) -> tuple[MountedTool, ...]:
        """Mount several tools, refusing two that share a name.

        Args:
            specs: The tools.

        Returns:
            The mounted tools, in order.

        Raises:
            ValueError: Two tools share a name, or :meth:`mount` refuses one.
        """
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"two tools share a name among {names}")
        return tuple(self.mount(spec) for spec in specs)


__all__ = [
    "ACTION_NAME",
    "DEFAULT_PREFIX",
    "HELP_ACTION",
    "PERMISSION_CLASSES",
    "REFUSAL_KINDS",
    "RESERVED_PARAMETERS",
    "TOOL_NAME",
    "Action",
    "ActionCatalogue",
    "ActionHandler",
    "ActionOutcome",
    "Caller",
    "ConformanceSample",
    "ConformanceStage",
    "HappyPath",
    "MountedTool",
    "PermissionClass",
    "Refusal",
    "RefusalKind",
    "ToolHints",
    "ToolSpec",
    "read_only_tools",
    "standard_tools",
]
