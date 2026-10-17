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
    EvalServiceError,
    ValidationFailedError,
)
from threetears.evals.kernel.refusals import refusal_kind
from threetears.evals.schema.refusal import REFUSAL_KINDS, KnownRefusalKind, RefusalKind
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

    :ivar scope_id: The scope every read and write of the call is in. The host decides it from the caller;
        no action takes it as a parameter, so an agent cannot reach another scope by naming it.
    :ivar identity: Who the caller is, as the host knows them — recorded where a write records an author.
    """

    scope_id: str
    identity: str

    def __post_init__(self) -> None:
        """Refuse a blank identity. The scope is opaque to the engine, which never reads its value.

        :raises ValueError: The identity is blank.
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

    :ivar template_id: A template in the caller's scope whose kind the host launches.
    :ivar subject_id: The subject its runs measure.
    :ivar models: The candidate models, one arm each; at least one, since a sweep's arm names its model.
    :ivar second_judge_model: A judge model other than the run's own, for a second judge's pass and a drift check.
    """

    template_id: str
    subject_id: str
    models: tuple[str, ...]
    second_judge_model: str

    def __post_init__(self) -> None:
        """Refuse a sample naming no model.

        :raises ValueError: ``models`` is empty.
        """
        if not self.models:
            raise ValueError("a conformance sample names at least one candidate model; a sweep's arm needs one")


@dataclass(frozen=True, kw_only=True)
class ConformanceStage:
    """What a :data:`HappyPath` is handed: the host the kit built, the caller, and the host's sample launch.

    :ivar host: The host the action's handler will run in, its completion clients guarded by the kit.
    :ivar caller: Who the kit calls as.
    :ivar sample: The host's sample launch (:class:`ConformanceSample`).
    """

    host: OpsHost
    caller: Caller
    sample: ConformanceSample

    async def finished_run(self) -> str:
        """Launch the sample and wait for its runs to end, as a script would; the first run's id.

        The runs execute as jobs, so their judges' requests are a job's. Arranged before the call under test.

        :return: The first run's id; it finished, and was judged when the host's judge answered.
        :rtype: str
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

    :ivar name: ``noun_verb`` (:data:`ACTION_NAME`), unique in the catalogue.
    :ivar summary: One line, shown in the help index.
    :ivar workflow: The help index's group: the step of the work this action belongs to.
    :ivar permission: Its class: ``read``, ``spend``, ``write`` or ``destructive``.
    :ivar params: Its parameters, as a flat pydantic model: every field described, none a nested model.
        A ``destructive`` action's model declares a required ``confirm``.
    :ivar result: The model its handler returns.
    :ivar handler: Carries it out.
    :ivar render: Its result as the text an agent reads. Long work need not declare one: every action returning
        :class:`~threetears.evals.ops.JobsStarted` reads back the same way, so it defaults to
        :func:`~threetears.evals.actions.render.render_jobs_started`, which only such an action may use.
    :ivar example: A call's parameters (without ``action``) that validate against ``params`` — shown in help
        and in every refusal of a bad call.
    :ivar long_running: Whether its work outlives the call; such an action returns
        :class:`~threetears.evals.ops.JobsStarted`, and no other may.
    :ivar detail: More help, shown on the action's own page.
    :ivar happy_path: For a ``spend`` action, required: what reaches its work, for the jobs conformance kit.
        ``None`` for every other class.
    """

    name: str
    summary: str
    workflow: str
    permission: PermissionClass
    params: type[BaseModel]
    result: type[BaseModel]
    handler: ActionHandler
    example: Mapping[str, Any]
    render: Callable[[Any], str] = render.render_jobs_started
    long_running: bool = False
    detail: str = ""
    happy_path: HappyPath | None = None

    def __post_init__(self) -> None:
        """Refuse an action no transport could mount honestly.

        :raises ValueError: The name is not ``noun_verb``; the summary is blank or more than one line; the
            class is not one of :data:`PERMISSION_CLASSES`; a parameter is reserved, undescribed or a
            nested model; a destructive action declares no required ``confirm``; ``long_running``
            disagrees with whether the result is ``JobsStarted``; an action whose result is anything else
            declares no ``render`` of its own; a spend action declares no ``happy_path``; or the example
            does not validate.
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
        if not self.long_running and self.render is render.render_jobs_started:
            raise ValueError(
                f"{where}: an action whose result is {self.result.__name__} declares its render — the text an agent "
                "reads it as; only long work, which returns JobsStarted, reads back as the jobs it started"
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

    :ivar name: The tool's name (:data:`TOOL_NAME`).
    :ivar description: What the tool is for, in a sentence or two; the tool adds how to get help.
    :ivar permissions: The classes of action it mounts.
    """

    name: str
    description: str
    permissions: frozenset[PermissionClass]

    def __post_init__(self) -> None:
        """Refuse a malformed name, a blank description or a set of classes that is empty or unknown.

        :raises ValueError: As described.
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

    :param prefix: The host's prefix for its eval tools.
    :ptype prefix: str
    :return: ``<prefix>`` — read, spend and write — and ``<prefix>_admin`` — the destructive actions.
    :rtype: tuple[ToolSpec, ToolSpec]
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

    :param prefix: The host's prefix for its eval tools.
    :ptype prefix: str
    :return: ``<prefix>``, read actions only.
    :rtype: tuple[ToolSpec]
    """
    return (
        ToolSpec(
            name=prefix,
            description="Read evals: templates, runs, jobs, campaigns and reports. Nothing here launches or changes.",
            permissions=frozenset({"read"}),
        ),
    )


@dataclass(frozen=True, kw_only=True)
class Refusal:
    """Why a call was refused, as data: the kind of error and the status it carries.

    The outcome's text is the refusal an agent reads; this is the same refusal for a host's code — a
    REST mapping's status, a metric's label, an audit row — which would otherwise parse that text, or
    lose the difference between a run that does not exist and a launch the process has no room for.

    :ivar kind: Which public error refused it (:data:`RefusalKind`).
    :ivar status: The HTTP-style status the error carries (``status_code``), so every surface reports one
        refusal with one status without owning a table of its own.
    """

    kind: KnownRefusalKind
    status: int

    @property
    def raised(self) -> bool:
        """Whether the call RAISED rather than was refused: a 5xx status, the engine or host failing to carry it out
        (a store, a provider, a host's own 5xx), never the call turned away (:func:`~threetears.evals.actions.render.raised`)."""
        return render.raised(self.status)

    def __post_init__(self) -> None:
        """Refuse a kind outside the closed set.

        :raises ValueError: ``kind`` is not one of :data:`REFUSAL_KINDS`.
        """
        if self.kind not in REFUSAL_KINDS:
            raise ValueError(f"refusal kind {self.kind!r} is none of {', '.join(REFUSAL_KINDS)}")

    @classmethod
    def of(cls, error: EvalServiceError) -> Refusal:
        """The refusal an engine error makes: the kind of its nearest public class, and its own status.

        :param error: The error a handler raised.
        :ptype error: EvalServiceError
        :return: The refusal. A host's own subclass of a public error takes that error's kind, and keeps the
            status it was raised with.
        :rtype: Refusal
        """
        return cls(kind=refusal_kind(error), status=error.status_code)


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

    **Refused or raised.** A call the catalogue, the engine or the host turned away (an unknown action, a value that
    does not validate, a run that does not exist, a conflict) is ``refused``; one the engine or host failed to carry
    out (a store that failed, a provider that refused the account: a 5xx :attr:`Refusal.status`) is ``raised``. Both
    carry a :attr:`refusal` and are errors; :attr:`ending` says which, and so does the text's head word. A host
    recording its candidate's refused calls (``CallLedger.record_refusal``) records an outcome whose ``ending`` is
    ``refused`` and no other: a raised one is the rig's fault, never the candidate's.

    :ivar text: The rendering — the result's, a help page's, or a refusal that teaches.
    :ivar structured: The result model's JSON form; ``None`` for help and for a refusal.
    :ivar refusal: Why the call was refused, as data; ``None`` for a call that was not.
    :ivar action: The action called, when the call named one the tool has.
    """

    text: str
    structured: dict[str, Any] | None = None
    refusal: Refusal | None = None
    action: str | None = None

    def __post_init__(self) -> None:
        """Refuse an outcome that is both a result and a refusal.

        :raises ValueError: ``structured`` and ``refusal`` are both set.
        """
        if self.structured is not None and self.refusal is not None:
            raise ValueError("an outcome carries a result or a refusal, never both")

    @property
    def is_error(self) -> bool:
        """Whether the call was refused or raised."""
        return self.refusal is not None

    @property
    def ending(self) -> Literal["ok", "refused", "raised"]:
        """How the call ended: ``ok`` (a result, or help), ``refused`` (turned away), or ``raised`` (failed by the
        engine or host)."""
        if self.refusal is None:
            return "ok"
        return "raised" if self.refusal.raised else "refused"


@dataclass(frozen=True)
class ToolHints:
    """The MCP behaviour hints a mounted tool's actions earn.

    :ivar read_only: Every action is ``read``.
    :ivar destructive: Some action is ``destructive``.
    :ivar open_world: Some action spends on a provider outside the host.
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

    :ivar spec: The tool.
    :ivar actions: Its actions, in catalogue order.
    :ivar catalogue: The catalogue it was cut from, so a refusal can say where an action it lacks lives.
    :ivar properties: Its actions' parameters, flat, by name — each one's schema without title or default.
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

        :return: A JSON Schema object. Nothing beyond ``action`` is required at the tool: each action holds
            the call to its own required parameters.
        :rtype: dict[str, Any]
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

        :param name: The action.
        :ptype name: str
        :return: The page.
        :rtype: str
        :raises KeyError: This tool has no such action.
        """
        action = self.action(name)
        if action is None:
            raise KeyError(name)
        return render.render_help_page(self, action)

    def refused_by_host(self, refused: EvalServiceError, *, action: object = None) -> ActionOutcome:
        """The outcome of a call the host refused before any action ran — while resolving who calls, or where.

        A transport asks the host who is calling and which :class:`~threetears.evals.ops.OpsHost` the call works in
        before it hands the call here, and a host may refuse there for a reason the caller can act on: it is still
        booting, the tenant is suspended, the token has lapsed. Raised as an
        :class:`~threetears.evals.kernel.EvalServiceError`, that refusal is an outcome built exactly as one the
        engine raises inside an action — the same ``refused:`` text with its ``remedy:`` line, the same
        :class:`Refusal` (:meth:`Refusal.of`: the kind of its nearest public class, the status it was raised with)
        — so a caller reads, and a client branches on, a host's refusal as it does the engine's, and no transport
        builds one of its own.

        :param refused: The error the host raised.
        :ptype refused: EvalServiceError
        :param action: The call's ``action`` argument, as given: it was never checked, so it may name nothing.
        :ptype action: object
        :return: The refused outcome, carrying ``action`` when it names one of this tool's actions and ``None``
            otherwise, as any outcome does.
        :rtype: ActionOutcome
        """
        named = self.action(action) if isinstance(action, str) else None
        return ActionOutcome(
            render.refuse_by_resolver(self, named, refused),
            refusal=Refusal.of(refused),
            action=named.name if named is not None else None,
        )

    async def call(self, arguments: Mapping[str, Any], *, host: OpsHost, caller: Caller) -> ActionOutcome:
        """Carry out one call: select the action, hold it to its parameters, run it, render what it returns.

        A refusal the engine raises (:class:`~threetears.evals.kernel.EvalServiceError`) is a refused
        call, rendered with its reason and carrying its kind and status (:meth:`Refusal.of`); a call the
        catalogue turns away is ``validation_failed`` (422). Anything else the handler raises propagates: it is a
        defect, and the transport's error path is where a defect belongs.

        :param arguments: The call's arguments, ``action`` among them.
        :ptype arguments: Mapping[str, Any]
        :param host: The host the action works in.
        :ptype host: OpsHost
        :param caller: Who is calling, and their scope.
        :ptype caller: Caller
        :return: The outcome.
        :rtype: ActionOutcome
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

        :param actions: The actions, in the order help lists them within a workflow.
        :ptype actions: Iterable[Action]
        :raises ValueError: Two actions share a name.
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

        :param actions: The host's actions.
        :ptype actions: Iterable[Action]
        :return: The new catalogue.
        :rtype: ActionCatalogue
        :raises ValueError: A host action's name is already in the catalogue, or two of them share one.
        """
        return ActionCatalogue([*self._actions.values(), *actions])

    def mount(self, spec: ToolSpec) -> MountedTool:
        """Cut one tool from the catalogue: the actions whose class the tool mounts.

        :param spec: The tool.
        :ptype spec: ToolSpec
        :return: The mounted tool.
        :rtype: MountedTool
        :raises ValueError: No action has a class the tool mounts, or two of its actions declare one
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

        :param specs: The tools.
        :ptype specs: Sequence[ToolSpec]
        :return: The mounted tools, in order.
        :rtype: tuple[MountedTool, ...]
        :raises ValueError: Two tools share a name, or :meth:`mount` refuses one.
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
