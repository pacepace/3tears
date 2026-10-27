"""What an agent reads back: help generated from the catalogue, refusals that teach, and each result as text.

Every string a mounted tool returns is made here, so the conventions hold in one place:

- **Help is generated, never written.** The index is the tool's actions grouped by workflow with each
  one's class and summary; a page is one action's parameters, what it returns and an example call —
  all read off the :class:`~threetears.evals.actions.catalogue.Action` and its models.
- **A refusal names what was wrong and what would be right**: the valid actions, the parameters the
  action accepts with what each means, and an example call that validates. A call the engine or host
  failed to carry out (a store that failed, a provider that refused the account: a 5xx status,
  :func:`raised`) reads ``raised:``, never ``refused:``, since the call itself was not turned away.
- **A page names the call to the next.** A paged action's result (:attr:`~threetears.evals.actions.Action.paged`)
  ends with ``next: action='…', …, offset=N`` — the arguments the caller sent, ``offset`` advanced — and carries the
  same call as ``next`` in its structured result; its last page carries neither, and no ``next_offset``
  (:func:`render_page`). Every page says in words how much of the whole it holds, in the unit its result pages
  (:func:`page_unit`: flips, options, cards…): ``all 77 options shown`` on a page holding every one, else which it
  holds and how many follow; a result with none to page says nothing of paging, and says in its own words what an
  empty list means where that carries meaning (a sweep with no flips: the call holds over the whole range). A complete
  result never reads as a cut one.
- **A result reads as every fact it carries.** One generic renderer prints any result model field by field
  (:func:`render_model`); a few results keep their own: a started job says how to poll it and a finished one
  what to read next, a page of results how to read the next, one result its counted goal verdicts, and a report
  is its own text. Every analyst action reads back through :func:`render_cited`, which writes each figure as
  ``key = reads``, so the key an analyst cites sits beside the figure as the page prints it.

This module reads the catalogue's types only for annotations, so the catalogue imports it freely.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ValidationError

from threetears.evals.schema import GoalStateOutcome
from threetears.evals.kernel import ResultOutcome, counted_goal_verdicts
from threetears.evals.kernel.errors import EvalServiceError
from threetears.evals.kernel.text_block import one_line
from threetears.evals.ops import (
    JobsStarted,
    JobStatus,
    JudgeTemperatureComparison,
    ReportDocument,
    ResultDetail,
    ResultLine,
    ResultListing,
    ScoreExport,
    SecondJudgeRead,
    dollars_text,
    export_text,
    format_number,
)
from threetears.evals.ops.analyst import CitedFigure

if TYPE_CHECKING:
    from threetears.evals.actions.catalogue import Action, HelpTopic, MountedTool


# --- help --------------------------------------------------------------------------------------------


def _type_word(schema: Mapping[str, Any]) -> str:
    """A parameter's type as a reader says it: ``string``, ``list of string``, ``one of a|b``, ``optional …``."""
    if "enum" in schema:
        return "one of " + "|".join(str(value) for value in schema["enum"])
    if "const" in schema:
        return str(schema["const"])
    if "anyOf" in schema:
        kinds = [option for option in schema["anyOf"] if option.get("type") != "null"]
        words = " or ".join(_type_word(option) for option in kinds)
        nullable = len(kinds) < len(schema["anyOf"])
        return f"optional {words}" if nullable else words
    kind = schema.get("type", "any")
    if kind == "array":
        return f"list of {_type_word(schema.get('items', {}))}"
    return str(kind)


def parameter_lines(action: Action) -> list[str]:
    """One line per parameter the action accepts: name, type, whether required, what it means."""
    schema = action.params.model_json_schema()
    required = set(schema.get("required", []))
    lines = []
    for name, prop in schema.get("properties", {}).items():
        need = "required" if name in required else f"default {json.dumps(prop['default'])}" if "default" in prop else ""
        lines.append(f"- {name} ({_type_word(prop)}{', ' + need if need else ''}): {prop.get('description', '')}")
    return lines


def example_call(action: Action) -> str:
    """The action's example, as the JSON arguments of a call to its tool: ``action`` first."""
    return json.dumps({"action": action.name, **dict(action.example)}, sort_keys=False)


def render_help_index(tool: MountedTool) -> str:
    """The tool's actions grouped by workflow, in catalogue order, each with its class and summary."""
    groups: dict[str, list[Action]] = {}
    for action in tool.actions:
        groups.setdefault(action.workflow, []).append(action)
    lines = [f"{tool.name}: {tool.spec.description}", ""]
    for workflow, actions in groups.items():
        lines.append(f"## {workflow}")
        for action in actions:
            job = ", starts a job" if action.long_running else ""
            lines.append(f"- {action.name} ({action.permission}{job}): {one_line(action.summary)}")
        lines.append("")
    if tool.spec.constraints:
        lines.append("## Rules")
        lines += [f"- {rule}" for rule in tool.spec.constraints]
        lines.append("")
    if tool.spec.topics:
        lines.append("## Reference")
        lines += [f"- {topic.name}: {one_line(topic.summary)}{_paged(topic)}" for topic in tool.spec.topics]
        lines.append("")
        lines.append(
            "For one action's parameters and an example, or a reference page: action='help', topic='<action or topic>'."
        )
    else:
        lines.append("For one action's parameters and an example: action='help', topic='<action>'.")
    return "\n".join(lines)


def _paged(topic: HelpTopic) -> str:
    """What the help index adds to a paged reference topic's line: that it is an index, and how a page is named."""
    return "" if topic.pages is None else f" An index: its pages are topic='{topic.name}.<page>'."


def render_primer(tools: Sequence[MountedTool]) -> str:
    """The primer :func:`~threetears.evals.actions.catalogue_primer` holds to its budget: help first, then the tools,
    then the rules."""
    lines = [
        "Eval tools. Each takes `action`: action='help' lists a tool's actions by workflow, and action='help', "
        "topic='<action>' gives one action's parameters and an example."
    ]
    topics = [f"{topic.name} ({tool.name})" for tool in tools for topic in tool.spec.topics]
    if topics:
        lines.append(f"Reference topics, as action='help', topic='<topic>': {', '.join(topics)}.")
    lines.append("Tools:")
    lines += [f"- {tool.name}: {one_line(tool.spec.description)} Call action='help' first." for tool in tools]
    rules = [
        "A refused call ends with recovery: {...}, the corrected call's fields (parameters at fault, accepted, "
        "example, valid_values or valid_values_from, see); the structured result carries them as recovery."
    ]
    if any(action.long_running for tool in tools for action in tool.actions):
        rules.append("Long work starts a job: poll it with action='job_poll' until it reads done.")
    for tool in tools:
        if any(action.permission == "destructive" for action in tool.actions):
            rules.append(f"{tool.name}: a destructive action needs `confirm` echoing the id it destroys.")
        rules += [f"{tool.name}: {rule}" for rule in tool.spec.constraints]
    lines.append("Rules:")
    lines += [f"- {rule}" for rule in rules]
    return "\n".join(lines)


def topic_description(tool: MountedTool) -> str:
    """What the tool's ``topic`` parameter says it takes: an action's name, and a reference topic where it has one."""
    if not tool.spec.topics:
        return "help only: the action to explain, by name."
    return "help only: the action to explain, or a reference topic the help index lists, by name."


def render_help_page(tool: MountedTool, action: Action) -> str:
    """One action's page: what it does, its class, its parameters, what it returns and an example call."""
    job = (
        " It starts long work and returns a job per piece of it; poll each with action='job_poll', job_id='<id>'."
        if action.long_running
        else ""
    )
    lines = [f"# {action.name}", "", action.summary + job]
    if action.detail:
        lines += ["", action.detail]
    lines += ["", f"Class: {action.permission} (tool {tool.name}).", "", "Parameters:"]
    lines += parameter_lines(action) or ["- none"]
    returned = ", ".join(action.result.model_fields)
    lines += ["", f"Returns {action.result.__name__}: {returned}.", "", f"Example: {example_call(action)}"]
    return "\n".join(lines)


# --- refusals ----------------------------------------------------------------------------------------


def _valid_actions(tool: MountedTool) -> str:
    return "valid actions: help, " + ", ".join(action.name for action in tool.actions)


def _accepted(action: Action) -> str:
    lines = parameter_lines(action) or ["- none"]
    return "\n".join([f"{action.name} accepts:", *lines, f"Example: {example_call(action)}"])


def refuse_no_action(tool: MountedTool) -> str:
    """A call that named no action."""
    return f"refused: name the action to run (`action`). {_valid_actions(tool)}. Start with action='help'."


def refuse_unknown_action(tool: MountedTool, name: str) -> str:
    """A call naming an action this tool does not carry — saying where it lives when the catalogue has it, and, for
    one of the tool's own cut that its role goes without (:attr:`~threetears.evals.actions.catalogue.ToolSpec.workflows`),
    what the role does instead (:attr:`~threetears.evals.actions.catalogue.ToolSpec.left_out`), never another tool."""
    elsewhere = tool.catalogue.get(name)
    if elsewhere is not None and elsewhere.cut != tool.spec.cut:
        return f"refused: {name} is not an action of tool {tool.name}: {_cut_of(elsewhere)}. {_valid_actions(tool)}."
    if elsewhere is not None and _left_out(tool, elsewhere):
        return f"refused: {name} is not on tool {tool.name}: {tool.spec.left_out}. {_valid_actions(tool)}."
    if elsewhere is not None:
        return (
            f"refused: {name} is a {elsewhere.permission} action, and tool {tool.name} carries "
            f"{', '.join(sorted(tool.spec.permissions))} actions only — call it on the tool that mounts "
            f"{elsewhere.permission} actions. {_valid_actions(tool)}."
        )
    return f"refused: there is no action {name!r}. {_valid_actions(tool)}."


def refuse_unknown_topic(tool: MountedTool, topic: str) -> str:
    """A help call naming a topic this tool has no page for: as an unknown action, with its reference topics named."""
    refused = refuse_unknown_action(tool, topic)
    if not tool.spec.topics:
        return refused
    return f"{refused} Reference topics: {', '.join(item.name for item in tool.spec.topics)}."


def refuse_unknown_page(topic: HelpTopic, page: str, named: Sequence[str]) -> str:
    """A help call naming a page a paged reference topic does not have: its pages, and its index."""
    return (
        f"refused: reference topic {topic.name} has no page {page!r}. Its pages: {', '.join(named)}; "
        f"topic='{topic.name}' is its index."
    )


def _left_out(tool: MountedTool, action: Action) -> bool:
    """Whether an action of the tool's own cut is one its role goes without: of a workflow the tool is not cut to."""
    workflows = tool.spec.workflows
    return action.cut == tool.spec.cut and workflows is not None and action.workflow not in workflows


def _cut_of(action: Action) -> str:
    """Which tools mount an action of another cut than the calling tool's, as a refusal says it."""
    if action.cut is None:
        return f"it is a {action.permission} action of the standard tools — call it there"
    return f"it is a {action.permission} action of the {action.cut} tools — call it on the {action.cut} tool"


def refuse_help_parameters(tool: MountedTool, undeclared: Sequence[str]) -> str:
    """A help call carrying parameters help does not take."""
    return (
        f"refused: help takes only `topic` (an action's name); it does not take {', '.join(undeclared)}. "
        f"{_valid_actions(tool)}."
    )


def refuse_undeclared(tool: MountedTool, action: Action, undeclared: Sequence[str]) -> str:
    """A call carrying parameters its action does not declare — refused rather than ignored."""
    del tool
    return f"refused: {action.name} does not take {', '.join(undeclared)}.\n{_accepted(action)}"


def refuse_invalid(tool: MountedTool, action: Action, invalid: ValidationError) -> str:
    """A call whose values do not validate against the action's parameters."""
    del tool
    problems = []
    for error in invalid.errors():
        where = ".".join(str(part) for part in error["loc"]) or "(the call)"
        problems.append(f"- {where}: {error['msg']}")
    return "\n".join(
        [
            f"refused: {action.name} was called with values it cannot take:",
            *problems,
            _accepted(action),
        ]
    )


def refuse_by_engine(tool: MountedTool, action: Action, refused: EvalServiceError) -> str:
    """A call the engine refused: the engine's own reason, then what to do next where the refusal names it.

    A remedy naming an action this tool does not carry says which tool does — the remedy is the engine's and
    names the catalogue's actions, while which of them a caller can reach here is this tool's — so an agent is
    never sent to a call its tool would refuse as unknown. A remedy needing an action of the tool's own cut that its
    role goes without is served as what the role does instead (``ToolSpec.left_out``), naming no action.
    """
    return _refused_with_remedy(tool, action.name, refused)


def refuse_by_resolver(tool: MountedTool, action: Action | None, refused: EvalServiceError) -> str:
    """A call the host refused while resolving it — who is calling, or the host it works in — before any action ran.

    Read exactly as a refusal the engine raises from inside an action (:func:`refuse_by_engine`): the host's own
    reason, then its remedy, so a host booting or a tenant suspended teaches the caller what to do as any refusal
    does. It is headed by the action the call named when this tool has it, and by the tool otherwise: the call was
    refused before its ``action`` was ever checked, so it may name none.
    """
    return _refused_with_remedy(tool, action.name if action is not None else tool.name, refused)


def recovery_line(recovery: Mapping[str, Any]) -> str:
    """A refusal's recovery fields as its text's last line, compact JSON (a client may forward only the text)."""
    return "recovery: " + json.dumps(dict(recovery), separators=(",", ":"), ensure_ascii=False, default=str)


#: The lowest status at which an error is the engine or host failing to carry a call out rather than turning the
#: call away: :class:`~threetears.evals.kernel.StorageError` (503), :class:`~threetears.evals.kernel.ProviderRefusedError`
#: (502), a host's own :class:`~threetears.evals.kernel.EvalServiceError` raised with a 5xx status.
_RAISED_FROM: Final = 500


def raised(status: int) -> bool:
    """Whether an error at ``status`` means the call RAISED (the engine or host failed it) rather than was REFUSED.

    The one rule both the text (``raised:`` or ``refused:``) and the outcome's data
    (:attr:`~threetears.evals.actions.ActionOutcome.ending`) read: a 5xx status is a failure to carry the call out,
    which a host recording its candidate's refused calls (``CallLedger.record_refusal``) must not record as one;
    every other status is the call turned away.

    :param status: The error's HTTP-style status (``status_code``).
    :ptype status: int
    :return: Whether it raised.
    :rtype: bool
    """
    return status >= _RAISED_FROM


def _refused_with_remedy(tool: MountedTool, subject: str, refused: EvalServiceError) -> str:
    """A refusal's reason under ``subject``, then its remedy, then where any action the remedy names is carried.

    Headed ``raised:`` when the error is the engine or host failing the call (:func:`raised`), ``refused:`` otherwise.
    """
    head = "raised" if raised(refused.status_code) else "refused"
    lines = [f"{head}: {subject}: {refused.message}"]
    if refused.remedy is not None:
        named = sorted(set(_ACTION_TOKEN.findall(refused.remedy)))
        actions = [action for action in map(tool.catalogue.get, named) if action is not None]
        if any(_left_out(tool, action) for action in actions):
            # The remedy needs an action this tool's role goes without: what the role does instead is the remedy.
            lines.append(f"remedy: {tool.spec.left_out}")
            return "\n".join(lines)
        lines.append(f"remedy: {refused.remedy}")
        for name in named:
            elsewhere = tool.catalogue.get(name)
            if elsewhere is not None and tool.action(name) is None and elsewhere.cut != tool.spec.cut:
                lines.append(f"{name} is not an action of tool {tool.name}: {_cut_of(elsewhere)}")
            elif elsewhere is not None and tool.action(name) is None:
                lines.append(
                    f"{name} is a {elsewhere.permission} action, which tool {tool.name} does not carry: call it on "
                    f"the tool that mounts {elsewhere.permission} actions"
                )
    return "\n".join(lines)


#: A word that could name an action: two or more lowercase parts joined by underscores (``templates_list``).
_ACTION_TOKEN = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")


# --- results -----------------------------------------------------------------------------------------


def render_model(result: BaseModel) -> str:
    """Any result as text: what it is, then every field it carries, one line each, in the model's own order.

    The one renderer for every result without a bespoke one below, so a field added to a result model is read
    back with no second edit here. A nested model's fields are indented under its name; a list of models is one
    bullet per model, each field ``name=value``; an unset field reads ``none``. Nothing is summarised, rounded
    past :func:`~threetears.evals.ops.format_number` or left out.
    """
    return "\n".join([_RESULT_WORDS.sub(r" \1", type(result).__name__).strip().lower(), *_field_lines(result, "")])


#: A capital letter that starts a word in a model's name (``RunListing`` -> ``run listing``).
_RESULT_WORDS = re.compile(r"(?<=[a-z0-9])([A-Z])")


def _field_lines(model: BaseModel, indent: str) -> list[str]:
    lines: list[str] = []
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            lines += [f"{indent}{name}:", *_field_lines(value, indent + "  ")]
        elif isinstance(value, list | tuple) and value and all(isinstance(item, BaseModel) for item in value):
            lines.append(f"{indent}{name} ({len(value)}):")
            lines += [f"{indent}- " + _item_line(item) for item in value]
        else:
            lines.append(f"{indent}{name}: {_value(value)}")
    return lines


def _item_line(item: BaseModel) -> str:
    """One model of a list on one line: each field ``name=value``."""
    return "; ".join(f"{field}={_value(getattr(item, field))}" for field in type(item).model_fields)


def _value(value: Any) -> str:
    """One field's value on one line: a number as :func:`format_number` writes it, a list joined, JSON otherwise."""
    if value is None:
        return "none"
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, bool | str):
        return str(value)
    if isinstance(value, int | float):
        return format_number(value)
    if isinstance(value, BaseModel):
        return _compact(value.model_dump(mode="json"))
    if isinstance(value, list | tuple):
        return f"[{', '.join(_value(item) for item in value)}]"
    return _compact(value)


def _compact(value: Any) -> str:
    """A stored JSON value on one line, keys sorted, as written — never re-typed or summarised."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def render_cited(result: BaseModel) -> str:
    """An analyst action's result as text: each figure ``key = reads``, grouped under the row that holds it.

    What the result is comes first, then its fields in the model's own order. A figure
    (:class:`~threetears.evals.ops.analyst.CitedFigure`) is one line, its key (with its filters, for a figure written
    through them) and the figure as the page writes it, so an analyst cites the key it read; the value, kind and scale stay in the structured result. A row of a list is one
    bullet of its plain fields, its figures and nested rows indented beneath it. Nothing is summarised or left out,
    but a page's ``next_offset``: the tool writes the next page's whole call in its place (:func:`render_page`).
    """
    return "\n".join([_RESULT_WORDS.sub(r" \1", type(result).__name__).strip().lower(), *_cited_body(result, "")])


def _cited_body(model: BaseModel, indent: str) -> list[str]:
    """Every field of one model, each as :func:`_cited_field` writes it; a page's ``next_offset`` at the top is the
    tool's to write, as the next page's call."""
    lines: list[str] = []
    for name in type(model).model_fields:
        if name == _NEXT_OFFSET and not indent:
            continue
        if _excluded(model, name):
            continue
        lines += _cited_field(name, getattr(model, name), indent)
    return lines


def _excluded(model: BaseModel, name: str) -> bool:
    """Whether the model leaves the field out of its document as it now stands (its ``exclude_if``): left out of its
    text too, so a field added unset writes a result's text as it was before the field existed."""
    excluded = type(model).model_fields[name].exclude_if
    return excluded is not None and bool(excluded(getattr(model, name)))


#: The field a page names the next page's offset by (:attr:`~threetears.evals.actions.Action.paged`).
_NEXT_OFFSET: Final = "next_offset"


def _next_page_call(action: str, params: BaseModel, next_offset: int) -> dict[str, Any]:
    """The call that reads the page after this one: the arguments the caller sent, ``offset`` advanced.

    :param action: The paged action.
    :ptype action: str
    :param params: The call's validated parameters; the ones the caller set are sent again, as validated.
    :ptype params: BaseModel
    :param next_offset: Where the next page starts.
    :ptype next_offset: int
    :return: The next call's arguments, ``action`` first and ``offset`` last.
    :rtype: dict[str, Any]
    """
    sent = params.model_dump(mode="json", include=set(params.model_fields_set) - {"offset"})
    return {
        "action": action,
        **{name: sent[name] for name in type(params).model_fields if name in sent},
        "offset": next_offset,
    }


def _next_page_line(call: Mapping[str, Any]) -> str:
    """The next page's call as a page's last line: ``next: action='readings_list', snapshot_id='…', offset=40``."""
    spelled = (
        f"{name}='{value}'" if isinstance(value, str) else f"{name}={_compact(value)}" for name, value in call.items()
    )
    return "next: " + ", ".join(spelled)


def render_page(
    action: str, params: BaseModel, text: str, structured: dict[str, Any], unit: tuple[str, str]
) -> tuple[str, dict[str, Any]]:
    """A paged action's text and structured result, each naming the next page's call, or ending on the last page.

    :param action: The paged action.
    :ptype action: str
    :param params: The call's validated parameters.
    :ptype params: BaseModel
    :param text: The result as its render writes it.
    :ptype text: str
    :param structured: The result's JSON.
    :ptype structured: dict[str, Any]
    :param unit: What its ``total`` counts, one and many (:func:`page_unit`): ``("flip", "flips")``.
    :ptype unit: tuple[str, str]
    :return: The text, with how much of the whole it holds in words (``all 77 options shown``; ``flips 1 to 40 of 77
        shown; 37 follow``; nothing where its total is none), then ``next: …`` as its last line while
        a page follows; the JSON, with ``next`` (the next call's arguments) beside ``next_offset`` while a page
        follows, and neither on the last page.
    :rtype: tuple[str, dict[str, Any]]
    """
    offset = structured.get(_NEXT_OFFSET)
    shown = _shown_line(structured.get("total"), structured.get("offset"), offset, unit)
    text = f"{text}\n{shown}" if shown else text
    if offset is None:
        return text, {key: value for key, value in structured.items() if key != _NEXT_OFFSET}
    call = _next_page_call(action, params, offset)
    return f"{text}\n{_next_page_line(call)}", {**structured, "next": call}


def page_unit(result: BaseModel) -> tuple[str, str]:
    """What a paged result's ``total`` counts, one and many, as the result says it: its ``page_unit``, a class
    variable (``("flip", "flips")``), or a property where it differs by instance (a block's rows are cards on an
    examples block).

    :param result: A paged result.
    :ptype result: BaseModel
    :return: The unit, one and many.
    :rtype: tuple[str, str]
    :raises TypeError: The result names no ``page_unit`` of two non-empty words.
    """
    unit = getattr(result, "page_unit", None)
    if not (isinstance(unit, tuple) and len(unit) == 2 and all(isinstance(word, str) and word for word in unit)):
        raise TypeError(
            f"{type(result).__name__} is a page and names no page_unit: what its total counts, one and many, such as "
            "('flip', 'flips')"
        )
    return unit[0], unit[1]


#: How a page says how much of the whole it holds (:func:`_shown_line`), ``{one}`` and ``{many}`` its unit.
_ALL_SHOWN: Final = "all {total} {many} shown"
_ONE_SHOWN: Final = "the one {one} shown"
_SOME_SHOWN: Final = "{many} {first} to {last} of {total} shown; {left} {follow}"
_LAST_SHOWN: Final = "{many} {first} to {total} of {total} shown; none follow"
_PAST_SHOWN: Final = "none from {one} {first}: {total} {counted} in all"


def _shown_line(total: Any, offset: Any, next_offset: int | None, unit: tuple[str, str]) -> str:
    """How much of the whole a page holds, in words and in its unit: every one (``all 77 options shown``), else which
    it holds and how many follow; empty for a result that states no total and offset, or whose total is none, since
    a line about paging nothing is no fact about the result."""
    if not isinstance(total, int) or not isinstance(offset, int):
        return ""
    one, many = unit
    if next_offset is not None:
        left = total - next_offset
        return _SOME_SHOWN.format(
            many=many,
            first=offset + 1,
            last=next_offset,
            total=total,
            left=left,
            follow="follows" if left == 1 else "follow",
        )
    if total == 0:
        return ""
    if offset >= total:
        return _PAST_SHOWN.format(one=one, first=offset + 1, total=total, counted=one if total == 1 else many)
    if offset == 0:
        return _ONE_SHOWN.format(one=one) if total == 1 else _ALL_SHOWN.format(total=total, many=many)
    return _LAST_SHOWN.format(many=many, first=offset + 1, total=total)


#: The most characters a page a caller does not size runs to: what one tool result may take of an analyst's context.
_PAGE_TEXT: Final = 12_000


def page_sizing(action: str, params: BaseModel, default: int) -> tuple[int, Callable[[BaseModel], bool] | None]:
    """The page size a paged operation takes, and the test of length it sizes a page by.

    The caller's ``limit`` as named, with no test; else up to ``default`` rows, as many as fit within 12,000
    characters (what one tool result may take of an analyst's context) as the tool writes the page
    (:func:`render_cited`, then :func:`render_page`, the next page's call included). An operation given the test
    returns the largest page that passes it, so a page of long rows is shorter rather than overlong.

    :param action: The action, as the next page's call names it.
    :ptype action: str
    :param params: The call's parameters, ``limit`` among them (``None``: size the page).
    :ptype params: BaseModel
    :param default: The most rows a sized page holds.
    :ptype default: int
    :return: The page's most rows, and the test, or ``None`` where the caller named the size.
    :rtype: tuple[int, Callable[[BaseModel], bool] | None]
    """
    named = getattr(params, "limit", None)
    if named is not None:
        return int(named), None
    return default, page_fits(action, params)


def page_fits(action: str, params: BaseModel) -> Callable[[BaseModel], bool]:
    """The test of length a page is sized by: whether it fits within 12,000 characters (what one tool result may take
    of an analyst's context) as the tool writes it (:func:`render_cited`, then :func:`render_page`, the next page's
    call included). :func:`page_sizing` applies it where the caller names no size; an action whose rows run long
    whatever its limit (a block's example cards) applies it to a named limit too, so a page holds the limit's rows
    where they fit and says how many follow where they do not.

    :param action: The action, as the next page's call names it.
    :ptype action: str
    :param params: The call's parameters.
    :ptype params: BaseModel
    :return: The test.
    :rtype: Callable[[BaseModel], bool]
    """

    def fits(page: BaseModel) -> bool:
        text, _structured = render_page(
            action, params, render_cited(page), page.model_dump(mode="json"), page_unit(page)
        )
        return len(text) <= _PAGE_TEXT

    return fits


def _cited_field(name: str, value: Any, indent: str) -> list[str]:
    """One field: a figure as ``key = reads``, a model or a map of them under its name, a list of rows as bullets."""
    if isinstance(value, CitedFigure):
        return [f"{indent}{value.cited_as()} = {value.reads}"]
    if isinstance(value, BaseModel):
        return [f"{indent}{name}:", *_cited_body(value, indent + "  ")]
    if isinstance(value, Mapping) and value and all(isinstance(item, BaseModel) for item in value.values()):
        lines = [f"{indent}{name}:"]
        for key, item in value.items():
            lines += _cited_field(str(key), item, indent + "  ")
        return lines
    if isinstance(value, list | tuple) and value and all(isinstance(item, BaseModel) for item in value):
        lines = [f"{indent}{name} ({len(value)}):"]
        for item in value:
            lines += _cited_row(item, indent + "  ")
        return lines
    return [f"{indent}{name}: {_value(value)}"]


def _cited_row(item: BaseModel, indent: str) -> list[str]:
    """One row of a list: a bullet of its plain fields, then its figures and nested rows beneath it."""
    if isinstance(item, CitedFigure):
        return [f"{indent}{item.cited_as()} = {item.reads}"]
    plain: list[str] = []
    beneath: list[str] = []
    for name in type(item).model_fields:
        if _excluded(item, name):
            continue
        value = getattr(item, name)
        written = _cited_field(name, value, indent + "  ")
        if len(written) == 1 and not isinstance(value, CitedFigure):
            plain.append(written[0].strip())
        else:
            beneath += written
    return [f"{indent}- {'; '.join(plain)}".rstrip(), *beneath]


def render_report(document: ReportDocument) -> str:
    """The report itself, in the form asked for — which already says whether it is an analysis or code-only."""
    return document.body


def _result_line(line: ResultLine) -> str:
    """One result on one line: where it sits, its condition, and its headline measures."""
    parts = [
        f"{line.condition.value} ({line.termination})",
        f"cost {dollars_text(line.cost_usd)}",
        f"total_ms {_ms(line.total_ms)}",
    ]
    if line.goal_checks:
        passed = "not counted" if line.goal_checks_passed is None else f"{line.goal_checks_passed}/{line.goal_checks}"
        parts.append(f"goal checks {passed}")
    if line.judge_scores:
        parts.append("judged " + ", ".join(f"{dim}={score}" for dim, score in line.judge_scores.items()))
    if line.host_measures:
        parts.append("measures " + ", ".join(f"{name}={value}" for name, value in line.host_measures.items()))
    if not line.has_trace:
        parts.append("no trace")
    return f"- {line.id}: case {line.test_case_id} k={line.k_iteration}, {line.model}: " + "; ".join(parts)


def render_results(listing: ResultListing, *, default_limit: int | None = None) -> str:
    """One page of a run's results, and how to read the next page and one result.

    :param listing: The page.
    :ptype listing: ResultListing
    :param default_limit: The page size the calling surface uses when none is named; the next-page hint names
        ``limit`` whenever this page's differs, so following the hint reads a page of the same size.
    :ptype default_limit: int | None
    :return: The text.
    :rtype: str
    """
    narrowed = "" if listing.condition_filter is None else f" {listing.condition_filter.value}"
    end = listing.offset + len(listing.results)
    lines = [
        f"results of run {listing.run_id}: {listing.total}{narrowed}, rows {listing.offset + 1}-{end} shown"
        if listing.results
        else f"results of run {listing.run_id}: {listing.total}{narrowed}, none from row {listing.offset + 1}",
        *(_result_line(line) for line in listing.results),
    ]
    if listing.next_offset is not None:
        hint = f"More: action='results_list', run_id='{listing.run_id}', offset={listing.next_offset}"
        if listing.condition_filter is not None:
            hint += f", condition_filter='{listing.condition_filter.value}'"
        if listing.limit is not None and listing.limit != default_limit:
            hint += f", limit={listing.limit}"
        lines.append(hint + ".")
    if listing.results:
        lines.append("Read one with action='result_get', result_id='<id>'.")
    return "\n".join(lines)


def render_export(export: ScoreExport) -> str:
    """An export as its text, and on a page that has a next, the call that reads it.

    The next call repeats the page's format and status wherever they are not the action's defaults, so following
    it reads the next page of the same export.

    :param export: The export, or one page of it.
    :ptype export: ScoreExport
    :return: The text.
    :rtype: str
    """
    page = export.page
    if page is None or page.next_after_run_id is None:
        return export_text(export)
    hint = f"More: action='scope_export', export_after_run_id='{page.next_after_run_id}'"
    if export.format != "csv":
        hint += f", export_format='{export.format}'"
    if page.run_status != "completed":
        hint += f", run_status='{page.run_status or 'all'}'"
    return export_text(export, next_call=hint + ".")


def _goal_check_line(outcome: GoalStateOutcome, counted: bool | None, condition: ResultOutcome) -> str:
    """One goal check as evaluated, and as every rate counts it when the two differ.

    A candidate failure counts every check failed and a harness fault counts none
    (:func:`~threetears.evals.kernel.counted_goal_verdicts`), so a check that evaluated True on such a
    result says both, rather than reading as a pass the listing's count does not hold.
    """
    evaluated = "passed" if outcome.passed else "failed"
    if counted is None:
        verdict = f"{evaluated} as evaluated; not counted (harness fault)"
    elif counted != outcome.passed:
        cause = "candidate failure" if condition is ResultOutcome.CANDIDATE_FAIL else condition.value
        verdict = f"{evaluated} as evaluated; counts {'passed' if counted else 'failed'} ({cause})"
    else:
        verdict = evaluated
    return f"goal check {outcome.expression}: {verdict}" + (f" — {outcome.detail}" if outcome.detail else "")


def _ms(value: float | None) -> str:
    return "absent" if value is None else f"{format_number(value)}ms"


def _latency_lines(detail: ResultDetail) -> list[str]:
    """The stored latency block, an absent component shown as absent, then the remainder as the partition read it."""
    latency = detail.result.latency
    if latency is None:
        stored = "latency: none recorded"
    else:
        stored = "latency: " + ", ".join(
            f"{name} {_ms(getattr(latency, name))}"
            for name in ("total_ms", "llm_ms", "tool_ms", "async_wait_ms", "judge_ms")
        )
    partition = detail.latency_partition
    if partition.withheld is not None:
        return [stored, f"orchestration_ms withheld: {partition.withheld}"]
    return [stored, f"orchestration_ms {_ms(partition.orchestration_ms)} (total_ms less llm_ms and tool_ms)"]


def _record_lines(detail: ResultDetail) -> list[str]:
    """The record part: errors, spend, latency and usage rows, checks and scores, then what the kind stored."""
    result, condition = detail.result, detail.condition
    lines = [
        f"{label}: {error}"
        for label, error in (
            ("candidate error", result.candidate_error),
            ("infra error", result.infra_error),
            ("judge error", result.judge_error),
        )
        if error
    ]
    lines.append(f"cost {dollars_text(result.cost_usd)} over {', '.join(result.cost_roles) or 'no role'}")
    lines += _latency_lines(detail)
    lines.append(f"usage ({len(result.usage)} row(s)):")
    lines += [f"- {_compact(row.model_dump(mode='json', exclude_none=True))}" for row in result.usage]
    counted = counted_goal_verdicts(result)
    lines += [
        _goal_check_line(outcome, None if counted is None else counted[index][1], condition.scoring)
        for index, outcome in enumerate(result.goal_state_outcomes)
    ]
    lines += [f"judged {score.dim}: {score.score} ({score.scale})" for score in result.judge_scores()]
    lines += [f"judge could not tell on {dim}: {reason}" for dim, reason in result.judge_cannot_tell.items()]
    if result.host_measures:
        lines.append(f"host measures: {_compact(result.host_measures)}")
    record = detail.record
    if record is None:
        return lines
    lines.append(f"output ({len(record.output)} document(s), as the kind stored them):")
    lines += [f"- {_compact(document)}" for document in record.output]
    if record.call_ledger is None:
        lines.append("call ledger: none kept")
    else:
        lines.append(f"call ledger ({len(record.call_ledger.calls)} call(s) the kind recorded as succeeded):")
        lines += [f"- {call.tool}.{call.action} {_compact(call.params)}" for call in record.call_ledger.calls]
        refused = record.call_ledger.refused
        if refused is None:
            lines.append("refused calls: no refusal record kept")
        else:
            lines.append(f"refused calls ({len(refused)} the kind recorded as refused, never among the calls above):")
            lines += [
                f"- {refusal.tool}.{refusal.action} {_compact(refusal.params)} after {refusal.position} call(s): "
                f"{one_line(refusal.reason)}"
                for refusal in refused
            ]
    if record.end_state is not None:
        lines.append(f"end state: {_compact(record.end_state)}")
    if record.judged_artifact is not None:
        lines.append(
            f"judge evidence ({record.judged_artifact}) left out: read it with action='result_get', "
            f"result_id='{result.id}', part='judge'."
        )
    if record.span_count:
        lines.append(
            f"{record.span_count} span(s) left out: read them with action='result_get', result_id='{result.id}', "
            "part='spans'."
        )
    return lines


def render_result(detail: ResultDetail) -> str:
    """One stored result and the part of its trace asked for.

    The record part prints the output documents one per line exactly as the kind stored them, so whatever the
    kind wrote about a call — that it failed, what the tool said — reads in the kind's own words; nothing here
    interprets them. The judge's evidence and the spans are printed only when their part is asked for, and the
    record part says how to ask.
    """
    result, condition = detail.result, detail.condition
    lines = [
        f"result {result.id} of run {result.eval_run_id}: case {result.test_case_id} k={result.k_iteration}, "
        f"model {result.model}, kind {result.candidate_kind}",
        f"condition: {condition.scoring.value} (termination {condition.termination}, judging {condition.judging})",
    ]
    if condition.disclosure:
        lines.append(f"  {condition.disclosure}")
    if detail.part == "record":
        lines += _record_lines(detail)
    if detail.trace_state == "none":
        lines.append("trace: none stored")
    elif detail.trace_state == "missing":
        lines.append("trace: recorded but its document is missing")
    elif detail.part == "judge":
        if detail.judge is None:
            lines.append("judge evidence: none stored — nothing was sent to a judge for this cell")
        else:
            evidence = detail.judge.evidence
            lines.append(f"judge evidence ({detail.judge.judged_artifact}):")
            if evidence.subject is not None:
                lines += ["subject:", evidence.subject]
            lines += ["case material:", evidence.case_material, "artifact:", evidence.artifact]
    elif detail.part == "spans":
        spans = detail.spans or []
        lines.append(f"spans ({len(spans)}):")
        lines += [f"- {_compact(span)}" for span in spans]
    return "\n".join(lines)


def render_jobs_started(started: JobsStarted) -> str:
    """What a start returned: each job, and how to poll it."""
    lines = [f"started {len(started.jobs)} job(s):"]
    lines += [
        f"- {job.job_id} ({job.kind} {job.target_id}{', ' + job.label if job.label else ''})" for job in started.jobs
    ]
    lines.append("Poll each with action='job_poll', job_id='<id>' until it reads done.")
    return "\n".join(lines)


def render_job_status(status: JobStatus) -> str:
    """Where a job stands, and what to read next when it has ended."""
    lines = [f"job {status.job_id}: {status.state} ({status.status}){'' if status.done else ', still going'}"]
    if status.progress:
        lines.append("progress: " + json.dumps(status.progress, sort_keys=True))
    if status.detail:
        lines.append(f"detail: {status.detail}")
    lines += [f"fault: {fault}" for fault in status.faults]
    if status.refusal is not None:
        lines.append(f"refusal: {status.refusal.kind} ({status.refusal.status})")
    if status.run_ids:
        lines.append("runs: " + ", ".join(status.run_ids))
    if status.result is not None:
        lines.append(_render_result(status))
    elif status.state == "completed" and status.analysis_id and status.campaign_id:
        lines.append(f"Read it with action='report_read', campaign_id='{status.campaign_id}'.")
    elif status.done and status.run_ids:
        lines.append("Read each with action='run_get', run_id='<id>'.")
    elif status.done and status.run_id and status.kind == "run":
        lines.append(f"Read it with action='run_get', run_id='{status.run_id}'.")
    return "\n".join(lines)


def _render_result(status: JobStatus) -> str:
    """A completed job's result as its own action renders it: a temperature comparison by its own words, else JSON."""
    assert status.result is not None  # only called with a result
    if status.kind == "judge_temperature":
        return JudgeTemperatureComparison.model_validate(status.result).render()
    if status.kind == "second_judge":
        return render_model(SecondJudgeRead.model_validate(status.result))
    return "result: " + json.dumps(status.result, sort_keys=True)


__all__ = [
    "example_call",
    "parameter_lines",
    "raised",
    "recovery_line",
    "refuse_by_engine",
    "refuse_by_resolver",
    "refuse_help_parameters",
    "refuse_invalid",
    "refuse_no_action",
    "refuse_undeclared",
    "refuse_unknown_action",
    "refuse_unknown_page",
    "refuse_unknown_topic",
    "render_cited",
    "render_help_index",
    "render_help_page",
    "render_job_status",
    "render_jobs_started",
    "render_export",
    "render_model",
    "page_fits",
    "page_unit",
    "render_page",
    "render_primer",
    "render_report",
    "render_result",
    "render_results",
    "topic_description",
]
