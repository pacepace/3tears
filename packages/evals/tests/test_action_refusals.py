"""A refused call carries its refusal as data: the kind of error and the status it carries.

The outcome's text teaches an agent; :class:`~threetears.evals.actions.Refusal` is the same refusal for
a host's code, which before it had to parse that text or lose the difference between a run that is not
there and a launch the process has no room for. Pinned here, through :meth:`MountedTool.call`:

- **The kinds are closed, and derived from the public error hierarchy**: one per class
  :mod:`threetears.evals.kernel.errors` exports, no more.
- **Every public error refuses as its kind, with its own status**, and a host's subclass as its nearest
  public ancestor's kind, keeping the status it was raised with.
- **A call the catalogue turns away** — no action, an unknown one, an undeclared parameter, a bad value,
  a help topic that is no action — is ``validation_failed`` with that error's status.
- **A result and help carry no refusal**, and an outcome cannot be both a result and a refusal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import Field

import threetears.evals.kernel.errors as errors
from threetears.evals.actions import (
    REFUSAL_KINDS,
    Action,
    ActionCatalogue,
    ActionOutcome,
    Caller,
    MountedTool,
    Refusal,
    ToolSpec,
)
from threetears.evals.kernel import (
    AdmissionRefusedError,
    ConflictError,
    EvalServiceError,
    NotFoundError,
    ProviderRefusedError,
    StorageError,
    ValidationFailedError,
)
from threetears.evals.ops import OpsHost
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema import CallLedger
from packages.evals.tests.ops_support import CALLER, ops_fixture
from packages.evals.tests.off_loop import off_loop


class _Raise(EvalBaseModel):
    """A host action's parameters."""

    note: str = Field(description="Anything; the action refuses whatever it is given.")


class _Never(EvalBaseModel):
    """A result the action never returns."""


class _HostMissing(NotFoundError):
    """A host's own not-found, as a host's code derives one."""


def _tool(error: Callable[[], Exception] | None) -> MountedTool:
    """A read tool whose one action raises ``error()``, or returns when it is ``None``."""

    async def handle(host: OpsHost, caller: Caller, params: _Raise) -> _Never:
        if error is not None:
            raise error()
        return _Never()

    action = Action(
        name="note_refuse",
        summary="Refuse a note.",
        workflow="Host",
        permission="read",
        params=_Raise,
        result=_Never,
        handler=handle,
        render=lambda result: "accepted",
        example={"note": "hello"},
    )
    return ActionCatalogue([action]).mount(
        ToolSpec(name="host", description="A host tool.", permissions=frozenset({"read"}))
    )


async def _call(tool: MountedTool, arguments: dict[str, Any]) -> ActionOutcome:
    return await tool.call(arguments, host=off_loop(ops_fixture).host, caller=CALLER)


#: One instance of every public error class, as a handler raises it, with the refusal it makes.
_PUBLIC: dict[type[EvalServiceError], tuple[Callable[[], EvalServiceError], str, int]] = {
    EvalServiceError: (lambda: EvalServiceError("no", status_code=418, remedy="ask the teapot"), "refused", 418),
    NotFoundError: (lambda: NotFoundError("run", "r-1", remedy="list them first"), "not_found", 404),
    ConflictError: (lambda: ConflictError("busy", remedy="try again"), "conflict", 409),
    ValidationFailedError: (
        lambda: ValidationFailedError("bad", details={"field": "x"}, remedy="send a good one"),
        "validation_failed",
        422,
    ),
    StorageError: (lambda: StorageError("down"), "storage_failed", 503),
    ProviderRefusedError: (lambda: ProviderRefusedError("rate limited"), "provider_refused", 502),
    AdmissionRefusedError: (
        lambda: AdmissionRefusedError(requested=3, admitted=4, limit=5, limit_name="EVAL_MAX"),
        "admission_refused",
        429,
    ),
}


#: The public exceptions of :mod:`~threetears.evals.kernel.errors` that no call is refused with: ``JobFenced`` stops a
#: job's own work when its record was written lost, and is caught at the job's boundary — no caller is told it; and
#: ``RefusedInput`` is a lower layer's refusal, which the surface calling it translates into one that is.
_NOT_REFUSALS = {"JobFenced", "RefusedInput"}


def test_the_kinds_are_one_per_public_error_class_and_no_more() -> None:
    exported = {name: getattr(errors, name) for name in errors.__all__}
    public = {cls for cls in exported.values() if issubclass(cls, EvalServiceError)}
    assert set(_PUBLIC) == public, (
        "a public error class was added or removed: give it a refusal kind here and in RefusalKind"
    )
    assert {name for name, cls in exported.items() if not issubclass(cls, EvalServiceError)} == _NOT_REFUSALS, (
        "a public exception that is no EvalServiceError is never a call's refusal: name it in _NOT_REFUSALS with why"
    )
    kinds = [Refusal.of(build()).kind for build, _, _ in _PUBLIC.values()]
    assert sorted(kinds) == sorted(REFUSAL_KINDS), "every kind is exactly one public class's"


@pytest.mark.parametrize(
    ("error", "kind", "status"),
    [*_PUBLIC.values(), (lambda: _HostMissing("widget", "w-1", remedy="widgets_list names them"), "not_found", 404)],
    ids=[*(cls.__name__ for cls in _PUBLIC), "host-subclass"],
)
async def test_an_engine_refusal_carries_its_kind_and_status(
    error: Callable[[], EvalServiceError], kind: str, status: int
) -> None:
    outcome = await _call(_tool(error), {"action": "note_refuse", "note": "x"})
    assert outcome.is_error and outcome.refusal == Refusal(kind=kind, status=status)  # type: ignore[arg-type]
    refused = error()
    ending = "raised" if status >= 500 else "refused"
    assert outcome.ending == ending, "a 5xx is the engine failing the call, never the call turned away"
    remedy = "" if refused.remedy is None else f"\nremedy: {refused.remedy}"
    assert outcome.text == f"{ending}: note_refuse: {refused.message}{remedy}", "the teaching text, then the remedy"
    assert (outcome.structured, outcome.action) == (None, "note_refuse")


@pytest.mark.parametrize(
    ("status", "kind", "ending"),
    [(409, "conflict", "refused"), (422, "validation_failed", "refused"), (503, "storage_failed", "raised")],
)
async def test_a_raised_call_and_a_refused_one_are_told_apart_in_data_and_in_text(
    status: int, kind: str, ending: str
) -> None:
    """A host recording its candidate's refused calls reads ``ending``: a raised call is the rig's, never recorded.

    The rule is the status alone (``render.raised``): the text's head word and the outcome's ``ending`` cannot
    disagree, and ``refusal.raised`` says the same of the data a transport serves.
    """
    built = {kind_: build for build, kind_, _ in _PUBLIC.values()}[kind]
    outcome = await _call(_tool(built), {"action": "note_refuse", "note": "x"})
    assert outcome.refusal is not None and outcome.refusal.status == status
    assert outcome.ending == ending and outcome.refusal.raised is (ending == "raised")
    assert outcome.text.startswith(f"{ending}: note_refuse: ")
    ledger = CallLedger(refused=[])
    if outcome.ending == "refused":
        ledger.record_refusal("evals", "note_refuse", {"note": "x"}, reason=outcome.text)
    assert len(ledger.refused or []) == (1 if ending == "refused" else 0)


async def test_a_result_and_help_end_ok() -> None:
    help_index = await _call(_tool(lambda: ConflictError("unused", remedy="unused")), {"action": "help"})
    assert help_index.ending == "ok" and not help_index.is_error


async def test_a_host_subclass_keeps_the_status_it_was_raised_with() -> None:
    class _Gone(EvalServiceError):
        """A host's own refusal with a status the engine never uses."""

    outcome = await _call(
        _tool(lambda: _Gone("gone for good", status_code=410, remedy="ask for another")),
        {"action": "note_refuse", "note": "x"},
    )
    assert outcome.refusal == Refusal(kind="refused", status=410)


async def test_a_defect_is_not_a_refusal_and_propagates() -> None:
    with pytest.raises(RuntimeError, match="a bug"):
        await _call(_tool(lambda: RuntimeError("a bug")), {"action": "note_refuse", "note": "x"})


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"action": "note_unheard"},
        {"action": "note_refuse", "note": "x", "extra": 1},
        {"action": "note_refuse", "note": 7},
        {"action": "help", "note": "x"},
        {"action": "help", "topic": "note_unheard"},
    ],
    ids=["no-action", "unknown-action", "undeclared", "invalid", "help-parameter", "help-topic"],
)
async def test_a_call_the_catalogue_turns_away_is_validation_failed(arguments: dict[str, Any]) -> None:
    outcome = await _call(_tool(None), arguments)
    assert outcome.is_error and outcome.refusal == Refusal(kind="validation_failed", status=422)
    assert outcome.refusal == Refusal.of(ValidationFailedError("any", remedy="any other")), (
        "the class's status, not a second copy"
    )


async def test_a_result_and_help_carry_no_refusal() -> None:
    tool = _tool(None)
    result = await _call(tool, {"action": "note_refuse", "note": "x"})
    index = await _call(tool, {"action": "help"})
    assert (result.refusal, result.is_error, result.structured) == (None, False, {})
    assert (index.refusal, index.is_error) == (None, False)


def test_a_kind_outside_the_closed_set_is_refused() -> None:
    with pytest.raises(ValueError, match="refusal kind 'teapot' is none of refused, not_found"):
        Refusal(kind="teapot", status=418)  # type: ignore[arg-type]


def test_an_outcome_is_a_result_or_a_refusal_never_both() -> None:
    with pytest.raises(ValueError, match="a result or a refusal, never both"):
        ActionOutcome("x", structured={}, refusal=Refusal(kind="conflict", status=409))
