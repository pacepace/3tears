"""Every refusal names what its caller can do next, and every call a remedy names is one the caller can make.

A consumer's end-to-end test met refusals that named nothing to do — an unknown template ("template 'x' not
found"), a cancel of a job that had already ended — and one whose remedy no caller could reach: a campaign bar
on a template's rubric dimension was refused because "this campaign names no template that could be read", from
an action that took no template. So, over every refusal the engine raises:

- **Every refusal carries a structured remedy.** Every :class:`~threetears.evals.kernel.EvalServiceError` requires
  one (:class:`~threetears.evals.kernel.StorageError`, :class:`~threetears.evals.kernel.ProviderRefusedError` and
  :class:`~threetears.evals.kernel.AdmissionRefusedError` supply their own), and so does every lower layer's
  :class:`~threetears.evals.kernel.RefusedInput` a surface translates into one. Read here from the source of every
  construction, so a refusal written without one fails here rather than in a consumer's agent loop.
- **A translation keeps the lower layer's remedy.** A refusal built from a caught :class:`RefusedInput` or a caught
  refusal hands its remedy on, never its words alone.
- **Every action a remedy names is in the engine's catalogue**, read from every remedy the engine writes, so no
  remedy points at a call that does not exist.
- **A refused call states its remedy**, and where the remedy names an action the tool does not carry, it says
  which tool does.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import pydantic

import pytest

import threetears.evals
import threetears.evals.kernel.errors as errors
from threetears.evals import PUBLIC_ROOTS
import threetears.evals.kernel.remedies as remedies
from threetears.evals.actions import ToolSpec
from threetears.evals.actions.engine import eval_catalogue
from threetears.evals.actions.render import refuse_by_engine
from threetears.evals.analysis.lenses.export import ExportError
from threetears.evals.analysis.lenses.frontier import FrontierError
from threetears.evals.analysis.lenses.history import HistoryError
from threetears.evals.analysis.lenses.pivot import PivotError
from threetears.evals.kernel import (
    AdmissionRefusedError,
    ConflictError,
    DesignRefused,
    EvalServiceError,
    NotFoundError,
    ProviderRefusedError,
    RefusedInput,
    StorageError,
    UndeclarableAxisError,
    ValidationFailedError,
)
from threetears.evals.kernel.errors import PROVIDER_REMEDY, STORAGE_REMEDY
from threetears.evals.kernel.status_filter import StatusFilterError
from threetears.evals.run import CeilingRaisedError

_SOURCE = Path(__file__).resolve().parents[1] / "src" / "threetears" / "evals"

#: Every refusal class whose construction names its remedy at the call: the service layer's, and every lower
#: layer's refusal a surface translates into one.
_REMEDY_AT_THE_CALL: dict[str, type[Exception]] = {
    cls.__name__: cls
    for cls in (
        EvalServiceError,
        NotFoundError,
        ConflictError,
        ValidationFailedError,
        RefusedInput,
        DesignRefused,
        UndeclarableAxisError,
        CeilingRaisedError,
        PivotError,
        HistoryError,
        ExportError,
        FrontierError,
        StatusFilterError,
    )
}

#: The refusal classes that supply their own remedy, so their constructions name none.
_REMEDY_OF_THEIR_OWN = {StorageError, ProviderRefusedError, AdmissionRefusedError}

#: A word that could name an action: lowercase parts joined by underscores.
_TOKEN = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")


def _calls() -> list[tuple[str, ast.Call]]:
    """Every construction of a refusal in the package's source, by where it is."""
    found = []
    for path in sorted(_SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _REMEDY_AT_THE_CALL:
                found.append((f"{path.relative_to(_SOURCE)}:{node.lineno}", node))
    return found


def _remedy_text(node: ast.expr) -> str:
    """A remedy's words as written: a literal's text, an f-string's literal parts and the constants it names."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return str(getattr(remedies, node.id, ""))
    if isinstance(node, ast.JoinedStr):
        return " ".join(
            _remedy_text(part.value if isinstance(part, ast.FormattedValue) else part) for part in node.values
        )
    if isinstance(node, ast.BinOp):
        return f"{_remedy_text(node.left)} {_remedy_text(node.right)}"
    return ""


def _action_like(token: str, verbs: set[str]) -> bool:
    return token.rsplit("_", 1)[-1] in verbs


class TestEveryRefusalNamesARemedy:
    def test_every_refusal_the_engine_raises_names_one_at_the_call(self) -> None:
        missing = [
            where
            for where, call in _calls()
            if not any(
                keyword.arg == "remedy" and not (isinstance(keyword.value, ast.Constant) and not keyword.value.value)
                for keyword in call.keywords
            )
        ]
        assert missing == [], f"refusals naming no remedy: {missing}"

    def test_a_translated_refusal_hands_on_its_remedy_never_its_words_alone(self) -> None:
        """``ValidationFailedError(str(e))`` over a caught refusal puts the remedy in the message: refused."""
        flattened = [
            where
            for where, call in _calls()
            if call.args
            and isinstance(call.args[0], ast.Call)
            and isinstance(call.args[0].func, ast.Name)
            and call.args[0].func.id == "str"
            and any(keyword.arg == "remedy" and isinstance(keyword.value, ast.Attribute) for keyword in call.keywords)
        ]
        assert flattened == [], f"translations repeating the remedy inside the message: {flattened}"

    def test_every_public_refusal_class_requires_one_or_supplies_its_own(self) -> None:
        public = {getattr(errors, name) for name in errors.__all__}
        refusals = {cls for cls in public if isinstance(cls, type) and issubclass(cls, EvalServiceError)}
        assert refusals == {*(cls for cls in _REMEDY_AT_THE_CALL.values() if issubclass(cls, EvalServiceError))} | (
            _REMEDY_OF_THEIR_OWN
        ), "a refusal class was added: decide whether it names its remedy at the call or supplies its own"

    @pytest.mark.parametrize(
        "build",
        [
            lambda: NotFoundError("template", "t-1"),  # type: ignore[call-arg]
            lambda: ConflictError("busy"),  # type: ignore[call-arg]
            lambda: ValidationFailedError("bad"),  # type: ignore[call-arg]
            lambda: EvalServiceError("no"),  # type: ignore[call-arg]
            lambda: PivotError("no such axis"),  # type: ignore[call-arg]
            lambda: DesignRefused("no such bar"),  # type: ignore[call-arg]
        ],
    )
    def test_a_refusal_cannot_be_built_without_one(self, build) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(TypeError):
            build()

    @pytest.mark.parametrize(
        "build",
        [
            lambda: ValidationFailedError("bad", remedy=" "),
            lambda: NotFoundError("template", "t-1", remedy=""),
            lambda: RefusedInput("bad", remedy=""),
        ],
    )
    def test_a_blank_remedy_is_refused(self, build) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="names its remedy"):
            build()

    def test_an_unknown_template_says_how_to_find_one(self) -> None:
        refused = NotFoundError("template", "t-1", remedy=remedies.FIND_TEMPLATE)
        assert str(refused) == "template 't-1' not found; templates_list names this scope's templates"

    def test_the_refusals_with_a_remedy_of_their_own_carry_it(self) -> None:
        assert StorageError("down").remedy == STORAGE_REMEDY
        assert ProviderRefusedError("rate limited").remedy == PROVIDER_REMEDY
        admission = AdmissionRefusedError(requested=3, admitted=4, limit=5, limit_name="EVAL_MAX")
        assert admission.remedy.startswith("wait for runs to finish") and "EVAL_MAX" in admission.remedy
        assert "wait for runs" not in admission.message.lower()

    def test_a_lower_layers_refusal_reads_as_both_halves(self) -> None:
        refused = PivotError("unknown factor 'x'", remedy="pivot on a lever")
        assert (refused.fault, refused.remedy, str(refused)) == (
            "unknown factor 'x'",
            "pivot on a lever",
            "unknown factor 'x'; pivot on a lever",
        )


#: A word written like a call or a field: lowercase parts joined by underscores, not qualified by a dotted prefix.
_IDENTIFIER = re.compile(r"(?<![\w.])[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?![\w])")

#: A public name written qualified by the public root a host imports it from (``run.start_run``,
#: ``analysis.update_campaign()``): the one way a remedy names a Python call, so its reader knows where it is.
_QUALIFIED = re.compile(r"(?<![\w.])(?:threetears\.evals\.)?((?:[a-z_]+\.)*[a-z_]+)\.([A-Za-z_][A-Za-z0-9_]*)(?![\w])")


#: A value quoted in a remedy, as an argument's value is written: ``'mean_score'``.
_QUOTED = re.compile(r"'[^'\s]*'")


def _module_strings() -> dict[str, str]:
    """Every module-level string constant in the package, by name: what a remedy written as a name says."""
    found: dict[str, str] = {}
    for path in sorted(_SOURCE.rglob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            value = getattr(node, "value", None)
            text = _literal(value) if value is not None else ""
            for target in targets:
                if isinstance(target, ast.Name) and text:
                    found[target.id] = text
    return found


def _literal(node: ast.expr | None) -> str:
    """A string expression's literal words: a constant, an f-string's literal parts, a join, both branches of a choice."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return " ".join(_literal(part.value if isinstance(part, ast.FormattedValue) else part) for part in node.values)
    if isinstance(node, ast.BinOp):
        return f"{_literal(node.left)} {_literal(node.right)}"
    if isinstance(node, ast.IfExp):
        return f"{_literal(node.body)} {_literal(node.orelse)}"
    if isinstance(node, ast.FormattedValue):
        return _literal(node.value)
    return ""


def _remedies_written() -> dict[str, str]:
    """Every remedy the package writes, by where: wherever it is built, not only at a refusal's construction.

    A remedy handed to any call as ``remedy=`` (a literal, an f-string, or a name, read at the constant it names);
    every constant, attribute or variable named for a remedy (``FIND_RUN``, ``self.remedy = ...``); and every
    string a function named for one returns from (``registry.axis_remedy(...)`` is read where it is defined). A
    remedy handed on from a caught refusal (``remedy=e.remedy``) is read where that refusal was built.
    """
    constants = _module_strings()
    written: dict[str, str] = {}
    for path in sorted(_SOURCE.rglob("*.py")):
        where = path.relative_to(_SOURCE)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.keyword) and node.arg == "remedy":
                text = constants.get(node.value.id, "") if isinstance(node.value, ast.Name) else _literal(node.value)
                if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
                    text = text or constants.get(node.value.attr, "")
                written[f"{where}:{node.value.lineno}"] = text
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                named = [
                    target.id if isinstance(target, ast.Name) else target.attr
                    for target in targets
                    if isinstance(target, ast.Name | ast.Attribute)
                ]
                if any("remedy" in name.lower() or name.startswith("FIND_") for name in named):
                    written[f"{where}:{node.lineno}"] = _literal(node.value)
            elif isinstance(node, ast.FunctionDef) and "remedy" in node.name:
                # Every string it builds its remedy from, its docstring aside: a remedy assembled from clauses is
                # read clause by clause.
                body = node.body[1:] if ast.get_docstring(node) is not None else node.body
                built = [
                    part
                    for statement in body
                    for part in ast.walk(statement)
                    if isinstance(part, ast.JoinedStr)
                    or (isinstance(part, ast.Constant) and isinstance(part.value, str))
                ]
                written[f"{where}:{node.lineno} {node.name}()"] = " ".join(_literal(part) for part in built)
    written |= {f"remedies.{name}": getattr(remedies, name) for name in remedies.__all__}
    written |= {"STORAGE_REMEDY": STORAGE_REMEDY, "PROVIDER_REMEDY": PROVIDER_REMEDY}
    return written


def _declared_words() -> set[str]:
    """What a remedy may name that is no call: a field or parameter some model, record or function declares, or a
    value the package spells as a constant (a kind's name)."""
    words: set[str] = set()
    for info in pkgutil.walk_packages(threetears.evals.__path__, "threetears.evals."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:
            continue  # an extra this environment lacks: its words are read where it is installed
        for value in vars(module).values():
            if isinstance(value, str) and _IDENTIFIER.fullmatch(value):
                words.add(value)
            if getattr(value, "__module__", None) != module.__name__:
                continue
            if isinstance(value, type) and issubclass(value, pydantic.BaseModel):
                words |= set(value.model_fields)
            elif isinstance(value, type) and dataclasses.is_dataclass(value):
                words |= {field.name for field in dataclasses.fields(value)}
            if callable(value):
                try:
                    words |= set(inspect.signature(value).parameters)
                except TypeError, ValueError:
                    continue  # a builtin or a class with no signature to read
    return words


class TestEveryRemedyIsReachable:
    def test_every_call_a_remedy_names_is_one_its_caller_can_make(self) -> None:
        """Each identifier is an action in the catalogue, a public name written qualified by its root, or a word the
        package declares (a parameter, a field, a kind's name) — never a bare Python name, which a caller reaching the
        refusal through an action cannot call, and which ``add_runs_to_campaign`` and ``start_run`` were."""
        catalogue = {action.name for action in eval_catalogue().actions}
        declared = _declared_words()
        exports = {root: set(importlib.import_module(root).__all__) for root in PUBLIC_ROOTS}
        unreachable: list[str] = []
        for where, text in _remedies_written().items():
            for qualifier, name in _QUALIFIED.findall(text):
                root = f"threetears.evals.{qualifier}"
                if root in exports and name not in exports[root]:
                    unreachable.append(f"{where}: {qualifier}.{name} (not exported from {root})")
            # A quoted word is a value the caller types ('mean_score'), not a call it makes.
            bare = _QUOTED.sub(" ", _QUALIFIED.sub(" ", text))
            unreachable += [
                f"{where}: {token}"
                for token in _IDENTIFIER.findall(bare)
                if token not in catalogue and token not in declared
            ]
        assert sorted(unreachable) == [], f"remedies naming a call nobody can make: {sorted(unreachable)}"

    def test_the_reach_check_reads_names_calls_and_functions(self) -> None:
        """Where a remedy is written as a name, a call or a function's return, its words are read, not skipped."""
        written = _remedies_written()
        assert any(text == remedies.FIND_RUN for text in written.values()), "a remedy=FIND_RUN reads as its words"
        assert any(where.endswith("axis_remedy()") and text for where, text in written.items()), (
            "a function building a remedy is read at its definition"
        )


class TestARefusedCallStatesItsRemedy:
    def _tool(self, permissions: frozenset[str]):  # type: ignore[no-untyped-def]
        return eval_catalogue().mount(ToolSpec(name="evals", description="Evals.", permissions=permissions))  # type: ignore[arg-type]

    def test_the_remedy_follows_the_reason(self) -> None:
        tool = self._tool(frozenset({"read"}))
        action = tool.action("run_get")
        assert action is not None
        text = refuse_by_engine(tool, action, NotFoundError("run", "r-1", remedy=remedies.FIND_RUN))
        assert text.splitlines() == ["refused: run_get: run 'r-1' not found", f"remedy: {remedies.FIND_RUN}"]

    def test_a_remedy_naming_an_action_on_another_tool_says_where_it_lives(self) -> None:
        tool = self._tool(frozenset({"read"}))
        action = tool.action("job_poll")
        assert action is not None
        refused = ValidationFailedError("job 'x' is still running", remedy="cancel it with job_cancel")
        text = refuse_by_engine(tool, action, refused)
        cancel = eval_catalogue().get("job_cancel")
        assert cancel is not None and tool.action("job_cancel") is None
        assert f"job_cancel is a {cancel.permission} action, which tool evals does not carry" in text
