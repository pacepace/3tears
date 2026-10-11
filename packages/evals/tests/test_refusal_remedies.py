"""Every refusal names what its caller can do next, and every call a remedy names is one the caller can make.

A consumer's end-to-end test met refusals that named nothing to do — an unknown template ("template 'x' not
found"), a cancel of a job that had already ended — and one whose remedy no caller could reach: a campaign bar
on a template's rubric dimension was refused because "this campaign names no template that could be read", from
an action that took no template. So, over every refusal the engine raises:

- **A not-found and a conflict carry a remedy** (:class:`~threetears.evals.kernel.NotFoundError` and
  :class:`~threetears.evals.kernel.ConflictError` require one), read here from the source of every construction,
  so a new refusal written without one fails here rather than in a consumer's agent loop.
- **Every action a remedy names is in the engine's catalogue**, read from every remedy the engine writes, so no
  remedy points at a call that does not exist.
- **A refused call states its remedy**, and where the remedy names an action the tool does not carry, it says
  which tool does.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import threetears.evals.kernel.remedies as remedies
from threetears.evals.actions import ToolSpec
from threetears.evals.actions.engine import eval_catalogue
from threetears.evals.actions.render import refuse_by_engine
from threetears.evals.kernel import ConflictError, NotFoundError, StorageError, ValidationFailedError
from threetears.evals.kernel.errors import PROVIDER_REMEDY, STORAGE_REMEDY

_SOURCE = Path(__file__).resolve().parents[1] / "src" / "threetears" / "evals"

#: The refusals whose construction must name a remedy.
_REMEDY_REQUIRED = {"NotFoundError", "ConflictError"}

#: A word that could name an action: lowercase parts joined by underscores.
_TOKEN = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")


def _calls() -> list[tuple[str, ast.Call]]:
    """Every construction of an engine refusal in the package's source, by where it is."""
    found = []
    for path in sorted(_SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {*_REMEDY_REQUIRED, "ValidationFailedError"}:
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
    return ""


def _action_like(token: str, verbs: set[str]) -> bool:
    return token.rsplit("_", 1)[-1] in verbs


class TestEveryRefusalNamesARemedy:
    def test_every_not_found_and_conflict_the_engine_raises_names_one(self) -> None:
        missing = [
            where
            for where, call in _calls()
            if call.func.id in _REMEDY_REQUIRED  # type: ignore[attr-defined]
            and not any(
                keyword.arg == "remedy" and not (isinstance(keyword.value, ast.Constant) and not keyword.value.value)
                for keyword in call.keywords
            )
        ]
        assert missing == [], f"refusals naming no remedy: {missing}"

    def test_a_not_found_and_a_conflict_cannot_be_built_without_one(self) -> None:
        with pytest.raises(TypeError):
            NotFoundError("template", "t-1")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            ConflictError("busy")  # type: ignore[call-arg]

    def test_an_unknown_template_says_how_to_find_one(self) -> None:
        refused = NotFoundError("template", "t-1", remedy=remedies.FIND_TEMPLATE)
        assert str(refused) == "template 't-1' not found; templates_list names this scope's templates"

    def test_a_storage_refusal_carries_its_class_remedy(self) -> None:
        assert StorageError("down").remedy == STORAGE_REMEDY


class TestEveryRemedyIsReachable:
    def test_every_action_a_remedy_names_is_in_the_catalogue(self) -> None:
        catalogue = {action.name for action in eval_catalogue().actions}
        verbs = {name.rsplit("_", 1)[-1] for name in catalogue} | {"update", "delete", "create", "list", "get"}
        texts = {
            where: _remedy_text(keyword.value)
            for where, call in _calls()
            for keyword in call.keywords
            if keyword.arg == "remedy"
        }
        texts |= {f"remedies.{name}": getattr(remedies, name) for name in remedies.__all__}
        texts |= {"STORAGE_REMEDY": STORAGE_REMEDY, "PROVIDER_REMEDY": PROVIDER_REMEDY}
        unreachable = sorted(
            f"{where}: {token}"
            for where, text in texts.items()
            for token in _TOKEN.findall(text)
            if _action_like(token, verbs) and token not in catalogue
        )
        assert unreachable == [], f"remedies naming a call nobody can make: {unreachable}"


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
