"""Question liveness: which declared questions still owe an answer.

`CampaignDesign.live_questions()` resolves two fields -- a question's `retired_at` and whether any
sibling `supersedes` it -- into whether that question still owes an answer. The answer is
load-bearing three times over: a campaign render prints the live count, the generator selects the
questions a memo must resolve from the same call, and the completeness rule refuses a memo silent
about one. A browser panel that restates this rule is pinned against it where that panel ships;
these are the server half's cases.
"""

from __future__ import annotations

from threetears.evals.contracts.declaration import CampaignDesign, ControlDeclaration, Question, SweptAxis
from threetears.evals.contracts.host.values import SweepableValue


def _design(questions: list[Question]) -> CampaignDesign:
    """A minimal declaration carrying the questions under test."""
    return CampaignDesign(
        axes=[SweptAxis(axis_id="chunk_tokens", values=[SweepableValue.of(256, display="256")], rationale="width")],
        questions=questions,
        controls=ControlDeclaration(stimulus="controlled", apparatus="commissioned"),
    )


class TestThePythonSideResolvesEveryCombination:
    """Every way a question can lose its obligation, and the one way it keeps it."""

    def test_a_plain_question_is_live(self) -> None:
        design = _design([Question(id="q1", text="is the wide chunk worth it?")])

        assert [q.id for q in design.live_questions()] == ["q1"]

    def test_a_retired_question_owes_nothing(self) -> None:
        design = _design(
            [Question(id="q1", text="is the wide chunk worth it?", retired_at="2026-08-01T00:00:00+00:00")]
        )

        assert design.live_questions() == []

    def test_a_superseded_question_owes_nothing_and_its_replacement_does(self) -> None:
        """Both halves in one case: the obligation MOVES rather than being duplicated or dropped."""
        design = _design(
            [
                Question(id="q1", text="is the wide chunk worth it?"),
                Question(id="q2", text="is the wide chunk worth it at this cost?", supersedes="q1"),
            ]
        )

        assert [q.id for q in design.live_questions()] == ["q2"]

    def test_liveness_is_read_in_declaration_order(self) -> None:
        """The generator and the MCP count both read this list; an unstable order is two answers."""
        design = _design([Question(id=f"q{n}", text=f"question {n}") for n in range(4)])

        assert [q.id for q in design.live_questions()] == ["q0", "q1", "q2", "q3"]


def test_a_question_retired_and_superseded_at_once_is_not_live() -> None:
    """The one input where the two branches could diverge: retired first, superseded too.

    `retired_at` is tested first and such a question is simply not live. What must never happen is
    it being treated as live: an obligation nothing else thinks exists.
    """
    design = _design(
        [
            Question(id="q1", text="original", retired_at="2026-08-01T00:00:00+00:00"),
            Question(id="q2", text="replacement", supersedes="q1"),
            Question(id="q3", text="also retired", retired_at="2026-08-02T00:00:00+00:00", supersedes="q2"),
        ]
    )

    assert design.live_questions() == [], "every question here has lost its obligation one way or the other"
