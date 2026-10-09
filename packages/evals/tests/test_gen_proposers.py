"""The gen package's rubric proposer, driven with text and a client and nothing of a host.

A host's own facade tests cover the proposers end to end through its
feeds. These pin the package contract a second client relies on: what is sent, in which
order, in which mode; that the server-owned axis is stamped whatever the model wrote; that a
refusal names the axis's proposer; and that the client is released on every exit. Each is
asserted on both axes.

The call runs outside any run, so it goes through an out-of-run budget: priced on the client before
it is made — refused, with nothing called, when the cap cannot pay for it or the client cannot price
it — and ledgered once made, the refused drafts included, since they were paid for.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from threetears.evals.contracts import EvalStorage, OutOfRunBudget
from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.models import RubricProposal
from threetears.evals.gen import propose_draft
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.llm_client_fakes import ReleasableClientMixin

_SCOPE = "proposals"

#: What one proposal call is priced at, and what it is reported to have cost.
_CEILING = 0.05
_COST = 0.0123


class _RecordingClient(ReleasableClientMixin):
    """Records the one prompt pair a proposal sends and answers with canned content, priced at ``ceiling``."""

    model_name = "proposer-model"

    def __init__(self, content: str, *, ceiling: float | None = _CEILING, raises: BaseException | None = None) -> None:
        self.content = content
        self.ceiling = ceiling
        self.raises = raises
        self.calls: list[dict[str, Any]] = []
        self.priced: list[dict[str, Any]] = []

    def price_ceiling(self, *, system: str, user: str, response_format: Any = None) -> float | None:
        self.priced.append({"system": system, "user": user, "response_format": response_format})
        return self.ceiling

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        self.calls.append({"system": system, "user": user, "response_format": response_format})
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(
            content=self.content,
            model=self.model_name,
            served_model="proposer-model-2026",
            input_tokens=900,
            output_tokens=300,
            reasoning_tokens=None,
            cost_usd=_COST,
            price_source="provider",
            stop_reason="end_turn",
        )


def _budget(*, cap_usd: float | None = 1.0, subject_id: str | None = None) -> tuple[OutOfRunBudget, EvalStorage]:
    storage = EvalStorage(InMemoryDocumentStore())
    return OutOfRunBudget(
        storage, scope_id=_SCOPE, cap_usd=cap_usd, subject_id=subject_id, blocking_executor=None
    ), storage


def _draft(*, suggestion_axis: str) -> str:
    """A valid draft whose one new-dim suggestion carries ``suggestion_axis``."""
    return json.dumps(
        {
            "template": {"name": "T", "intent": "i", "rubric": [], "variation_axes": []},
            "reused_dim_keys": [],
            "new_dim_suggestions": [
                {
                    "key": "graceful_decline",
                    "dim": {
                        "name": "boundary.graceful_decline",
                        "description": "declines without stonewalling",
                        "scale": "pass_fail",
                    },
                    "axis": suggestion_axis,
                    "universal": True,
                }
            ],
        }
    )


_AXES = [
    pytest.param("capability", id="capability"),
    pytest.param("boundary", id="boundary"),
]


@pytest.mark.parametrize("axis", _AXES)
async def test_the_feeds_are_sent_verbatim_subject_first_in_json_mode(axis):
    """The user message is the subject feed, then the catalog feed, ending in one newline."""
    client = _RecordingClient(_draft(suggestion_axis=axis))
    budget, _storage = _budget()

    proposal, _spend = await propose_draft(
        client,
        budget=budget,
        axis=axis,
        subject_id="subject-1",
        system_prompt="SYSTEM TEXT",
        subject_feed="# Subject\n\nName: S\n",
        catalog_feed="# Catalog\n\n(empty)\n\n",
    )

    assert isinstance(proposal, RubricProposal)
    assert client.calls == [
        {
            "system": "SYSTEM TEXT",
            "user": "# Subject\n\nName: S\n\n# Catalog\n\n(empty)\n",
            "response_format": {"type": "json_object"},
        }
    ]
    assert client.aclose_calls == 1


@pytest.mark.parametrize(
    ("axis", "written"),
    [
        pytest.param("capability", "boundary", id="capability-overrides-boundary"),
        pytest.param("boundary", "capability", id="boundary-overrides-capability"),
        pytest.param("capability", "graceful_decline", id="capability-overrides-a-dim-key"),
        pytest.param("boundary", "graceful_decline", id="boundary-overrides-a-dim-key"),
    ],
)
async def test_the_proposer_stamps_its_own_axis_whatever_the_model_wrote(axis, written):
    """The axis is the one the proposer ran on, never the model's call — both directions on one draft.

    Each axis is fed the other one's literal as well as a dim key, so a proposer that kept a
    valid literal it was handed, or stamped a fixed axis whatever it ran on, fails here.
    """
    client = _RecordingClient(_draft(suggestion_axis=written))

    proposal, _spend = await propose_draft(
        client, budget=_budget()[0], axis=axis, subject_id="s", system_prompt="", subject_feed="F", catalog_feed="C"
    )

    assert [s.axis for s in proposal.new_dim_suggestions] == [axis]


@pytest.mark.parametrize("axis", ["capability", "boundary"])
async def test_the_drafted_rubric_dims_take_the_proposers_axis_too(axis):
    """A boundary battery's own rubric dims are guardrails, so the judge must stamp them so; a draft that
    left them capability would put them in the composite they must stay out of."""
    draft = json.loads(_draft(suggestion_axis="graceful_decline"))
    dim = draft["new_dim_suggestions"][0]["dim"]
    draft["template"]["rubric"] = [{**dim, "axis": "capability" if axis == "boundary" else "boundary"}]
    client = _RecordingClient(json.dumps(draft))

    proposal, _spend = await propose_draft(
        client, budget=_budget()[0], axis=axis, subject_id="s", system_prompt="", subject_feed="F", catalog_feed="C"
    )

    assert [d.axis for d in proposal.template.rubric] == [axis]
    assert [s.dim.axis for s in proposal.new_dim_suggestions] == [axis]


@pytest.mark.parametrize(
    ("axis", "refused_by"),
    [
        pytest.param("capability", "proposer", id="capability"),
        pytest.param("boundary", "boundary proposer", id="boundary"),
    ],
)
@pytest.mark.parametrize(
    ("content", "reason"),
    [
        pytest.param("I can't help with that.", "was not valid JSON", id="not-json"),
        pytest.param(
            json.dumps({"template": {"name": "x"}, "reused_dim_keys": []}), "failed draft validation", id="not-a-draft"
        ),
    ],
)
async def test_a_refused_draft_raises_and_still_releases_the_client(axis, refused_by, content, reason):
    """Both refusal paths raise the structured error, naming the axis's proposer and the call's cost, after release.

    The call was paid for whatever the reply said, so its ledger row is written before the reply is read.
    """
    client = _RecordingClient(content)
    budget, storage = _budget()

    with pytest.raises(
        ValidationFailedError, match=rf"^{refused_by} LLM output {reason} \(the call cost \$0\.0123\): "
    ):
        await propose_draft(
            client, budget=budget, axis=axis, subject_id="s", system_prompt="", subject_feed="F", catalog_feed="C"
        )

    assert client.aclose_calls == 1
    [spend] = storage.query_out_of_run_spend(_SCOPE)
    assert (spend.purpose, spend.outcome, spend.cost_usd) == ("proposer", "completed", _COST)


# =============================================================================
# The call is out-of-run spend: priced before it is made, ledgered once it is
# =============================================================================


async def _propose(client: _RecordingClient, budget: OutOfRunBudget, *, subject_id: str = "s") -> Any:
    return await propose_draft(
        client,
        budget=budget,
        axis="capability",
        subject_id=subject_id,
        system_prompt="SYS",
        subject_feed="F",
        catalog_feed="C",
    )


async def test_a_proposal_returns_and_ledgers_what_its_call_cost():
    client = _RecordingClient(_draft(suggestion_axis="capability"))
    budget, storage = _budget(subject_id="s")

    _proposal, spend = await _propose(client, budget)

    assert client.priced == client.calls, "the call priced is the call made"
    assert storage.query_out_of_run_spend(_SCOPE) == [spend]
    assert storage.query_out_of_run_spend(_SCOPE, purpose="proposer") == [spend]
    assert storage.query_out_of_run_spend(_SCOPE, purpose="variation") == []
    assert spend.model_dump(include={"purpose", "model", "served_model", "outcome", "stop_reason"}) == {
        "purpose": "proposer",
        "model": "proposer-model",
        "served_model": "proposer-model-2026",
        "outcome": "completed",
        "stop_reason": "end_turn",
    }
    assert (spend.prompt_tokens, spend.completion_tokens, spend.reasoning_tokens) == (900, 300, None)
    assert (spend.cost_usd, spend.price_source) == (_COST, "provider")
    assert (spend.priced_ceiling_usd, spend.cap_usd, spend.subject_id) == (_CEILING, 1.0, "s")


async def test_a_call_priced_above_the_cap_is_refused_before_it_is_made():
    client = _RecordingClient(_draft(suggestion_axis="capability"), ceiling=2.5)
    budget, storage = _budget(cap_usd=1.0)

    with pytest.raises(ValidationFailedError, match=r"priced at up to \$2\.5000, above the out-of-run cap \$1\.00"):
        await _propose(client, budget)

    assert client.calls == [] and storage.query_out_of_run_spend(_SCOPE) == []
    assert client.aclose_calls == 1, "a refused admission still releases the client"


async def test_a_call_the_client_cannot_price_is_refused_under_an_enforced_cap():
    client = _RecordingClient(_draft(suggestion_axis="capability"), ceiling=None)
    budget, storage = _budget(cap_usd=1.0)

    with pytest.raises(ValidationFailedError, match="cannot be priced before they are made.*unknown is not \\$0"):
        await _propose(client, budget)

    assert client.calls == [] and storage.query_out_of_run_spend(_SCOPE) == []


async def test_with_no_enforced_cap_an_unpriced_call_is_made_and_ledgered_unpriced():
    client = _RecordingClient(_draft(suggestion_axis="capability"), ceiling=None)
    budget, storage = _budget(cap_usd=None)

    _proposal, spend = await _propose(client, budget)

    assert len(client.calls) == 1
    assert (spend.priced_ceiling_usd, spend.cap_usd) == (None, None)
    assert storage.query_out_of_run_spend(_SCOPE) == [spend]


async def test_a_call_that_raises_is_ledgered_with_its_failure_class_and_no_usage():
    client = _RecordingClient("", raises=TimeoutError("provider said: account acct_123 timed out"))
    budget, storage = _budget()

    with pytest.raises(TimeoutError):
        await _propose(client, budget)

    [spend] = storage.query_out_of_run_spend(_SCOPE)
    assert (spend.outcome, spend.failure, spend.cost_usd, spend.stop_reason) == ("raised", "TimeoutError", None, None)
    assert "acct_123" not in spend.model_dump_json(), "the exception's text, which can carry the account, is never kept"
    assert client.aclose_calls == 1


async def test_a_budget_for_another_subject_is_refused_before_anything_is_priced():
    client = _RecordingClient(_draft(suggestion_axis="capability"))
    budget, storage = _budget(subject_id="someone-else")

    with pytest.raises(ValueError, match="ledgers its calls under subject 'someone-else'"):
        await _propose(client, budget, subject_id="s")

    assert client.priced == [] and client.calls == [] and storage.query_out_of_run_spend(_SCOPE) == []
