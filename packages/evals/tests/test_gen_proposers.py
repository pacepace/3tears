"""The gen package's rubric proposer, driven with text and a client and nothing of a host.

A host's own facade tests cover the proposers end to end through its
feeds. These pin the package contract a second client relies on: what is sent, in which
order, in which mode; that the server-owned axis is stamped whatever the model wrote; that a
refusal names the axis's proposer; and that the client is released on every exit. Each is
asserted on both axes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.models import RubricProposal
from threetears.evals.gen import propose_draft
from packages.evals.tests.llm_client_fakes import ReleasableClientMixin


class _RecordingClient(ReleasableClientMixin):
    """Records the one prompt pair a proposal sends and answers with canned content."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        self.calls.append({"system": system, "user": user, "response_format": response_format})
        return SimpleNamespace(content=self.content)


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

    proposal = await propose_draft(
        client,
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

    proposal = await propose_draft(
        client, axis=axis, subject_id="s", system_prompt="", subject_feed="F", catalog_feed="C"
    )

    assert [s.axis for s in proposal.new_dim_suggestions] == [axis]


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
    """Both refusal paths raise the structured error, naming the axis's proposer, after the client is released."""
    client = _RecordingClient(content)

    with pytest.raises(ValidationFailedError, match=f"^{refused_by} LLM output {reason}: "):
        await propose_draft(client, axis=axis, subject_id="s", system_prompt="", subject_feed="F", catalog_feed="C")

    assert client.aclose_calls == 1
