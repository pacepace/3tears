"""The ladder's rung directories under ``examples/``, and a model-calling file run against its recording.

The file is one that builds its client as the generate_analysis fixture (``fixtures/generate_analysis/llm_analysis.py``)
does: with ``ANTHROPIC_API_KEY`` it calls Claude through :class:`threetears.evals.providers.Model`, and without it it
replays its recording. A test runs the second way only: :func:`replaying` takes the key out of the environment, makes
building a live client fail the test, and counts every call the file's clients serve from the recording, keeping each
refusal a replay raised so the test can name it rather than read it as a missing result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from threetears.evals.schema import ClientRequestSettings
from threetears.evals.testing import RecordedCompletions, RecordingRefused

#: The examples directory.
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

#: The ladder: every example under ``examples/`` lives in one of these directories, a rung each.
RUNGS = ("1-triage", "2-replies", "3-invoices", "4-booking")


@dataclass
class Replay:
    """What an example's clients served from its recording while a test ran it.

    Attributes:
        calls: One ``(settings, system, user)`` per call served, in the order asked.
        refusals: Each refusal a replay raised mid-run (a prompt the recording does not hold), in the order raised.
    """

    calls: list[tuple[ClientRequestSettings, str, str]] = field(default_factory=list)
    refusals: list[RecordingRefused] = field(default_factory=list)

    def served(self) -> int:
        """How many calls were served, after failing the test by name on any refusal met mid-run.

        Returns:
            The number of calls served.
        """
        if self.refusals:
            pytest.fail(f"the recording no longer serves the example: {self.refusals[0]}")
        return len(self.calls)


def replaying(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Replay:
    """Run ``module`` against its recording: no key, no live client, and every served call counted.

    Args:
        module: A loaded model-calling example.
        monkeypatch: The test's monkeypatch.

    Returns:
        The record of what its clients serve once its ``main`` runs.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("RECORD", raising=False)

    def no_live_client(model: str, **_: Any) -> Any:
        pytest.fail(f"a replay built a live client for {model}")

    monkeypatch.setattr(module, "Model", no_live_client)
    replay, build = Replay(), module.client

    def counted(model: str, settings: ClientRequestSettings) -> Any:
        llm = build(model, settings)
        assert isinstance(llm, RecordedCompletions) and llm.mode == "replay"

        async def generate(*, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
            try:
                completion = await llm.generate(system=system, user=user, response_format=response_format)
            except RecordingRefused as refused:
                replay.refusals.append(refused)  # kept to be named by served(), and raised on as the example meets it
                raise
            replay.calls.append((settings, system, user))
            return completion

        return SimpleNamespace(generate=generate, aclose=llm.aclose, model_name=llm.model_name, mode=llm.mode)

    monkeypatch.setattr(module, "client", counted)
    return replay
