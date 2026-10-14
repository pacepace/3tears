"""Load a file under ``examples/`` as a module, and run a model-calling example against its recording.

Run as a script, an example reads ``ANTHROPIC_API_KEY``: with it the example calls Claude through
:class:`threetears.evals.providers.Model`, and without it it replays ``examples/recordings/<example>.jsonl``.
A test runs the second way only: :func:`replaying` takes the key out of the environment, makes building a live
client fail the test, and counts every call the example's clients serve from the recording, keeping each refusal a
replay raised so the test can name it rather than read it as a missing result.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from threetears.evals.schema import ClientRequestSettings
from threetears.evals.testing import RecordedCompletions, RecordedPrompt, RecordingRefused

#: The examples directory.
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

#: Every example that calls a model, and so builds its clients live or from its recording.
MODEL_CALLING = ("llm_judge.py",)


def load_example(name: str) -> ModuleType:
    """Execute ``examples/<name>`` as a fresh module.

    Args:
        name: The file's name, ``"llm_judge.py"``.

    Returns:
        The module, its ``main`` not yet run.
    """
    path = EXAMPLES / name
    spec = importlib.util.spec_from_file_location(f"{path.stem}_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def check_covers(module: ModuleType, prompts: Iterable[RecordedPrompt], *, settings: ClientRequestSettings) -> None:
    """Refuse, naming each prompt, when ``module``'s recording no longer covers ``prompts``, before anything runs.

    Args:
        module: A loaded model-calling example.
        prompts: The calls it will make that are known before it runs (each repeat listed once per repeat).
        settings: The request settings the client making them is built with.

    Raises:
        RecordingRefused: The recording does not exist, or no longer covers one of the prompts.
    """
    RecordedCompletions.check_covers(module.RECORDING, list(prompts), settings=settings, rerecord=module.RERECORD)
