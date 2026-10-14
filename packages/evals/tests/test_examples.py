"""What every example under ``examples/`` holds to, read across all of them.

Each example that calls a model builds its clients one way: with ``ANTHROPIC_API_KEY`` set,
``Model(MODEL, provider="anthropic", ...)`` from the request settings it declares, recorded into
``recordings/<example>.jsonl`` when ``RECORD=1``; without it, a replay of that file. There is no other path: no
scripted stand-in and no shared helper module. The examples call one model, ``claude-haiku-5-5``, and send it no
temperature, which it takes none of. No test here calls the API or touches a committed recording.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from threetears.evals.schema import ClientRequestSettings
from threetears.evals.testing import RecordedCompletions
from packages.evals.tests.example_loader import EXAMPLES, MODEL_CALLING, load_example
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

#: The one model the examples call.
MODEL = "claude-haiku-5-5"


def _settings_of(module: ModuleType) -> list[ClientRequestSettings]:
    """Every request setting ``module`` builds a client with."""
    declared = [getattr(module, name, None) for name in ("SETTINGS", "JUDGE_SETTINGS")]
    return [settings for settings in declared if settings is not None] + list(getattr(module, "EFFORTS", {}).values())


def test_every_example_reaches_the_engine_only_through_public_roots() -> None:
    examples = [(path.relative_to(EXAMPLES).as_posix(), path) for path in sorted(EXAMPLES.rglob("*.py"))]
    assert {"1-triage/triage.py", "2-replies/replies.py"} <= {name for name, _ in examples}
    assert public_root_violations(SOURCE_ROOT, examples, consumer_root=REPO_ROOT) == []


def test_the_model_calling_examples_are_exactly_those_that_build_a_client() -> None:
    builds = {path.name for path in EXAMPLES.glob("*.py") if "live = Model(" in path.read_text(encoding="utf-8")}
    assert builds == set(MODEL_CALLING)
    assert not list(EXAMPLES.glob("_*.py")), "every example stands alone: no shared module beside them"


def test_the_examples_call_one_model() -> None:
    named = {
        found
        for path in [*EXAMPLES.glob("*.py"), EXAMPLES / "README.md"]
        for found in re.findall(r"claude-[a-z0-9.-]*\d", path.read_text(encoding="utf-8"))
    }
    assert named == {MODEL}
    assert {load_example(name).MODEL for name in MODEL_CALLING} == {MODEL}


@pytest.mark.parametrize("name", MODEL_CALLING)
def test_each_replays_its_own_recording_and_names_the_command_that_records_it(name: str) -> None:
    module = load_example(name)
    assert module.RECORDING == EXAMPLES / "recordings" / f"{Path(name).stem}.jsonl"
    assert module.RERECORD == f"RECORD=1 python {name}"  # the refusal adds "the provider's key set"
    assert f"RECORD=1 python {name}" in (EXAMPLES / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", MODEL_CALLING)
def test_live_each_client_is_built_from_its_request_settings_with_no_temperature(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = load_example(name)
    monkeypatch.setattr(module, "RECORDING", tmp_path / "recording.jsonl")  # never the committed one
    built: list[tuple[str, dict[str, Any]]] = []

    def model_client(model: str, **kwargs: Any) -> Any:
        built.append((model, kwargs))
        return object()

    monkeypatch.setattr(module, "Model", model_client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")
    monkeypatch.delenv("RECORD", raising=False)
    for settings in _settings_of(module):
        assert not isinstance(module.client(module.MODEL, settings), RecordedCompletions), "live and not recording"
        monkeypatch.setenv("RECORD", "1")
        recorder = module.client(module.MODEL, settings)
        assert isinstance(recorder, RecordedCompletions) and recorder.mode == "record"
        assert recorder.path == module.RECORDING
        monkeypatch.delenv("RECORD")
    assert built, "every model-calling example declares its request settings"
    for model, kwargs in built:
        assert model == MODEL and kwargs["provider"] == "anthropic" and kwargs["api_key"] == "sk-not-a-real-key"
        assert "temperature" not in kwargs, f"{MODEL} takes no temperature"
        sent = (kwargs.get("max_tokens"), kwargs.get("effort"))
        declared = [(s.max_tokens, s.reasoning_effort or s.reasoning_max_tokens) for s in _settings_of(module)]
        assert sent in declared, f"{name} built a client from settings it does not declare: {sent}"


@pytest.mark.parametrize("name", MODEL_CALLING)
def test_recording_with_no_key_is_refused_rather_than_replayed(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_example(name)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("RECORD", "1")
    with pytest.raises(SystemExit, match="RECORD=1 records a live run, which needs ANTHROPIC_API_KEY"):
        module.client(module.MODEL, _settings_of(module)[0])


def test_the_recordings_have_a_home() -> None:
    assert (EXAMPLES / "recordings" / "README.md").is_file()
