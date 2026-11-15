"""Re-key a recorded judge's calls from their prompts to their inputs, in place, for $0: recording format 3 to 4.

Dev tooling for ``3tears-evals``, run once by hand; it lives beside the package, not in it, and is not installed.

    uv run python packages/evals/scripts/rekey_judge_recording.py            # rung 5's judge.jsonl, rewritten
    uv run python packages/evals/scripts/rekey_judge_recording.py --check    # say what it would do, write nothing

A judge call over evidence that names its inputs (a report writer's page) is keyed by those inputs since format 4
(``testing/recorded_completions.py``), so a word printed differently on the page moves no recording. A recording made
before then keys each call by its prompt, which a format-3 line does not keep: only its key (the prompt's digest) and
the prompt's first 80 characters. So the request behind each line is recovered by replaying the example at a commit
whose pages still match every stored key, which is what makes this a one-off. For each call the replay asks, it:

1. records the call once, with no network, into a scratch recording through ``RecordedCompletions.record`` with no
   inputs bound: the key the recorder writes must be one the file holds, which proves the call is that line's request;
2. records it again with the call's inputs bound: the key the recorder writes is the line's new key.

Every line must be matched by exactly one call and every call by one line, the new keys must be distinct, and a line's
other members are written back exactly as read. Each line then gains ``inputs``, ``prompt_key`` (its old key, the
key its prompt had, so a replay can tell a page rendered otherwise since) and the format it is now in, 4; its
``result``, ``extra``, ``cell``, ``index``, ``settings``, ``sent_settings``, ``user_head`` and ``recorded_at`` are
unchanged, so every replayed response is byte for byte what was recorded. Nothing is sent to a provider: the replay
serves the recording, and the scratch recordings answer from a stub.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from threetears.evals.quick.files import main as run_example
from threetears.evals.schema import ClientRequestSettings
from threetears.evals.schema.call_inputs import bind_call_inputs, current_call_inputs
from threetears.evals.testing import RecordedCompletions

#: The package's directory.
PACKAGE = Path(__file__).resolve().parents[1]

#: The example whose judge recording this re-keys, and the recording, relative to it.
EXAMPLE = PACKAGE / "examples" / "5-writer"
RECORDING = Path("recordings") / "judge.jsonl"

#: The environment a replay must run without, so nothing can reach a provider.
_LIVE = ("RECORD", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")


@dataclass(frozen=True, slots=True)
class _Asked:
    """One call the replay asked the judge: its prompt and the inputs bound around it."""

    system: str
    user: str
    response_format: dict[str, Any] | None
    inputs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Stub:
    """What a scratch recording is answered with: the recording's model and temperature, nothing else."""

    content: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    price_source: str | None
    model: str
    served_model: str | None
    reasoning_tokens: int | None
    temperature: float | None
    stop_reason: str


class _StubClient:
    """A client that answers every call with one stub, so a scratch recording costs nothing."""

    def __init__(self, model: str, temperature: float | None) -> None:
        self._reply = _Stub("", None, None, None, None, model, None, None, temperature, "end_turn")

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Stub:
        return self._reply

    async def aclose(self) -> None:
        return None


def _asked_calls(example: Path) -> list[_Asked]:
    """Every call the example's replay asks its judge, in a scratch copy of it, with no key and no network."""
    captured: list[_Asked] = []
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / example.name
        shutil.copytree(example, copy, ignore=shutil.ignore_patterns(".evals", "report", "__pycache__"))
        target = (copy / RECORDING).resolve()
        replayed = RecordedCompletions.generate

        async def capturing(
            self: RecordedCompletions, *, system: str, user: str, response_format: dict[str, Any] | None = None
        ) -> Any:
            if self.path.resolve() == target:
                inputs = current_call_inputs()
                if inputs is None:
                    raise SystemExit(f"a judge call named no inputs ({user[:80]!r}): nothing to re-key it by")
                captured.append(_Asked(system, user, response_format, json.loads(json.dumps(dict(inputs)))))
            return await replayed(self, system=system, user=user, response_format=response_format)

        RecordedCompletions.generate = capturing  # type: ignore[method-assign]
        here = Path.cwd()
        for name in _LIVE:
            os.environ.pop(name, None)
        try:
            os.chdir(copy)
            sys.path.insert(0, str(copy))
            code = run_example(["writer.py"])
        finally:
            os.chdir(here)
            sys.path.remove(str(copy))
            RecordedCompletions.generate = replayed  # type: ignore[method-assign]
        if code != 0:
            raise SystemExit(f"the example's replay exited {code}: its pages no longer match every recorded prompt")
    return captured


async def _key(
    call: _Asked,
    scratch: Path,
    settings: ClientRequestSettings | None,
    model: str,
    temperature: float | None,
    *,
    by_inputs: bool,
) -> str:
    """The key the recorder writes for ``call``: by its prompt, or with its inputs bound."""
    path = scratch / f"{len(list(scratch.iterdir()))}.jsonl"
    async with RecordedCompletions.record(_StubClient(model, temperature), path, settings=settings) as recorder:
        if by_inputs:
            with bind_call_inputs(call.inputs):
                await recorder.generate(system=call.system, user=call.user, response_format=call.response_format)
        else:
            await recorder.generate(system=call.system, user=call.user, response_format=call.response_format)
    (line,) = path.read_text(encoding="utf-8").splitlines()
    key: str = json.loads(line)["key"]
    return key


async def _rekeyed(lines: list[dict[str, Any]], asked: list[_Asked]) -> dict[str, tuple[str, dict[str, Any]]]:
    """Each stored key's new key and inputs, refusing anything that is not one line to one call."""
    model, temperature = lines[0]["result"]["model"], lines[0]["result"]["temperature"]
    settings_of = {line["key"]: line["settings"] for line in lines}
    stored = set(settings_of)
    mapping: dict[str, tuple[str, dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory() as scratch:
        for call in asked:
            # Every line under one key carries one settings value; try each the file holds until one matches.
            matched = None
            for raw in {json.dumps(value, sort_keys=True): value for value in settings_of.values()}.values():
                settings = None if raw is None else ClientRequestSettings.model_validate(raw)
                old = await _key(call, Path(scratch), settings, model, temperature, by_inputs=False)
                if old in stored:
                    matched = (old, settings)
                    break
            if matched is None:
                raise SystemExit(f"a call the replay asked matches no stored key ({call.user[:80]!r})")
            old, settings = matched
            new = await _key(call, Path(scratch), settings, model, temperature, by_inputs=True)
            if old in mapping and mapping[old] != (new, call.inputs):
                raise SystemExit(f"stored key {old[:12]} was asked under two sets of inputs")
            mapping[old] = (new, call.inputs)
    if missing := stored - set(mapping):
        raise SystemExit(f"{len(missing)} stored key(s) no call asked: {sorted(k[:12] for k in missing)}")
    if len({new for new, _ in mapping.values()}) != len(mapping):
        raise SystemExit("two stored keys re-key to one: the inputs do not tell their calls apart")
    if clash := {new for new, _ in mapping.values()} & stored:
        raise SystemExit(f"a new key is already a stored one: {sorted(k[:12] for k in clash)}")
    return mapping


def _rewrite(text: str, mapping: Mapping[str, tuple[str, dict[str, Any]]]) -> tuple[str, list[dict[str, Any]]]:
    """The recording re-keyed, and its lines as read; every member but ``key``, ``recorder_version`` and the added
    ``inputs`` and ``prompt_key`` (the line's old key) written back as read."""
    raw = text.splitlines()
    read = [json.loads(line) for line in raw]
    out: list[str] = []
    for original, line in zip(raw, read, strict=True):
        if json.dumps(line, ensure_ascii=False, allow_nan=False) != original:
            raise SystemExit("a line does not serialise back to its own bytes; refusing to rewrite it")
        if line["recorder_version"] != 3:
            raise SystemExit(f"a line is in format {line['recorder_version']}, not 3: already re-keyed?")
        new, inputs = mapping[line["key"]]
        rekeyed = {**line, "recorder_version": 4, "key": new, "inputs": inputs, "prompt_key": line["key"]}
        out.append(json.dumps(rekeyed, ensure_ascii=False, allow_nan=False))
    return "".join(f"{line}\n" for line in out), read


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="say what would change, and write nothing")
    args = parser.parse_args(argv)
    path = EXAMPLE / RECORDING
    text = path.read_text(encoding="utf-8")
    lines = [json.loads(line) for line in text.splitlines()]
    asked = _asked_calls(EXAMPLE)
    mapping = asyncio.run(_rekeyed(lines, asked))
    rewritten, _ = _rewrite(text, mapping)
    after = [json.loads(line) for line in rewritten.splitlines()]
    unchanged = all(
        {k: v for k, v in a.items() if k not in ("key", "recorder_version")}
        == {k: v for k, v in b.items() if k not in ("key", "recorder_version", "inputs", "prompt_key")}
        for a, b in zip(lines, after, strict=True)
    )
    print(f"{path.relative_to(PACKAGE)}: {len(lines)} line(s) before, {len(after)} after; {len(asked)} call(s) asked")
    print(f"members other than key, recorder_version and the added inputs and prompt_key unchanged: {unchanged}")
    if not unchanged or len(lines) != len(after):
        return 1
    if not args.check:
        path.write_text(rewritten, encoding="utf-8")
        print("rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
