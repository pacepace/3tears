"""``Log``: the runner for an app that cannot be wrapped as a function — its outputs pushed, then graded.

Some apps cannot be called once per case from an eval: a pipeline, a service in another language, a UI. Their
outputs are pushed instead. ``Log("outputs.jsonl")`` is a runner whose outputs are what was logged:

- **The app logs each output** where it appears: ``log(case, output, cost_usd=..., model=...)``, once per repeat. The
  case is named by its ``id``, so a case logged to carries one.
- **Another process, in any language, may append** ``{"id": ..., "output": ...}`` lines to the same file (with
  ``cost_usd`` and ``model`` when it knows them).
- **The eval grades what was logged.** The ``k``-th repeat of a case is graded on the ``k``-th output logged for it,
  in the order the file holds them. A case with fewer than ``k`` outputs has each missing repeat excluded with cause
  ``not_logged`` — counted and named among the run's exclusions, never dropped and never read as a failure of the app.

An output logged with its spend reports it, as an :class:`~threetears.evals.quick.answer.Answer` would; one logged
without it reports none, which the run states.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from typing import Any

from threetears.evals.kernel.eval_file import beside_eval_file
from threetears.evals.kernel.host import ApparatusError
from threetears.evals.quick.answer import Answer

#: What an exclusion for a repeat no output was logged for is called, first in its reason.
NOT_LOGGED = "not_logged"


class NotLogged(ApparatusError):
    """A repeat of a case no output was logged for: the rig's gap, so its cell is excluded, never failed."""


class Log:
    """A runner whose outputs an app pushed to a JSONL file, and the call that pushes one.

    :ivar path: The file.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Name the file the outputs are logged to and read from.

        :param path: The JSONL file; created on the first output logged. A relative path is read beside the eval file
            being loaded, else in the working directory.
        :ptype path: str | os.PathLike[str]
        """
        self.path = beside_eval_file(path)
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        """The log, by file."""
        return f"Log({str(self.path)!r})"

    def __call__(
        self, case: Mapping[str, Any], output: Any, *, cost_usd: float | None = None, model: str | None = None
    ) -> None:
        """Log one output for one case: one repeat of it.

        :param case: The case it answers, named by its ``id``.
        :ptype case: Mapping[str, Any]
        :param output: The output, a JSON value.
        :ptype output: Any
        :param cost_usd: What producing it cost, in US dollars; ``None`` when unpriced, which the run states.
        :ptype cost_usd: float | None
        :param model: The model that produced it, when there was one.
        :ptype model: str | None
        :raises ValueError: A case with no ``id``, an output JSON cannot hold, or a spend no usage row can hold.
        """
        name = case.get("id") if isinstance(case, Mapping) else None
        if isinstance(name, bool) or not isinstance(name, str | int) or not str(name).strip():
            raise ValueError(f"a logged output names its case by the case's id, and {case!r} carries none")
        Answer(value=output, model=model, cost_usd=cost_usd)
        line = {"id": str(name), "output": output, "cost_usd": cost_usd, "model": model}
        try:
            text = json.dumps(line, sort_keys=True)
        except (TypeError, ValueError) as unencodable:
            raise ValueError(f"the output logged for case {name} is not JSON: {unencodable}") from unencodable
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as logged:
                logged.write(text + "\n")

    def outputs(self) -> dict[str, list[Answer]]:
        """Every output logged, by case id, in the order the file holds them.

        :return: Each case's outputs, as answers carrying their spend; empty when nothing has been logged yet.
        :rtype: dict[str, list[Answer]]
        :raises ValueError: A line that is not a JSON object with a non-blank ``id`` and an ``output``, named by number.
        """
        if not self.path.is_file():
            return {}
        logged: dict[str, list[Answer]] = {}
        for number, text in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not text.strip():
                continue
            try:
                line = json.loads(text)
            except ValueError as unreadable:
                raise ValueError(f"{self.path} line {number} is not JSON: {unreadable}") from unreadable
            name = line.get("id") if isinstance(line, dict) else None
            unnamed = isinstance(name, bool) or not isinstance(name, str | int) or not str(name).strip()
            if unnamed or "output" not in line:
                raise ValueError(f"{self.path} line {number} is not an {{'id', 'output'}} object: {text[:80]}")
            logged.setdefault(str(name), []).append(
                Answer(value=line["output"], model=line.get("model"), cost_usd=line.get("cost_usd"))
            )
        return logged

    def runner(self, logged: Mapping[str, list[Answer]], case: str, repeat: int) -> Any:
        """The runner one cell is served by: the ``repeat``-th output logged for ``case``, or its gap.

        :param logged: What :meth:`outputs` read when the run started.
        :ptype logged: Mapping[str, list[Answer]]
        :param case: The case's name.
        :ptype case: str
        :param repeat: Which repeat the cell is, from 1.
        :ptype repeat: int
        :return: An async function of the case returning the output.
        :rtype: Any
        """
        held = logged.get(case, [])

        async def logged_output(_case: Mapping[str, Any]) -> Answer:
            if repeat > len(held):
                raise NotLogged(
                    f"{NOT_LOGGED}: case {case} has {len(held)} output(s) logged in {self.path}, and this is repeat "
                    f"{repeat}"
                )
            return held[repeat - 1]

        logged_output.__name__ = self.path.stem
        return logged_output


__all__ = ["NOT_LOGGED", "Log", "NotLogged"]
