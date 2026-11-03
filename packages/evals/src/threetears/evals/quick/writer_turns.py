"""Recorded writer models: an arm's report writer sessions, replayed from one file per case, or recorded once.

A writer arm's model factory (:data:`~threetears.evals.ops.report_writer.writer_contract.WriterModels`, by the
candidate model's id) built by :func:`writer_turns`. Each cell's model routes every session of a report (the
analyst's, each editor round's) to the recording of the case the cell writes over, found by the snapshot id the
session reads:

    ``<recordings>/<arm>-<skill12>-<model>-<effort>/<case>.jsonl``

where ``skill12`` is the first 12 hex digits of the arm's skill digest and ``effort`` is the effort word, or
``default`` for none.

- **The directory carries what a recording does not check.** A turn recording keeps the replies alone and never what
  was sent (:class:`~threetears.evals.testing.RecordedTurns`), so one recorded under one skill, model or effort would
  replay under another without a word. Keying the directory by all three keeps each recording to the condition that
  made it.
- **A file that exists is replayed, with no key.** It is read through
  :meth:`RecordedTurns.replay <threetears.evals.testing.RecordedTurns.replay>` directly, never through
  :class:`~threetears.evals.providers.Model`, which asks for ``ANTHROPIC_API_KEY`` even to replay.
- **A missing file is recorded when the key is set**, through ``Model(id, provider="anthropic",
  turn_recording=path)``, and refused when it is not: the refusal names the file and ``ANTHROPIC_API_KEY``. A refusal
  is a :class:`~threetears.evals.testing.RecordingRefused`, an apparatus fault, so its cell is excluded, never scored.
- **One file serves one cell.** A second cell on the same case would replay the first one's report as a repeat, so
  it is refused: a recorded arm runs each case once.
- **Drift is disclosed, never silent.** :meth:`WriterTurns.unconsumed` names each session a replay left recorded
  lines of, for the caller to print once its cells have run.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from threetears.evals.analysis.analyst.skill import AnalystSkill
from threetears.evals.kernel.eval_file import beside_eval_file
from threetears.evals.ops.report_writer.loop import TurnModel
from threetears.evals.ops.report_writer.writer_contract import EffortWord, WriterModels
from threetears.evals.schema.turns import ToolTurnClient
from threetears.evals.testing.recorded_completions import RecordingRefused
from threetears.evals.testing.recorded_turns import RecordedTurns

if TYPE_CHECKING:
    from threetears.evals.providers import Model

#: The variable a recording run reads its key from.
KEY_VARIABLE: Final = "ANTHROPIC_API_KEY"

#: How many hex digits of the skill digest a recording directory carries.
_SKILL_DIGITS: Final = 12

#: The directory's effort for an arm that names none.
_NO_EFFORT: Final = "default"


class _Sessions(Protocol):
    """What hands out a case's named sessions: a replay, or a model recording one."""

    def turn_session(self, name: str, *, snapshot: str) -> ToolTurnClient: ...


@dataclass(frozen=True, slots=True)
class TurnsLeft:
    """One session a replay left recorded lines of: the code now stops sooner than it did when it was recorded.

    :ivar case: The case the recording is for.
    :ivar session: The session's name (``analyst``, ``editor-1``).
    :ivar left: How many recorded lines were never asked: replies never served and price ceilings never read.
    :ivar path: The recording.
    """

    case: str
    session: str
    left: int
    path: Path

    def __str__(self) -> str:
        return (
            f"case {self.case!r}: session {self.session!r} left {self.left} recorded lines unasked in {self.path}, "
            "so the code now stops sooner than it did when it was recorded"
        )


def _segment(value: str, what: str) -> str:
    """``value`` as one path segment, or the refusal naming ``what`` it is."""
    if not value.strip() or value in {".", ".."} or any(sep in value for sep in ("/", "\\", "\0")):
        raise ValueError(f"the {what} {value!r} names a recording path segment, so it must be non-blank with no '/'")
    return value


class WriterTurns:
    """One arm's model factory over its recordings: by candidate model id, the arm's
    :data:`~threetears.evals.ops.report_writer.writer_contract.WriterModels`.

    Built by :func:`writer_turns`. Read :meth:`unconsumed` after the arm's cells have run, and close it
    (:meth:`aclose`) to close every model it built to record.
    """

    def __init__(self, recordings: Path, *, arm: str, skill: AnalystSkill, cases: Mapping[str, str]) -> None:
        self._recordings = recordings
        self._arm = _segment(arm, "arm label")
        self._skill = skill.digest[:_SKILL_DIGITS]
        named: dict[str, str] = {}
        for snapshot, case in cases.items():
            if (other := named.get(_segment(case, "case name"))) is not None:
                raise ValueError(f"snapshots {other} and {snapshot} are both case {case!r}, which is one recording")
            named[case] = snapshot
        self._cases = dict(cases)
        self._served: set[Path] = set()
        self._replays: list[tuple[str, RecordedTurns]] = []
        self._recorders: list[Model] = []

    def path(self, model: str, effort: EffortWord | None, case: str) -> Path:
        """The recording one case's sessions are routed to, for a model at an effort.

        :param model: The candidate model's id.
        :ptype model: str
        :param effort: The arm's reasoning effort, or ``None`` for the model's own.
        :ptype effort: EffortWord | None
        :param case: The case's name.
        :ptype case: str
        :return: ``<recordings>/<arm>-<skill12>-<model>-<effort>/<case>.jsonl``.
        :rtype: Path
        :raises ValueError: The model id or case name is no single path segment.
        """
        condition = f"{self._arm}-{self._skill}-{_segment(model, 'model id')}-{effort or _NO_EFFORT}"
        return self._recordings / condition / f"{_segment(case, 'case name')}.jsonl"

    def __call__(self, model: str) -> WriterModels:
        """The arm's model factory for one candidate model: a fresh routed model per cell, at the arm's effort.

        :param model: The candidate model's id.
        :ptype model: str
        :return: The factory, by effort.
        :rtype: WriterModels
        :raises ValueError: The model id is no single path segment.
        """
        _segment(model, "model id")

        def at(effort: EffortWord | None) -> TurnModel:
            missing = self._missing(model, effort) if os.environ.get(KEY_VARIABLE) else []
            # A recording run's settings are refused here, when the cell is prepared, before any turn is sent.
            probe = self._recorder(model, effort, missing[0]) if missing else None
            return _CellModel(model, lambda snapshot: self._sessions(model, effort, snapshot), probe)

        return at

    def unconsumed(self) -> list[TurnsLeft]:
        """Each session a replay left recorded lines of, over every replay opened so far.

        :return: The sessions, in the order their replays were opened; empty when every replay ran as recorded.
        :rtype: list[TurnsLeft]
        """
        return [
            TurnsLeft(case, session, left, replay.path)
            for case, replay in self._replays
            for session, left in replay.unconsumed().items()
        ]

    async def aclose(self) -> None:
        """Close every model built to record a case, and to price or check a recording run's settings."""
        recorders, self._recorders = self._recorders, []
        for recorder in recorders:
            await recorder.aclose()

    def _missing(self, model: str, effort: EffortWord | None) -> list[Path]:
        """The recordings of the arm's cases that do not exist yet."""
        paths = (self.path(model, effort, case) for case in self._cases.values())
        return [path for path in paths if not path.is_file()]

    def _recorder(self, model: str, effort: EffortWord | None, path: Path) -> Model:
        """A model that records ``path``'s sessions live, on the key the environment holds; closed by :meth:`aclose`."""
        from threetears.evals.providers import Model  # the completion client, loaded only to record a missing case

        built = Model(model, provider="anthropic", effort=effort, turn_recording=path)
        self._recorders.append(built)
        return built

    def _sessions(self, model: str, effort: EffortWord | None, snapshot: str) -> _Sessions:
        """What serves one cell's sessions over ``snapshot``: its case's replay, or a model recording it."""
        case = self._cases.get(snapshot)
        if case is None:
            raise RecordingRefused(
                self.path(model, effort, "_").parent,
                f"snapshot {snapshot} is no case of arm {self._arm!r}",
                "map every snapshot the arm writes over to its case's name",
            )
        path = self.path(model, effort, case)
        if path in self._served:
            raise RecordingRefused(
                path,
                f"case {case!r} was already served to a cell of this arm, and a second cell would repeat its report",
                "run each recorded case once (k_runs=1)",
            )
        self._served.add(path)
        if path.is_file():
            replay = RecordedTurns.replay(path)
            self._replays.append((case, replay))
            return replay
        if not os.environ.get(KEY_VARIABLE):
            raise RecordingRefused(
                path,
                f"there is no such turn recording, and {KEY_VARIABLE} is not set to record it",
                f"restore the file, or set {KEY_VARIABLE} to record it once against the live model",
            )
        return self._recorder(model, effort, path)


class _CellModel:
    """One cell's model: each session routed to its case's recording by the snapshot it reads."""

    def __init__(self, model: str, route: Callable[[str], _Sessions], probe: Model | None) -> None:
        self.model_name = model
        self._route = route
        self._probe = probe
        self._by_snapshot: dict[str, _Sessions] = {}

    async def turn_pricing_refusal(self) -> str | None:
        """``None`` when every session replays, which costs nothing; else the recording model's own answer."""
        if self._probe is None:
            return None
        return await self._probe.turn_pricing_refusal()

    def turn_session(self, name: str, *, snapshot: str) -> ToolTurnClient:
        """The named session over ``snapshot``, from its case's replay or recording."""
        sessions = self._by_snapshot.get(snapshot)
        if sessions is None:
            sessions = self._by_snapshot[snapshot] = self._route(snapshot)
        return sessions.turn_session(name, snapshot=snapshot)


def writer_turns(
    recordings: str | os.PathLike[str], *, arm: str, skill: AnalystSkill, cases: Mapping[str, str]
) -> WriterTurns:
    """One writer arm's recorded model factory, for the writer kind's ``models``.

    :param recordings: The directory every arm's recordings sit under; a relative path is read beside the eval file
        being loaded, else in the working directory.
    :ptype recordings: str | os.PathLike[str]
    :param arm: The arm's label, which leads its recordings' directory name.
    :ptype arm: str
    :param skill: The skill the arm runs, whose digest's first 12 hex digits the directory name carries.
    :ptype skill: AnalystSkill
    :param cases: Snapshot id -> case name, for every case the arm writes over; a case's recording is named for it.
    :ptype cases: Mapping[str, str]
    :return: The factory.
    :rtype: WriterTurns
    :raises ValueError: The arm label or a case name is no single path segment, or two snapshots name one case.
    """
    return WriterTurns(beside_eval_file(recordings).absolute(), arm=arm, skill=skill, cases=cases)


__all__ = ["KEY_VARIABLE", "TurnsLeft", "WriterTurns", "writer_turns"]
