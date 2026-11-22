"""Recorded web search: a web search port that records a live search once and replays it exactly, offline.

Claims grading's web check sends its queries to a host's port (:class:`~threetears.evals.schema.WebSearch`).
:meth:`RecordedWebSearch.record` wraps the host's real port and appends every search it answers to a JSONL file;
:meth:`RecordedWebSearch.replay` reads that file and answers each search from it, with no network.

**What a search is matched on: the port's identity and the query, exactly.** A recording holds one port's searches,
under its identity, which the replay answers under too, so a replayed run's spend names the same search. A query the
judge writes the same way again is answered with what it was answered first: a recording keeps one answer per query,
the first it saw, so replays are the same whatever order a run's cells search in. Recording, a query the file already
holds is answered from it too, so the recording run grades on what its replay serves.

**A miss is refused, never searched live.** A query the recording does not hold, a missing or unreadable file, a line
from a newer format, or a second identity recorded onto one file, raises
:class:`~threetears.evals.testing.RecordingRefused`, a :class:`~threetears.evals.kernel.host.ReplayMiss`: the cell is
recorded ``apparatus_failed`` naming the search, and the run fails with it, as a replayed completion's miss does.

**Replayed grades say so.** Every answer the recording serves says it was replayed
(:attr:`~threetears.evals.schema.WebSearchResult.replayed`), and a grade it decided is stored as replayed
(:attr:`~threetears.evals.schema.ClaimGrade.web_replayed`): no live search stands behind it. An answer the recording
serves reports it cost nothing (``usd`` 0.0), which is what it is billed at whatever rate the run declares; replaying,
:meth:`RecordedWebSearch.price_ceiling` is ``0.0``.

Line format 1: ``{"format": 1, "identity": str, "query": str, "pages": [{"url", "title", "text", "snippet_only"}]}``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Final, Self

from threetears.evals.schema.web_search import WebPage, WebSearch, WebSearchResult
from threetears.evals.testing.recorded_completions import RecordingRefused

__all__ = ["RecordedWebSearch"]

#: The line format this build writes and reads.
_FORMAT: Final = 1

#: Serialises every append in this process, so two recorders on one file never interleave a line.
_APPEND_LOCK = threading.Lock()


class RecordedWebSearch:
    """A web search port that records a host's searches to a file, or replays them from one.

    Build it with :meth:`record` or :meth:`replay`, and hand it in where the port goes
    (:attr:`~threetears.evals.run.RunnerOptions.web_search`).
    """

    def __init__(
        self,
        path: Path,
        *,
        inner: WebSearch | None,
        identity: str | None,
        recorded: dict[str, tuple[WebPage, ...]],
        rerecord: str | None = None,
    ) -> None:
        """Bind the recording, the port it records (None to replay), and what it holds.

        Build it through :meth:`record` or :meth:`replay`.

        :param path: The recording.
        :param inner: The host's port, recorded; None when replaying.
        :param identity: The identity the recording holds; None for a recording that holds none yet.
        :param recorded: Each query's first answer, as the recording holds it.
        :param rerecord: The command that re-records ``path``, quoted in every refusal; None for the general remedy.
        """
        self._path = path
        self._inner = inner
        self._identity = identity
        self._recorded = recorded
        self._remedy = (
            f"re-record it ({rerecord})"
            if rerecord is not None
            else "re-record it: build RecordedWebSearch.record over the live port and run once"
        )
        self._lock = threading.Lock()

    @classmethod
    def record(cls, inner: WebSearch, path: Path, *, replace: bool = False) -> Self:
        """A port that sends every search to ``inner`` and appends each query's first answer to ``path``.

        A recording that exists is read first and appended to: one this build cannot read, or one of another
        identity, is refused before any search is sent.

        :param inner: The host's real port.
        :param path: The recording. Its directory is created when missing.
        :param replace: Remove ``path`` first and record afresh.
        :return: The recording port.
        :raises RecordingRefused: ``path`` holds a line this build cannot read, or another port's searches.
        """
        if replace:
            path.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        remedy = "remove it, or record with replace=True"
        identity, recorded = _read(path, remedy) if path.exists() else (None, {})
        if identity is not None and identity != inner.identity:
            raise RecordingRefused(
                path,
                f"it holds the searches of {identity!r}, and {inner.identity!r} would record onto it",
                "record each port into a file of its own",
            )
        return cls(path, inner=inner, identity=inner.identity, recorded=recorded)

    @classmethod
    def replay(cls, path: Path, *, rerecord: str | None = None) -> Self:
        """A port that answers every search from the recording at ``path``, with no network.

        :param path: The recording.
        :param rerecord: The command that re-records ``path``, quoted in every refusal; None for the general remedy.
        :return: The replaying port.
        :raises RecordingRefused: ``path`` does not exist, holds no search, holds a line this build cannot read, or
            holds one from a newer format.
        """
        remedy = f"re-record it ({rerecord})" if rerecord is not None else "re-record it over the live port"
        if not path.exists():
            raise RecordingRefused(path, "it does not exist", remedy)
        identity, recorded = _read(path, remedy)
        if identity is None:
            raise RecordingRefused(path, "it holds no search", remedy)
        return cls(path, inner=None, identity=identity, recorded=recorded, rerecord=rerecord)

    @property
    def identity(self) -> str:
        """The recorded port's identity: what a replayed run's spend names its searches under."""
        if self._identity is None:
            raise RuntimeError("a recorded web search is built through record or replay, which name its identity")
        return self._identity

    def price_ceiling(self) -> float | None:
        """The most one search can cost: the live port's ceiling when recording, ``0.0`` when replaying."""
        if self._inner is None:
            return 0.0
        return self._inner.price_ceiling()

    async def search(self, query: str) -> WebSearchResult:
        """Answer ``query``: from the recording when it holds it, else from the live port, whose answer is recorded.

        An answer the recording serves, replaying or recording, costs nothing and says it was replayed, so a
        recording run grades a repeated query on the answer its replay will serve. A live answer is the inner port's,
        as it returned it.

        :param query: The search.
        :return: Its result.
        :raises RecordingRefused: Replaying, and the recording holds no answer for ``query``.
        """
        with self._lock:
            held = self._recorded.get(query)
        if held is not None:
            return WebSearchResult(pages=held, usd=0.0, replayed=True)
        if self._inner is None:
            raise RecordingRefused(
                self._path, f"it holds no search {query!r} for the web search {self.identity!r}", self._remedy
            )
        result = await self._inner.search(query)
        with self._lock:
            if query not in self._recorded:
                self._recorded[query] = result.pages
                _append(self._path, self.identity, query, result.pages)
        return result


def _append(path: Path, identity: str, query: str, pages: tuple[WebPage, ...]) -> None:
    line = {
        "format": _FORMAT,
        "identity": identity,
        "query": query,
        "pages": [
            {"url": page.url, "title": page.title, "text": page.text, "snippet_only": page.snippet_only}
            for page in pages
        ],
    }
    with _APPEND_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")


def _read(path: Path, remedy: str) -> tuple[str | None, dict[str, tuple[WebPage, ...]]]:
    """The identity a recording holds and each query's first answer.

    :raises RecordingRefused: A line will not parse, is of a newer format, or names a second identity.
    """
    identity: str | None = None
    recorded: dict[str, tuple[WebPage, ...]] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as unreadable:
        raise RecordingRefused(path, f"it cannot be read ({unreadable})", remedy) from unreadable
    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            line: Any = json.loads(raw)
            if not isinstance(line, dict):
                raise ValueError("a line is a JSON object")
            if line.get("format") != _FORMAT:
                raise ValueError(f"format {line.get('format')!r}, and this build reads format {_FORMAT}")
            named = line["identity"]
            query = line["query"]
            if not isinstance(named, str) or not isinstance(query, str):
                raise ValueError("its identity and query are strings")
            pages = tuple(
                WebPage(url=page["url"], title=page["title"], text=page["text"], snippet_only=page["snippet_only"])
                for page in line["pages"]
            )
        except (ValueError, KeyError, TypeError) as broken:
            raise RecordingRefused(path, f"line {number} cannot be read ({broken})", remedy) from broken
        if identity is not None and named != identity:
            raise RecordingRefused(
                path, f"line {number} names {named!r}, and the recording holds {identity!r}'s searches", remedy
            )
        identity = named
        recorded.setdefault(query, pages)
    return identity, recorded
