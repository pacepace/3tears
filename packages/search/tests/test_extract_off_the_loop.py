"""Extract's CPU work off the event loop, and a page read from one parse.

A page is hundreds of milliseconds of CPU on a small machine: decoding, the
text, the title and the date. Run on the caller's loop, a few concurrent
reads hold it long enough that their own timers fire late and the reads die
at their bound after the network has answered. A caller that passes an
executor gets that work off the loop; one that passes none gets it inline.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import BrokenExecutor, Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import pytest

from threetears.interfaces.media import EXTRACTION_STATUS_COMPLETE
from threetears.search.contracts.candidate import Candidate, Locator
from threetears.search.extract import (
    EXTRACTION_NOTES_FACET,
    EXTRACTION_STATUS_FACET,
    extract,
)
from threetears.search.testing import RecordedFetchTransport, RecordedPage

from packages.search.tests.search_instances import PROVENANCE

SITE = "https://journal.example.org"
ROBOTS = f"{SITE}/robots.txt"
ALLOW = RecordedPage(body=b"User-agent: *\nAllow: /\n", content_type="text/plain")

#: Paragraphs enough for trafilatura to call it an article; distinct, since it folds repeats into one.
PARAGRAPHS = b"".join(
    f"<p>Capybaras range across most of South America, favouring wetlands and slow rivers "
    f"(field note {n}).</p>".encode()
    for n in range(1, 31)
)

ARTICLE = (
    b"<html><head><title>Capybara habitats</title>"
    b'<meta property="article:published_time" content="2026-09-14T08:00:00Z"></head><body><article>'
    + PARAGRAPHS
    + b"</article></body></html>"
)

#: How long a held stage waits for its release before giving up, so a broken test cannot hang the run.
HOLD_SECONDS = 10.0


def candidate(url: str) -> Candidate:
    return Candidate(identity=url, locators=(Locator(url=url, rel="canonical"),), provenance=PROVENANCE)


class HeldExecutor(Executor):
    """Runs what it is given on a thread, only once the test releases it; counts what it was given."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.submitted = 0
        self._threads = ThreadPoolExecutor(max_workers=1)

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        self.submitted += 1

        def held() -> Any:
            if not self.release.wait(HOLD_SECONDS):
                raise TimeoutError("the test never released the held stage")
            return fn(*args, **kwargs)

        return self._threads.submit(held)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.release.set()
        self._threads.shutdown(wait=wait, cancel_futures=cancel_futures)


class BrokenPool(Executor):
    """An executor that has died: every future it hands back has failed, as a crashed process pool's do."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        failed: Future[Any] = Future()
        failed.set_exception(BrokenExecutor("a worker process died"))
        return failed


@pytest.fixture
def processes() -> Iterator[ProcessPoolExecutor]:
    pool = ProcessPoolExecutor(max_workers=1)
    yield pool
    pool.shutdown(wait=True, cancel_futures=True)


@pytest.fixture
def held() -> Iterator[HeldExecutor]:
    executor = HeldExecutor()
    yield executor
    executor.shutdown()


async def until_submitted(executor: HeldExecutor) -> None:
    """Yield to the loop until the stage reaches the executor, within the hold bound."""
    async with asyncio.timeout(HOLD_SECONDS):
        while executor.submitted == 0:
            await asyncio.sleep(0.01)


# --- the executor -----------------------------------------------------------


async def test_a_page_read_in_a_process_pool_matches_the_page_read_inline(processes: ProcessPoolExecutor) -> None:
    """The work crosses a process boundary: its functions and its result pickle, and nothing is lost."""
    page = f"{SITE}/article/42"
    inline = await extract(
        candidate(page), transport=RecordedFetchTransport({ROBOTS: ALLOW, page: RecordedPage(body=ARTICLE)})
    )

    pooled = await extract(
        candidate(page),
        transport=RecordedFetchTransport({ROBOTS: ALLOW, page: RecordedPage(body=ARTICLE)}),
        executor=processes,
    )

    assert pooled.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_COMPLETE
    assert pooled.content is not None and inline.content is not None
    assert pooled.content.text == inline.content.text and "South America" in pooled.content.text
    assert pooled.title == inline.title == "Capybara habitats"
    assert pooled.published_at == inline.published_at
    assert pooled.published_at is not None and pooled.published_at.date().isoformat() == "2026-09-14"
    assert pooled.facets == inline.facets


async def test_a_thin_page_s_frame_is_found_and_read_in_a_process_pool(processes: ProcessPoolExecutor) -> None:
    """The frame scan and the frame's own read go through the same executor."""
    page, frame = f"{SITE}/article/42", f"{SITE}/viewer/42.html"
    menu = (
        b"<html><head><title>Journal</title></head><body><nav><a href='/'>Home</a> <a href='/about'>About</a>"
        b"</nav><iframe src='/viewer/42.html'></iframe></body></html>"
    )
    transport = RecordedFetchTransport(
        {ROBOTS: ALLOW, page: RecordedPage(body=menu), frame: RecordedPage(body=ARTICLE)}
    )

    result = await extract(candidate(page), transport=transport, thin_page_chars=1000, executor=processes)

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_COMPLETE
    assert result.content is not None and "South America" in result.content.text
    notes = result.facets[EXTRACTION_NOTES_FACET]
    assert isinstance(notes, list) and any("held only its menu" in str(note) for note in notes)


async def test_text_read_as_written_goes_through_the_executor_too(held: HeldExecutor) -> None:
    """Decoding and indenting a large JSON answer is CPU work as well, not only the article path."""
    address = f"{SITE}/api/habitats.json"
    transport = RecordedFetchTransport(
        {ROBOTS: ALLOW, address: RecordedPage(body=b'{"habitats": ["wetland"]}', content_type="application/json")}
    )
    held.release.set()

    result = await extract(candidate(address), transport=transport, executor=held)

    assert held.submitted == 1
    assert result.content is not None and result.content.text == '{\n  "habitats": [\n    "wetland"\n  ]\n}'


async def test_the_loop_is_free_while_a_page_is_parsed(held: HeldExecutor) -> None:
    """While the stage is held on the executor, other work on the loop runs: here, this test itself."""
    page = f"{SITE}/article/42"
    transport = RecordedFetchTransport({ROBOTS: ALLOW, page: RecordedPage(body=ARTICLE)})
    reading = asyncio.create_task(extract(candidate(page), transport=transport, executor=held))

    await until_submitted(held)
    assert not reading.done(), "the read finished before its stage was released"
    held.release.set()
    result = await reading

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_COMPLETE
    assert result.title == "Capybara habitats"


async def test_a_read_cancelled_while_its_page_is_parsed_is_cancelled_and_nothing_stands_in(
    held: HeldExecutor,
) -> None:
    """A bound that fires mid-parse cancels the read: no half-read candidate comes back in its place."""
    page = f"{SITE}/article/42"
    transport = RecordedFetchTransport({ROBOTS: ALLOW, page: RecordedPage(body=ARTICLE)})
    reading = asyncio.create_task(extract(candidate(page), transport=transport, executor=held))
    await until_submitted(held)

    reading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reading
    assert reading.cancelled()


async def test_a_broken_executor_is_raised_not_recorded_as_the_page_s_failure() -> None:
    """A dead pool fails every page the same way, so it reaches the caller; it says nothing about the page."""
    page = f"{SITE}/article/42"
    transport = RecordedFetchTransport({ROBOTS: ALLOW, page: RecordedPage(body=ARTICLE)})

    with pytest.raises(BrokenExecutor, match="a worker process died"):
        await extract(candidate(page), transport=transport, executor=BrokenPool())


# --- the date ---------------------------------------------------------------


async def test_a_page_with_no_stated_date_is_dated_by_its_address() -> None:
    """The day in a dated article's address is read; a citation's date in the body is not taken for it."""
    page = f"{SITE}/2020/03/25/obituary/"
    body = (
        b"<html><head><title>Obituary</title></head><body><article>"
        + PARAGRAPHS
        + b"<p>Interview with the drummer, Chain DLK, May 29, 2005.</p></article></body></html>"
    )
    transport = RecordedFetchTransport({ROBOTS: ALLOW, page: RecordedPage(body=body)})

    result = await extract(candidate(page), transport=transport)

    assert result.published_at is not None and result.published_at.date().isoformat() == "2020-03-25"


async def test_a_date_found_only_in_running_text_is_not_taken_for_the_page_s() -> None:
    """A page that states no date of its own has none; a date it mentions in passing is not its date."""
    page = f"{SITE}/members/atkins.html"
    body = (
        b"<html><head><title>Members</title></head><body><article>"
        + PARAGRAPHS
        + b"<p>Interview with the drummer, Chain DLK, May 29, 2005.</p></article></body></html>"
    )
    transport = RecordedFetchTransport({ROBOTS: ALLOW, page: RecordedPage(body=body)})

    result = await extract(candidate(page), transport=transport)

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_COMPLETE
    assert result.published_at is None


# --- what a page's read absorbs ----------------------------------------------

LOGGER = "threetears.search.extract"


class PickyError(Exception):
    """An exception whose class does not survive pickling: its constructor needs a keyword ``args`` lacks."""

    def __init__(self, message: str, *, detail: str) -> None:
        super().__init__(message)
        self.detail = detail


def break_the_title_read_with_a_runtime_error() -> None:
    """A worker initializer: the page's title and date read raises, in the worker only."""
    import trafilatura.metadata

    def broken(tree: object) -> None:
        raise RuntimeError("the meta-tag reader broke")

    trafilatura.metadata.examine_meta = broken


def break_the_title_read_with_an_unpicklable_error() -> None:
    """A worker initializer: the read raises an exception that cannot be sent back as it is."""
    import trafilatura.metadata

    def broken(tree: object) -> None:
        raise PickyError("the meta-tag reader broke", detail="unpicklable")

    trafilatura.metadata.examine_meta = broken


def contained_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The records ``contain()`` wrote, in this process, from the extract module."""
    return [
        record
        for record in caplog.records
        if record.name == LOGGER and "contained_site" in (getattr(record, "extra_data", None) or {})
    ]


@pytest.mark.parametrize(
    ("initializer", "exc_type", "raised"),
    [
        pytest.param(break_the_title_read_with_a_runtime_error, "RuntimeError", "RuntimeError", id="picklable"),
        pytest.param(break_the_title_read_with_an_unpicklable_error, "RuntimeError", "PickyError", id="unpicklable"),
    ],
)
async def test_a_failure_absorbed_in_a_worker_process_is_recorded_in_the_caller_s(
    caplog: pytest.LogCaptureFixture, initializer: Callable[[], None], exc_type: str, raised: str
) -> None:
    """A worker's own log and counters are not the host's: the host records what the worker absorbed."""
    page = f"{SITE}/article/42"
    transport = RecordedFetchTransport({ROBOTS: ALLOW, page: RecordedPage(body=ARTICLE)})
    pool = ProcessPoolExecutor(max_workers=1, initializer=initializer)
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = await extract(candidate(page), transport=transport, executor=pool)
    finally:
        pool.shutdown(wait=True)

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_COMPLETE
    assert result.content is not None and "South America" in result.content.text
    assert result.title is None and result.published_at is None
    assert result.facets[EXTRACTION_NOTES_FACET] == [
        f"The page's own title and date could not be read ({raised}); it was read without them."
    ]
    [record] = contained_records(caplog)
    assert record.getMessage() == "a page's metadata could not be read; the page is read without it"
    extra = record.extra_data  # type: ignore[attr-defined]
    assert extra["exc_type"] == exc_type
    assert extra["stage_pid"] != os.getpid()
    assert raised in extra["stage_traceback"] and "the meta-tag reader broke" in extra["stage_traceback"]


async def test_a_failure_absorbed_on_a_thread_is_recorded_once(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A thread shares the caller's process, so the stage's own record is the host's, and is not repeated."""
    import trafilatura.metadata

    def broken(tree: object) -> None:
        raise RuntimeError("the meta-tag reader broke")

    monkeypatch.setattr(trafilatura.metadata, "examine_meta", broken)
    page = f"{SITE}/article/42"
    transport = RecordedFetchTransport({ROBOTS: ALLOW, page: RecordedPage(body=ARTICLE)})
    with ThreadPoolExecutor(max_workers=1) as threads, caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await extract(candidate(page), transport=transport, executor=threads)

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_COMPLETE
    assert len(contained_records(caplog)) == 1


async def test_a_page_whose_json_ld_cannot_be_read_keeps_its_text_title_and_date(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JSON-LD is the part of a page's metadata that breaks; the title comes from its other markup instead."""
    import trafilatura.metadata

    def broken(tree: object, metadata: object) -> None:
        raise RuntimeError("the JSON-LD reader broke")

    monkeypatch.setattr(trafilatura.metadata, "extract_meta_json", broken)
    page = f"{SITE}/article/42"
    transport = RecordedFetchTransport({ROBOTS: ALLOW, page: RecordedPage(body=ARTICLE)})
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await extract(candidate(page), transport=transport)

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_COMPLETE
    assert result.content is not None and "South America" in result.content.text
    assert result.title == "Capybara habitats"
    assert result.published_at is not None and result.published_at.date().isoformat() == "2026-09-14"
    assert EXTRACTION_NOTES_FACET not in result.facets
    [record] = contained_records(caplog)
    assert record.getMessage() == "a page's JSON-LD could not be read; its title is read from its other markup"
