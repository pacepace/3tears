"""The web search port: what claims grading's web check asks a host's search for, and what it reads back.

A claim criterion with a web budget (:class:`~threetears.evals.schema.models.WebBudget`) checks, on the web, the claims
its judge nominated by its own suspicion: claims no case evidence bears on that the judge doubts. The engine writes
the searches, cuts windows from the pages a search returns, and asks its judge whether a window states the claim; a
host's port does only the search. The engine never calls a search provider itself and never names one.

A run hands its port in on its options (:attr:`~threetears.evals.run.RunnerOptions.web_search`). The engine records
every page window it read on the cell's trace and never searches for a stored grade again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

__all__ = ["WebPage", "WebSearch", "WebSearchResult"]


@dataclass(frozen=True)
class WebPage:
    """One result a search returned, as the port read it.

    :ivar url: Where the page is, as the search named it.
    :ivar title: Its title; empty when the search named none.
    :ivar text: What the port read of it: the page's text, or the result's snippet alone (``snippet_only``).
    :ivar snippet_only: The port read only the search result's snippet, not the page.
    """

    url: str
    title: str
    text: str
    snippet_only: bool = False

    def __post_init__(self) -> None:
        """Refuse a page with no address, or fields of the wrong type.

        :raises TypeError: A field is not of its type.
        :raises ValueError: ``url`` is blank.
        """
        for name in ("url", "title", "text"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"a web page's {name} is a string, and this one's is {type(getattr(self, name))}")
        if not isinstance(self.snippet_only, bool):
            raise TypeError(f"a web page's snippet_only is a bool, and this one's is {type(self.snippet_only)}")
        if not self.url.strip():
            raise ValueError("a web page names where it is: its url is blank")


@dataclass(frozen=True)
class WebSearchResult:
    """What one search answered: its results, and what the port knows of how it answered.

    :ivar pages: The results, in the search's own order.
    :ivar usd: What this search actually cost, in dollars, when the port knows it; None when it does not, and the
        search is billed at a rate the run declares, else at the port's ``price_ceiling``.
    :ivar replayed: The results came from a recording, not a live search: the grades they decide are marked
        replayed, so a check that needs a live search over them is not made.
    """

    pages: tuple[WebPage, ...]
    usd: float | None = None
    replayed: bool = False

    def __post_init__(self) -> None:
        """Refuse results that are not pages, a cost that is no amount, or a flag that is not a bool.

        :raises TypeError: ``pages`` is not a tuple of :class:`WebPage`, or ``replayed`` is not a bool.
        :raises ValueError: ``usd`` is negative or not finite.
        """
        if not isinstance(self.pages, tuple) or not all(isinstance(page, WebPage) for page in self.pages):
            raise TypeError("a search result's pages are a tuple of WebPage")
        if not isinstance(self.replayed, bool):
            raise TypeError(f"a search result's replayed is a bool, and this one's is {type(self.replayed)}")
        usd = self.usd
        if usd is not None and (
            isinstance(usd, bool) or not isinstance(usd, int | float) or not math.isfinite(usd) or usd < 0
        ):
            raise ValueError(f"a search result's usd is a finite amount, 0 or more, or None; got {usd!r}")


class WebSearch(Protocol):
    """A host's web search, which claims grading's web check sends its queries to.

    **Identity.** ``identity`` names the search as the run's spend records it: the provider of the cell's ``judge``
    row for its searches (``provider_unit`` ``query``), and the provider a declared rate is looked up under
    (``ExternalRateTable`` keyed ``(identity, "query")``). A replay of a recorded search answers under the same
    identity. Keep it stable for one search set-up; change it when what a query returns could change (another
    provider, another index).

    **Price.** ``price_ceiling`` is the most one query can cost, in dollars, or ``None`` when the port cannot say. The
    run's cost cap admits each search on it, or on a rate the run declares for ``(identity, "query")``, before the
    search is sent. A search is billed at what its result reports it cost (:attr:`WebSearchResult.usd`), else at the
    declared rate, else at the ceiling, which is then an upper bound on the spend, not an observation. Under an
    enforcing cost cap a port that states no ceiling, with no declared rate, is refused at launch.

    **Failures.** A search that raises fails the cell's web step (``port_error``): the claims it had not decided read
    unverifiable, and every verdict case evidence decided stands. The engine never retries it, so a port that wants
    retries makes them inside ``search``. Return the results in the search's own order; the engine reads at most its
    budget's ``results_per_query`` of them.

    A port that serves recorded results rather than searching says so on each result
    (:attr:`WebSearchResult.replayed`), as :class:`~threetears.evals.testing.RecordedWebSearch` does in replay.

    ``search`` must be safe to call concurrently on one instance: a run's cells may search at once.
    """

    @property
    def identity(self) -> str:
        """The search's name, as the run's spend and its rate table key it."""
        ...

    async def search(self, query: str) -> WebSearchResult:
        """Search the web for ``query``.

        :param query: The search, as the judge wrote it.
        :return: Its results, in order, with what the port knows of the search.
        """
        ...

    def price_ceiling(self) -> float | None:
        """The most one query can cost, in dollars; ``None`` when it cannot say."""
        ...
