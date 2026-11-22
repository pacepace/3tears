"""The web search port: what claims grading's web check asks a host's search for, and what it reads back.

A claim criterion with a web budget (:class:`~threetears.evals.schema.models.WebBudget`) checks, on the web, the claims
its judge nominated by its own suspicion: claims no case evidence bears on that the judge doubts. The engine writes
the searches, cuts windows from the pages a search returns, and asks its judge whether a window states the claim; a
host's port does only the search. The engine never calls a search provider itself and never names one.

A run hands its port in on its options (:attr:`~threetears.evals.run.RunnerOptions.web_search`). The engine records
every page window it read on the cell's trace and never searches for a stored grade again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

__all__ = ["WebPage", "WebSearch"]


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


class WebSearch(Protocol):
    """A host's web search, which claims grading's web check sends its queries to.

    **Identity.** ``identity`` names the search as the run's spend records it: the provider of the cell's ``judge``
    row for its searches (``provider_unit`` ``query``), and the provider a declared rate is looked up under
    (``ExternalRateTable`` keyed ``(identity, "query")``). A replay of a recorded search answers under the same
    identity. Keep it stable for one search set-up; change it when what a query returns could change (another
    provider, another index).

    **Price.** ``price_ceiling`` is the most one query can cost, in dollars, or ``None`` when the port cannot say. The
    run's cost cap admits a criterion's web check on it before any query exists, and, where the run declares no rate
    for ``(identity, "query")``, each query is billed at it. Under an enforcing cost cap a port that states none, with
    no declared rate, is refused at launch.

    **Failures.** A search that raises is the criterion's loss for that cell (``port_error``): the engine never
    retries it, so a port that wants retries makes them inside ``search``. Return the results in the search's own
    order; the engine reads at most its budget's ``results_per_query`` of them.

    A port that serves recorded results rather than searching may say so with a ``replayed`` attribute that is
    ``True`` (as :class:`~threetears.evals.testing.RecordedWebSearch` does in replay): the grades it served are marked
    replayed, so a check that needs a live search over them is not made.

    ``search`` must be safe to call concurrently on one instance: a run's cells may search at once.
    """

    @property
    def identity(self) -> str:
        """The search's name, as the run's spend and its rate table key it."""
        ...

    async def search(self, query: str) -> tuple[WebPage, ...]:
        """Search the web for ``query``.

        :param query: The search, as the judge wrote it.
        :return: Its results, in order.
        """
        ...

    def price_ceiling(self) -> float | None:
        """The most one query can cost, in dollars; ``None`` when it cannot say."""
        ...
