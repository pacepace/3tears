"""Record what a tool answered once, and replay it so every arm of a comparison sees the same answers.

The candidates here call a search tool, whose live results change from call to call, so two arms run one
after the other would be graded on different results. A cassette fixes that: a ``capture`` run calls the
tools live and records their answers, and a ``replay`` run is served that recording instead of calling them.
It records the tools only, never the candidate. New since ``compare_two_prompts.py``: ``tools=``, which
hands the candidate its tools as ``candidate(case, tools)``, and ``cassette_mode=`` / ``cassette_corpus_id=``.
How replay matches each ask, and what a miss does, is in ``docs/adopting-a-host.md`` (Cassettes).

Run it with ``python packages/evals/examples/cassettes.py``.
It calls no model and needs no API key: the candidates are plain code and the search is an offline stand-in.
"""

import asyncio
import random
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from threetears.evals.quick import CandidateTools, Comparison, callable_host, compare, run_eval

# -----------------------------------------------------------------------------
# 1. The cases, and the pages a search finds for each: one current page and two stale ones.
# -----------------------------------------------------------------------------

CASES = [
    {"query": "refund window", "expected": "14 days"},
    {"query": "pro plan price", "expected": "$24/month"},
    {"query": "team plan seats", "expected": "10 seats"},
    {"query": "upload size limit", "expected": "2 GB"},
    {"query": "log retention", "expected": "90 days"},
    {"query": "api rate limit", "expected": "600/min"},
]

PAGES = {
    case["query"]: [
        {"date": "2026-09-01", "says": case["expected"]},
        {"date": "2025-03-15", "says": "what was true in 2025"},
        {"date": "2024-06-30", "says": "what was true in 2024"},
    ]
    for case in CASES
}

# -----------------------------------------------------------------------------
# 2. The tool: a plain function of keyword arguments that returns JSON.
#
# This offline stand-in returns two of a query's three pages, re-ranked every time the query is
# asked again, as a live index would. Seeded, so the script prints the same thing every run.
# -----------------------------------------------------------------------------

live_searches: Counter[str] = Counter()


def offline_search(query: str) -> list[dict[str, str]]:
    live_searches[query] += 1
    return random.Random(f"{query} #{live_searches[query]}").sample(PAGES[query], k=2)


# -----------------------------------------------------------------------------
# 3. The two candidates: both search once, and differ only in which page they trust.
# -----------------------------------------------------------------------------

served: dict[str, dict[str, list[Any]]] = {}  # arm -> query -> the results it was handed, per repeat


def reader(arm: str, pick: Callable[[list[dict[str, str]]], dict[str, str]]) -> Callable[..., Awaitable[str]]:
    """A candidate that searches the case's query and answers with what the page ``pick`` chooses says."""

    async def answer(case: Mapping[str, Any], tools: CandidateTools) -> str:
        hits = await tools["search"](query=case["query"])  # the tool, as the engine hands it over
        served.setdefault(arm, {}).setdefault(case["query"], []).append(hits)
        return pick(hits)["says"]

    return answer


trust_top_hit = reader("top_hit", lambda hits: hits[0])
trust_newest_hit = reader("newest_hit", lambda hits: max(hits, key=lambda hit: hit["date"]))


def correct(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer is what the current page says."""
    return answer == case["expected"]


# -----------------------------------------------------------------------------
# 4. Capture the search once, then replay that recording for both arms.
# -----------------------------------------------------------------------------


async def main() -> Comparison:
    """Record the search's answers, compare the two candidates on that recording, and print the report."""
    print("No model is called: the candidates are plain code, and the search is an OFFLINE stand-in.\n")
    tools = {"search": offline_search}  # each tool by the name the candidate calls it by
    # The replay reads the capture's recording from the same host and scope.
    host, scope = callable_host([correct]), "cassettes-example"

    capture = await run_eval(
        CASES,
        trust_top_hit,
        [correct],
        scope_id=scope,
        host=host,
        k=1,  # one recorded session per case
        tools=tools,
        cassette_mode="capture",  # call the search live, and record every answer
    )
    print(f"captured run {capture.run_id}: {live_searches.total()} live search(es) recorded")

    live_searches.clear()
    served.clear()
    comparison = await compare(
        CASES,
        {"top_hit": trust_top_hit, "newest_hit": trust_newest_hit},
        [correct],
        control="top_hit",
        scope_id=scope,
        host=host,
        k=2,
        tools=tools,
        cassette_mode="replay",  # serve the recording instead of calling the search
        cassette_corpus_id=capture.run_id,  # which capture to serve
    )
    print(f"replayed both arms: {live_searches.total()} live search(es)\n")
    for case in CASES:
        dates = {arm: [hit["date"] for hit in served[arm][case["query"]][0]] for arm in ("top_hit", "newest_hit")}
        print(f"  {case['query']:<18} top_hit was served {dates['top_hit']}, newest_hit {dates['newest_hit']}")
    print(f"identical for every case and repeat: {served['top_hit'] == served['newest_hit']}\n")

    print(comparison.render())
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
