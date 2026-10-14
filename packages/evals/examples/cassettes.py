"""How do I compare candidates fairly when the tool they call answers differently every time?

The candidates call a search whose results drift from call to call, so two arms run one after the other
would be graded on different results. A recording fixes that: the first run calls the tools live and records what
they answered, and every run after it is served that recording instead; it records the tools, never the candidate.
New here: a ``World`` with no state, only tools, which hands the candidate its tools as ``candidate(case, tools)``,
and its ``recording=``: a file that is there is replayed, one that is not is recorded (``RECORD=1`` records it
again). Each answer is keyed by the case, the tool, its arguments and which time the cell asked it.

Run it with ``python cassettes.py`` (``README.md`` beside it says how). It calls no model, so it needs no API key
and costs nothing: the candidates are plain code and the search is an offline stand-in.
"""

import asyncio
import random
import tempfile
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from threetears.evals import Eval, Result, World

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
# 2. The tool: a plain function of typed keyword arguments that returns JSON, named by its def.
#
# This offline stand-in returns two of a query's three pages, re-ranked every time the query is
# asked again, as a live index would. Seeded, so the script prints the same thing every run.
# -----------------------------------------------------------------------------

live_searches: Counter[str] = Counter()


def search(query: str) -> list[dict[str, str]]:
    """Search the help pages for a query."""
    live_searches[query] += 1
    return random.Random(f"{query} #{live_searches[query]}").sample(PAGES[query], k=2)


# -----------------------------------------------------------------------------
# 3. The two candidates: both search once, and differ only in which page they trust.
# -----------------------------------------------------------------------------

served: dict[str, dict[str, list[Any]]] = {}  # arm -> query -> the results it was handed, per repeat


def reader(arm: str, pick: Callable[[list[dict[str, str]]], dict[str, str]]) -> Callable[..., Awaitable[str]]:
    """A candidate that searches the case's query and answers with what the page ``pick`` chooses says."""

    async def answer(case: Mapping[str, Any], tools: Mapping[str, Callable[..., Awaitable[Any]]]) -> str:
        hits = await tools["search"](query=case["query"])  # the tool, as the world hands it over
        served.setdefault(arm, {}).setdefault(case["query"], []).append(hits)
        return pick(hits)["says"]

    return answer


trust_top_hit = reader("top_hit", lambda hits: hits[0])
trust_newest_hit = reader("newest_hit", lambda hits: max(hits, key=lambda hit: hit["date"]))


# The engine reads this docstring's first line: it is the measure's description. A world's runner is graded on its
# outcome: the answer, beside the calls it made.
def correct(case: Mapping[str, Any], out: Any) -> bool:
    """Whether the answer is what the current page says."""
    return out.answer == case["expected"]


# -----------------------------------------------------------------------------
# 4. Record the search once, then replay that recording to both arms, twice per case, and print the verdict.
# -----------------------------------------------------------------------------


async def main(recording: Path | None = None) -> Result:
    """Record the search into ``recording`` (a fresh temporary file when ``None``), then replay it to both arms."""
    print("No model is called: running OFFLINE, with plain-code candidates and a stand-in search.\n")
    with tempfile.TemporaryDirectory() as scratch:
        # The search, and where its answers are kept: the first run records them, every later run replays them.
        lookups = World(tools=[search], recording=recording or Path(scratch) / "search.jsonl")

        capture = await Eval(CASES, lookups(trust_top_hit), [correct], k=1).run(store=":memory:")  # one per case
        print(f"captured run {capture.run_id}: {live_searches.total()} live search(es) recorded")

        live_searches.clear()
        served.clear()
        comparison = await Eval(
            CASES,
            {"top_hit": lookups(trust_top_hit), "newest_hit": lookups(trust_newest_hit)},  # the first is the control
            [correct],
            k=2,
        ).run(store=":memory:", scope="cassettes-example")
    print(f"replayed both arms: {live_searches.total()} live search(es)\n")
    for case in CASES:
        dates = {arm: [hit["date"] for hit in served[arm][case["query"]][0]] for arm in ("top_hit", "newest_hit")}
        print(f"  {case['query']:<18} top_hit was served {dates['top_hit']}, newest_hit {dates['newest_hit']}")
    print(f"identical for every case and repeat: {served['top_hit'] == served['newest_hit']}\n")

    # The verdict, each arm against the control, now on the same search results.
    for row in comparison.contrasts():
        p = "" if row["p_adjusted"] is None else f", Holm-adjusted p {row['p_adjusted']:.2g}"  # none if nothing varied
        print(f"{row['arm']} vs {comparison.control} on {row['reading']}: delta {row['delta']:+.2f}, ", end="")
        print(f"interval {row['interval']}{p}: {row['verdict']}")
    print("\nThe full report: print(comparison.render(full=True)), or reports.py to write it to files.")
    return comparison


if __name__ == "__main__":
    asyncio.run(main())
