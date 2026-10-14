"""How do I turn a finished campaign into the files people read? ``result.write(dir)``.

It writes one page that needs no script, the same page as Markdown, and the evidence bundle every number on it was
computed from. New here: ``Result.write`` and ``Result.gate``. Reading the page: ``docs/reading-reports.md``.

Run it with ``python reports.py``; the files go to ``./eval-report/``. It calls no model: both arms are keyword
stand-ins, so it runs offline and costs nothing.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from threetears.evals import Eval

URGENT = ["Checkout is down since 09:00.", "We lost yesterday's orders.", "The API returns 500.", "Nobody can sign in."]
ROUTINE = ["Could you add a dark mode?", "Please update our billing address asap.", "The pricing page has a typo."]
CASES = [{"ticket": text, "label": "urgent"} for text in URGENT] + [{"ticket": t, "label": "routine"} for t in ROUTINE]


def keywords(*words: str) -> Callable[[dict], Awaitable[str]]:
    """An OFFLINE stand-in for a model: urgent when the ticket mentions any of ``words``."""

    async def triage(case: dict) -> str:
        return "urgent" if any(word in case["ticket"].lower() for word in words) else "routine"

    return triage


async def main(out_dir: Path = Path("eval-report")) -> dict[str, Path]:
    arms = {"current": keywords("asap"), "more-keywords": keywords("down", "lost", "500", "nobody")}
    # The first arm is the control.
    comparison = await Eval(CASES, arms, ["label"], k=2).run(store=":memory:")
    print(comparison.gate().render())  # the verdicts as a CI step gates them
    files = comparison.write(out_dir)  # report.html (no script), report.md, bundle.json
    print("Wrote", ", ".join(str(path) for path in files.values()))
    return dict(files)


if __name__ == "__main__":
    asyncio.run(main())
