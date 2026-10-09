"""Does my function give the right answer on my cases?

The first rung: cases, an async function, and what a right answer is. ``expected=`` grades it as a classifier
(confusion matrix, per-label precision, recall and F1); ``decisive`` is a scorer. New here: ``run_eval``.

Run it with ``python packages/evals/examples/rung_zero.py``. It calls no model: it runs offline, for free.
"""

import asyncio

from threetears.evals.quick import EvalSummary, run_eval

# -----------------------------------------------------------------------------
# 1. The cases: each review, and the sentiment a person gave it.
# -----------------------------------------------------------------------------

CASES = [
    {"text": "The delivery came two days late and the box was crushed.", "expected": "negative"},
    {"text": "Exactly what I ordered, and it arrived early.", "expected": "positive"},
    {"text": "It works.", "expected": "neutral"},
    {"text": "Great price, terrible battery.", "expected": "negative"},
    {"text": "Not bad at all.", "expected": "positive"},
]

# -----------------------------------------------------------------------------
# 2. The function under test, and a scorer.
# -----------------------------------------------------------------------------


async def classify(case: dict) -> str:
    """Label a review's sentiment as positive, negative or neutral."""
    words = case["text"].lower()
    if any(word in words for word in ("late", "crushed", "terrible", "broken")):
        return "negative"
    if any(word in words for word in ("exactly", "early", "great", "love")):
        return "positive"
    return "neutral"


def decisive(case: dict, label: str) -> bool:  # the engine reads its docstring's first line: the measure's description
    """Whether the classifier committed to a polarity rather than answering neutral."""
    return label != "neutral"


# -----------------------------------------------------------------------------
# 3. Run it over every case twice, and print what the run measured.
# -----------------------------------------------------------------------------


async def main() -> EvalSummary:
    summary = await run_eval(
        CASES, classify, [decisive], expected=lambda case: case["expected"], scope_id="rung-zero", k=2
    )
    print(summary.render())
    return summary


if __name__ == "__main__":
    asyncio.run(main())
