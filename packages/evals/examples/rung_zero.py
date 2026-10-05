"""Rung zero: evaluate one async function against a list of cases, in one file.

Everything a product needs to bring is below: its cases, the function under test, and the code
that grades an answer. ``run_eval`` builds the rest — a host, a store, a kind over the function —
launches one run through the engine, and returns its summary. Run it with
``python packages/evals/examples/rung_zero.py``.
"""

import asyncio

from threetears.evals.quick import EvalSummary, run_eval

CASES = [
    {"text": "The delivery came two days late and the box was crushed.", "expected": "negative"},
    {"text": "Exactly what I ordered, and it arrived early.", "expected": "positive"},
    {"text": "It works.", "expected": "neutral"},
    {"text": "Great price, terrible battery.", "expected": "negative"},
    {"text": "Not bad at all.", "expected": "positive"},
]

NEGATIVE = ("late", "crushed", "terrible", "broken")
POSITIVE = ("exactly", "early", "great", "love")


async def classify(case: dict) -> str:
    """Label a review's sentiment as positive, negative or neutral."""
    words = case["text"].lower()
    if any(word in words for word in NEGATIVE):
        return "negative"
    if any(word in words for word in POSITIVE):
        return "positive"
    return "neutral"


def correct(case: dict, label: str) -> bool:
    """Whether the label is the expected one."""
    return label == case["expected"]


def decisive(case: dict, label: str) -> bool:
    """Whether the classifier committed to a polarity rather than answering neutral."""
    return label != "neutral"


async def main() -> EvalSummary:
    """Run the classifier over every case twice and print what the run measured."""
    summary = await run_eval(CASES, classify, [correct, decisive], scope_id="rung-zero", k=2)
    print(summary.render())
    return summary


if __name__ == "__main__":
    asyncio.run(main())
