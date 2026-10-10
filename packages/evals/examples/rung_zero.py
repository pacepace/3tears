"""Does my function give the right answer on my cases?

The first rung: cases, an async function, and what a right answer is. New here: ``run_eval``, and code grading
each answer two ways: ``expected=`` grades a classifier (accuracy, confusion matrix, each label's precision and
recall), and a scorer checks one thing. Then ``summary.misses()``: every miss, and why. Most answers have no
exact right label; ``llm_judge.py`` grades those with a model. Run ``python packages/evals/examples/rung_zero.py``.
It calls no model, so it is free.
"""

import asyncio

from threetears.evals.quick import EvalSummary, run_eval

# 1. The cases: each review, and the sentiment a person gave it.
CASES = [
    {"text": "The delivery came two days late and the box was crushed.", "expected": "negative"},
    {"text": "Exactly what I ordered, and it arrived early.", "expected": "positive"},
    {"text": "It works.", "expected": "neutral"},
    {"text": "Great price, terrible battery.", "expected": "negative"},
    {"text": "Not bad at all.", "expected": "positive"},
    {"text": "It arrived a day late, but I love it.", "expected": "positive"},
]


# 2. The function under test, and a scorer.
async def classify(case: dict) -> str:
    """Label a review's sentiment as positive, negative or neutral."""
    words = case["text"].lower()
    if any(word in words for word in ("late", "crushed", "terrible", "broken")):
        return "negative"
    if any(word in words for word in ("exactly", "early", "great", "love")):
        return "positive"
    return "neutral"


def no_false_alarm(case: dict, label: str) -> bool:  # the engine reads its docstring's first line: the description
    """Whether the review was labelled negative only when a person called it negative."""
    return label != "negative" or case["expected"] == "negative"


# 3. Run it over every case twice, print what the run measured, then read every miss.
async def main() -> EvalSummary:
    summary = await run_eval(
        CASES, classify, [no_false_alarm], expected=lambda case: case["expected"], scope_id="rung-zero", k=2
    )
    print(summary.render())
    print("\nWhat it got wrong, each case once per repeat:")
    for miss in summary.misses():
        print(f"  {miss.input['text']!r}, repeat {miss.repeat}: {'; '.join(miss.missed_because)}")
    return summary


if __name__ == "__main__":
    asyncio.run(main())
