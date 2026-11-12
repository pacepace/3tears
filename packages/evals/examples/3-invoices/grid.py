"""Rung 3e: two levers at once. Name each arm as k=v pairs and each pair is a factor of the design.

The two prompts at two reasoning efforts, four arms over the 30 invoices; the first is the control. The low-effort
cells replay ``recordings/invoices.jsonl``, the high-effort cells ``recordings/invoices-high.jsonl``. As recorded on
deepseek/deepseek-v4-flash at high effort: 180 calls, $0.0188 (the recording's own cost_usd fields, summed).
"""

import asyncio

from invoice import V1, V2, Invoice

from threetears.evals import Eval, Model

low = Model("deepseek/deepseek-v4-flash", max_tokens=8192, effort="low", recording="recordings/invoices.jsonl")
high = Model("deepseek/deepseek-v4-flash", max_tokens=8192, effort="high", recording="recordings/invoices-high.jsonl")

grid = Eval(
    "invoices.jsonl",
    runner={  # each arm is one cell of the grid: prompt is one factor, effort the other
        "prompt=v1, effort=low": low.prompt(V1, Invoice),  # the control
        "prompt=v2, effort=low": low.prompt(V2, Invoice),
        "prompt=v1, effort=high": high.prompt(V1, Invoice),
        "prompt=v2, effort=high": high.prompt(V2, Invoice),
    },
    grader=["currency", "number", "due", "total"],
    name="grid",
)


async def main() -> None:
    result = await grid.run()
    print(result)  # each arm against the control: each lever's effect with the other at its control level
    print(await result.against("prompt=v1, effort=high"))  # the prompt's effect at high effort, no run repeated


if __name__ == "__main__":
    asyncio.run(main())
