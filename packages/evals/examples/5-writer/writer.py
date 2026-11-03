"""Rung 5: evaluate the evaluator. Run ``python -m threetears.evals examples/5-writer/writer.py``.

The report writer (an analyst model and its editor) is a model too, so it gets an eval of its own: two arms that
differ in one file of the analyst's skill, over eleven frozen campaigns (the four ladder rungs', a larger one the
package ships, six misfits), scored by a judge of another family. The control is a frozen copy of the shipped skill (``skills/control/``); the candidate puts the lead
claim first (``skills/lead-first.md``). Each arm writes the held-out cases once, and the engine reads the promotion.

It replays ``recordings/`` with no key. Until they are recorded, it refuses, naming the file and the key that would
record it; ``recordings/README.md`` holds the one command that records them.
"""

from pathlib import Path

from threetears.evals import Model
from threetears.evals.harness import AnalystSkill, Writer, WriterEval

control = AnalystSkill.at("skills/control")  # frozen: recordings are keyed by the skill's digest
variant = (Path(__file__).parent / "skills" / "lead-first.md").read_text(encoding="utf-8")  # SKILL.md, lead claim first
lead_first = control.with_file("SKILL.md", variant)
judge = Model("google/gemini-3.8-flash", temperature=0, recording="recordings/judge.jsonl")  # another family

writer_eval = WriterEval(
    "cases",  # eleven frozen campaigns: {name, group, stratum, withhold, snapshot}, one .json.gz each
    arms={  # the first is the control; the arms differ in the skill alone
        "shipped": Writer("claude-sonnet-5-5", skill=control, effort="medium"),
        "lead-first": Writer("claude-sonnet-5-5", skill=lead_first, effort="medium"),
    },
    judge=judge,
    recordings="recordings/turns",  # one file per arm and case, keyed by skill, model, effort and snapshot
    budget_usd=1.0,  # each report's cap
    max_cost_usd=15.0,  # the whole eval's, judge included: each arm runs under half
    held_out_fraction=0.35,  # 7 held-out cases of 11: the 0.3 default leaves 4, under the 5 a rate needs
    name="writer",
)
