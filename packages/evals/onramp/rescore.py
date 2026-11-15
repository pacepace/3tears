"""A stored run graded again with today's harness: no model call, no agent run, every command in a trial container.

``python -m packages.evals.onramp --regrade RUN_DIR`` reads each stored trial as the agent left it (its ``steps.jsonl``,
its directory, and each reply from the run's recordings), reads its turns again with today's reply rules
(:func:`~.agent.read_reply`, :func:`~.agent.done_report`), takes the report the first ``done`` it now finds gives,
runs the regrade again in containers of the trial's own image, and grades it with today's checks. It writes
``rescore.jsonl`` and ``rescore.txt`` beside the run, and prints each check's count before and after.

A turn that today's rules read as a tool call the run never made (a reply refused then and taken now) cannot be
replayed, since the agent never saw its result: it is counted and named, and the trial is graded on what it did.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .agent import Act, done_report, parse_report, read_reply
from .campaign import CASES, failure_causes, slug
from .checks import GRADERS, regrade
from .isolation import Container
from .sandbox import DocsBuild, Sandbox


def _replies(run_dir: Path, arm: str) -> dict[tuple[str, int], list[str]]:
    """Each trial's replies in turn order, by its app and repeat, from the run's recording of ``arm``."""
    apps = [json.loads(line)["app"] for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    recording = run_dir / "recordings" / f"{slug(arm)}.jsonl"
    found: dict[tuple[str, int], list[str]] = defaultdict(list)
    if not recording.is_file():
        return found
    for line in recording.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        index = int(row["cell"]["case"].rsplit("-", 1)[1])
        found[(apps[index], int(row["cell"]["repeat"]))].append(row["result"]["content"])
    return found


def read_again(steps: Sequence[Mapping[str, Any]], replies: Sequence[str]) -> dict[str, Any]:  # Any: JSON
    """A trial's turns read with today's rules: the turn its run now ends on (``None`` when none says done), the
    report and where it was read from, the turns still refused as no tool call, and the turns now read as a tool call
    the run never made."""
    unmade: list[int] = []
    refused: list[int] = []
    for step in steps:
        turn = int(step["turn"])
        if step.get("act") is not None:
            act = Act.model_validate(step["act"])
            if act.tool == "done":
                report, where = done_report(act)
                return {"ended": turn, "report": report, "report_from": where, "unmade": unmade, "refused": refused}
            continue
        reply = step.get("reply") or (replies[turn - 1] if turn <= len(replies) else "")
        read, _, how = read_reply(reply)
        if read is None:
            refused.append(turn)
            continue
        if read.tool == "done":
            ended = {"ended": turn, "report": read.text, "report_from": how or "text"}
            return {**ended, "unmade": unmade, "refused": refused}
        unmade.append(turn)
    return {"ended": None, "report": None, "report_from": "", "unmade": unmade, "refused": refused}


async def _rescore(row: Mapping[str, Any], replies: Sequence[str], docs: Path) -> dict[str, Any]:  # Any: JSON
    record = row["record"]
    trial = Path(record["sandbox"]).parent
    logged = [json.loads(line) for line in (trial / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    again = read_again([step for step in logged if "observation" in step], replies)
    ended = again["ended"]
    ran = [command for command in record["commands"] if ended is None or command["turn"] < ended]
    later = [command["turn"] for command in record["commands"] if ended is not None and command["turn"] > ended]
    report, problem = parse_report(again["report"])
    built = record["build"]
    build = DocsBuild(
        built["name"], Path(Container.python), docs, docs / "llms.txt", built["source"], built["image"],
        Path(built["package_dir"]),
    )  # fmt: skip
    sandbox = Sandbox(Path(record["sandbox"]), record["app"], build, Container(built["image"]))
    sandbox.written = list(record["written"])
    out = trial / "rescore"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    started = time.monotonic()
    checks = await regrade(sandbox, out, report, ran)
    took = round(time.monotonic() - started, 1)
    rescored = {
        **record,
        "stopped": "done" if ended is not None else record["stopped"],
        "commands": ran,
        "format_error_turns": again["refused"],
        "report": report,
        "report_problem": problem,
        "report_from": again["report_from"],
        "checks": checks,
    }
    case = {"app": record["app"]}
    scores = {grader.__name__: float(grader(case, rescored)) for grader in GRADERS}
    return {
        "trial": trial.name,
        "app": record["app"],
        "repeat": row["repeat"],
        "before": row["scores"],
        "after": scores,
        "causes": failure_causes({"record": rescored, "scores": scores}),
        "ended_on": ended,
        "report_from": again["report_from"],
        "unmade_turns": again["unmade"],
        "commands_after_the_end": later,
        "regrade_s": took,
        "checks": checks,
    }


def rescore(run_dir: Path) -> list[str]:
    """Grade every stored trial of ``run_dir`` again (see the module); the lines it prints, also in ``rescore.txt``."""
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    docs = next((run_dir.parent / "images").glob("docs-*"), run_dir)
    replies = {arm: _replies(run_dir, arm) for arm in {row["arm"] for row in rows}}

    async def every() -> list[dict[str, Any]]:  # Any: JSON
        return [
            await _rescore(row, replies[row["arm"]].get((row["record"]["app"], row["repeat"]), []), docs)
            for row in rows
        ]

    done = asyncio.run(every())
    (run_dir / "rescore.jsonl").write_text("".join(json.dumps(item) + "\n" for item in done), encoding="utf-8")
    lines = ["| check | before | after | after, failures the harness caused |", "|---|---|---|---|"]
    for grader in GRADERS:
        name = grader.__name__
        before = sum(1 for item in done if item["before"].get(name) == 1.0)
        after = sum(1 for item in done if item["after"][name] == 1.0)
        harness = sum(1 for item in done if item["causes"].get(name) == "harness")
        lines.append(f"| {name} | {before} of {len(done)} | {after} of {len(done)} | {harness} |")
    for item in done:
        moved = [name for name in item["after"] if item["after"][name] != item["before"].get(name)]
        lines.append(
            f"{item['trial']}: ends on turn {item['ended_on']}, report from {item['report_from'] or 'nowhere'}, "
            f"regrade {item['regrade_s']}s"
            + (f"; moved: {', '.join(moved)}" if moved else "")
            + (f"; turns read now as acts never made: {item['unmade_turns']}" if item["unmade_turns"] else "")
            + (f"; commands after its end: {item['commands_after_the_end']}" if item["commands_after_the_end"] else "")
        )
    (run_dir / "rescore.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines
