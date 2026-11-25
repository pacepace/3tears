"""A stored run graded again with today's harness: no model call, no agent run, every command in a trial container.

``python -m packages.evals.onramp --regrade RUN_DIR`` reads each stored trial as the agent left it (its ``steps.jsonl``,
its directory, and each reply from the run's recordings), reads its turns again with today's reply rules
(:func:`~.agent.read_reply`, :func:`~.agent.done_report`), takes the report the first ``done`` it now finds gives,
runs the regrade again in containers of the trial's own image, and grades it with today's checks. It writes
``rescore.jsonl`` and ``rescore.txt`` beside the run, and prints each check's count before and after.

A trial is graded on the directory as it stood at the turn it now ends on, which its report was made about. When
nothing after that turn changed the directory (no write, no ``python`` command), that is the directory the run left,
and the grade is exact. Otherwise the directory is rebuilt at that turn: the app as shipped, the agent's writes up to
it replayed from ``steps.jsonl``, and its ``python`` commands up to it run again in containers. A run that logged its
directory's state after each such turn (``files`` in ``steps.jsonl``, since 2026-11-16) has the rebuild checked
against it: a rebuild that matches is exact. One that does not, or one of a run that logged no state (the baseline of
2026-11-15), is labelled not comparable, and the table counts it apart.

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

from .agent import Act, Step, done_report, parse_report, read_reply, scored_door_turns
from .campaign import failure_causes, registry, slug
from .checks import GRADERS, regrade
from .isolation import Container
from .sandbox import DocsBuild, Observation, Sandbox


def _run_json(run_dir: Path) -> Path | None:
    """The ``run.json`` of the run ``run_dir`` is, or one tier of (a run of several tiers keeps each in its own
    directory beneath it); ``None`` for a run from before it was written."""
    found = [path for path in (run_dir / "run.json", run_dir.parent / "run.json") if path.is_file()]
    return found[0] if found else None


def _replies(run_dir: Path, arm: str) -> dict[tuple[str, int], list[str]]:
    """Each trial's replies in turn order, by its app and repeat, from the run's recording of ``arm``: a recorded
    case is named by its place among the run's cases, which ``run.json`` lists (a run before it ran ``apps.jsonl``'s)."""
    ran = _run_json(run_dir)
    apps = json.loads(ran.read_text(encoding="utf-8"))["apps"] if ran is not None else [c["app"] for c in registry()]
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


#: How a trial's directory was had at the turn it now ends on.
EXACT = "the directory as the run left it"
REBUILT_EXACT = "rebuilt at its end, and it matches the state the run logged there"
REBUILT_DIFFERS = "rebuilt at its end, and it differs from the state the run logged there: not comparable"
REBUILT_UNLOGGED = "rebuilt at its end; the run logged no per-turn state to check it against: not comparable"
COMPARABLE = (EXACT, REBUILT_EXACT)


def _changes(step: Mapping[str, Any]) -> bool:  # Any: a logged step
    """Whether a logged turn may have changed the directory: a write, or a ``python`` command."""
    act = step.get("act") or {}
    words = str(act.get("command", "")).split()[:1]
    return act.get("tool") == "write" or (act.get("tool") == "run" and words in (["python"], ["python3"]))


async def _rebuilt(
    logged: Sequence[Mapping[str, Any]], ended: int, record: Mapping[str, Any], build: DocsBuild, dest: Path
) -> tuple[Sandbox, str]:  # Any: a logged step
    """The trial's directory at turn ``ended``: the app as shipped, its writes replayed and its ``python`` commands run
    again, in order; and how far it can be trusted (:data:`REBUILT_EXACT` and its kin)."""
    sandbox = Sandbox.create(dest, record["app"], build, Container(build.image or ""))
    state: dict[str, str] | None = None
    for step in logged:
        if int(step["turn"]) > ended or not _changes(step):
            continue
        act = step["act"]
        if act["tool"] == "write":
            sandbox.write(act["path"], act["text"])
        else:
            await sandbox.run(act["command"])
        state = step.get("files", state)
    if state is None:
        return sandbox, REBUILT_UNLOGGED
    return sandbox, REBUILT_EXACT if sandbox.manifest() == state else REBUILT_DIFFERS


def _steps(logged: Sequence[Mapping[str, Any]]) -> list[Step]:  # Any: a logged step
    """The logged turns as steps: their acts and what each did."""
    return [
        Step(int(step["turn"]), None if step.get("act") is None else Act.model_validate(step["act"]),
             Observation(**step["observation"]))
        for step in logged
    ]  # fmt: skip


async def _rescore(row: Mapping[str, Any], replies: Sequence[str], images: Path) -> dict[str, Any]:  # Any: JSON
    record = row["record"]
    docs = images / f"docs-{record['build']['image'].rsplit(':', 1)[-1]}"
    trial = Path(record["sandbox"]).parent
    logged = [json.loads(line) for line in (trial / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    steps = [step for step in logged if "observation" in step]
    again = read_again(steps, replies)
    ended = again["ended"]
    ran = [command for command in record["commands"] if ended is None or command["turn"] < ended]
    later = [command["turn"] for command in record["commands"] if ended is not None and command["turn"] > ended]
    changed_after = [
        int(step["turn"]) for step in steps if ended is not None and step["turn"] > ended and _changes(step)
    ]
    report, problem = parse_report(again["report"])
    built = record["build"]
    build = DocsBuild(
        built["name"], Path(Container.python), docs, docs / "llms.txt", built["source"], built["image"],
        Path(built["package_dir"]),
    )  # fmt: skip
    out = trial / "rescore"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    started = time.monotonic()
    if ended is not None and changed_after:
        sandbox, state = await _rebuilt(steps, ended, record, build, out / f"at-turn-{ended}")
    else:
        sandbox, state = Sandbox(Path(record["sandbox"]), record["app"], build, Container(built["image"])), EXACT
        sandbox.written = list(record["written"])
    checks = await regrade(sandbox, out, report, ran)
    took = round(time.monotonic() - started, 1)
    acts = _steps(steps)
    rescored = {
        **record,
        "stopped": "done" if ended is not None else record["stopped"],
        "commands": ran,
        "scored_door_turns": scored_door_turns([step for step in acts if ended is None or step.turn < ended]),
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
        "arm": row["arm"],
        "before": row["scores"],
        "before_causes": failure_causes(
            {"record": {**record, "scored_door_turns": scored_door_turns(acts)}, "scores": row["scores"]}
        ),
        "after": scores,
        "causes": failure_causes({"record": rescored, "scores": scores}),
        "ended_on": ended,
        "report": report,
        "report_from": again["report_from"],
        "unmade_turns": again["unmade"],
        "commands_after_the_end": later,
        "changed_after_the_end": changed_after,
        "state": state,
        "regrade_s": took,
        "checks": checks,
    }


def rescore(run_dir: Path) -> list[str]:
    """Grade every stored trial of ``run_dir`` again (see the module); the lines it prints, also in ``rescore.txt``."""
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    # The workdir the run's images are kept in: the run's parent, or a tier's grandparent.
    ran = _run_json(run_dir)
    images = (run_dir.parent if ran is None or ran.parent == run_dir else run_dir.parent.parent) / "images"
    replies = {arm: _replies(run_dir, arm) for arm in {row["arm"] for row in rows}}

    async def every() -> list[dict[str, Any]]:  # Any: JSON
        return [
            await _rescore(row, replies[row["arm"]].get((row["record"]["app"], row["repeat"]), []), images)
            for row in rows
        ]

    done = asyncio.run(every())
    (run_dir / "rescore.jsonl").write_text("".join(json.dumps(item) + "\n" for item in done), encoding="utf-8")
    lines = [
        "| arm | check | before (harness-caused failures) | after (harness-caused) | after, comparable trials only |",
        "|---|---|---|---|---|",
    ]
    for arm in sorted({item["arm"] for item in done}):
        ours = [item for item in done if item["arm"] == arm]
        kept = [item for item in ours if item["state"] in COMPARABLE]
        for grader in GRADERS:
            name = grader.__name__
            before = sum(1 for item in ours if item["before"].get(name) == 1.0)
            was = sum(1 for item in ours if item["before_causes"].get(name) == "harness")
            after = sum(1 for item in ours if item["after"][name] == 1.0)
            now = sum(1 for item in ours if item["causes"].get(name) == "harness")
            comparable = sum(1 for item in kept if item["after"][name] == 1.0)
            lines.append(
                f"| {arm} | {name} | {before} of {len(ours)} ({was}) | {after} of {len(ours)} ({now}) | "
                f"{comparable} of {len(kept)} |"
            )
    apart = [item["trial"] for item in done if item["state"] not in COMPARABLE]
    if apart:
        lines.append(
            f"not comparable ({', '.join(apart)}): graded on a rebuild of the directory at the turn it ends on"
        )
    for item in done:
        moved = [name for name in item["after"] if item["after"][name] != item["before"].get(name)]
        lines.append(
            f"{item['arm']} {item['trial']}: ends on turn {item['ended_on']}, report from {item['report_from'] or 'nowhere'}, "
            f"graded on {item['state']}, regrade {item['regrade_s']}s"
            + (f"; moved: {', '.join(moved)}" if moved else "")
            + (f"; turns read now as acts never made: {item['unmade_turns']}" if item["unmade_turns"] else "")
            + (
                f"; changed after its end: turns {item['changed_after_the_end']}"
                if item["changed_after_the_end"]
                else ""
            )
        )
    (run_dir / "rescore.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines
