"""The plan-scoring dataset (``fixtures/report_plans/``): the Kairo acceptance bundle, the rubric, the anchors and the
score records.

Contracts held here:

- **Kairo.** The committed Kairo round 2 bundle is under two megabytes, and ``report_input(bundle)`` on it matches the
  r5 report input on every field r5 carries a value for, each number at the precision r5 writes it; r5's contrasts
  are matched by arm, measure and kind, since the bundle carries every contrast and r5 a subset.
- **Rubric.** Every rubric version is immutable: ``rubric/DIGESTS.json`` pins the sha256 of each version's two files,
  and lists every version there is. v1 holds R1 (code's) to R11, each once; v2 holds R1 to R12, changing only R4 and
  R10 and adding R12; the dataset reads the newest by default.
- **Anchors.** ``anchors/LABELS.json`` labels every anchor and weakened copy, and nothing else; each is pending the
  owner or the owner's. Each input's anchor passes the plan checks and R1. Each refused copy fails exactly the faults
  its label lists (one rule, nothing more), and every rule of the plan checks has one. Each copy weakened on an item
  passes the plan checks with its anchor's structure, only prose changed, and every model-scored item has one.
  ``weaken_anchors.py`` writes exactly the committed copies and labels.
- **Scores.** Every run plan the dataset keeps, current or historical, has exactly one score record beside it, and
  every record is sound (its digests are its files', every item scored once, every quote text the plan holds) and
  scores a plan the dataset keeps; a run recorded and not yet scored is red here until it is. A scorer's reply becomes
  a record only when it is sound, its planner is the run's recorded provenance, and records pool only by planner prompt
  digest, rubric digest and scorer.
- **The brief** carries the rubric, R1's verdict, the plan rendered and every item to score, and names neither the
  plan's file nor its planner.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans import dataset, weaken_anchors
from threetears.evals.analysis.errors import PlanRefusal
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.report_plan_checks import validate_plan
from threetears.evals.kernel.report_plan import ReportPlan
from threetears.evals.schema.prose import field_is_prose

HERE = dataset.HERE

#: The plan checks' rules (``validate_plan``), each of which some refused copy must break alone.
RULES = frozenset(
    {
        "facts",
        "mode",
        "allowed_mode",
        "ids",
        "recommendation",
        "answers",
        "confidence",
        "flag",
        "coverage",
        "charts",
        "keys",
        "literal_digits",
        "weakest",
        "reader_ids",
    }
)


# --- Kairo -----------------------------------------------------------------------------------------------------------


def _significant(value: float) -> int:
    digits = re.sub(r"e.*$", "", repr(abs(value))).replace(".", "").lstrip("0")
    return max(1, len(digits))


def _differences(carried: Any, built: Any, path: str = "") -> list[str]:
    """Where ``built`` differs from ``carried`` on a value ``carried`` holds, numbers at ``carried``'s precision."""
    if carried is None:
        return []
    if isinstance(carried, dict):
        if not isinstance(built, dict):
            return [f"{path}: {built!r} is not an object"]
        return [
            found for key, value in carried.items() for found in _differences(value, built.get(key), f"{path}.{key}")
        ]
    if isinstance(carried, list):
        if not isinstance(built, list):
            return [f"{path}: {built!r} is not a list"]
        if path == ".contrasts":
            found: list[str] = []
            for entry in carried:
                same = [
                    b
                    for b in built
                    if (b["arm"], b["measure"], b["kind"]) == (entry["arm"], entry["measure"], entry["kind"])
                ]
                if len(same) != 1:
                    found.append(f"{path}: {len(same)} built contrasts for {entry['arm']}/{entry['measure']}")
                    continue
                found += _differences(entry, same[0], f"{path}[{entry['arm']}/{entry['measure']}]")
            return found
        if len(carried) != len(built):
            return [f"{path}: {len(built)} entries, r5 holds {len(carried)}"]
        return [
            f
            for index, (c, b) in enumerate(zip(carried, built, strict=True))
            for f in _differences(c, b, f"{path}[{index}]")
        ]
    if isinstance(carried, float) and isinstance(built, int | float) and not isinstance(built, bool):
        if carried == built or carried == float(f"{built:.{_significant(carried)}g}"):
            return []
    elif carried == built:
        return []
    return [f"{path}: built {built!r}, r5 holds {carried!r}"]


def test_the_kairo_bundle_is_committed_under_two_megabytes() -> None:
    assert dataset.KAIRO_BUNDLE.stat().st_size < 2_000_000


def test_the_kairo_bundles_report_input_matches_r5_on_every_field_r5_carries() -> None:
    r5: dict[str, Any] = json.loads((HERE / "r5" / "input.json").read_text(encoding="utf-8"))
    built = report_input(dataset.kairo_bundle(), provenance=r5["meta"]["provenance"])
    assert _differences(r5, built.model_dump(mode="json", by_alias=True)) == []


def test_the_comparison_sees_a_difference() -> None:
    r5: dict[str, Any] = json.loads((HERE / "r5" / "input.json").read_text(encoding="utf-8"))
    built = report_input(dataset.kairo_bundle(), provenance=r5["meta"]["provenance"]).model_dump(
        mode="json", by_alias=True
    )
    built["arms"][1]["served_model"] = "something else"
    built["aspects"][0]["readings"][0]["change"] += 0.01
    built["contrasts"] = built["contrasts"][1:]
    found = _differences(r5, built)
    assert any(".arms[1].served_model" in line for line in found)
    assert any(".aspects[0].readings[0].change" in line for line in found)
    assert any(line.startswith(".contrasts: 0 built contrasts") for line in found)


# --- the rubric ------------------------------------------------------------------------------------------------------


def test_every_rubric_version_is_pinned_by_its_digests() -> None:
    pinned: dict[str, dict[str, str]] = json.loads((dataset.RUBRICS / "DIGESTS.json").read_text(encoding="utf-8"))
    on_disk = {path.stem for path in dataset.RUBRICS.glob("v*.*")}
    assert set(pinned) == on_disk
    for version, files in pinned.items():
        assert set(files) == {f"{version}.md", f"{version}.json"}
        for name, digest in files.items():
            assert dataset.sha256_file(dataset.RUBRICS / name) == digest, f"{name} changed: add a new version instead"


def test_rubric_v1_holds_r1_to_r11_once_with_r1_scored_by_code() -> None:
    items = dataset.rubric("v1")["items"]
    assert [item["id"] for item in items] == [f"R{n}" for n in range(1, 12)]
    assert [item["id"] for item in items if item["scored_by"] == "code"] == ["R1"]
    text = (dataset.RUBRICS / "v1.md").read_text(encoding="utf-8")
    assert all(f"| {item['id']} |" in text for item in items)


def test_rubric_v2_changes_r4_and_r10_and_adds_r12_scored_by_a_model() -> None:
    old = {item["id"]: item for item in dataset.rubric("v1")["items"]}
    new = {item["id"]: item for item in dataset.rubric("v2")["items"]}
    assert list(new) == [f"R{n}" for n in range(1, 13)]
    assert new["R12"]["scored_by"] == "model"
    assert {item_id for item_id in old if old[item_id] != new[item_id]} == {"R4", "R10"}
    assert "Method's required entries" in new["R4"]["pass_when"]
    assert "clear from context" in new["R10"]["pass_when"]
    text = (dataset.RUBRICS / "v2.md").read_text(encoding="utf-8")
    assert all(f"| {item_id} |" in text for item_id in new)


def test_the_dataset_reads_the_newest_rubric_by_default() -> None:
    assert dataset.rubric_versions() == ["v1", "v2"]
    assert dataset.latest_rubric() == "v2"
    assert dataset.model_items() == dataset.model_items("v2")
    assert dataset.rubric_digest() == dataset.rubric_digest("v2")


# --- anchors ---------------------------------------------------------------------------------------------------------


def _label(kind: str) -> list[dict[str, Any]]:
    return [entry for entry in dataset.labels() if entry["expect"] == kind]


def test_the_labels_cover_every_anchor_and_nothing_else() -> None:
    labeled = [entry["plan"] for entry in dataset.labels()]
    assert len(labeled) == len(set(labeled))
    on_disk = {path.relative_to(HERE).as_posix() for path in HERE.glob("r*/anchor.plan.json")}
    on_disk |= {path.relative_to(HERE).as_posix() for path in (HERE / "anchors" / "weak").glob("*.plan.json")}
    assert set(labeled) == on_disk
    assert {entry["labeled_by"] for entry in dataset.labels()} <= {"pending_owner", "owner"}


def test_each_inputs_anchor_passes_the_checks_and_r1() -> None:
    anchors = _label("pass")
    assert sorted(entry["input"] for entry in anchors) == ["r1", "r2", "r3", "r4", "r5"]
    for entry in anchors:
        report = dataset.report_input_of(entry["input"])
        plan = dataset.load_plan(HERE / entry["plan"])
        validate_plan(plan, report)
        assert dataset.code_score(plan, report)[0] == "pass", entry["plan"]


def test_each_refused_copy_fails_exactly_its_rule_and_every_rule_has_one() -> None:
    refused = _label("refused")
    assert {entry["rule"] for entry in refused} == RULES
    for entry in refused:
        plan = dataset.load_plan(HERE / entry["plan"])
        with pytest.raises(PlanRefusal) as caught:
            validate_plan(plan, dataset.report_input_of(entry["input"]))
        found = caught.value.faults
        assert [path for path, _ in found] == [path for path, _ in entry["faults"]], entry["plan"]
        assert all(fragment in why for (_, why), (_, fragment) in zip(found, entry["faults"], strict=True)), found
        assert dataset.code_score(plan, dataset.report_input_of(entry["input"]))[0] == "fail"


def _structure(plan: ReportPlan) -> Any:
    """A plan with every prose field blanked: what is left is its structure."""
    dumped = plan.model_dump(mode="json", by_alias=True)
    for path, _ in dataset.prose_fields(plan):
        if path.startswith("masthead"):
            continue
        node: Any = dumped
        steps = re.findall(r"[^.\[\]]+|\[\d+\]", path)
        for step in steps[:-1]:
            node = node[int(step[1:-1])] if step.startswith("[") else node[step]
        node[steps[-1]] = ""
    return dumped


def test_each_item_copy_changes_only_prose_and_every_model_item_has_one() -> None:
    weak = _label("fails_item")
    assert sorted(entry["item"] for entry in weak) == sorted(dataset.model_items())
    for entry in weak:
        plan = dataset.load_plan(HERE / entry["plan"])
        validate_plan(plan, dataset.report_input_of(entry["input"]))
        anchor = dataset.load_plan(HERE / entry["from"])
        assert plan != anchor, entry["plan"]
        assert _structure(plan) == _structure(anchor), entry["plan"]


def test_blanking_prose_leaves_a_structural_change_visible() -> None:
    anchor = dataset.load_plan(HERE / "r5" / "anchor.plan.json")
    moved = anchor.model_copy(update={"shape": "A"})
    assert _structure(moved) != _structure(anchor)
    assert all(field_is_prose(type(anchor).model_fields[name]) for name in ("headline", "lede"))


def test_weaken_anchors_writes_exactly_the_committed_copies_and_labels() -> None:
    files, labels = weaken_anchors.build()
    for path, text in files.items():
        assert (HERE / path).read_text(encoding="utf-8") == text, path
    assert json.loads(dataset.LABELS.read_text(encoding="utf-8")) == labels


# --- score records ---------------------------------------------------------------------------------------------------


def test_every_run_plan_has_one_sound_score_record() -> None:
    plans = dataset.run_plans()
    assert plans, "the dataset keeps no run plan"
    unscored: list[str] = []
    for plan in plans:
        if not dataset.scores_path(plan).is_file():
            unscored.append(plan.relative_to(HERE).as_posix())
            continue
        scores = dataset.PlanScores.model_validate_json(dataset.scores_path(plan).read_text(encoding="utf-8"))
        assert dataset.score_faults(scores, plan) == [], plan
    assert unscored == [], f"run plans with no score record (score them: dataset.py brief, then record): {unscored}"


def test_every_score_record_scores_a_plan_the_dataset_keeps() -> None:
    records = sorted(HERE.glob("**/*.scores.json"))
    assert records
    for path in records:
        assert path.with_name(path.name.removesuffix(".scores.json") + ".plan.json").is_file(), path


def test_a_runs_record_names_the_planner_its_provenance_records() -> None:
    for plan in dataset.run_plans():
        if dataset.scores_path(plan).is_file():
            scores = dataset.PlanScores.model_validate_json(dataset.scores_path(plan).read_text(encoding="utf-8"))
            assert scores.planner == dataset.planner_of(plan), plan


R5_ANCHOR = HERE / "r5" / "anchor.plan.json"


def _reply(**changes: dict[str, Any]) -> str:
    """A scorer's reply passing every item on the r5 anchor, with ``changes`` replacing items by id."""
    headline = dataset.load_plan(R5_ANCHOR).headline
    items = [
        {"id": item_id, "verdict": "pass", "reason": "It holds.", "quote": headline[:20], "labeled_by": "model"}
        for item_id in dataset.model_items()
    ]
    items = [{**item, **changes.get(item["id"], {})} for item in items]
    return json.dumps({"items": items, "scored_at": "2026-10-25T00:00:00+00:00"})


def test_a_sound_reply_becomes_a_record_with_every_digest_set() -> None:
    scores = dataset.record(R5_ANCHOR, "```json\n" + _reply() + "\n```", scorer="owner")
    assert scores.run_id == "anchor" and scores.input == "r5" and scores.scorer == "owner"
    assert scores.plan_digest == dataset.sha256_file(R5_ANCHOR)
    assert scores.rubric == dataset.RubricRef(version="v2", digest=dataset.rubric_digest("v2"))
    assert scores.planner.model == "hand-written"
    assert dataset.score_faults(scores, R5_ANCHOR) == []


def test_a_reply_quoting_the_rendered_prose_is_sound() -> None:
    rendered = "Keep September. Both candidates cost"
    scores = dataset.record(R5_ANCHOR, _reply(R2={"verdict": "fail", "quote": rendered}), scorer="claude-opus-5-5")
    assert scores.items[0].quote == rendered


@pytest.mark.parametrize(
    ("changes", "fault"),
    [
        ({"R3": {"quote": "words the plan never says"}}, "which no prose field of the plan holds"),
        ({"R4": {"verdict": "fail", "quote": ""}}, "fails with no quote"),
        ({"R5": {"id": "R1"}}, "item R5 is scored 0 times"),
    ],
)
def test_an_unsound_reply_is_refused(changes: dict[str, Any], fault: str) -> None:
    with pytest.raises(ValueError, match=re.escape(fault)):
        dataset.record(R5_ANCHOR, _reply(**changes), scorer="claude-opus-5-5")


def test_a_record_of_a_changed_plan_or_rubric_is_unsound(tmp_path: Path) -> None:
    scores = dataset.record(R5_ANCHOR, _reply(), scorer="owner")
    stale = scores.model_copy(update={"plan_digest": "0" * 64, "rubric": dataset.RubricRef(version="v2", digest="x")})
    faults = dataset.score_faults(stale, R5_ANCHOR)
    assert any("plan changed after it was scored" in fault for fault in faults)
    assert any("rubric digest" in fault for fault in faults)


def test_records_pool_only_by_prompt_rubric_and_scorer() -> None:
    base = dataset.record(R5_ANCHOR, _reply(), scorer="claude-opus-5-5")
    other_scorer = base.model_copy(update={"scorer": "owner"})
    other_prompt = base.model_copy(update={"planner": base.planner.model_copy(update={"prompt_digest": "p2"})})
    other_rubric = base.model_copy(update={"rubric": dataset.RubricRef(version="v1", digest="r1")})
    same = base.model_copy(update={"run_id": "another", "scored_at": "later"})
    grouped = dataset.pools([base, other_scorer, other_prompt, other_rubric, same])
    assert len(grouped) == 4
    assert grouped[dataset.pool_key(base)] == [base, same]


# --- the brief -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("plan", ["anchors/weak/item-R7.plan.json", "r5 current run"])
def test_the_brief_is_blind_and_carries_what_a_scorer_needs(plan: str) -> None:
    path = dataset.current_run("r5") if plan == "r5 current run" else HERE / plan
    brief = dataset.scoring_brief(path)
    assert path.name.removesuffix(".plan.json") not in brief
    assert "gemini" not in brief and "item-R" not in brief
    assert (dataset.RUBRICS / "v2.md").read_text(encoding="utf-8") in brief
    assert "R1 is code's and already scored: pass" in brief
    assert all(f'"id": "{item_id}"' in brief for item_id in dataset.model_items())
    assert "against rubric v2" in brief
    plan_model = dataset.load_plan(path)
    assert f"as written: {plan_model.headline}" in brief
