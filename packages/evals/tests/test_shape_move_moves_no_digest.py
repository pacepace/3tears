"""The reading-state shape move adds fields and moves nothing: no digest, no fingerprint, no dumped document.

The report input and the bundle gained every optional field the first-screen work fills later (a reading's state, the
caveat list, the first-screen value, the shapes of a check), all unset. Contracts held here:

- **Built fixtures digest as they did, but for the list.** For each hot-path campaign whose bundle is the same on
  every platform, the report input built from it (as a page reads it, and as an analyst's snapshot counts it) is the
  document it was before the fields existed, by the digest of the whole document with its floats rounded as a
  fingerprint rounds them (``fixtures/shape_move/input_digests.json``, written from the tree before the move), once
  the two keys the builder now fills are set apart: the caveat list and the first-screen value, both derived and
  outside the input's own digest; and each aspect's looks, which judges D1 fills on the bundle's reading scope and the
  input's aspect (held by ``test_looks.py``). What the list holds is held by ``test_caveat_list.py``. The bundles' own
  fingerprints and every page are held by the hot-path oracle.
- **A document with no new field set holds no new key.** No built input, but for those keys, and no assembled
  bundle, but for its looks, on any hot-path campaign, dumps a key the move added, at any depth.
- **A stored document reads as before.** Every stored report input fixture validates, carries none of the new
  fields, and digests as the model before the move digested it, whatever order its keys come in.
- **The first-screen value is outside the digest**, set or not.
- **The builder is on the numbers side**: it imports nothing from ``analysis/report/``.

``SHAPE_MOVE_PINS=1`` rewrites the pins, which is right only on a tree before the move.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.test_hot_path_oracle import PLATFORM_OVERRIDES, WITH_BUNDLES, subject
from threetears.evals.analysis import report_input_build
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.report_input import (
    CallRead,
    FirstScreen,
    ReportInput,
    report_input_digest,
)
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.schema.hashing import identity_json

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PINS = FIXTURES / "shape_move" / "input_digests.json"

#: The campaigns whose bundle is one document on every platform, so whose input's digest can be pinned once.
PINNED = tuple(name for name in WITH_BUNDLES if name not in PLATFORM_OVERRIDES)

#: Every key the move added to the report input, at any depth, that no input held before under another meaning
#: (``checks`` is also a pass rule's, and is looked for where the move put it).
INPUT_KEYS = frozenset(
    {
        "aa",
        "arm_id",
        "baseline",
        "baseline_n_cases",
        "caveat_list",
        "cell",
        "corrected",
        "cost",
        "delta_note",
        "first_screen",
        "flip_margin",
        "identity",
        "looks",
        "margin_informative",
        "no_variation",
        "ordered",
        "spelling",
        "spread",
        "state",
    }
)

#: Every key the move added to a bundle, at any depth, that no bundle held before under another meaning (``checks``,
#: ``state`` and ``family`` are looked for where the move put them).
BUNDLE_KEYS = frozenset(
    {
        "baseline",
        "baseline_n_cases",
        "corrected",
        "delta_note",
        "identity",
        "looks",
        "no_variation",
        "rescore",
        "rescore_share",
        "spread",
    }
)


def _campaign(name: str) -> CampaignBundle:
    read = subject(name)
    return assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id)


def _digests(name: str) -> dict[str, str]:
    bundle = _campaign(name)
    page, analyst = report_input(bundle), report_input(bundle, counted=True, per_slice=2, excerpt="failure")
    assert report_input_digest(page) != report_input_digest(analyst)
    return {"page": _rounded_digest(page), "analyst": _rounded_digest(analyst)}


#: The keys of a built input the builder fills since the move: the one caveat list and the first-screen value.
FILLED = frozenset({"caveat_list", "first_screen"})

#: What a chunk after the move fills where the move put it: each aspect's looks (judges D1), on the input's aspect and
#: on the bundle's reading scope. Set apart as the list is, and for the same reason: it is derived from stored facts
#: about the campaign, so it is outside ``report_input_digest`` (no input's digest or snapshot id moves with it) and,
#: on the bundle, outside the v2 fingerprint; ``test_looks.py`` holds both and what it holds.
FILLED_LOOKS = "looks"


def _without_looks(document: dict[str, Any]) -> dict[str, Any]:
    """An input's or a bundle's document with each aspect's filled looks set apart."""
    if "aspects" in document:
        aspects = [
            {key: value for key, value in aspect.items() if key != FILLED_LOOKS} for aspect in document["aspects"]
        ]
        return {**document, "aspects": aspects}
    if "reading_scope" in document:
        scope = {key: value for key, value in document["reading_scope"].items() if key != FILLED_LOOKS}
        return {**document, "reading_scope": scope}
    return document


def _rounded_digest(built: ReportInput) -> str:
    """The digest of an input's whole document as a bundle's fingerprint reads one, the keys the builder fills since
    the move set apart: every float rounded to twelve significant figures first (``identity_json``), since the last
    bits of a computed float differ between the platforms the suite runs on, and the pins are one file."""
    document = built.model_dump(mode="json", by_alias=True)
    assert FILLED <= set(document), "the builder fills the caveat list and the first screen"
    document = _without_looks({key: value for key, value in document.items() if key not in FILLED})
    return hashlib.sha256(identity_json(document).encode("utf-8")).hexdigest()


def _keys(node: Any) -> set[str]:
    """Every key of every object in a JSON document."""
    if isinstance(node, dict):
        return set(node).union(*(_keys(value) for value in node.values()))
    if isinstance(node, list):
        return set().union(*(_keys(value) for value in node))
    return set()


def _objects_with(node: Any, key: str) -> list[dict[str, Any]]:
    """Every object of a JSON document that holds ``key``."""
    if isinstance(node, dict):
        inner = [found for value in node.values() for found in _objects_with(value, key)]
        return [node, *inner] if key in node else inner
    return [found for value in node for found in _objects_with(value, key)] if isinstance(node, list) else []


def test_each_built_fixture_s_input_digest_is_what_it_was_before_the_move() -> None:
    built = {name: _digests(name) for name in PINNED}
    if os.environ.get("SHAPE_MOVE_PINS") == "1":
        PINS.parent.mkdir(exist_ok=True)
        PINS.write_text(json.dumps({"built": built, "stored": _stored_digests()}, indent=1, sort_keys=True) + "\n")
        pytest.fail("pins rewritten: review the diff, and never rewrite them on a tree after the move")
    assert PINNED, "no hot-path campaign is pinned"
    assert built == json.loads(PINS.read_text())["built"]


def _stored_digests() -> dict[str, str]:
    return {
        path.parent.name: report_input_digest(ReportInput.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        for path in _stored_inputs()
    }


def test_each_stored_input_s_digest_is_what_it_was_before_the_move() -> None:
    """A stored document names none of the new fields, and the model that reads it digests it as the model before the
    move did."""
    assert _stored_digests() == json.loads(PINS.read_text())["stored"]


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_a_built_input_and_bundle_hold_no_key_the_move_added(name: str) -> None:
    bundle = _campaign(name)
    for built in (report_input(bundle), report_input(bundle, counted=True)):
        document = _without_looks(
            {key: value for key, value in built.model_dump(mode="json", by_alias=True).items() if key not in FILLED}
        )
        assert not _keys(document) & INPUT_KEYS, sorted(_keys(document) & INPUT_KEYS)
        assert not any("checks" in judged for judged in document["judged"].values())
    assert not set(bundle.frame()) & FILLED, "the campaign bundle holds its list in memory and dumps none of it"
    for aspect in bundle.aspects:
        held = _keys(_without_looks(aspect.to_dict()))
        assert not held & BUNDLE_KEYS, sorted(held & BUNDLE_KEYS)
        assert "checks" not in aspect.to_dict()
        assert not any("checks" in judged for judged in aspect.to_dict()["judged_measures"])


def test_no_built_comparison_holds_a_state_or_a_family() -> None:
    comparisons = [
        comparison
        for name in WITH_BUNDLES
        for aspect in _campaign(name).aspects
        for comparison in _objects_with(aspect.to_dict(), "hedges_g")
    ]
    assert comparisons, "no hot-path campaign holds a comparison, so this checks nothing"
    assert not any({"state", "family"} & set(comparison) for comparison in comparisons)


def _stored_inputs() -> list[Path]:
    found = sorted((FIXTURES / "report_inputs").glob("*/input.json"))
    assert len(found) >= 5
    return found


@pytest.mark.parametrize("path", _stored_inputs(), ids=lambda path: path.parent.name)
def test_a_stored_input_without_the_new_fields_reads_as_before(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    read = ReportInput.model_validate(document)
    dumped = read.model_dump(mode="json", by_alias=True)

    assert read.caveat_list is None and read.first_screen is None
    assert not _keys(dumped) & INPUT_KEYS, sorted(_keys(dumped) & INPUT_KEYS)
    assert all(contrast.state is None for contrast in read.contrasts), "a contrast with no state reads counted"
    assert ReportInput.model_validate(dumped) == read


def test_a_stored_input_reads_whatever_order_its_keys_come_in() -> None:
    """A host's store may hand a document back with its keys in any order."""
    document = json.loads(_stored_inputs()[-1].read_text(encoding="utf-8"))

    def reverse(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: reverse(node[key]) for key in reversed(list(node))}
        return [reverse(value) for value in node] if isinstance(node, list) else node

    assert list(reverse(document)) != list(document)
    assert report_input_digest(ReportInput.model_validate(reverse(document))) == report_input_digest(
        ReportInput.model_validate(document)
    )


def test_the_first_screen_value_is_outside_the_input_s_digest() -> None:
    plain = ReportInput.model_validate(json.loads(_stored_inputs()[0].read_text(encoding="utf-8")))
    arm = plain.arms[0].id
    shown = plain.model_copy(
        update={"first_screen": FirstScreen(open_report_caveats=2, call=CallRead(arm=arm, made_by="model"))}
    )

    assert "first_screen" in shown.model_dump(mode="json", by_alias=True)
    assert report_input_digest(shown) == report_input_digest(plain)


def test_the_builder_imports_nothing_from_the_page() -> None:
    tree = ast.parse(Path(report_input_build.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    page = sorted(module for module in imported if module.startswith("threetears.evals.analysis.report."))
    assert page == []
    assert report_input_build.__name__ == "threetears.evals.analysis.report_input_build"
