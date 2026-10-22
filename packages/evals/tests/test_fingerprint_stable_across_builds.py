"""A bundle's fingerprint, and the page JSON a campaign serves, are the same on every Python build.

The bundle's intervals and multipliers come from C maths (``math.lgamma``, ``math.exp``, ``NormalDist().inv_cdf``)
whose last bits differ between two builds of one Python: a uv-managed CPython 3.14.6 and Debian's give
``NormalDist().inv_cdf(0.975)`` as ``...0534`` and ``...0536``. Hashed in full, one evidence fingerprinted
differently on a laptop and in a container, and a reporter case re-frozen on the other was refused as moved evidence.
Pinned here:

- **The helper**: :func:`identity_json` is canonical JSON with every float, at any depth, to twelve significant
  figures; ``-0.0`` is ``0.0``; NaN and the infinities are their one token each; nothing else moves.
- **A bundle moved by a ULP fingerprints alike**: every nonzero float of the toy corpus's bundle nudged one ULP up, or down,
  leaves the ``v2:`` fingerprint as it was, the aspect's and the campaign's; and the same holds for the readings the
  page JSON serves.
- **A build whose C maths differs fingerprints alike**: a fresh interpreter whose ``math`` and ``NormalDist`` answer
  one ULP off assembles the toy campaign to the same ``v2:`` fingerprint and serves the same page JSON, while the
  unrounded digest moves, which is the defect.
- **A real difference still moves it**: a statistic changed in its tenth figure is another fingerprint.
- **A fingerprint stored before the rounding still matches**: an unprefixed (version 1) fingerprint is compared as it
  was taken, by a frozen case's own bundle, by a re-freeze of the same evidence, by the limits and by an inspection.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from typing import Any

import pytest

from threetears.evals.analysis import assemble_campaign_bundle, campaign_report
from threetears.evals.analysis.bundle.schema import BundleInspection, evidence_digest
from threetears.evals.analysis.campaign_bundle import CampaignBundle, campaign_evidence_digest, writer_view
from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.report.model import CampaignReport
from threetears.evals.analysis.reporter_bank import case_limits
from threetears.evals.analysis.reporter_kind import rebuild_bundle, reporter_case_of, reporter_case_payload
from threetears.evals.schema.hashing import (
    UnhashableContentError,
    canonical_digest,
    canonical_json,
    identity_digest,
    identity_json,
    round_figures,
)
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.ops_support import TOYHOST_SCOPE
from packages.evals.tests.report_support import toy_campaign_host
from packages.evals.tests.reporter_case_support import freeze_corpus_campaign, reporter_fixture, reporter_template
from packages.evals.tests.test_package_matrix import REPO_ROOT

#: What the bundle's fingerprint leaves out, as version 1 did.
_NOT_EVIDENCE = ("package_version", "first_result_at")


def _v1_aspect(document: dict[str, Any]) -> str:
    """An aspect bundle's fingerprint as version 1 took it: every float in full, no prefix."""
    return canonical_digest({key: value for key, value in document.items() if key not in _NOT_EVIDENCE})


def _v1_campaign(document: dict[str, Any]) -> str:
    """A campaign bundle's fingerprint as version 1 took it, from its stored document."""
    frame = {key: value for key, value in document.items() if key != "aspects"}
    frame["aspects"] = [
        {"aspect": aspect["aspect"], "fingerprint": _v1_aspect(aspect)} for aspect in document["aspects"]
    ]
    return canonical_digest(frame)


#: What a person declared on a campaign's readings: the same on every build, so never nudged.
_DECLARED = frozenset({"alpha", "bar", "margin"})


def _nudged(node: Any, toward: float, *, declared: frozenset[str] = frozenset()) -> Any:
    """``node`` with every nonzero float, at any depth, one ULP toward ``toward``, keys in ``declared`` apart.

    Zero stays zero: one ULP of it is a subnormal, an absolute step no relative rounding absorbs, and a zero in a
    bundle is an exact result (a mean of zero costs, an empty share), never a C maths one whose last bit varies.
    """
    if isinstance(node, float):
        return math.nextafter(node, toward) if node != 0 else node
    if isinstance(node, dict):
        return {
            key: value if key in declared else _nudged(value, toward, declared=declared) for key, value in node.items()
        }
    if isinstance(node, list):
        return [_nudged(value, toward, declared=declared) for value in node]
    return node


def _floats(node: Any) -> list[float]:
    """Every float in ``node``, at any depth."""
    if isinstance(node, float):
        return [node]
    if isinstance(node, dict):
        return [found for value in node.values() for found in _floats(value)]
    if isinstance(node, list):
        return [found for value in node for found in _floats(value)]
    return []


def _toy_campaign() -> tuple[Any, Any, CampaignBundle]:
    """The toy host holding its corpus campaign, the campaign, and the campaign bundle a writer would read."""
    host, campaign = toy_campaign_host()
    return host, campaign, assemble_campaign_bundle(host, campaign.id, campaign.scope_id)


# =============================================================================
# The helper
# =============================================================================


def test_a_float_is_rounded_to_twelve_significant_figures_and_a_typed_one_is_unchanged() -> None:
    assert round_figures(1.9599639845400536) == round_figures(1.9599639845400534) == 1.95996398454
    assert round_figures(0.030000000000000002) == 0.03
    assert round_figures(123456789.01234567) == 123456789.012
    assert round_figures(1.2345678901234e-20) == 1.23456789012e-20
    for typed in (0.7, 2.5, 0.05, 100.0, 1e-9, 0.0):
        assert round_figures(typed) == typed


def test_minus_zero_is_zero_and_nan_and_the_infinities_are_one_token_each() -> None:
    assert math.copysign(1.0, round_figures(-0.0)) == 1.0
    assert identity_json({"x": -0.0}) == identity_json({"x": 0.0}) == '{"x":0.0}'
    assert identity_json([math.nan, math.inf, -math.inf]) == "[NaN,Infinity,-Infinity]"
    assert identity_json([math.nan]) == canonical_json([math.nan]), "NaN hashes as it did"


def test_every_float_at_any_depth_is_rounded_and_nothing_else_moves() -> None:
    nested = {"b": [1, True, None, "0.10000000000000001", (0.1 + 0.2, {"c": 1 / 3})], "a": 2.5}
    assert identity_json(nested) == ('{"a":2.5,"b":[1,true,null,"0.10000000000000001",[0.3,{"c":0.333333333333}]]}'), (
        "keys sorted, ints, bools, None and strings as they were, a tuple as a list"
    )
    assert identity_json({"b": 1, "a": 2}) == identity_json({"a": 2, "b": 1}), "key order is irrelevant"
    assert identity_digest(nested) == identity_digest(
        {"a": 2.5, "b": [1, True, None, "0.10000000000000001", [0.3, {"c": 0.333333333333}]]}
    )
    assert identity_json({"n": 7}) == canonical_json({"n": 7}), "an int is not a float"


def test_a_value_json_cannot_encode_is_refused_as_canonical_json_refuses_it() -> None:
    with pytest.raises(UnhashableContentError):
        identity_json({"when": object()})


# =============================================================================
# One ULP: the same fingerprint, the same page JSON
# =============================================================================


@pytest.mark.parametrize("toward", [math.inf, -math.inf], ids=["up", "down"])
def test_every_float_of_a_bundle_moved_one_ulp_leaves_its_fingerprint(toward: float) -> None:
    bundle = toyhost_bundle()
    document = bundle.to_dict()
    nudged = _nudged(document, toward)
    assert len(_floats(document)) > 100, "the toy corpus's bundle carries its statistics"
    assert _v1_aspect(nudged) != _v1_aspect(document), "a ULP moves the unrounded digest: the defect"

    assert bundle.fingerprint().startswith("v2:")
    assert evidence_digest(nudged) == evidence_digest(document) == bundle.fingerprint()


@pytest.mark.parametrize("toward", [math.inf, -math.inf], ids=["up", "down"])
def test_every_float_of_a_campaign_bundle_moved_one_ulp_leaves_its_fingerprint(toward: float) -> None:
    _, _, bundle = _toy_campaign()
    document = bundle.to_dict()
    nudged = _nudged(document, toward)
    assert _v1_campaign(nudged) != _v1_campaign(document)

    assert campaign_evidence_digest(nudged) == campaign_evidence_digest(document) == bundle.fingerprint()


@pytest.mark.parametrize("toward", [math.inf, -math.inf], ids=["up", "down"])
def test_readings_moved_one_ulp_serve_the_same_page_json(toward: float) -> None:
    host, campaign, _ = _toy_campaign()
    report = campaign_report(host, campaign.id, campaign.scope_id)
    assert isinstance(report, CampaignReport)
    readings = report.readings.to_dict()
    nudged = report.model_copy(
        update={"readings": CampaignReadings.model_validate(_nudged(readings, toward, declared=_DECLARED))}
    )
    assert nudged.readings.to_dict() != readings, "the readings did move"

    assert json.dumps(nudged.json_document()) == json.dumps(report.json_document())


def test_the_page_json_serves_each_computed_figure_to_twelve_figures_and_a_declared_one_as_declared() -> None:
    host, campaign, _ = _toy_campaign()
    report = campaign_report(host, campaign.id, campaign.scope_id)
    assert isinstance(report, CampaignReport)
    served = report.json_document()["readings"]
    assert isinstance(served, dict)
    computed = report.readings.to_dict()
    assert [value for value in _floats(computed) if round_figures(value) != value], "the toy's readings run long"
    assert all(round_figures(value) == value for value in _floats(served))
    assert served["alpha"] == computed["alpha"]
    assert [bar["bar"] for bar in served["bars"]] == [bar["bar"] for bar in computed["bars"]]
    assert [round_figures(bar["value"]) for bar in computed["bars"] if bar["value"] is not None] == [
        bar["value"] for bar in served["bars"] if bar["value"] is not None
    ]


def test_the_writer_reads_a_bars_figures_and_a_rows_costs_to_its_four_figures() -> None:
    _, _, bundle = _toy_campaign()
    view = writer_view(bundle)
    verdicts = [verdict for aspect in view["aspects"] for bar in aspect["bars"] for verdict in bar["verdicts"]]
    figures = [verdict[key] for verdict in verdicts for key in ("value", "sem", "ci_low", "ci_high")]
    figures += [
        row[key]
        for aspect in view["aspects"]
        for row in aspect["scoreboard"]
        for key in ("test_cost_usd", "reply_cost_usd")
    ]
    held = [figure for figure in figures if figure is not None]
    assert held, "the toy campaign holds a bar and a priced reply"
    assert all(float(f"{figure:.4g}") == figure for figure in held)
    assert all(round_figures(value) == value for value in _floats(view)), "nothing the writer reads runs past 12"


# =============================================================================
# A build whose C maths answers one ULP off
# =============================================================================

#: Run in a fresh interpreter, since the stats module caches what it computed: ``math``'s transcendental functions
#: and ``NormalDist``'s two maths methods answer one ULP toward ``argv[1]`` (or as they do, for ``none``), then the
#: toy campaign is assembled and its page served.
_ANOTHER_BUILD = """
import json, math, sys
from statistics import NormalDist

step = sys.argv[1]
if step != "none":
    toward = math.inf if step == "up" else -math.inf

    def nudged(compute):
        def call(*args, **kwargs):
            value = compute(*args, **kwargs)
            return math.nextafter(value, toward) if math.isfinite(value) and value != 0 else value
        return call

    for name in ("lgamma", "exp", "log", "log1p", "expm1", "erf", "erfc"):
        setattr(math, name, nudged(getattr(math, name)))
    NormalDist.inv_cdf = nudged(NormalDist.inv_cdf)
    NormalDist.cdf = nudged(NormalDist.cdf)

from threetears.evals.analysis import assemble_campaign_bundle, campaign_report
from packages.evals.tests.report_support import toy_campaign_host

host, campaign = toy_campaign_host()
bundle = assemble_campaign_bundle(host, campaign.id, campaign.scope_id)
print(json.dumps({
    "fingerprint": bundle.fingerprint(),
    "document": bundle.to_dict(),
    "page": campaign_report(host, campaign.id, campaign.scope_id).json_document(),
}))
"""


def _on_another_build(step: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-c", _ANOTHER_BUILD, step],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    answer: dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
    return answer


def test_a_build_whose_c_maths_answers_one_ulp_off_fingerprints_and_serves_alike() -> None:
    this = _on_another_build("none")
    for step in ("up", "down"):
        other = _on_another_build(step)
        assert _v1_campaign(other["document"]) != _v1_campaign(this["document"]), (
            f"one ULP {step} moves the unrounded digest, so this build differs as two real ones do"
        )
        assert other["fingerprint"] == this["fingerprint"]
        assert json.dumps(other["page"]) == json.dumps(this["page"])


# =============================================================================
# A real difference is still a difference
# =============================================================================


def test_a_statistic_changed_in_its_tenth_figure_is_another_fingerprint() -> None:
    bundle = toyhost_bundle()
    document = bundle.to_dict()
    moved = False

    def change_one(node: Any) -> Any:
        nonlocal moved
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                if not moved and key == "ci_high" and isinstance(value, float) and value != 0:
                    out[key] = value * (1 + 1e-10)
                    moved = True
                else:
                    out[key] = change_one(value)
            return out
        if isinstance(node, list):
            return [change_one(value) for value in node]
        return node

    changed = change_one(document)
    assert moved
    assert evidence_digest(changed) != bundle.fingerprint()


# =============================================================================
# A fingerprint stored before the rounding
# =============================================================================


def test_an_unprefixed_fingerprint_is_compared_as_it_was_taken() -> None:
    bundle = toyhost_bundle()
    v1 = _v1_aspect(bundle.to_dict())

    assert bundle.fingerprint(like=v1) == v1
    assert evidence_digest(bundle.to_dict(), like=v1) == v1
    assert bundle.fingerprint(like=bundle.fingerprint()) == bundle.fingerprint() != v1
    inspected = BundleInspection.over(bundle, analysis_id="a", recorded_fingerprint=v1)
    assert (inspected.reproduces_generation, inspected.fingerprint) == (True, bundle.fingerprint())


def test_a_frame_stored_with_unprefixed_fingerprints_still_names_its_bundles() -> None:
    _, _, bundle = _toy_campaign()
    frame = bundle.frame()
    assert all(entry["fingerprint"].startswith("v2:") for entry in frame["aspects"])
    stored = {
        **frame,
        "aspects": [{"aspect": a.aspect, "fingerprint": _v1_aspect(a.to_dict())} for a in bundle.aspects],
    }

    assert CampaignBundle.from_frame(stored, bundle.aspects).fingerprint() == bundle.fingerprint()
    v1 = _v1_campaign(bundle.to_dict())
    assert bundle.fingerprint(like=v1) == v1
    assert not [limit for limit in case_limits(bundle, recorded_fingerprint=v1) if "does not reproduce" in limit]
    assert [limit for limit in case_limits(bundle, recorded_fingerprint=_another(v1)) if "does not reproduce" in limit]


def test_a_reporter_case_frozen_before_the_rounding_still_holds_its_bundle_and_is_its_evidences_case() -> None:
    fixture = reporter_fixture()
    storage = fixture.host.eval_host.storage
    receipt = freeze_corpus_campaign(fixture)
    assert receipt.bundle_fingerprint.startswith("v2:")
    (stored,) = storage.query_test_cases(TOYHOST_SCOPE, template_id=reporter_template().id)
    case = reporter_case_of(stored)
    assert case is not None
    # As a build before the rounding froze it: the same case, its fingerprint unprefixed.
    before = case.model_copy(update={"bundle_fingerprint": _v1_campaign(case.bundle)})
    storage.save_test_case(stored.model_copy(update={"host_payload": reporter_case_payload(before)}))

    assert rebuild_bundle(before).fingerprint() == case.bundle_fingerprint
    again = freeze_corpus_campaign(fixture)
    assert again.test_case_id == stored.id, "the same evidence re-frozen on the same machine is its live case"
    assert again.bundle_fingerprint == before.bundle_fingerprint


def _another(fingerprint: str) -> str:
    """``fingerprint`` with its last digit changed: a fingerprint of some other bundle, in the same version."""
    return fingerprint[:-1] + ("0" if fingerprint[-1] != "0" else "1")


def test_a_stored_fingerprint_of_another_bundle_does_not_match_in_either_version() -> None:
    bundle = toyhost_bundle()
    for recorded in (bundle.fingerprint(), _v1_aspect(bundle.to_dict())):
        inspected = BundleInspection.over(bundle, analysis_id="a", recorded_fingerprint=_another(recorded))
        assert inspected.reproduces_generation is False
