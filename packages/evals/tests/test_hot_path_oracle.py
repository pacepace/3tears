"""The report and planning reads give the same output across every change that makes them faster.

Each campaign below is read the way a host reads it, and every output is compared with its golden as the house rule
compares it (:func:`~threetears.evals.schema.hashing.round_figures`: a computed float is compared or served to twelve
significant figures):

- every aspect's assembled bundle through :func:`~threetears.evals.schema.hashing.identity_json`, the pre-image its
  fingerprint hashes, and its ``fingerprint()``;
- the campaign bundle's frame (what a stored analysis keeps of it) through ``identity_json``, and the campaign
  fingerprint;
- every ``report_read`` body: markdown, html, the static html page and json, each with ``trials="table"`` and
  ``trials="all"``, byte for byte as served, and the read's other fields beside its body;
- the planning block a launch estimate carries (``detectable_effects``), for a campaign with a template.

The campaigns:

- ``scaled``: the production-shaped campaign of :mod:`~packages.evals.tests.hot_path_support` at ``scale=1``: 41
  runs, about 3,100 results, a code-only report;
- ``scaled_two_aspects``: the same runs split over two templates, one aspect each, so assembly runs once per aspect;
- ``toyhost``: the toy host's corpus campaign, code-only (``ops_support.ops_fixture``);
- ``toyhost_doc``: the same campaign with its report written by ``analysis_generate`` and stored, so the read is the
  stored document's page (doc basis). Read back from a capture (``inputs/toyhost_doc.json``), because a generation
  mints its analysis's id and dates its made line;
- ``courierhost``: the courier's planner bake-off, run through the real trial loop, read back from its capture
  (``inputs/courierhost.json``) for the same reason.

**What a golden holds.** A body of up to 4 KiB is held whole. A larger one is held as the sha256 of its UTF-8 bytes,
its length, and its first and last 1,024 characters, so a red names the body and shows where it starts and ends
without the goldens growing to megabytes. Each campaign's goldens are one file, ``fixtures/hot_path_oracle/<name>.json``.

**Regenerating** is for an owner-approved change of output, never a routine re-baseline: a red here means a change
moved what a host reads. With ``HOT_PATH_ORACLE_REGEN=1`` every test writes its golden instead of comparing it (run it
locally, on one process: ``TESTAMENT=off HOT_PATH_ORACLE_REGEN=1 ./scripts/test.sh evals -k hot_path_oracle``). A
capture under ``inputs/`` is written only when it is missing; it is the campaign read, not an output, so it is never
rewritten by a regeneration.

**One golden set, the pool's, holds on every platform but for one named exception.** The last digits of some statistics (a t interval's bounds and everything
computed from them, read through the platform's maths library) differ between platforms: the same code reads
``21.28666989160534`` on the pool and ``21.28666989160533`` in the aarch64 dev container on glibc 2.39. By house rule
those digits are not a difference: a bundle is compared through ``identity_json``, as its fingerprint is, and a served
page rounds every computed float it serializes (its JSON, the kit's ``data-chart``, the static page's SVG
coordinates and interval glyphs), so a body is compared as served, byte for byte. A value that falls on a
twelfth-figure rounding boundary is the rule's one stated exception, and the fixture holds one: a ``hedges_g`` of
``scaled_two_aspects`` (its small-sample factor subtracts two large log-gammas) rounds to neighbouring twelfth figures
on the aarch64 dev container on glibc 2.39 and on the pool, so that aspect's fingerprint, the campaign's and every
body that prints them differ there. Those entries, and only those (:data:`PLATFORM_OVERRIDES`), carry a second golden
for that platform until a fingerprint over inputs retires the case (#1062); an override of any other entry fails. The
goldens are the pool's (GitHub CI's x86_64 runner on glibc 2.39 reads them too). One red lists every entry of its test that moved, carrying the record it read
(between :data:`OBSERVED` marks). A regeneration is local only: it refuses in CI, on xdist and on testament's runners,
and every regenerating test fails, naming what it wrote.

**Where a reordered fold is caught.** Comparing past twelve figures no longer catches a fold whose float order moved
its last place. That is guarded where it is exact: ``test_readings_match_results.py``,
``test_aggregates_both_paths.py`` and ``test_bundle_both_paths.py`` compare the fast and the reference paths for exact equality, and the scaling counts
(``test_hot_path_scaling.py``) hold the work each read does. Floats are never compared at a tolerance here: twelve
figures is the form the house rule compares and serves, not a tolerance.

The report's clock is pinned (:data:`REPORT_NOW`), the one wall-clock value a code-only page prints.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import platform
from collections.abc import Callable, Coroutine, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis import service as analysis_service
from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops import report_read
from threetears.evals.ops.lenses import detectable_effects
from threetears.evals.schema.hashing import identity_json
from packages.evals.tests.fixtures.courierhost import COURIER_SCOPE, COURIER_TEMPLATE_ID, courier_cases
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.hot_path_support import (
    ARM_MODELS,
    CASES_PER_RUN,
    HOT_PATH_SCOPE,
    K_RUNS,
    campaign_id_of,
    courierhost_capture,
    courierhost_from,
    hot_path_corpus,
    toyhost_doc_capture,
    toyhost_from,
)
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture

#: Where the goldens live, one file per campaign, and the captured campaigns they are read from.
GOLDENS = Path(__file__).parent / "fixtures" / "hot_path_oracle"
INPUTS = GOLDENS / "inputs"

#: Set to write every golden instead of comparing it (see the module docstring).
REGENERATE = os.environ.get("HOT_PATH_ORACLE_REGEN") == "1"

#: The instant every code-only page is assembled at.
REPORT_NOW = "2026-10-30T09:00:00+00:00"

#: A body up to this many characters is held whole; a longer one by digest, length, head and tail.
WHOLE_UP_TO = 4096

#: How much of a long body's start and end its golden keeps.
EDGE = 1024

#: The platform a read ran on, for a red's message: the system, the machine, its C library and the Python build. Every
#: platform is held to the same goldens; a red names where it was read so a platform-only one is seen as such.
PLATFORM = f"{platform.system()}-{platform.machine()}-{'-'.join(platform.libc_ver())}-py{platform.python_version()}"

#: The platform as far as the one override goes: the machine and its C library.
PLATFORM_KEY = f"{platform.machine()}-{'-'.join(platform.libc_ver())}"

#: The one platform an override is held for: the aarch64 dev container on glibc 2.39.
OVERRIDE_PLATFORM = "aarch64-glibc-2.39"

#: The entries held a second time for :data:`OVERRIDE_PLATFORM`, by campaign, and no others. A ``hedges_g`` of
#: ``scaled_two_aspects``' ``tpl-support`` sits on a twelfth-figure rounding boundary that the platform's ``lgamma``
#: decides, so the aspect's bundle (compared through ``identity_json``), its fingerprint, the campaign's frame and
#: fingerprint, and every body printing a fingerprint differ there. #1062 (a fingerprint over inputs) is the fix; this
#: table, :data:`OVERRIDE_PLATFORM` and every ``@`` entry in the goldens are deleted when it lands.
PLATFORM_OVERRIDES: dict[str, frozenset[str]] = {
    "scaled_two_aspects": frozenset(
        {
            "bundle:tpl-support",
            "fingerprint:tpl-support",
            "frame",
            "campaign_fingerprint",
            *(
                f"body:{form}:{trials}"
                for form in ("markdown", "html", "html_static", "json")
                for trials in ("table", "all")
            ),
        }
    ),
}

#: What brackets the observed record in a red's message, so a pool run's report carries the record it read.
OBSERVED = "<<observed-record>>"

#: Every form a report is read in: ``(format, static)``.
FORMS = (("markdown", False), ("html", False), ("html", True), ("json", False))

#: Every trials detail a report is read with.
TRIALS = ("table", "all")


@dataclass(frozen=True)
class Planning:
    """The launch a campaign's planning block is read for: its template, its cases per arm, its repeats, its arms."""

    template_id: str
    n_cases: int | None
    k_runs: int
    n_arms: int


@dataclass(frozen=True)
class Subject:
    """One campaign the oracle reads: its host, its id and scope, and what besides its report is read of it.

    :ivar bundles: Whether its bundles are compared (a doc-basis read shares its campaign's bundles with the code-only
        one, so they are compared once).
    """

    host: EvalHost
    campaign_id: str
    scope_id: str
    bundles: bool
    planning: Planning | None


def _captured(name: str, capture: Callable[[], Coroutine[Any, Any, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """The captured campaign ``inputs/<name>.json``, captured first when it is missing and goldens are being written."""
    path = INPUTS / f"{name}.json"
    if not path.exists():
        if not REGENERATE:
            raise AssertionError(f"{path} is missing; it is captured once, under HOT_PATH_ORACLE_REGEN=1")
        INPUTS.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asyncio.run(capture()), indent=1, sort_keys=True) + "\n")
    documents: list[dict[str, Any]] = json.loads(path.read_text())
    return documents


@functools.cache
def subject(name: str) -> Subject:
    """The campaign ``name`` (see the module docstring), built or loaded once per process."""
    if name in ("scaled", "scaled_two_aspects"):
        corpus = hot_path_corpus(scale=1, two_aspects=name == "scaled_two_aspects")
        return Subject(
            corpus.host,
            corpus.campaign.id,
            HOT_PATH_SCOPE,
            bundles=True,
            planning=(
                Planning(corpus.template_ids[0], n_cases=CASES_PER_RUN, k_runs=K_RUNS, n_arms=len(ARM_MODELS))
                if name == "scaled"
                else None
            ),
        )
    if name == "toyhost":
        fixture = ops_fixture()
        return Subject(
            fixture.host.eval_host,
            fixture.campaign.id,
            TOYHOST_SCOPE,
            bundles=True,
            planning=Planning(toyhost_template().id, n_cases=12, k_runs=3, n_arms=2),
        )
    if name == "toyhost_doc":
        documents = _captured(name, toyhost_doc_capture)
        return Subject(toyhost_from(documents), campaign_id_of(documents), TOYHOST_SCOPE, bundles=False, planning=None)
    if name == "courierhost":
        documents = _captured(name, courierhost_capture)
        return Subject(
            courierhost_from(documents),
            campaign_id_of(documents),
            COURIER_SCOPE,
            bundles=True,
            planning=Planning(COURIER_TEMPLATE_ID, n_cases=len(courier_cases()), k_runs=2, n_arms=2),
        )
    raise KeyError(name)


#: Every campaign, and those whose bundles and planning blocks are read.
CAMPAIGNS = ("scaled", "scaled_two_aspects", "toyhost", "toyhost_doc", "courierhost")
WITH_BUNDLES = ("scaled", "scaled_two_aspects", "toyhost", "courierhost")
WITH_PLANNING = ("scaled", "toyhost", "courierhost")


def record(text: str) -> dict[str, Any]:
    """What a golden holds of ``text`` (see the module docstring)."""
    if len(text) <= WHOLE_UP_TO:
        return {"text": text}
    data = text.encode()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "head": text[:EDGE], "tail": text[-EDGE:]}


def refuse_unattended_regeneration() -> None:
    """Refuse to write goldens anywhere but one local process: never in CI, on an xdist worker or a testament runner.

    Parallel writers would race on each campaign's one file, and goldens written where nobody reviews the diff are a
    silent re-baseline.
    """
    unattended = sorted(
        name for name in os.environ if name in ("CI", "PYTEST_XDIST_WORKER") or name.startswith("TESTAMENT_")
    )
    if unattended:
        pytest.fail(
            f"HOT_PATH_ORACLE_REGEN refuses to write goldens here ({', '.join(unattended)} set): run it locally"
        )


class Goldens:
    """One campaign's goldens, and every entry a test compares against them.

    Every entry a test compares is checked before it fails, so one red names every entry that moved, and every entry
    the file has no golden for, each with the record it read. The file holds one golden per entry, the pool's, and,
    for an entry :data:`PLATFORM_OVERRIDES` names, ``<entry>@<OVERRIDE_PLATFORM>`` beside it, read there instead.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.path = GOLDENS / f"{name}.json"
        self.stored: dict[str, Any] = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.problems: list[str] = []
        self.written: list[str] = []

    def entries(self) -> list[str]:
        """The entries the file holds, without the overrides."""
        return sorted(key for key in self.stored if "@" not in key)

    def overrides(self) -> list[str]:
        """The overrides the file holds, each ``<entry>@<platform>``."""
        return sorted(key for key in self.stored if "@" in key)

    def key(self, entry: str) -> str:
        """Where ``entry``'s golden is held for this platform: its override where :data:`PLATFORM_OVERRIDES` names it
        and this is :data:`OVERRIDE_PLATFORM`, else the entry itself."""
        overridden = PLATFORM_KEY == OVERRIDE_PLATFORM and entry in PLATFORM_OVERRIDES.get(self.name, frozenset())
        return f"{entry}@{OVERRIDE_PLATFORM}" if overridden else entry

    def check(self, entry: str, text: str) -> None:
        """Compare ``text`` with ``entry``'s golden, or write it under :data:`REGENERATE`."""
        observed = record(text)
        if REGENERATE:
            self.regenerate(entry, observed)
            return
        carried = json.dumps(
            {"campaign": self.name, "entry": entry, "platform": PLATFORM, "record": observed}, ensure_ascii=False
        )
        golden = self.stored.get(self.key(entry))
        if golden is None:
            self.problems.append(
                f"{self.name} {entry}: no golden; a regeneration writes it {OBSERVED}{carried}{OBSERVED}"
            )
            return
        if observed != golden:
            moved = next(
                (
                    index
                    for index, (was, now) in enumerate(zip(golden.get("text", golden.get("head", "")), text))
                    if was != now
                ),
                None,
            )
            self.problems.append(
                f"{self.name} {entry} moved on {PLATFORM}: {observed.get('bytes', len(text))} bytes now, "
                f"{golden.get('bytes', len(golden.get('text', '')))} in the golden; first difference in its head at "
                f"character {moved}; head now {text[:200]!r} {OBSERVED}{carried}{OBSERVED}"
            )

    def regenerate(self, entry: str, observed: dict[str, Any]) -> None:
        """Write ``observed`` as ``entry``'s golden for this platform (:meth:`key`): on :data:`OVERRIDE_PLATFORM` an
        overridden entry's override, never the pool's golden beside it, which a pool red's record supplies."""
        refuse_unattended_regeneration()
        self.stored[self.key(entry)] = observed
        GOLDENS.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.stored, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        self.written.append(entry)

    def assert_unchanged(self) -> None:
        """Fail naming every entry that moved or has no golden; under :data:`REGENERATE`, fail naming what was written,
        so a regenerating run never reads as green."""
        if self.written:
            pytest.fail(f"{self.name}: goldens written on {PLATFORM} for {', '.join(self.written)}: review the diff")
        assert not self.problems, "\n".join(self.problems)


@pytest.fixture
def pinned_clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every code-only page assembled at :data:`REPORT_NOW`."""
    monkeypatch.setattr(analysis_service, "utc_now_iso", lambda: REPORT_NOW)
    yield


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_every_aspect_s_bundle_and_the_campaign_frame_are_unchanged(name: str, pinned_clock: None) -> None:
    read = subject(name)
    bundle = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id)

    goldens = Goldens(name)

    assert bundle.aspects, f"{name} assembled no aspect"
    if not REGENERATE:
        pinned = sorted(entry.removeprefix("bundle:") for entry in goldens.entries() if entry.startswith("bundle:"))
        assert sorted(aspect.aspect for aspect in bundle.aspects) == pinned, f"{name}'s aspects are not the pinned ones"
    for aspect in bundle.aspects:
        goldens.check(f"bundle:{aspect.aspect}", identity_json(aspect.to_dict()))
        goldens.check(f"fingerprint:{aspect.aspect}", aspect.fingerprint())
    goldens.check("frame", identity_json(bundle.frame()))
    goldens.check("campaign_fingerprint", bundle.fingerprint())
    goldens.assert_unchanged()


@pytest.mark.parametrize("trials", TRIALS)
@pytest.mark.parametrize(("form", "static"), FORMS, ids=["markdown", "html", "html_static", "json"])
@pytest.mark.parametrize("name", CAMPAIGNS)
def test_every_report_read_body_is_unchanged(
    name: str, form: Any, static: bool, trials: Any, pinned_clock: None
) -> None:
    read = subject(name)
    document = report_read(read.host, read.campaign_id, read.scope_id, format=form, static=static, trials=trials)
    form_id = f"{form}_static" if static else form
    goldens = Goldens(name)

    goldens.check(f"body:{form_id}:{trials}", document.body)
    goldens.check(f"read:{form_id}:{trials}", _json(document.model_dump(mode="json", exclude={"body"})))
    goldens.assert_unchanged()


@pytest.mark.parametrize("name", WITH_PLANNING)
def test_the_planning_block_a_launch_estimate_carries_is_unchanged(name: str) -> None:
    read = subject(name)
    planning = read.planning
    assert planning is not None
    block = detectable_effects(
        read.host,
        read.scope_id,
        planning.template_id,
        n_cases=planning.n_cases,
        k_runs=planning.k_runs,
        n_arms=planning.n_arms,
    )

    goldens = Goldens(name)

    goldens.check("planning", block.model_dump_json())
    goldens.assert_unchanged()


def test_the_scaled_page_states_how_many_trials_it_lists_of_how_many_ran(pinned_clock: None) -> None:
    """The bundle's trials are a bounded sample (at most ``MAX_TRIALS``, failed and lost first): the page says how many
    more ran than it lists, so a reader never takes the listed trials for the whole campaign."""
    read = subject("scaled")
    bundle = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id)
    (aspect,) = bundle.aspects
    assert aspect.trials_omitted > 0, "the scaled campaign must run past the trial cap for this to test anything"
    body = report_read(read.host, read.campaign_id, read.scope_id, format="markdown").body

    assert f"{aspect.trials_omitted} more trials count on the scoreboard and are not listed here" in body


@pytest.mark.parametrize("name", CAMPAIGNS)
def test_every_campaign_holds_one_golden_per_form_and_overrides_only_the_named_entries(name: str) -> None:
    """Every form of every campaign's page has its golden, and the only goldens held for one platform are the
    overrides :data:`PLATFORM_OVERRIDES` names, each beside the pool's: an override of any other entry, or for any
    other platform, fails here (see the module docstring)."""
    goldens = Goldens(name)
    entries = set(goldens.entries())
    expected = {f"body:{form}:{trials}" for form in ("markdown", "html", "html_static", "json") for trials in TRIALS}
    expected |= {entry.replace("body:", "read:", 1) for entry in expected}

    assert expected <= entries, f"{name} lacks goldens for {sorted(expected - entries)}"
    named = PLATFORM_OVERRIDES.get(name, frozenset())
    assert set(goldens.overrides()) == {f"{entry}@{OVERRIDE_PLATFORM}" for entry in named}, (
        f"{name}'s overrides are not the named ones (#1062): {goldens.overrides()}"
    )
    assert named <= entries, f"{name} overrides an entry with no pool golden beside it"


@pytest.mark.parametrize("variable", ["CI", "PYTEST_XDIST_WORKER", "TESTAMENT_RUNNER"])
def test_a_regeneration_refuses_to_write_goldens_where_nobody_reviews_them(
    variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regeneration writes goldens on one local process only: in CI, on an xdist worker or on a testament runner it
    refuses, naming why."""
    for name in [name for name in os.environ if name in ("CI", "PYTEST_XDIST_WORKER") or name.startswith("TESTAMENT_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv(variable, "1")

    with pytest.raises(pytest.fail.Exception, match=rf"refuses to write goldens here \({variable} set\)"):
        refuse_unattended_regeneration()
