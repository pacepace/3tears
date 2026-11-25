"""The report and planning reads give the same output across every change that makes them faster.

Each campaign below is read the way a host reads it, and every output is compared with its golden as the house rule
compares it (:func:`~threetears.evals.schema.hashing.round_figures`: a computed float is compared or served to twelve
significant figures):

- every aspect's assembled bundle through :func:`~threetears.evals.schema.hashing.identity_json`, and its
  ``fingerprint()``, the digest of the evidence it was assembled from;
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

**The build is not compared.** Which version of 3tears-evals read a page is the build's identity, not the page's
content, and a release bump would otherwise move every bundle and every code-only page. A bundle carries it as
``package_version``, which its fingerprint already leaves out (a build is not evidence), and a code-only
page prints it in one place, its methods table's Evidence row (``3tears-evals <version>,``; on an HTML page the
package's hyphenated name held whole, ``3tears-evals</span> <version>,``). The oracle puts
:data:`GOLDEN_VERSION` in both places before it compares or writes, so the goldens hold that version whatever build
reads them. Nothing else the fingerprint leaves out is the build's: ``first_result_at`` is when the results were
written, and the presentation and pricing fields are what the page prints, so all of them stay compared.

**Goldens move only when a change to a page is intended** (owner ruling, 2026-11-06): regenerate with
``HOT_PATH_ORACLE_REGEN=1``, review the golden diff line by line, state the reason in the commit and the chunk report,
and never regenerate to make a red go away: a red here means a change moved what a host reads. A regeneration writes
every golden instead of comparing it (run it locally, on one process: ``TESTAMENT=off
TESTAMENT_OFF_REASON="regenerating the hot-path goldens" HOT_PATH_ORACLE_REGEN=1 ./scripts/test.sh evals -k
hot_path_oracle``, the reason ``test.sh`` asks of every local run), and each regenerating test fails naming the entries it moved, with their
sizes before and after, and how many it left as they were. A capture under ``inputs/`` is written only when it is
missing; it is the campaign read, not an output, so it is never rewritten by a regeneration.

**One golden set, the pool's, holds on every platform but for one named entry.** The last digits of some statistics
(a t interval's bounds and everything computed from them, read through the platform's maths library) differ between
platforms: the same code reads ``21.28666989160534`` on the pool and ``21.28666989160533`` in the aarch64 dev container
on glibc 2.39. By house rule those digits are not a difference: a bundle is compared through ``identity_json``, and a
served page rounds every computed float it serializes (its JSON, the kit's ``data-chart``, the static page's SVG
coordinates and interval glyphs), so a body is compared as served, byte for byte.

**A fingerprint reads none of them.** It is the digest of the evidence the bundle was assembled from (``v3:``, #1062),
which no maths library touches, so every fingerprint here, the campaign's frame (which names each aspect by its
fingerprint) and every body that prints one is one value for every platform. Until version 3 a fingerprint hashed
the bundle's statistics, and ``scaled_two_aspects`` held twelve entries a second time for the dev container.

**One entry is still held twice**, and it is no fingerprint: ``scaled_two_aspects``' ``tpl-support`` bundle, compared
whole as computed, holds one corrected p-value (``multiple_comparisons.families[0].comparisons[7].p_adjusted``,
``1.5029458234250586e-05`` on the pool and ``1.5029458234249729e-05`` on the aarch64 dev container on glibc 2.39) that
falls on a twelfth-figure rounding boundary, the house rule's one stated exception. That entry, and only that
(:data:`PLATFORM_OVERRIDES`), carries a second golden for that platform; an override of any other entry fails. The
goldens are the pool's (GitHub CI's x86_64 runner on glibc 2.39 reads them too). One red lists every entry of its test that moved, carrying the record it read
(between :data:`OBSERVED` marks). A regeneration is local only: it refuses in CI, on xdist and on testament's runners,
and every regenerating test fails, naming what it moved.

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
from types import SimpleNamespace
from typing import Any

import pytest

from threetears.evals.analysis import service as analysis_service
from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.kernel import build_stamp
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops import report_read
from threetears.evals.ops.lenses import detectable_effects
from threetears.evals.schema.hashing import identity_json
from threetears.evals.kernel.numbers import spell_count
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

#: The version every golden holds as the build that read it (see the module docstring): the version the goldens were
#: written at when the build was first left out of the comparison. A read's own version is put in its place.
GOLDEN_VERSION = "0.90.0"

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

#: The entries held a second time for :data:`OVERRIDE_PLATFORM`, by campaign, and no others: one bundle, compared whole
#: as computed, whose ``p_adjusted`` of one comparison sits on a twelfth-figure rounding boundary that the platform's
#: maths library decides (see the module docstring). No fingerprint, frame or body is among them: a fingerprint is
#: over the evidence (#1062), so the statistic reaches none of them.
PLATFORM_OVERRIDES: dict[str, frozenset[str]] = {"scaled_two_aspects": frozenset({"bundle:tpl-support"})}

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


def size(stored: dict[str, Any]) -> int:
    """How many UTF-8 bytes the body a golden's record holds has."""
    return int(stored["bytes"]) if "bytes" in stored else len(stored["text"].encode())


def bundle_without_build(aspect: AnalysisContextBundle) -> str:
    """``aspect``'s bundle through ``identity_json``, the version that assembled it put as :data:`GOLDEN_VERSION`."""
    document = aspect.to_dict()
    assert "package_version" in document, "the bundle no longer carries package_version: update the oracle"
    return identity_json({**document, "package_version": GOLDEN_VERSION})


def body_without_build(body: str) -> str:
    """``body`` with the version the running build printed in its methods table put as :data:`GOLDEN_VERSION`.

    Only the one token a code-only page prints the build in (``3tears-evals <version>,``, its name held whole on an
    HTML page: :func:`printed_build`) is replaced; a page that prints no such token (the JSON, a stored document's
    page) comes back as it was.
    """
    version = build_stamp.package_version()
    if version is None or version == GOLDEN_VERSION:
        return body
    for running, golden in zip(printed_build(version), printed_build(GOLDEN_VERSION), strict=True):
        body = body.replace(running, golden)
    return body


def printed_build(version: str) -> tuple[str, str]:
    """The token a code-only page prints the build ``version`` in: as a Markdown page prints it, and as an HTML page
    does, whose prose cell holds the package's hyphenated name whole in a span."""
    return f"3tears-evals {version},", f"3tears-evals</span> {version},"


#: The ``TESTAMENT_`` variables a person sets to run one file locally (``TESTAMENT=off`` takes a stated reason): a
#: regeneration's own, never a runner's.
_LOCAL_RUN = frozenset({"TESTAMENT_OFF", "TESTAMENT_OFF_REASON"})


def refuse_unattended_regeneration() -> None:
    """Refuse to write goldens anywhere but one local process: never in CI, on an xdist worker or a testament runner.

    Parallel writers would race on each campaign's one file, and goldens written where nobody reviews the diff are a
    silent re-baseline. A testament runner is told by any ``TESTAMENT_`` variable but the two a person sets to run
    locally (:data:`_LOCAL_RUN`), which ``test.sh`` requires of every local run.
    """
    unattended = sorted(
        name
        for name in os.environ
        if name in ("CI", "PYTEST_XDIST_WORKER") or (name.startswith("TESTAMENT_") and name not in _LOCAL_RUN)
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
        self.moved: list[str] = []
        self.unchanged = 0

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
        before = self.stored.get(self.key(entry))
        if before == observed:
            self.unchanged += 1
        else:
            was = "new" if before is None else f"{size(before)} bytes"
            self.moved.append(f"{entry} ({was} -> {size(observed)} bytes)")
        self.stored[self.key(entry)] = observed
        GOLDENS.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.stored, indent=1, sort_keys=True, ensure_ascii=False) + "\n")

    def assert_unchanged(self) -> None:
        """Fail naming every entry that moved or has no golden; under :data:`REGENERATE`, fail naming every entry the
        regeneration moved, with its size before and after, and how many it left as they were, so a regenerating run
        never reads as green and its diff is reviewed where it moved."""
        if self.moved or self.unchanged:
            pytest.fail(
                f"{self.name}: goldens written on {PLATFORM}; {len(self.moved)} moved: "
                f"{', '.join(self.moved) or 'none'}; {self.unchanged} unchanged. Review `git diff {self.path}` line by "
                "line and state the reason in the commit"
            )
        assert not self.problems, "\n".join(self.problems)


@pytest.fixture
def pinned_clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every code-only page assembled at :data:`REPORT_NOW`."""
    monkeypatch.setattr(analysis_service, "utc_now_iso", lambda: REPORT_NOW)
    yield


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def check_bundles(goldens: Goldens, name: str) -> list[AnalysisContextBundle]:
    """Check campaign ``name``'s aspect bundles, their fingerprints, its frame and its fingerprint; return the aspects."""
    read = subject(name)
    bundle = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id)
    assert bundle.aspects, f"{name} assembled no aspect"
    if not REGENERATE:
        pinned = sorted(entry.removeprefix("bundle:") for entry in goldens.entries() if entry.startswith("bundle:"))
        assert sorted(aspect.aspect for aspect in bundle.aspects) == pinned, f"{name}'s aspects are not the pinned ones"
    for aspect in bundle.aspects:
        goldens.check(f"bundle:{aspect.aspect}", bundle_without_build(aspect))
        goldens.check(f"fingerprint:{aspect.aspect}", aspect.fingerprint())
    goldens.check("frame", identity_json(bundle.frame()))
    goldens.check("campaign_fingerprint", bundle.fingerprint())
    return list(bundle.aspects)


def check_body(goldens: Goldens, name: str, form: Any, static: bool, trials: Any) -> str:
    """Check campaign ``name``'s report read in ``form`` with ``trials``, its body and its other fields; return the
    body as served."""
    read = subject(name)
    document = report_read(read.host, read.campaign_id, read.scope_id, format=form, static=static, trials=trials)
    form_id = f"{form}_static" if static else form
    goldens.check(f"body:{form_id}:{trials}", body_without_build(document.body))
    goldens.check(f"read:{form_id}:{trials}", _json(document.model_dump(mode="json", exclude={"body"})))
    return document.body


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_every_aspect_s_bundle_and_the_campaign_frame_are_unchanged(name: str, pinned_clock: None) -> None:
    goldens = Goldens(name)

    check_bundles(goldens, name)

    goldens.assert_unchanged()


@pytest.mark.parametrize("trials", TRIALS)
@pytest.mark.parametrize(("form", "static"), FORMS, ids=["markdown", "html", "html_static", "json"])
@pytest.mark.parametrize("name", CAMPAIGNS)
def test_every_report_read_body_is_unchanged(
    name: str, form: Any, static: bool, trials: Any, pinned_clock: None
) -> None:
    goldens = Goldens(name)

    check_body(goldens, name, form, static, trials)

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

    # Each count reads grouped by thousands, as every count in the package's prose does.
    assert f"{spell_count(aspect.trials_omitted)} more trials count on the scoreboard and are not listed here" in body
    ran = sum(board.ran for board in aspect.scoreboard)
    assert ran >= 1000, "the scaled campaign must run a thousand trials for the grouping to show"
    assert f"{spell_count(ran)} trials ran. Of them " in body and f" {ran} trials ran" not in body


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
        f"{name}'s overrides are not the named ones: {goldens.overrides()}"
    )
    assert named <= entries, f"{name} overrides an entry with no pool golden beside it"


def test_no_fingerprint_frame_or_body_is_held_for_one_platform() -> None:
    """A fingerprint is over the evidence (#1062), so nothing that is one or prints one depends on where it was
    computed: the only entry held a second time for a platform is a bundle compared whole as computed."""
    overridden = {entry for entries in PLATFORM_OVERRIDES.values() for entry in entries}

    assert overridden == {"bundle:tpl-support"}
    assert not {entry for entry in overridden if not entry.startswith("bundle:")}


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_every_pinned_fingerprint_is_over_the_evidence(name: str) -> None:
    """The golden vectors: each aspect's pinned fingerprint is ``v3:``, the digest of its evidence manifest, one value
    for every platform."""
    goldens = Goldens(name)
    pinned = {entry: goldens.stored[entry]["text"] for entry in goldens.entries() if entry.startswith("fingerprint:")}

    assert pinned, f"{name} pins no aspect's fingerprint"
    assert all(len(text) == 67 and text.startswith("v3:") for text in pinned.values()), pinned


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


#: The version :func:`bumped_build` installs: any version but :data:`GOLDEN_VERSION`.
BUMPED_VERSION = "99.1.0"


@pytest.fixture
def bumped_build(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """The installed 3tears-evals distribution reporting :data:`BUMPED_VERSION`, as it would after a release bump:
    its metadata is what :func:`~threetears.evals.kernel.build_stamp.package_version` reads, so every reader of the
    version sees the bump."""
    installed = build_stamp.distribution

    def bumped(name: str) -> Any:
        distribution = installed(name)
        return SimpleNamespace(version=BUMPED_VERSION, read_text=distribution.read_text)

    monkeypatch.setattr(build_stamp, "distribution", bumped)
    build_stamp.package_version.cache_clear()
    yield BUMPED_VERSION
    build_stamp.package_version.cache_clear()


def test_a_version_bump_moves_no_golden(bumped_build: str, pinned_clock: None) -> None:
    """A build at another version reads the toy host's campaign against the same goldens: its bundles carry the bumped
    version and its code-only pages print it, and neither is compared (see the module docstring)."""
    goldens = Goldens("toyhost")

    aspects = check_bundles(goldens, "toyhost")
    bodies = [check_body(goldens, "toyhost", form, static, trials) for form, static in FORMS for trials in TRIALS]

    assert {aspect.package_version for aspect in aspects} == {bumped_build}, "the bump did not reach the bundle"
    printed = sum(any(token in body for token in printed_build(bumped_build)) for body in bodies)
    assert printed == 6, f"the bump reached {printed} of the six markdown and html bodies"
    goldens.assert_unchanged()


def test_a_regeneration_names_every_entry_it_moved_with_its_size_before_and_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regeneration fails naming what it moved, so its diff is reviewed where it moved: each moved or new entry with
    its size before and after, and how many it left as they were."""
    for name in [name for name in os.environ if name in ("CI", "PYTEST_XDIST_WORKER") or name.startswith("TESTAMENT_")]:
        monkeypatch.delenv(name)
    monkeypatch.setitem(globals(), "GOLDENS", tmp_path)
    (tmp_path / "campaign.json").write_text(json.dumps({"body:a": record("kept"), "body:b": record("before")}))
    goldens = Goldens("campaign")

    goldens.regenerate("body:a", record("kept"))
    goldens.regenerate("body:b", record("after, longer"))
    goldens.regenerate("body:c", record("new"))

    with pytest.raises(pytest.fail.Exception) as failed:
        goldens.assert_unchanged()
    assert "2 moved: body:b (6 bytes -> 13 bytes), body:c (new -> 3 bytes); 1 unchanged" in str(failed.value)
    assert json.loads((tmp_path / "campaign.json").read_text())["body:b"] == record("after, longer")
