"""The report and planning reads give byte-identical output across every change that makes them faster.

Each campaign below is read the way a host reads it, and every output is compared, byte for byte and unrounded, with
the golden captured from the code before any of that work began:

- every aspect's assembled bundle as JSON, and its ``fingerprint()``;
- the campaign bundle's frame (what a stored analysis keeps of it), and the campaign fingerprint;
- every ``report_read`` body: markdown, html, the static html page and json, each with ``trials="table"`` and
  ``trials="all"``, and the read's other fields beside its body;
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

**Every platform is held to its own golden.** The last unrounded digits of some statistics (a t interval's bounds and
everything computed from them, read through the platform's maths library) differ between C libraries: the same code
reads ``21.28666989160534`` on glibc 2.41 and ``21.28666989160533`` on glibc 2.39, whatever the machine. A golden
file's generic entries were captured on the platform it names under ``@captured_on`` (the pool's, glibc 2.41). Every
other platform (:data:`PLATFORM`: the system and its C library) is compared against its own ``<entry>@<PLATFORM>``, or
against the generic entry where the file lists it under ``@confirmed`` for that platform, never silently against
another platform's. A platform with no golden for an entry is a red that says so, carrying the record it read (between
:data:`OBSERVED` marks) to add once the base commit is confirmed to give it there. Every entry has a golden on every
platform in :data:`DECLARED_PLATFORMS`, which a test checks. One red lists every entry of its test that moved or has
no golden. A regeneration (local only: it refuses in CI, on xdist and on testament's runners, and every regenerating
test fails, naming what it wrote) writes its own platform's goldens; on the capture platform it drops the other
platforms' overrides and confirmations of each entry it rewrites, so they read as having none until confirmed again.

Floats are not compared at a tolerance, not even those a maths library computes: a tolerance would also pass a sum
whose order moved its last place, which is what this oracle exists to catch.

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

#: The platform a read ran on, as far as its floats' last digits go: the system and its C library, whose maths
#: library moves them. The machine and the Python build do not: the pool's x86_64 and aarch64 runners, both on glibc
#: 2.41, read every entry alike, and the dev container on glibc 2.39 reads some apart from them.
PLATFORM = f"{platform.system()}-{'-'.join(platform.libc_ver())}"

#: The platform in full, for a red's message: the machine and the Python build too.
PLATFORM_DETAIL = f"{PLATFORM}-{platform.machine()}-py{platform.python_version()}"

#: Every platform the oracle is held on, each with a golden for every entry: the pool's runners, and the development
#: container a coordinator may read it in.
DECLARED_PLATFORMS = ("Linux-glibc-2.41", "Linux-glibc-2.39")

#: The key under which a golden file names the platform its generic entries were captured on.
CAPTURED_ON = "@captured_on"

#: The key under which a golden file lists, per platform, the entries that platform reads exactly as the generic one.
CONFIRMED = "@confirmed"

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
    """One campaign's goldens, read as this platform holds them, and every entry a test compares against them.

    Every entry a test compares is checked before it fails, so one red names every entry that moved, and every
    entry this platform has no golden for, each with the record it read.

    The file's generic entries were captured on the platform it names under :data:`CAPTURED_ON`. Another platform is
    compared against its own override, ``<entry>@<PLATFORM>``, or against the generic entry where the file lists the
    entry as confirmed for that platform (:data:`CONFIRMED`): never silently against a golden of a platform whose
    floats may read differently in their last place.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.path = GOLDENS / f"{name}.json"
        self.stored: dict[str, Any] = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.problems: list[str] = []
        self.written: list[str] = []

    def golden(self, entry: str, on: str = PLATFORM) -> dict[str, Any] | None:
        """Platform ``on``'s golden of ``entry``: its override, else the generic entry where it holds there."""
        override = self.stored.get(f"{entry}@{on}")
        if override is not None:
            return dict(override)
        holds = self.stored.get(CAPTURED_ON) == on or entry in self.stored.get(CONFIRMED, {}).get(on, [])
        return dict(self.stored[entry]) if holds and entry in self.stored else None

    def entries(self) -> list[str]:
        """The generic entries the file holds."""
        return sorted(key for key in self.stored if not key.startswith("@") and "@" not in key)

    def check(self, entry: str, text: str) -> None:
        """Compare ``text`` with ``entry``'s golden on this platform, or write it under :data:`REGENERATE`."""
        observed = record(text)
        if REGENERATE:
            self.regenerate(entry, observed)
            return
        carried = json.dumps(
            {
                "campaign": self.name,
                "entry": entry,
                "platform": PLATFORM,
                "detail": PLATFORM_DETAIL,
                "record": observed,
            },
            ensure_ascii=False,
        )
        golden = self.golden(entry)
        if golden is None:
            self.problems.append(
                f"{self.name} {entry}: no golden for this platform ({PLATFORM}; the generic goldens were captured on "
                f"{self.stored.get(CAPTURED_ON)}): add {entry}@{PLATFORM} = the record below, or list the entry as "
                f"confirmed for it, after confirming the base commit gives it on this platform "
                f"{OBSERVED}{carried}{OBSERVED}"
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
        """Write ``observed`` as this platform's golden of ``entry``.

        On the capture platform it replaces the generic entry, and every other platform's override and confirmation of
        it is dropped: they described the output it replaces, so each reads as having no golden until confirmed again.
        On another platform it confirms the generic entry where it is equal, and is written as this platform's
        override where it is not.
        """
        refuse_unattended_regeneration()
        captured = self.stored.setdefault(CAPTURED_ON, PLATFORM)
        confirmed: dict[str, list[str]] = self.stored.setdefault(CONFIRMED, {})
        if captured == PLATFORM:
            self.stored[entry] = observed
            for key in [key for key in self.stored if key.startswith(f"{entry}@")]:
                del self.stored[key]
            for listed in confirmed.values():
                if entry in listed:
                    listed.remove(entry)
        elif self.stored.get(entry) == observed:
            self.stored.pop(f"{entry}@{PLATFORM}", None)
            confirmed[PLATFORM] = sorted({*confirmed.get(PLATFORM, []), entry})
        else:
            self.stored[f"{entry}@{PLATFORM}"] = observed
            confirmed[PLATFORM] = [listed for listed in confirmed.get(PLATFORM, []) if listed != entry]
        GOLDENS.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.stored, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        self.written.append(entry)

    def assert_unchanged(self) -> None:
        """Fail naming every entry that moved or has no golden on this platform; under :data:`REGENERATE`, fail naming
        what was written, so a regenerating run never reads as green."""
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
        goldens.check(f"bundle:{aspect.aspect}", _json(aspect.to_dict()))
        goldens.check(f"fingerprint:{aspect.aspect}", aspect.fingerprint())
    goldens.check("frame", _json(bundle.frame()))
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
def test_every_golden_entry_has_a_golden_on_every_declared_platform(name: str) -> None:
    """Coverage is checked here, not found by which runner a read lands on: every entry a campaign's goldens hold has
    a golden on every platform in :data:`DECLARED_PLATFORMS`."""
    goldens = Goldens(name)
    entries = goldens.entries()
    expected = {f"body:{form}:{trials}" for form in ("markdown", "html", "html_static", "json") for trials in TRIALS}
    expected |= {entry.replace("body:", "read:", 1) for entry in expected}

    assert expected <= set(entries), f"{name} lacks goldens for {sorted(expected - set(entries))}"
    missing = {on: [entry for entry in entries if goldens.golden(entry, on) is None] for on in DECLARED_PLATFORMS}
    assert not any(missing.values()), f"{name}: entries with no golden, by platform: {missing}"
