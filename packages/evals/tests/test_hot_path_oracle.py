"""The report and planning reads give byte-identical output across every change that makes them faster (W4b oracle).

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

**A golden is per platform where the platform moves it.** The last unrounded digits of some statistics (a t
interval's bounds, a sum's last place, read through the platform's maths library) differ between platforms: the same
code reads ``21.28666989160534`` on one and ``21.28666989160533`` on another. So an entry a platform reads differently
from where it was captured carries an override, ``<entry>@<PLATFORM>`` (:data:`PLATFORM`: the machine, the C library
and the Python build), and each platform is held byte for byte to its own. A red's message names the platform and
carries the record it observed (between :data:`OBSERVED` marks), so an override, or an approved change's goldens, can
be taken from a pool run's report (``testament run --json``).

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

#: The platform a read ran on, as far as its floats' last digits go: the machine, the C library and the Python build.
PLATFORM = f"{platform.system()}-{platform.machine()}-{'-'.join(platform.libc_ver())}-py{platform.python_version()}"

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


def _golden_path(name: str) -> Path:
    return GOLDENS / f"{name}.json"


def _goldens(name: str) -> dict[str, Any]:
    path = _golden_path(name)
    if not path.exists():
        return {}
    goldens: dict[str, Any] = json.loads(path.read_text())
    return goldens


def check(name: str, entry: str, text: str) -> None:
    """``text`` against campaign ``name``'s golden ``entry`` on this platform, or written as it under
    :data:`REGENERATE`.

    The golden is the entry this platform overrides (``<entry>@<PLATFORM>``) where there is one, else the entry
    itself.
    """
    observed = record(text)
    goldens = _goldens(name)
    if REGENERATE:
        goldens[entry] = observed
        GOLDENS.mkdir(parents=True, exist_ok=True)
        _golden_path(name).write_text(json.dumps(goldens, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        return
    golden = goldens.get(f"{entry}@{PLATFORM}", goldens.get(entry))
    assert golden is not None, f"{name} has no golden {entry!r}; goldens are written under HOT_PATH_ORACLE_REGEN=1"
    if observed != golden:
        moved = next(
            (
                index
                for index, (was, now) in enumerate(zip(golden.get("text", golden.get("head", "")), text))
                if was != now
            ),
            None,
        )
        raise AssertionError(
            f"{name} {entry} moved on {PLATFORM}: {observed.get('bytes', len(text))} bytes now, "
            f"{golden.get('bytes', len(golden.get('text', '')))} in the golden; first difference in its head at "
            f"character {moved}; head now {text[:200]!r}; observed record "
            f"{OBSERVED}{json.dumps(observed, ensure_ascii=False)}{OBSERVED}"
        )


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

    assert bundle.aspects, f"{name} assembled no aspect"
    for aspect in bundle.aspects:
        check(name, f"bundle:{aspect.aspect}", _json(aspect.to_dict()))
        check(name, f"fingerprint:{aspect.aspect}", aspect.fingerprint())
    check(name, "frame", _json(bundle.frame()))
    check(name, "campaign_fingerprint", bundle.fingerprint())


@pytest.mark.parametrize("trials", TRIALS)
@pytest.mark.parametrize(("form", "static"), FORMS, ids=["markdown", "html", "html_static", "json"])
@pytest.mark.parametrize("name", CAMPAIGNS)
def test_every_report_read_body_is_unchanged(
    name: str, form: Any, static: bool, trials: Any, pinned_clock: None
) -> None:
    read = subject(name)
    document = report_read(read.host, read.campaign_id, read.scope_id, format=form, static=static, trials=trials)
    form_id = f"{form}_static" if static else form

    check(name, f"body:{form_id}:{trials}", document.body)
    check(name, f"read:{form_id}:{trials}", _json(document.model_dump(mode="json", exclude={"body"})))


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

    check(name, "planning", block.model_dump_json())


def test_the_scaled_page_states_how_many_trials_it_lists_of_how_many_ran(pinned_clock: None) -> None:
    """The bundle's trials are a bounded sample (at most ``MAX_TRIALS``, failed and lost first): the page says how many
    more ran than it lists, so a reader never takes the listed trials for the whole campaign."""
    read = subject("scaled")
    bundle = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id)
    (aspect,) = bundle.aspects
    assert aspect.trials_omitted > 0, "the scaled campaign must run past the trial cap for this to test anything"
    body = report_read(read.host, read.campaign_id, read.scope_id, format="markdown").body

    assert f"{aspect.trials_omitted} more trials count on the scoreboard and are not listed here" in body
