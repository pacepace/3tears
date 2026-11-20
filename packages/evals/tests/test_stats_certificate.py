"""The statistics certificate is current and every claim in it clears its target (``stats_certificate.py``).

The evals reading's large error-rate claims under the null (a comparison continued to five reads separates at most α,
an interval covers what it states, an engine comparison drained by its operator does not lean) are certified rather
than simulated on every build: the certification command counts each claim and writes the certificate with the digest
of the files that decide it: every repository file the certification imports, derived from the code. Here, in seconds,
the certificate is held to those files: when any has changed since it was certified its claims are stale, and this
fails naming the files, the claims and the command that certifies again. Every claim's count is
read again against its target, so a certificate written with a claim that fails cannot pass either.
"""

from __future__ import annotations

import os
import subprocess
import sys

from packages.evals.tests.stats_certificate import (
    CERTIFICATE,
    CLAIMS,
    COMMAND,
    ENTRY_POINTS,
    ROOT,
    closure,
    judge,
    moved_files,
    read,
    stale,
)


def test_the_digest_is_the_whole_import_closure_of_the_certification() -> None:
    """The digest is no list anyone keeps: it is every repository file the certification imports, from its entry
    points, so it holds the reading's files whatever they come to be. It reaches the files a reading is decided by,
    beyond the bet itself: the per-case values, the question's lean, the family, the runner, the engine's operator path.
    """
    files = set(closure())
    assert set(ENTRY_POINTS) <= files
    src = "packages/evals/src/threetears/evals/"
    for decides in (
        "analysis/stats.py",
        "analysis/bundle/comparisons.py",
        "analysis/bundle/measures.py",
        "analysis/bundle/schema.py",
        "analysis/bundle/design.py",
        "analysis/bundle/cell_reads.py",
        "analysis/bundle/trials.py",
        "kernel/scoring.py",
        "kernel/offload.py",
        "run/runner.py",
        "quick/one_call.py",
        "storage/memory.py",
        "actions/engine.py",
        "ops/analysis.py",
        "ops/jobs.py",
    ):
        assert src + decides in files, decides


#: What one product newcomer and one engine comparison load when run alone: printed by a fresh interpreter, so no other
#: test's imports are counted.
_RUN_ALONE = """
import asyncio, sys
from pathlib import Path
from packages.evals.tests import engine_take_up, null_newcomers
asyncio.run(null_newcomers.small_continued_product("binary", "closure-0", "gate"))
asyncio.run(null_newcomers.rereadable_product("closure-1", continued=True))
asyncio.run(engine_take_up.one_comparison("closure-2", "looks", "b-slower-20x", "check", "second"))
root = Path.cwd().resolve()
for module in list(sys.modules.values()):
    found = getattr(module, "__file__", None)
    if found and Path(found).resolve().is_relative_to(root) and ".venv" not in Path(found).parts:
        print(Path(found).resolve().relative_to(root))
"""


def test_every_repository_module_the_simulations_load_is_in_the_digest() -> None:
    """Run a gate newcomer, a continued no-question newcomer and an interrupted engine comparison in a fresh
    interpreter: every repository file it loads, imported at module level, inside a function or by name on first use,
    is in the closure the digest covers."""
    loaded = subprocess.run(
        [sys.executable, "-c", _RUN_ALONE],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    ).stdout.split()
    missing = sorted(set(loaded) - set(closure()))
    assert loaded and not missing, f"loaded by the simulations, outside the digest: {missing}"


def test_the_certificate_holds_every_claim_the_certification_states() -> None:
    document = read()
    held = [entry["id"] for entry in document["claims"]]
    stated = [claim.id for claim in CLAIMS]
    assert held == stated, (
        f"the certificate holds {sorted(set(held) ^ set(stated))} differently from the claims stated; "
        f"re-certify: {COMMAND}"
    )


def test_the_certificate_is_current() -> None:
    """No file the certification imports has changed, joined or left the closure since the certificate was written."""
    document = read()
    out_of_date = stale(document)
    assert not out_of_date, (
        f"{len(out_of_date)} certified claims are stale: files the certification imports changed since "
        f"{CERTIFICATE.name} was written:\n  "
        + "\n  ".join(moved_files(document))
        + "\nstale claims:\n  "
        + "\n  ".join(out_of_date)
        + f"\nRe-certify from the repository's root (at most 15 minutes; docs/testing.md): {COMMAND}"
    )


def test_every_certified_claim_clears_its_target() -> None:
    """Each claim's count, read again here on its Clopper-Pearson 95% bounds, clears its target as its test says."""
    by_id = {claim.id: claim for claim in CLAIMS}
    failing = []
    for entry in read()["claims"]:
        claim = by_id[entry["id"]]
        verdict = judge(claim, entry["count"], entry["replicates"])
        if not verdict.holds or not entry["holds"]:
            failing.append(
                f"{claim.id}: {entry['count']}/{entry['replicates']}, 95% {verdict.bounds}, "
                f"{claim.test} {claim.target:.5g}: {claim.says}"
            )
    assert not failing, "certified claims that do not clear their target:\n  " + "\n  ".join(failing)
