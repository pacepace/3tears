"""The statistics certificate is current and every claim in it clears its target (``stats_certificate.py``).

The evals reading's large error-rate claims under the null (a comparison continued to five reads separates at most α,
an interval covers what it states, an engine comparison drained by its operator does not lean) are certified rather
than simulated on every build: the certification command counts each claim and writes the certificate with the digest
of the files that decide it. Here, in seconds, the certificate is held to those files: a claim whose files have changed
since it was certified is stale, and this fails naming it and the command that certifies again. Every claim's count is
read again against its target, so a certificate written with a claim that fails cannot pass either.
"""

from __future__ import annotations

from packages.evals.tests.stats_certificate import (
    CERTIFICATE,
    CLAIMS,
    COMMAND,
    ROOT,
    SOURCES,
    judge,
    read,
    stale,
)


def test_every_file_a_claim_names_exists() -> None:
    """A source moved or renamed must be named again, or its claims would rest on nothing."""
    missing = sorted(path for paths in SOURCES.values() for path in paths if not (ROOT / path).is_file())
    assert not missing, f"the certificate names files that are gone: {missing}; update SOURCES in stats_certificate.py"


def test_the_certificate_holds_every_claim_the_certification_states() -> None:
    document = read()
    held = [entry["id"] for entry in document["claims"]]
    stated = [claim.id for claim in CLAIMS]
    assert held == stated, (
        f"the certificate holds {sorted(set(held) ^ set(stated))} differently from the claims stated; "
        f"re-certify: {COMMAND}"
    )


def test_the_certificate_is_current() -> None:
    """Every claim's digest is its source files' digest now: nothing that decides it has changed since it was
    certified."""
    out_of_date = stale(read())
    assert not out_of_date, (
        f"{len(out_of_date)} certified claims are stale, their sources changed since {CERTIFICATE.name} was written:\n  "
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
