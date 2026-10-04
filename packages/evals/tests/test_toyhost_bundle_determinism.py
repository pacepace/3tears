"""Two builds of the toy-host fixture must be the same bundle.

`AnalysisContextBundle.fingerprint` is the invariant the prompt-tuning A/B loop
depends on, and that loop is: hold the bundle still, vary the prompt, compare. A fixture that builds a
different bundle every call cannot be held still, so the loop is not runnable
over it — not slower, not noisier, *not runnable*.

It once was not. `EvalRun.id`, `EvalResult.id` and `EvalCampaign.id`
default to `uuid4`, and `created_at` / `scored_at` / `declared_at` / `asked_at`
default to `utc_now_iso` — so the fixture minted new identity and new wall-clock on
every call. **Eleven leaves of the assembled bundle differed between two
consecutive builds**, and the assembler multiplies that: it derives the
measurement windows and the window disclosure from the result timestamps, so one
unpinned default reached five more leaves on its own — both windows' `start` and
`end`, plus the disclosure prose rendered from them.

The bundle docstring's reason for the invariant — "the bundle carries no
wall-clock value" — is true of one assembled from **stored** runs, whose
timestamps are data. It is false of one assembled from freshly constructed
objects, which is what every fixture in this repo does. That is why this is a
test about the fixture rather than a correction to the engine.

Asserted over the WHOLE dump rather than the fingerprint alone: a fingerprint
comparison says two bundles differ, and this says *where*, which is the
difference between a five-minute fix and the afternoon the first one took.
"""

from __future__ import annotations


from threetears.evals.analysis.generator import build_user_message
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile


def _leaf_diffs(a, b, path=""):
    """Every differing leaf between two JSON-shaped structures.

    Args:
        a: Left structure.
        b: Right structure.
        path: Accumulated dotted path.

    Returns:
        ``(path, left, right)`` triples, empty when identical.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in _leaf_diffs(a.get(k), b.get(k), f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [(path, f"len {len(a)}", f"len {len(b)}")]
        return [d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in _leaf_diffs(x, y, f"{path}[{i}]")]
    return [] if a == b else [(path, a, b)]


def test_two_builds_of_the_toy_host_bundle_are_the_same_bundle() -> None:
    """Not merely the same fingerprint — the same content, field by field.

    Any new default that mints identity or reads the clock lands here with its
    own path named, which is what makes this a guard against the CLASS rather
    than against the four fields that were wrong once.
    """
    profile = toyhost_profile()
    from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle

    first, second = toyhost_bundle(profile=profile), toyhost_bundle(profile=profile)

    diffs = _leaf_diffs(first.model_dump(mode="json"), second.model_dump(mode="json"))
    assert not diffs, (
        "two builds of the toy-host fixture produced different bundles, so nothing can be "
        "held still across them — the fixed-bundle prompt A/B and any replay loop over the "
        "fixture both stop working.\n  "
        + "\n  ".join(f"{p}: {x!r} != {y!r}" for p, x, y in diffs[:10])
        + "\n\nPin it in tests/fixtures/toyhost/ — derive ids (uuid5) and use TOYHOST_INSTANT "
        "for anything that would otherwise default to now()."
    )


def test_the_fingerprint_agrees_with_the_content() -> None:
    """The fingerprint is what the A/B keys on, so it must move with the content and not otherwise."""
    profile = toyhost_profile()
    from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle

    assert toyhost_bundle(profile=profile).fingerprint() == toyhost_bundle(profile=profile).fingerprint()


def test_the_generated_user_message_is_reproducible() -> None:
    """The bundle is not the payload — the user message built from it is.

    Replay keys on the exact request text, so a bundle that was stable while the
    message built from it was not would leave the loop refusing every replay with
    nothing visibly wrong. Asserted separately because the two can diverge: the
    message is a rendering, and a renderer is free to introduce its own clock.
    """
    profile = toyhost_profile()
    from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle

    assert build_user_message(toyhost_bundle(profile=profile)) == build_user_message(toyhost_bundle(profile=profile))
