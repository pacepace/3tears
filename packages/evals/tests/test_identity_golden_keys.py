"""Golden identity keys: a key derivation cannot move without ``IDENTITY_VERSION`` moving, and a bump cannot land unrecorded.

``identity.py`` promises that "golden-vector tests pin the digests, so changing the hashed inputs
breaks a test and forces the bump to be deliberate". This is that test. It derives the context and
variant identities of representative runs of the two fixture hosts — the toy extractor, whose
context is partial (it wires no subject state), and the courier planner, whose context is complete —
through the same front doors a reader uses, and holds every key and every context component to the
digest recorded here under the version recorded here.

**When this goes red**, one of three things happened:

- **A key moved and the version did not.** If the derivation changed, that is the silent regrouping
  the version exists to prevent: bump ``IDENTITY_VERSION``, record why in the ledger above it in
  ``identity.py``, and re-pin here under the new version. A change to a fixture host's own
  registrations moves that host's keys too, exactly as a real host growing a coordinate does
  (the ledger's v13-v19 are such bumps); re-pin, and say which in the commit.
- **The version moved and the pins did not.** Re-derive and re-pin every golden under the new
  version, even where a key held, so this file states what it was checked under.
- **The version moved and the ledger has no entry for it.** Write the entry.

The components are pinned beside the keys so a failure names WHICH condition moved.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass

import pytest

from threetears.evals.contracts import IDENTITY_VERSION, EvalRun
from threetears.evals.contracts.host import HostProfile
from threetears.evals.contracts.identity import derive_context_identity, derive_variant_identity
import threetears.evals.contracts.identity as identity_module
from packages.evals.tests.fixtures.courierhost import COURIER_MODELS, courier_cases, courier_host, courier_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import ToyExtractorKind
from packages.evals.tests.fixtures.toyhost.run import (
    RUN_MODELS,
    ScriptedExtractionClient,
    toyhost_run,
    toyhost_template,
)

#: The ``IDENTITY_VERSION`` every golden below was derived under.
PINNED_IDENTITY_VERSION = 24

#: The toy host's context, shared by both its arms: one condition, two contestants.
_TOY_CONTEXT = {
    "subject_state": None,
    "seeded_world": "082338b40e87dda25cf60934744cc37392a995447545889bcfb8db3dbe977e16",
    "case_basis": "0ee4684cf928e3aef19a726e51eaf1713606616aacbb42db60bde0650d8d4734",
    "roles": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    "cassette": "c7978efdb5bf6764fcf4785ba9fc5d0464e9d3e39dc798d06349188b85803d28",
    "tool_permissions": "baf5287548d904d9d942c97182d06b2bbc9ba2c35129e6cc9f1302922224e915",
    "world": "35efc88198267d603fa880ed25f36644d541fd4a5884a401dfb08b60867e7742",
    "apparatus_settings": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    "scope": "toyhost-scope",
}
_TOY_CONTEXT_KEY = "a784a3ab06c1c496edf240e8ba7528beb16f6df44ed3eacd1cd936c1cbcc376d"

_COURIER_CONTEXT = {
    "subject_state": {},
    "seeded_world": "6f3e213054b28d30c9ea12f8c692cd67ebcd062d2577194505cb87c9ffbf38fd",
    "case_basis": "36a84f34b71d98d70fd81b4ff2fa9e944bcfe6ddaa017e1b5fc1fb1038c2dc04",
    "roles": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    "cassette": "c7978efdb5bf6764fcf4785ba9fc5d0464e9d3e39dc798d06349188b85803d28",
    "tool_permissions": "baf5287548d904d9d942c97182d06b2bbc9ba2c35129e6cc9f1302922224e915",
    "world": "0868b4f01fb7c86e57c3a9270c1115d89d309d6e72418e664a57f4dc79679b54",
    "apparatus_settings": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    "scope": "courier-depot-north",
}
_COURIER_CONTEXT_KEY = "89a065de810ffb5d09bc8a3cbaaaaf7a5f8ed57bb2980b0a3bda086797ba932e"


@dataclass(frozen=True)
class Golden:
    """One run's pinned identity."""

    host: str
    model: str
    context_components: dict[str, object]
    context_key: str
    missing_components: tuple[str, ...]
    variant_key: str


GOLDENS: tuple[Golden, ...] = (
    Golden(
        "toy",
        "extractor-v2",
        _TOY_CONTEXT,
        _TOY_CONTEXT_KEY,
        ("subject_state",),
        "48516ba0ede3616a3d28349faddd20a6a8822fd89beb14416ccc031eb7610d91",
    ),
    Golden(
        "toy",
        "extractor-v3",
        _TOY_CONTEXT,
        _TOY_CONTEXT_KEY,
        ("subject_state",),
        "a52d0e50c2b68544b54a781512ab9dea117177697492eefdfbad69cddd2635c2",
    ),
    Golden(
        "courier",
        "planner-lite",
        _COURIER_CONTEXT,
        _COURIER_CONTEXT_KEY,
        (),
        "7bc3d84de64ee94e10646a34abc8c56a56ee7df4815b0530f286ab668f481af5",
    ),
    Golden(
        "courier",
        "planner-pro",
        _COURIER_CONTEXT,
        _COURIER_CONTEXT_KEY,
        (),
        "45bbd9683e14a1d9f83600b2984e40811a28902fc0a03b8408c507cffae59a1c",
    ),
)


def _toy_run(model: str) -> tuple[EvalRun, HostProfile]:
    host = toyhost_host()
    world = host.profile.world
    assert world is not None
    template = toyhost_template()
    kind = ToyExtractorKind(
        client=ScriptedExtractionClient(), world=world, judged=False, goal_checks=tuple(template.goal_state_checks)
    )
    return toyhost_run(model=model, template=template, kind=kind, world=world), host.profile


def _courier_run(model: str) -> tuple[EvalRun, HostProfile]:
    host = courier_host()
    world = host.profile.world
    assert world is not None
    return courier_run(model, cases=courier_cases(), world=world), host.profile


_BUILDERS = {"toy": _toy_run, "courier": _courier_run}

_REPIN = (
    "a key moved under IDENTITY_VERSION {version} — if the derivation changed, bump IDENTITY_VERSION, record why in "
    "the ledger in identity.py and re-pin; if a fixture host's registrations changed, re-pin and say so"
)


def test_the_goldens_cover_every_arm_of_both_fixture_hosts():
    """So a fixture growing an arm cannot leave it unpinned while this file stays green."""
    assert {(g.host, g.model) for g in GOLDENS} == {
        *(("toy", model) for model in RUN_MODELS),
        *(("courier", model) for model in COURIER_MODELS),
    }


def test_the_goldens_were_pinned_under_the_current_identity_version():
    assert IDENTITY_VERSION == PINNED_IDENTITY_VERSION, (
        f"IDENTITY_VERSION is {IDENTITY_VERSION} and the goldens were pinned under {PINNED_IDENTITY_VERSION}: "
        "re-derive and re-pin every golden in this file under the new version"
    )


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: f"{g.host}-{g.model}")
def test_a_runs_keys_are_the_golden_ones(golden: Golden):
    run, profile = _BUILDERS[golden.host](golden.model)

    context = derive_context_identity(run, profile)
    variant = derive_variant_identity(run=run, profile=profile)

    message = _REPIN.format(version=IDENTITY_VERSION)
    assert context.identity_version == variant.identity_version == PINNED_IDENTITY_VERSION
    # Components first, so a failure says which condition moved before it says that the key did.
    assert context.context_components.model_dump(mode="json") == golden.context_components, message
    assert tuple(context.missing_components) == golden.missing_components, message
    assert context.context_key == golden.context_key, message
    assert variant.variant_key == golden.variant_key, message


def test_arms_differ_in_variant_and_share_their_context():
    """The goldens' own shape: one condition per host, one contestant per arm — so a pin cannot be a copy."""
    for host in _BUILDERS:
        pinned = [g for g in GOLDENS if g.host == host]
        assert len({g.context_key for g in pinned}) == 1
        assert len({g.variant_key for g in pinned}) == len(pinned)
    assert _TOY_CONTEXT_KEY != _COURIER_CONTEXT_KEY


#: A ledger entry, as the comment block above ``IDENTITY_VERSION`` writes one: ``#: - **v22** — …``.
_LEDGER_ENTRY = re.compile(r"^#: - \*\*v(\d+)\*\*", re.MULTILINE)


def test_every_identity_version_has_a_recorded_reason_in_the_ledger():
    """A bump without a reason is the regrouping the ledger exists to explain, left unexplained.

    The ledger is the ``#:`` block above ``IDENTITY_VERSION`` in ``identity.py``. Every version from
    the first bump (v2) to the current one has exactly one entry, in order, and none is written
    ahead of the version it describes.
    """
    source = inspect.getsource(identity_module)
    ledger = source[: source.index("IDENTITY_VERSION: int =")]
    versions = [int(match) for match in _LEDGER_ENTRY.findall(ledger)]

    assert versions == list(range(2, IDENTITY_VERSION + 1)), (
        f"the identity ledger records versions {versions}; IDENTITY_VERSION is {IDENTITY_VERSION} — "
        "every bump needs its own entry, in order, saying what moved and why"
    )
