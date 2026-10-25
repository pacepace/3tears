"""What importing each base module costs on the base install, as an allowlist.

``test_imports_without_core`` proves the base modules work with core, NATS and the agent
layer refused. This test asks the question from the other side: not "did it avoid what we
thought of" but **"is everything it loaded on the permitted list"** -- the stdlib, the
distributions iam's ruled floor declares, and the three family packages that floor names.
A denylist only catches the weight somebody predicted; the next dependency to arrive here
will be one nobody wrote down.

The floor is the promise to a consumer outside the platform (testament's coordinator takes
``github``, ``oidc``, ``tokens`` and ``clientip`` and runs neither NATS nor Postgres): importing
a base module costs what the base install declares and no more.

Measured by :mod:`threetears.enforcement.import_floor`, in a fresh interpreter with what the base
install leaves out refused. Refused rather than left installed because ``totp`` imports core's
sealing at its top under the guard naming the ``core`` extra (lazy imports are banned in serving
code, owner ruling 2026-10-24): with core installed, importing it loads core, which is the cost
the extra's buyer chose; the base install's buyer pays none of it.
"""

from __future__ import annotations

from functools import cache

import pytest

from threetears.enforcement.import_floor import (
    ImportFloorConfig,
    modules_loaded_by_importing,
    run_import_floor_enforcement,
)

#: Third-party import roots the base floor pays for: the declared distributions, and the
#: support modules each loads of its own accord. Enumerated rather than derived, so a change
#: in what a dependency drags in is a visible event to read rather than weight to absorb.
_PERMITTED_THIRD_PARTY_ROOTS = frozenset(
    {
        # argon2-cffi and its bindings, and the cffi backend it and cryptography share
        "argon2",
        "_argon2_cffi_bindings",
        "_cffi_backend",
        "bcrypt",
        # cryptography, and its compiled OpenSSL binding
        "cryptography",
        "_openssl",
        # pyjwt
        "jwt",
        "joserfc",
        "pyotp",
        # httpx and what it imports on load; httpcore and anyio arrive with the first request
        "httpx",
        "httpcore",
        "anyio",
        "sniffio",
        "certifi",
        "idna",
        "h11",
        # httpx imports its CLI (click, rich, pygments) and its zstd and brotli decoders when
        # they happen to be installed, and does without them otherwise. A lean install has none
        # of them; a development environment has all of them.
        "click",
        "rich",
        "pygments",
        "markdown_it",
        "mdurl",
        "zstandard",
        "brotli",
        "brotlicffi",
        # pydantic and the tree its own distribution installs
        "pydantic",
        "pydantic_core",
        "annotated_types",
        "typing_extensions",
        "typing_inspection",
    }
)

#: Family trees the base floor names: the namespace packages on the way down, iam itself,
#: observe, and the access-contracts leaf.
_PERMITTED_FAMILY_MODULES = frozenset({"threetears", "threetears.access"})
_PERMITTED_FAMILY_PREFIXES = ("threetears.iam", "threetears.observe", "threetears.access.contracts")

#: Every module the base install imports. Listed by name rather than discovered, so a new
#: module is a deliberate addition to the promise; ``dpop``, ``stores.attempt_limiter`` (the
#: ``core`` extra) and ``saml`` (the ``saml`` extra) are out of it on purpose.
_BASE_MODULES = (
    "threetears.iam",
    "threetears.iam.apikeys",
    "threetears.iam.breach",
    "threetears.iam.buckets",
    "threetears.iam.claim_mapping",
    "threetears.iam.clientip",
    "threetears.iam.connection_types",
    "threetears.iam.dpop_client",
    "threetears.iam.github",
    "threetears.iam.oauth_state",
    "threetears.iam.oidc",
    "threetears.iam.passwords",
    "threetears.iam.pkce",
    "threetears.iam.rotation",
    "threetears.iam.stepup",
    "threetears.iam.stores",
    "threetears.iam.stores.memory",
    "threetears.iam.stores.nats_kv",
    "threetears.iam.stores.postgres",
    "threetears.iam.tokens",
    "threetears.iam.totp",
    "threetears.iam.webauthn",
)

#: What the base install leaves out, refused in the probe as an install without it refuses it: the
#: same roots ``tests/enforcement/test_package_boundaries.py`` holds the source to.
_REFUSED_ROOTS = (
    "threetears.core",
    "threetears.agent",
    "threetears.nats",
    "sqlalchemy",
    "asyncpg",
    "nats",
    "nkeys",
)

_FLOOR = ImportFloorConfig(
    base_modules=_BASE_MODULES,
    third_party_roots=_PERMITTED_THIRD_PARTY_ROOTS,
    family_prefixes=_PERMITTED_FAMILY_PREFIXES,
    namespace_modules=_PERMITTED_FAMILY_MODULES,
    refused_roots=_REFUSED_ROOTS,
)


@cache
def _modules_loaded_by_importing(module: str) -> tuple[str, ...]:
    """Every module a fresh base-install interpreter loads for *module*'s import.

    :param module: the dotted module name to import in the probe
    :ptype module: str
    :return: the module names loaded, sorted
    :rtype: tuple[str, ...]
    """
    return modules_loaded_by_importing(module, refused=_REFUSED_ROOTS)


@pytest.mark.parametrize("module", _BASE_MODULES)
def test_nothing_outside_the_base_floor_is_loaded(module: str) -> None:
    """No core, no NATS, no agent layer, no Postgres driver: an allowlist, not a denylist.

    A change to the floor itself moves the ruling in test_dependency_alignment.py, the package's
    dependencies and this config together.
    """
    run_import_floor_enforcement(_FLOOR, module)


def test_the_contracts_leaf_is_what_stepup_loads_for_its_taxonomy() -> None:
    """The permitted-family half is exercised: stepup reaches the leaf and not agent-acl."""
    loaded = _modules_loaded_by_importing("threetears.iam.stepup")

    assert "threetears.access.contracts" in loaded
    assert not any(m.startswith("threetears.agent") for m in loaded)
