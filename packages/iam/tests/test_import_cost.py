"""What importing each base module costs, as an allowlist.

``test_imports_without_core`` proves the base modules work with core, NATS and the agent
layer refused. This test asks the question from the other side: not "did it avoid what we
thought of" but **"is everything it loaded on the permitted list"** -- the stdlib, the
distributions iam's ruled floor declares, and the three family packages that floor names.
A denylist only catches the weight somebody predicted; the next dependency to arrive here
will be one nobody wrote down.

The floor is the promise to a consumer outside the platform (testament's coordinator takes
``github``, ``oidc``, ``tokens`` and ``clientip`` and runs neither NATS nor Postgres): importing
a base module costs what the base install declares and no more.

Measured in a fresh interpreter, because whatever the pytest process already imported would
otherwise hide the cost entirely.
"""

from __future__ import annotations

import json
import subprocess
import sys
from functools import cache

import pytest

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

#: CPython's build configuration module embeds the ABI flags and platform in its name, so it
#: is the interpreter rather than a dependency, yet it is absent from ``sys.stdlib_module_names``.
_INTERPRETER_ARTIFACT_PREFIXES = ("_sysconfigdata",)

_PROBE = """
import json
import sys

before = set(sys.modules)
import {module}  # noqa: F401
print(json.dumps(sorted(set(sys.modules) - before)))
"""


@cache
def _modules_loaded_by_importing(module: str) -> tuple[str, ...]:
    """Every module a fresh interpreter loads for *module*'s import.

    :param module: the dotted module name to import in the probe
    :ptype module: str
    :return: the module names loaded, sorted
    :rtype: tuple[str, ...]
    """
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"probe for {module} failed:\n{result.stderr}"
    loaded: list[str] = json.loads(result.stdout.strip())
    return tuple(loaded)


def _is_permitted(module: str) -> bool:
    """Whether the base floor pays for *module*.

    :param module: a dotted module name from ``sys.modules``
    :ptype module: str
    :return: true when the module is stdlib, a declared distribution's, or permitted family
    :rtype: bool
    """
    root = module.split(".")[0]
    if root in sys.stdlib_module_names or module.startswith(_INTERPRETER_ARTIFACT_PREFIXES):
        return True
    if root in _PERMITTED_THIRD_PARTY_ROOTS:
        return True
    if any(module == prefix or module.startswith(f"{prefix}.") for prefix in _PERMITTED_FAMILY_PREFIXES):
        return True
    return module in _PERMITTED_FAMILY_MODULES


@pytest.mark.parametrize("module", _BASE_MODULES)
class TestBaseModuleImportCost:
    def test_the_probe_actually_imported_the_module(self, module: str) -> None:
        """An allowlist over an empty set passes for the wrong reason."""
        assert module in _modules_loaded_by_importing(module)

    def test_nothing_outside_the_base_floor_is_loaded(self, module: str) -> None:
        """No core, no NATS, no agent layer, no Postgres driver: an allowlist, not a denylist."""
        offenders = sorted(m for m in _modules_loaded_by_importing(module) if not _is_permitted(m))

        assert offenders == [], (
            f"importing {module} loaded modules outside iam's base floor: {offenders}. Either the "
            "import belongs behind TYPE_CHECKING, inside a function or behind an extra, or the floor "
            "itself is changing -- in which case the ruling in test_dependency_alignment.py, the "
            "package's dependencies and this list move together"
        )


def test_the_contracts_leaf_is_what_stepup_loads_for_its_taxonomy() -> None:
    """The permitted-family half is exercised: stepup reaches the leaf and not agent-acl."""
    loaded = _modules_loaded_by_importing("threetears.iam.stepup")

    assert "threetears.access.contracts" in loaded
    assert not any(m.startswith("threetears.agent") for m in loaded)
