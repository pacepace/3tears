"""iam's base install imports and runs without core, NATS or the agent layer.

Only ``dpop``, ``stores.attempt_limiter`` and TOTP seed sealing build on core, and they are the
``core`` extra. Everything else -- sign-in, OIDC, sessions, client IPs, the store Protocols and
the in-memory stores -- must import and work with none of core, the NATS client, the agent
layer, sqlalchemy or asyncpg installed, and the three that need core must say which extra is
missing rather than fail three frames down.

Each check runs in a fresh interpreter with those modules refused at import, as an install
without them refuses them.
"""

from __future__ import annotations

import subprocess
import sys

#: refuses what the base install leaves out, as an install without it would
BLOCK_CORE = """
import importlib.abc, sys
_REFUSED = (
    "threetears.core", "threetears.nats", "threetears.agent",
    "sqlalchemy", "asyncpg", "nats", "nkeys",
)
class _NoCore(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == refused or name.startswith(refused + ".") for refused in _REFUSED):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _NoCore())

def _assert_none_loaded():
    loaded = sorted(
        name for name in sys.modules
        if any(name == refused or name.startswith(refused + ".") for refused in _REFUSED)
    )
    assert not loaded, loaded
"""

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

#: the base install at work: a session token round trip, TOTP, backup codes, an attempt
#: limiter and step-up, each over nothing but the base floor
USE_THE_BASE = """
import asyncio
import importlib
for module in MODULES:
    importlib.import_module(module)

from threetears.access.contracts import ImpersonationCategory
from threetears.iam import stepup, totp
from threetears.iam.stores.memory import MemoryAttemptLimiter
from threetears.iam.tokens import HmacSigner, HmacVerifier, TokenType, mint_token_pair, verify_session_token

pair = mint_token_pair(
    subject="user-1", session_id="sid-1", issuer="https://iam.test", audience="console",
    signer=HmacSigner("a-test-secret-of-adequate-length"),
)
claims = verify_session_token(
    pair.access_token,
    verifier=HmacVerifier("a-test-secret-of-adequate-length", issuer="https://iam.test", audience="console"),
    expected_type=TokenType.ACCESS,
)
assert claims.sub == "user-1", claims

import pyotp
seed = totp.generate_seed()
assert totp.verify_code(seed, pyotp.TOTP(seed).now()).matched
assert not totp.verify_code(seed, "not-a-code").matched
codes = totp.generate_backup_codes()
hashes = tuple(totp.hash_backup_code(code) for code in codes)
assert totp.match_backup_code(codes[3], hashes) == hashes[3]

async def limit():
    limiter = MemoryAttemptLimiter(max_attempts=2)
    await limiter.record_failure("k")
    window = await limiter.record_failure("k")
    assert window.limited, window
    await limiter.clear("k")
    assert not (await limiter.check("k")).limited
asyncio.run(limit())

assert stepup.is_step_up_satisfied(auth_time=100, step_up_window=60, now=150)
try:
    stepup.require_fresh_auth(claims, category=ImpersonationCategory.EMAIL_CHANGE, now=claims.auth_time + 10)
except stepup.StepUpRequiredError as exc:
    assert exc.category is ImpersonationCategory.EMAIL_CHANGE
else:
    raise AssertionError("a session with no step-up window passed step-up")

_assert_none_loaded()
print("ran")
"""


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter.

    :param code: the program
    :ptype code: str
    :return: the finished process
    :rtype: subprocess.CompletedProcess[str]
    """
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=120)


def _expect_the_core_extra_named(statement: str) -> str:
    """A program that runs *statement* without core and asserts it names the ``core`` extra.

    :param statement: one line of Python expected to raise
    :ptype statement: str
    :return: the program
    :rtype: str
    """
    return BLOCK_CORE + (
        "try:\n"
        f"    {statement}\n"
        "except ModuleNotFoundError as exc:\n"
        "    raise AssertionError(f'a bare ModuleNotFoundError, not the named extra: {exc}')\n"
        "except ImportError as exc:\n"
        '    assert "3tears-iam[core]" in str(exc), str(exc)\n'
        "else:\n"
        f"    raise AssertionError({statement!r} + ' worked without core')\n"
        "print('refused')\n"
    )


def test_the_base_modules_import_and_work_without_core_nats_or_the_agent_layer() -> None:
    result = _run(BLOCK_CORE + f"MODULES = {_BASE_MODULES!r}\n" + USE_THE_BASE)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ran"


def test_dpop_without_core_names_the_extra() -> None:
    result = _run(_expect_the_core_extra_named("import threetears.iam.dpop"))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused"


def test_the_collection_attempt_limiter_without_core_names_the_extra() -> None:
    result = _run(_expect_the_core_extra_named("import threetears.iam.stores.attempt_limiter"))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused"


def test_sealing_a_seed_without_core_names_the_extra() -> None:
    code = "from pydantic import SecretStr; from threetears.iam import totp; totp.seal_seed('JBSWY3DPEHPK3PXP', SecretStr('k' * 32))"
    result = _run(_expect_the_core_extra_named(code))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused"


def test_unsealing_a_seed_without_core_names_the_extra() -> None:
    code = (
        "from pydantic import SecretStr; from threetears.iam import totp; totp.unseal_seed('v1.x', SecretStr('k' * 32))"
    )
    result = _run(_expect_the_core_extra_named(code))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused"


def test_decryption_error_without_core_names_the_extra() -> None:
    result = _run(_expect_the_core_extra_named("from threetears.iam.totp import DecryptionError"))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused"


def test_a_module_missing_inside_an_installed_core_is_not_relabelled_as_the_extra() -> None:
    """core present, sqlalchemy gone: a broken environment, which an install hint would misdirect."""
    code = (
        "import importlib.abc, sys\n"
        "class _NoSqlalchemy(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'sqlalchemy' or name.startswith('sqlalchemy.'):\n"
        "            raise ModuleNotFoundError(f'No module named {name!r}', name=name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _NoSqlalchemy())\n"
        "try:\n"
        "    import threetears.iam.stores.attempt_limiter\n"
        "except ModuleNotFoundError as exc:\n"
        "    assert exc.name == 'sqlalchemy', exc.name\n"
        "    assert '3tears-iam[core]' not in str(exc), str(exc)\n"
        "else:\n"
        "    raise AssertionError('imported with sqlalchemy refused; the test no longer exercises core')\n"
        "print('passed through')\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "passed through"


def test_decryption_error_with_core_is_core_s_own_class() -> None:
    """``except totp.DecryptionError`` must catch what ``unseal_seed`` raises."""
    from threetears.core.security import DecryptionError

    from threetears.iam import totp

    assert totp.DecryptionError is DecryptionError
    assert "DecryptionError" in dir(totp)
