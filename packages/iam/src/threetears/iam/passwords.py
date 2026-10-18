"""Password hashing, verification, and set-time policy.

argon2id for every password this module writes. bcrypt is accepted on the
VERIFY path only, for credentials migrated in from an older system, and a
successful bcrypt verify hands the caller a fresh argon2id hash to write back
(upgrade-on-login). Nothing here ever produces a new bcrypt hash.

**NFKC normalization is applied to the password always**, before hashing,
before verifying, and before measuring length. This is not a knob. Without it
the same passphrase typed through a different keyboard layout or IME -- one
producing composed diacritics, the other decomposed -- fails to match its own
stored hash, and the failure is invisible to the user and to the operator.
Measuring length on the normalized form matters for the same reason: a policy
that counts one string and hashes another is a policy that rejects passwords it
would have accepted.

**Sync core, async wrappers.** argon2 is deliberately CPU-expensive, which on
an event loop means it is deliberately loop-blocking: at default cost
parameters a single verify stalls every other coroutine in the process for tens
of milliseconds, and a burst of login attempts becomes a self-inflicted denial
of service. So the hashing functions come in pairs -- a sync one for callers
that already run off the loop, and an ``_async`` one that offloads via
:func:`asyncio.to_thread`. Async callers must use the async pair. The
:class:`~argon2.PasswordHasher` is module-level and shared: it is thread-safe
and holds only cost parameters.

**Failures look uniform to the user, and are not uniform to the operator.** A
wrong password, a malformed stored hash, and an unsupported hash format all
return ``matched=False``, and none of them raise: a corrupt stored hash is an
authentication failure, not a 500, and an exception that escapes only on *some*
rejection paths is a side channel that tells an attacker which accounts have
unusual stored material. But a corrupt hash is not a wrong password. It locks
the account out for good, and read as one it looks like a user who keeps
mistyping. So the result names it in :attr:`PasswordVerifyResult.stored_hash_fault`
and it is logged at ERROR (without the hash) and counted on
``contained_errors_total``, for the caller to tell its operator and never its user.
"""

from __future__ import annotations

import asyncio
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import bcrypt
from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from threetears.iam.breach import BreachCorpus
from threetears.observe import contain, get_logger

__all__ = [
    "DEFAULT_MAX_PASSWORD_LENGTH",
    "DEFAULT_MIN_PASSWORD_LENGTH",
    "PasswordPolicyError",
    "PasswordVerifyResult",
    "equalize_verify_cost",
    "equalize_verify_cost_async",
    "hash_password",
    "hash_password_async",
    "normalize_password",
    "resolve_password_length_bounds",
    "validate_new_password",
    "verify_password",
    "verify_password_async",
]

#: The shared argon2id hasher (``Type.ID`` = argon2id, explicit rather than relying on the
#: library default staying argon2id across upgrades). Cost parameters are argon2-cffi's
#: maintained RFC 9106-aligned defaults. Thread-safe; reused across every call.
_HASHER: Final[PasswordHasher] = PasswordHasher(type=Type.ID)

log = get_logger(__name__)

_ARGON2_PREFIX: Final[str] = "$argon2id$"
_BCRYPT_PREFIXES: Final[tuple[str, ...]] = ("$2a$", "$2b$", "$2y$")

#: NIST 800-63B-aligned floors. A caller may raise either bound but never lower it below the
#: floor: a long minimum is the single highest-value password rule, and a short maximum
#: silently defeats passphrases and password managers.
DEFAULT_MIN_PASSWORD_LENGTH: Final[int] = 16
DEFAULT_MAX_PASSWORD_LENGTH: Final[int] = 64

#: A precomputed argon2id hash of a value no caller can supply, used only to spend the
#: hashing cost on paths where there is no real hash to verify against (see
#: :func:`equalize_verify_cost`). Computed once at import so the first request does not pay
#: for it.
_DUMMY_HASH: Final[str] = _HASHER.hash("threetears-iam/timing-equalization")


class PasswordPolicyError(Exception):
    """A candidate password was rejected at set/change time -- too short, too long, or
    present in the breach corpus.

    The breach-corpus message is deliberately generic: it says the password cannot be used
    without confirming *why*. A rejection that distinguishes "breached" from "too short"
    turns this call into an oracle an attacker can query against the corpus.
    """


@dataclass(frozen=True, slots=True)
class PasswordVerifyResult:
    """The outcome of :func:`verify_password`.

    :ivar matched: whether the password verified.
    :ivar upgraded_hash: set ONLY on a successful verify against a legacy bcrypt hash -- a
        fresh argon2id hash of the same password, which the caller is responsible for
        persisting in place of the bcrypt one. ``None`` on every other outcome (the stored
        hash was already argon2id, or the verify failed). This module never writes to a
        store; the write-back is the caller's, and skipping it is not a correctness bug --
        it just means the credential stays on bcrypt until the next successful login.
    :ivar stored_hash_fault: set when the STORED hash could not be read -- malformed
        argon2id, malformed bcrypt, or neither format -- saying which, never quoting the hash.
        ``matched`` is always ``False`` then: access is still denied. ``None`` for a wrong
        password against a readable hash. For the operator, not the user: the account cannot
        log in until its credential is reset, and telling the user why would tell an attacker
        which accounts hold unusual stored material.
    """

    matched: bool
    upgraded_hash: str | None = None
    stored_hash_fault: str | None = None


def normalize_password(password: str) -> str:
    """NFKC-normalize a password.

    Applied by every function in this module before hashing, verifying, or measuring; see
    the module docstring for why it is mandatory rather than configurable.
    """
    return unicodedata.normalize("NFKC", password)


def resolve_password_length_bounds(
    config: Mapping[str, Any] | None,
    *,
    min_floor: int = DEFAULT_MIN_PASSWORD_LENGTH,
    max_floor: int = DEFAULT_MAX_PASSWORD_LENGTH,
) -> tuple[int, int]:
    """Resolve ``(min_length, max_length)`` from a caller-supplied policy mapping, floored.

    A caller may configure a STRICTER minimum or a LONGER maximum, never the reverse: both
    bounds are clamped upward to their floor. A missing or non-``int`` entry falls back to
    the floor. This is why both clamps use :func:`max` -- raising ``max_length`` is a
    relaxation for the user and a non-event for security, so it is allowed; lowering it
    would break password managers and is not.

    :param config: an opaque policy mapping, e.g. a tenant's connection config. ``None`` is
        treated as empty -- both bounds fall back to their floors.
    :ptype config: Mapping[str, Any] | None
    :param min_floor: the lowest ``min_length`` any caller may configure.
    :ptype min_floor: int
    :param max_floor: the lowest ``max_length`` any caller may configure.
    :ptype max_floor: int
    :return: the resolved ``(min_length, max_length)``.
    :rtype: tuple[int, int]
    """
    configured_min = config.get("min_length") if config else None
    configured_max = config.get("max_length") if config else None
    min_length = max(configured_min, min_floor) if isinstance(configured_min, int) else min_floor
    max_length = max(configured_max, max_floor) if isinstance(configured_max, int) else max_floor
    return min_length, max_length


def validate_new_password(
    password: str,
    *,
    config: Mapping[str, Any] | None = None,
    breach_corpus: BreachCorpus | None = None,
    min_floor: int = DEFAULT_MIN_PASSWORD_LENGTH,
    max_floor: int = DEFAULT_MAX_PASSWORD_LENGTH,
) -> None:
    """Enforce set/change-time password policy. Call this BEFORE :func:`hash_password` on
    every path that writes new password material.

    Skipping it on one write path while a login-time breach check exists on the read path
    produces an account-bricking loop with no visible cause: the corpus-listed password is
    accepted at set time, then rejected at every subsequent login with a deliberately
    non-enumerating generic failure that tells the user nothing. Every write path must call
    this, or none should.

    Length is measured on the NFKC-normalized password, matching what will actually be
    hashed (module docstring).

    :param password: the candidate plaintext.
    :ptype password: str
    :param config: policy mapping for :func:`resolve_password_length_bounds`.
    :ptype config: Mapping[str, Any] | None
    :param breach_corpus: the corpus to screen against. ``None`` skips screening -- an
        explicit opt-out for callers that have no corpus, not a silent default.
    :ptype breach_corpus: BreachCorpus | None
    :param min_floor: passed through to :func:`resolve_password_length_bounds`.
    :ptype min_floor: int
    :param max_floor: passed through to :func:`resolve_password_length_bounds`.
    :ptype max_floor: int
    :raises PasswordPolicyError: too short, too long, or a breach-corpus match.
    """
    normalized = normalize_password(password)
    min_length, max_length = resolve_password_length_bounds(config, min_floor=min_floor, max_floor=max_floor)
    if len(normalized) < min_length:
        raise PasswordPolicyError(f"password must be at least {min_length} characters long.")
    if len(normalized) > max_length:
        raise PasswordPolicyError(f"password must be at most {max_length} characters long.")
    if breach_corpus is not None and breach_corpus.is_breached(normalized):
        # Deliberately generic -- never confirms corpus membership specifically (class docstring).
        raise PasswordPolicyError("this password can't be used; choose a different one.")


def hash_password(password: str) -> str:
    """Hash ``password`` with argon2id, normalizing first.

    Blocking and CPU-bound. On an event loop use :func:`hash_password_async`.

    :param password: the plaintext to hash. Never logged, never stored.
    :ptype password: str
    :return: the argon2id PHC string (``$argon2id$v=19$...``) to persist.
    :rtype: str
    """
    return _HASHER.hash(normalize_password(password))


async def hash_password_async(password: str) -> str:
    """:func:`hash_password`, offloaded to a worker thread so it does not stall the loop."""
    return await asyncio.to_thread(hash_password, password)


def verify_password(password: str, stored_hash: str) -> PasswordVerifyResult:
    """Verify ``password`` against ``stored_hash`` -- argon2id, or a migrated bcrypt hash.

    Never raises for a wrong password, a malformed hash, or an unrecognized format: all
    three return ``matched=False``, so no rejection reason is distinguishable by exception
    shape (module docstring). The last two also set ``stored_hash_fault`` and log at ERROR,
    so a corrupt credential is not mistaken for a mistyped password. Blocking and CPU-bound -- on an event loop use
    :func:`verify_password_async`.

    :param password: the candidate plaintext.
    :ptype password: str
    :param stored_hash: the stored hash to verify against.
    :ptype stored_hash: str
    :return: the outcome, carrying an ``upgraded_hash`` when a legacy bcrypt hash verified.
    :rtype: PasswordVerifyResult
    """
    normalized = normalize_password(password)
    if stored_hash.startswith(_ARGON2_PREFIX):
        return _verify_argon2(normalized, stored_hash)
    if stored_hash.startswith(_BCRYPT_PREFIXES):
        return _verify_bcrypt_and_upgrade(normalized, stored_hash)
    if not stored_hash:
        # No password is set: an SSO-only account, or one never given a password. Denied and
        # named like any other unusable hash, but not an ERROR -- nothing is broken, and an
        # operator paged for every password attempt on an SSO account would learn to ignore it.
        fault = "no password is set for this account (the stored hash is empty)"
        log.info("password verify: %s; denied", fault)
        return PasswordVerifyResult(matched=False, stored_hash_fault=fault)
    fault = "the stored hash is neither argon2id nor a migrated bcrypt hash"
    log.error("password verify: %s; denied, and the credential needs resetting", fault)
    return PasswordVerifyResult(matched=False, stored_hash_fault=fault)


async def verify_password_async(password: str, stored_hash: str) -> PasswordVerifyResult:
    """:func:`verify_password`, offloaded to a worker thread so it does not stall the loop."""
    return await asyncio.to_thread(verify_password, password, stored_hash)


def equalize_verify_cost() -> None:
    """Spend one argon2 verify against a throwaway hash, discarding the result.

    Call this on the "no such user" branch of a login. Without it, an unknown username
    returns as fast as the database lookup and a known one takes an argon2 verify longer,
    which is a user-enumeration oracle measurable over the network. This is best-effort
    rather than a constant-time guarantee -- it equalizes the dominant term, not every term.

    The verify is expected to FAIL (nothing hashes to :data:`_DUMMY_HASH`); the mismatch is
    swallowed because the point is the elapsed time, not the answer.
    """
    try:
        _HASHER.verify(_DUMMY_HASH, "")
    # NOSILENT: the verify is EXPECTED to fail; this call is timed, not consulted
    except VerifyMismatchError, VerificationError, InvalidHashError:
        pass


async def equalize_verify_cost_async() -> None:
    """:func:`equalize_verify_cost`, offloaded to a worker thread."""
    await asyncio.to_thread(equalize_verify_cost)


def _verify_argon2(normalized_password: str, stored_hash: str) -> PasswordVerifyResult:
    """Verify against an argon2id hash. Blocking; fails closed on malformed material, and names it."""
    try:
        _HASHER.verify(stored_hash, normalized_password)
    except VerifyMismatchError:
        return PasswordVerifyResult(matched=False)
    except (VerificationError, InvalidHashError) as exc:
        # A malformed argon2 hash denies, never raises past this module (module docstring),
        # and is told apart from a wrong password. Neither error's message quotes the hash.
        fault = f"the stored argon2id hash could not be read ({type(exc).__name__})"
        contain(
            exc, log=log, what=f"password verify: {fault}; denied, and the credential needs resetting", level="error"
        )
        return PasswordVerifyResult(matched=False, stored_hash_fault=fault)
    return PasswordVerifyResult(matched=True)


def _verify_bcrypt_and_upgrade(normalized_password: str, stored_hash: str) -> PasswordVerifyResult:
    """Verify against a migrated bcrypt hash, returning a fresh argon2id hash on success.

    bcrypt silently truncates at 72 BYTES, so a long passphrase's tail was never part of
    the stored hash. Passing the untruncated password back to ``checkpw`` reproduces that
    truncation identically, which is what makes the migrated credential verify at all --
    but it also means the upgrade is the only chance to stop inheriting the weakness, and
    the argon2id hash returned here covers the FULL password.
    """
    try:
        matched = bcrypt.checkpw(normalized_password.encode("utf-8"), stored_hash.encode("utf-8"))
    except ValueError as exc:
        # A malformed bcrypt hash fails closed and is named, like the argon2 path. bcrypt's
        # message ("Invalid salt") does not quote the hash.
        fault = f"the stored bcrypt hash could not be read ({exc})"
        contain(
            exc, log=log, what=f"password verify: {fault}; denied, and the credential needs resetting", level="error"
        )
        return PasswordVerifyResult(matched=False, stored_hash_fault=fault)
    if not matched:
        return PasswordVerifyResult(matched=False)
    return PasswordVerifyResult(matched=True, upgraded_hash=_HASHER.hash(normalized_password))
