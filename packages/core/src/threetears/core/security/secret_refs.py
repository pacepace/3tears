"""pluggable secret resolution: a ``scheme://locator`` reference, resolved at use time.

A secret is referenced by a ``scheme://locator`` string, never by value; the value is
resolved at *use* time by the backend the scheme selects -- so the secret never lands
(plaintext) in a config file or DB and never sits in a long-lived process variable.
This is the canonical home (any 3tears app consumes it from ``core``);
``threetears.datasources.secrets`` re-exports it for back-compat.

shipped backends:

- ``env://NAME``      -- read process env var ``NAME``. the dev backend.
- ``k8s://rel/path``  -- read the file at ``<secrets-dir>/rel/path``, where
  ``<secrets-dir>`` defaults to ``/var/run/secrets/aibots`` and is overridable via
  ``AIBOTS_DATASOURCE_SECRETS_DIR``. the prod shape: a k8s ``Secret`` projected as a
  volume (projected secrets update in-place, so a new credential needs no redeploy).

registered-but-unimplemented (raise a clear error until a deployment needs them, so the
scheme surface is stable for config authors today): ``vault://``,
``aws-secretsmanager://``, ``gcp-sm://``.

Apps add their own scheme with :func:`register_scheme` -- e.g. scriob registers
``scriob://<repo_id>`` to resolve an encrypted-at-rest deploy key. Every resolution
failure raises :class:`SecretResolutionError` whose message names the *reference*
(scheme + locator -- safe to log) but never the resolved secret value.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import SecretStr

# a backend resolves a scheme's locator to a SecretStr. ``None`` in the
# registry marks a recognised-but-unimplemented scheme.
Resolver = Callable[[str], SecretStr]

__all__ = [
    "Resolver",
    "SecretResolutionError",
    "parse_ref",
    "register_scheme",
    "resolve_secret",
    "validate_ref",
]

# a reference is ``scheme://locator``. scheme is lowercase alnum + a
# few separators; locator is everything after the first ``://`` and
# is interpreted per-scheme.
_REF_RE = re.compile(r"^(?P<scheme>[a-z][a-z0-9+._-]*)://(?P<locator>.+)$")

# a registrable scheme name (the ``scheme`` half of a reference, anchored).
_SCHEME_NAME_RE = re.compile(r"^[a-z][a-z0-9+._-]*$")

# env-var name validity (mirrors config.py's _ENV_VAR_NAME_RE).
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# default mount point for the k8s Secret-as-volume backend. override
# via env so a deployment can relocate the projected-secret mount.
_K8S_SECRETS_DIR_ENV = "AIBOTS_DATASOURCE_SECRETS_DIR"
_K8S_SECRETS_DIR_DEFAULT = "/var/run/secrets/aibots"


class SecretResolutionError(ValueError):
    """raised when a ``scheme://locator`` reference cannot be resolved.

    subclasses :class:`ValueError` so existing call sites that catch the
    ``ValueError`` raised by the old ``_resolve_env_to_secret`` keep working. the
    message names the reference (safe) and never the secret value.
    """


def parse_ref(ref: str) -> tuple[str, str]:
    """split a ``scheme://locator`` reference into its parts.

    :param ref: the credential reference string
    :ptype ref: str
    :return: ``(scheme, locator)`` -- scheme lowercased, locator verbatim
    :rtype: tuple[str, str]
    :raises SecretResolutionError: when ``ref`` is not ``scheme://locator``
    """
    match = _REF_RE.match(ref)
    if match is None:
        raise SecretResolutionError(
            f"invalid secret reference {ref!r}: expected "
            f"'scheme://locator' (e.g. 'env://MY_VAR' or 'k8s://my-datasource/password').",
        )
    return match.group("scheme"), match.group("locator")


def _resolve_env(locator: str) -> SecretStr:
    """``env://NAME`` backend -- read process env var ``NAME``.

    :raises SecretResolutionError: when the name is malformed or unset
    """
    if not _ENV_VAR_NAME_RE.match(locator):
        raise SecretResolutionError(
            f"invalid env reference 'env://{locator}': {locator!r} is not a valid environment variable name.",
        )
    raw = os.environ.get(locator)
    if raw is None:
        raise SecretResolutionError(
            f"env://{locator} is not set. export {locator} before the secret is used.",
        )
    return SecretStr(raw)


def _resolve_k8s(locator: str) -> SecretStr:
    """``k8s://rel/path`` backend -- read a projected-Secret file.

    reads ``<secrets-dir>/rel/path`` where ``<secrets-dir>`` is
    ``AIBOTS_DATASOURCE_SECRETS_DIR`` or ``/var/run/secrets/aibots``. the file content
    is the exact secret (no newline stripping -- k8s Secret volumes store exact bytes).

    :raises SecretResolutionError: on path traversal, missing file, or read error
    """
    if ".." in Path(locator).parts:
        raise SecretResolutionError(
            f"invalid k8s reference 'k8s://{locator}': path traversal ('..') is not allowed.",
        )
    base = Path(os.environ.get(_K8S_SECRETS_DIR_ENV, _K8S_SECRETS_DIR_DEFAULT))
    secret_path = base / locator
    try:
        raw = secret_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SecretResolutionError(
            f"k8s://{locator} not found at {secret_path}. confirm the Secret is mounted "
            f"at {base} (set {_K8S_SECRETS_DIR_ENV} to relocate).",
        ) from None
    except OSError as exc:
        raise SecretResolutionError(
            f"k8s://{locator}: failed to read {secret_path}: {type(exc).__name__}.",
        ) from None
    return SecretStr(raw)


def _resolve_unimplemented(scheme: str) -> SecretResolutionError:
    """build the error for a registered-but-unimplemented backend."""
    return SecretResolutionError(
        f"secret backend {scheme!r} is recognised but not implemented in this build. "
        f"register it with threetears.core.security.register_scheme, or add it in core "
        f"when a deployment needs it.",
    )


# registered schemes. value is either a resolver callable or ``None``
# (recognised-but-unimplemented -- raises a clear error rather than the
# generic 'unknown scheme' so config authors know the scheme is on the
# roadmap).
_BACKENDS: dict[str, Resolver | None] = {
    "env": _resolve_env,
    "k8s": _resolve_k8s,
    "vault": None,
    "aws-secretsmanager": None,
    "gcp-sm": None,
}


def register_scheme(scheme: str, resolver: Resolver) -> None:
    """register a new ``scheme://`` resolver, so apps extend the secret surface.

    The resolver is called with the reference's locator and returns a ``SecretStr``.
    Refuses to overwrite an already-registered scheme (including the built-ins) -- a
    silent override of, say, ``env`` would be a security footgun.

    :param scheme: the scheme name (matches ``^[a-z][a-z0-9+._-]*$``)
    :ptype scheme: str
    :param resolver: ``locator -> SecretStr``
    :ptype resolver: Resolver
    :raises ValueError: when ``scheme`` is malformed or already registered
    """
    if not _SCHEME_NAME_RE.match(scheme):
        raise ValueError(
            f"invalid secret scheme name {scheme!r}: must match {_SCHEME_NAME_RE.pattern!r}.",
        )
    if scheme in _BACKENDS:
        raise ValueError(
            f"secret scheme {scheme!r} is already registered; refusing to overwrite it.",
        )
    _BACKENDS[scheme] = resolver


def validate_ref(ref: str) -> str:
    """validate a reference at config-load time without resolving it.

    checks the ``scheme://locator`` shape, that the scheme is registered, and (for
    ``env``) that the locator is a valid env-var name. does NOT touch the environment
    or filesystem -- resolution stays a use-time concern.

    :return: ``ref`` unchanged (so it can be used inline in a validator)
    :raises SecretResolutionError: when the reference is malformed or names an unknown scheme
    """
    scheme, locator = parse_ref(ref)
    if scheme not in _BACKENDS:
        known = ", ".join(sorted(_BACKENDS))
        raise SecretResolutionError(
            f"unknown secret scheme {scheme!r} in {ref!r}. known schemes: {known}.",
        )
    if scheme == "env" and not _ENV_VAR_NAME_RE.match(locator):
        raise SecretResolutionError(
            f"invalid env reference {ref!r}: {locator!r} is not a valid environment variable name.",
        )
    return ref


def resolve_secret(ref: str) -> SecretStr:
    """resolve a ``scheme://locator`` reference to a :class:`SecretStr`.

    dispatches on the scheme to the registered backend. the returned value is only ever
    held inside a ``SecretStr`` and unwrapped at the last moment.

    :raises SecretResolutionError: on malformed ref, unknown / unimplemented scheme, or
        backend resolution failure
    """
    scheme, locator = parse_ref(ref)
    backend = _BACKENDS.get(scheme)
    if backend is None:
        if scheme in _BACKENDS:
            raise _resolve_unimplemented(scheme)
        known = ", ".join(sorted(_BACKENDS))
        raise SecretResolutionError(
            f"unknown secret scheme {scheme!r} in {ref!r}. known schemes: {known}.",
        )
    return backend(locator)
