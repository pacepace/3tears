"""policy-driven access-control primitive; Sandbox ABC + PathSandbox concrete.

generalizes the "check before act" pattern into a reusable ABC:

- :class:`Sandbox` — abstract contract; subclasses implement ``check`` to
  return :class:`SandboxDecision`, inherit :meth:`Sandbox.enforce` which
  raises :class:`SandboxDenied` on DENY.
- :class:`PathSandbox` — concrete filesystem-shaped sandbox with named
  roots (root-jail with symlink-escape rejection), virtual-key syntactic
  validation, and per-mode glob allow-lists.

future subclasses (``WorkspaceSandbox``, ``NetworkSandbox``,
``SubprocessSandbox``, ``QuerySandbox``) extend the same contract, each
defining their own action vocabulary and deny-reason messages via
:meth:`Sandbox.deny_reason`.

design notes:

- action vocabulary is open: :meth:`Sandbox.check` takes arbitrary
  action strings; subclasses deny unknown actions by default.
- :class:`SandboxDenied` extends :class:`PermissionError` so callers may
  catch broadly or narrowly.
- path matching uses :class:`pathlib.PurePosixPath.match` semantics for
  correct ``**`` recursive wildcard support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal

__all__ = [
    "PathSandbox",
    "Sandbox",
    "SandboxDecision",
    "SandboxDenied",
]


class SandboxDecision(StrEnum):
    """policy outcome for a single :meth:`Sandbox.check` call.

    members:

    - ``ALLOW`` — requested action on target is permitted
    - ``DENY`` — requested action on target is forbidden
    """

    ALLOW = "allow"
    DENY = "deny"


class SandboxDenied(PermissionError):
    """raised by :meth:`Sandbox.enforce` when policy denies requested action.

    inherits :class:`PermissionError` so callers may catch at that level or
    at this narrower class. exposes three attributes for programmatic
    handling and structured logging:

    :ivar action: action verb that was denied (e.g. ``"read"``, ``"write"``)
    :ivar target: target identifier (e.g. relative key, path, url)
    :ivar reason: human-readable explanation of why policy denied
    """

    def __init__(self, action: str, target: str, reason: str) -> None:
        """populate attributes and formatted message on exception.

        :param action: action verb being denied
        :ptype action: str
        :param target: target identifier being denied
        :ptype target: str
        :param reason: explanation of denial for log and error surface
        :ptype reason: str
        :return: None
        :rtype: None
        """
        self.action = action
        self.target = target
        self.reason = reason
        super().__init__(f"sandbox denied {action!r} on {target!r}: {reason}")


class Sandbox(ABC):
    """policy-driven access-control abstract base class.

    subclasses implement :meth:`check` returning a :class:`SandboxDecision`.
    :meth:`enforce` is concrete and calls :meth:`check`, raising
    :class:`SandboxDenied` on DENY. subclasses may override
    :meth:`deny_reason` to provide richer error messages without
    overriding :meth:`check` or :meth:`enforce`.
    """

    @abstractmethod
    def check(self, action: str, target: str) -> SandboxDecision:
        """evaluate policy for given action and target; return decision.

        subclasses implement their own action vocabulary; unknown actions
        must return :attr:`SandboxDecision.DENY` (fail-closed).

        :param action: action verb being evaluated
        :ptype action: str
        :param target: target identifier being evaluated
        :ptype target: str
        :return: ALLOW if permitted, DENY otherwise
        :rtype: SandboxDecision
        """

    def enforce(self, action: str, target: str) -> None:
        """raise :class:`SandboxDenied` if :meth:`check` returns DENY.

        :param action: action verb to enforce
        :ptype action: str
        :param target: target identifier to enforce
        :ptype target: str
        :return: None
        :rtype: None
        :raises SandboxDenied: if policy denies requested action on target
        """
        if self.check(action, target) is SandboxDecision.DENY:
            raise SandboxDenied(action, target, self.deny_reason(action, target))

    def deny_reason(self, action: str, target: str) -> str:
        """return human-readable reason why policy denies this call.

        default implementation returns generic ``"policy denied"``.
        subclasses override to return richer, actionable messages (which
        rule failed, which glob did not match, etc.).

        :param action: action verb being denied
        :ptype action: str
        :param target: target identifier being denied
        :ptype target: str
        :return: reason string embedded in :class:`SandboxDenied.reason`
        :rtype: str
        """
        return "policy denied"


class PathSandbox(Sandbox):
    """filesystem-shaped sandbox with named roots and glob allow-lists.

    handles two distinct concerns:

    - **virtual relative-key validation** — reject absolute paths, ``..``
      parent references, NUL/control chars, empty, oversize strings; then
      match against per-mode glob allow-lists using
      :meth:`pathlib.PurePosixPath.match` (supports ``**``).
    - **filesystem root-jail** — :meth:`resolve_fs_path` resolves a
      caller-supplied relative path under a named root, detecting escape
      via ``..`` traversal or symlink resolution via
      :meth:`pathlib.Path.resolve`.

    action vocabulary is ``{"read", "write"}``; unknown actions deny by
    default.

    :cvar _MAX_KEY_LEN: maximum accepted length for any relative key;
        keys longer than this are rejected before pattern matching
    """

    _MAX_KEY_LEN = 512

    def __init__(
        self,
        *,
        fs_roots: dict[str, Path],
        allow_read: list[str],
        allow_write: list[str],
    ) -> None:
        """capture named roots (resolved) and per-mode allow-lists.

        each root value is resolved via :meth:`Path.resolve` at
        construction time so the containment check in
        :meth:`resolve_fs_path` compares resolved paths against a resolved
        root (both sides free of symlinks).

        :param fs_roots: mapping of logical root name to base filesystem
            :class:`Path`; keys become valid ``root_name`` arguments to
            :meth:`resolve_fs_path`
        :ptype fs_roots: dict[str, Path]
        :param allow_read: shell-style globs against which ``read`` keys
            are matched using :meth:`PurePosixPath.match`
        :ptype allow_read: list[str]
        :param allow_write: shell-style globs against which ``write``
            keys are matched using :meth:`PurePosixPath.match`
        :ptype allow_write: list[str]
        :return: None
        :rtype: None
        """
        self._fs_roots: dict[str, Path] = {name: Path(p).resolve() for name, p in fs_roots.items()}
        self._allow_read: list[str] = list(allow_read)
        self._allow_write: list[str] = list(allow_write)

    def check(self, action: str, target: str) -> SandboxDecision:
        """dispatch ``read``/``write`` to :meth:`check_relative_key`; deny else.

        :param action: action verb; only ``"read"`` and ``"write"`` are
            recognized, all others deny
        :ptype action: str
        :param target: relative key to validate and match against globs
        :ptype target: str
        :return: ALLOW when key passes validation and matches a glob;
            DENY otherwise
        :rtype: SandboxDecision
        """
        if action not in {"read", "write"}:
            return SandboxDecision.DENY
        mode: Literal["read", "write"] = "read" if action == "read" else "write"
        return self.check_relative_key(target, mode)

    def check_relative_key(self, key: str, mode: Literal["read", "write"]) -> SandboxDecision:
        """validate relative key syntactically then match against mode globs.

        validation sequence (first failing rule returns DENY):

        1. key is non-empty
        2. ``len(key) <= _MAX_KEY_LEN``
        3. no NUL byte nor control character (``ord(c) < 32``) except
           ``\\t`` and ``\\n``
        4. key is not absolute (no leading ``/``, no windows drive letter)
        5. no path segment equal to ``".."``
        6. at least one allow-list glob matches via
           :meth:`PurePosixPath.match`

        :param key: relative key to validate
        :ptype key: str
        :param mode: selects which allow-list to match against
        :ptype mode: Literal["read", "write"]
        :return: ALLOW if all rules pass, DENY otherwise
        :rtype: SandboxDecision
        """
        return self._classify_relative_key(key, mode)[0]

    def resolve_fs_path(self, path: str | Path, root_name: str) -> Path:
        """resolve ``path`` under named root; reject escape via ``..`` or symlink.

        algorithm:

        1. look up ``root = self._fs_roots[root_name]`` — raises
           :class:`KeyError` if root not configured (this is a programmer
           error, not a policy denial).
        2. if ``path`` is absolute, raise :class:`SandboxDenied` with
           action ``"access"`` (the read/write distinction is orthogonal
           to path resolution; callers pair this call with
           :meth:`enforce` / :meth:`check` for action-specific policy).
        3. compute ``candidate = (root / path).resolve()`` — this follows
           symlinks so the final candidate reflects real disk location.
        4. verify ``candidate`` is under ``root`` via
           :meth:`Path.relative_to`; escape is raised as
           :class:`SandboxDenied` with action ``"access"``.

        :param path: relative path (or :class:`Path`) under ``root_name``
        :ptype path: str | Path
        :param root_name: logical name of root registered at construction
        :ptype root_name: str
        :return: resolved absolute path known to be inside the named root
        :rtype: Path
        :raises KeyError: if ``root_name`` was not registered in
            ``fs_roots`` at construction time
        :raises SandboxDenied: if ``path`` is absolute, if the resolved
            candidate escapes the named root via ``..`` traversal, or if
            symlink resolution lands outside the root
        """
        root = self._fs_roots[root_name]
        as_path = Path(path)
        if as_path.is_absolute():
            raise SandboxDenied("access", str(path), "absolute path not allowed")
        candidate = (root / as_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise SandboxDenied("access", str(path), "path escapes root") from exc
        return candidate

    def deny_reason(self, action: str, target: str) -> str:
        """return actionable reason by re-running :meth:`check_relative_key`.

        computed on demand rather than cached on a per-call mutable attr
        so there is no race risk when the same sandbox is shared across
        threads or tasks.

        :param action: action verb that was denied
        :ptype action: str
        :param target: target identifier that was denied
        :ptype target: str
        :return: short explanation of which rule failed
        :rtype: str
        """
        if action not in {"read", "write"}:
            result = f"action {action!r} not in allowed actions (read, write)"
        else:
            mode: Literal["read", "write"] = "read" if action == "read" else "write"
            _, reason = self._classify_relative_key(target, mode)
            result = reason
        return result

    def _classify_relative_key(self, key: str, mode: Literal["read", "write"]) -> tuple[SandboxDecision, str]:
        """joint implementation of decision + reason for a relative key.

        single source of truth behind :meth:`check_relative_key` and
        :meth:`deny_reason`; returns ``(decision, reason)`` so callers
        that only need the decision discard the reason.

        :param key: relative key to validate and match
        :ptype key: str
        :param mode: selects which allow-list to match against
        :ptype mode: Literal["read", "write"]
        :return: pair of decision and reason string; reason is
            ``"allowed"`` on ALLOW
        :rtype: tuple[SandboxDecision, str]
        """
        result: tuple[SandboxDecision, str]
        if not key:
            result = (SandboxDecision.DENY, "key is empty")
        elif len(key) > self._MAX_KEY_LEN:
            result = (
                SandboxDecision.DENY,
                f"key length {len(key)} exceeds max {self._MAX_KEY_LEN}",
            )
        elif self._contains_control_char(key):
            result = (
                SandboxDecision.DENY,
                "key contains NUL or control character",
            )
        elif Path(key).is_absolute():
            result = (SandboxDecision.DENY, "absolute path not allowed")
        elif ".." in Path(key).parts:
            result = (SandboxDecision.DENY, "parent-ref (..) not allowed in key")
        else:
            patterns = self._allow_read if mode == "read" else self._allow_write
            if self._matches_any(key, patterns):
                result = (SandboxDecision.ALLOW, "allowed")
            else:
                result = (
                    SandboxDecision.DENY,
                    f"no {mode}-mode allow glob matched {key!r}",
                )
        return result

    @staticmethod
    def _contains_control_char(key: str) -> bool:
        """return True when any char in ``key`` is NUL or low-ASCII control.

        tab (``\\t``) and newline (``\\n``) are tolerated since they are
        not path-traversal hazards and some legitimate keys may contain
        them (defensive: this is about rejecting unprintable junk).

        :param key: candidate relative key
        :ptype key: str
        :return: True if rejection rule 3 fires, False otherwise
        :rtype: bool
        """
        result = False
        for ch in key:
            if ord(ch) < 32 and ch not in ("\t", "\n"):
                result = True
                break
        return result

    @staticmethod
    def _matches_any(key: str, patterns: list[str]) -> bool:
        """return True when ``key`` matches any of ``patterns`` as posix glob.

        uses :meth:`PurePosixPath.full_match` so the pattern matches
        against the entire path rather than just the right side. this
        gives the expected semantics:

        - ``*.yaml`` matches ``foo.yaml`` but NOT ``sub/foo.yaml``
          (``*`` is single-segment and anchored)
        - ``**/*.yaml`` matches both ``foo.yaml`` and ``a/b/c.yaml``
          (``**`` is recursive)
        - ``**/*`` matches any non-empty relative key including dotfiles

        posix variant forces forward-slash segment semantics regardless
        of host OS; match is case-sensitive by default on posix.

        :param key: relative key being matched
        :ptype key: str
        :param patterns: list of shell-style globs to try
        :ptype patterns: list[str]
        :return: True if any pattern matches, False otherwise
        :rtype: bool
        """
        posix_key = PurePosixPath(key)
        result = False
        for pattern in patterns:
            if posix_key.full_match(pattern):
                result = True
                break
        return result
