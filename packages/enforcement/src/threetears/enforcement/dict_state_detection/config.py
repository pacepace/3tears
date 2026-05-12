"""configuration dataclasses for dict-state-detection enforcement.

the dict-state-detection domain enforces a single contract: persistent
state attached to ``self`` in ``__init__`` methods must not be a raw
``dict`` / ``OrderedDict``. shared / cached state belongs in a 3tears L1
backend (``SQLiteBackend``) for pod-local cache or NATS KV for
cross-instance sharing. raw dicts are flagged so they cannot silently
become a hidden cache layer that disagrees with the rest of the system
on lifetime, eviction, and persistence.

the rule is universal — it does not vary per repo — but two per-repo
allowlist surfaces let consumers acknowledge legitimate exceptions
without forking the walker:

- :attr:`DictStateConfig.allowlist`: tuples of
  :class:`DictStateAllowlistEntry` describing **legitimately-ephemeral**
  state. live LangChain instances, circuit-breaker state machines,
  per-process counter dicts -- things that genuinely cannot be
  serialised. these are *allowed forever*.
- :attr:`DictStateConfig.known_violations`: tuples of
  :class:`DictStateAllowlistEntry` describing violations that are
  *tracked for migration*. semantically different from ``allowlist``
  but the same shape, so the dataclass is reused. listed entries are
  filtered out of the failing-violation set; their stale-entry check
  is reported under a different category so the two surfaces are
  auditable independently.

both surfaces apply rationale discipline (minimum length, blanket-phrase
rejection) at construction time so the lists cannot become silent
test-disablers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DictStateAllowlistEntry",
    "DictStateConfig",
    "AllowlistRationaleError",
]


_MIN_RATIONALE_LENGTH = 30

_BLANKET_RATIONALE_PHRASES: frozenset[str] = frozenset(
    {
        "internal access",
        "tests need this",
        "tests need access",
        "temporary",
        "todo",
        "fixme",
        "needed",
        "required",
        "necessary",
    }
)


class AllowlistRationaleError(ValueError):
    """raised when an allowlist / known-violation entry's rationale is invalid.

    inherits from :class:`ValueError` so dataclass ``__post_init__``
    failures surface idiomatically, and so callers can choose to catch
    either the specific subclass (audit log) or the broader stdlib
    type (defensive try-except in fixture builders).
    """


@dataclass(frozen=True)
class DictStateAllowlistEntry:
    """one allowlist or known-violation entry.

    instances are matched against detected violations by the triple
    ``(file, line, attr_name)``. the ``rationale`` is required and
    validated at construction time — empty rationales, blanket phrases,
    and rationales below the shared-domain minimum length all raise
    :class:`AllowlistRationaleError`. this is the same discipline the
    common :func:`~threetears.enforcement.common.exemptions.parse_exemptions_with_rationale`
    parser applies to file-based exemptions; centralising the rule in
    the dataclass means in-code allowlists cannot drift away from
    file-based ones.

    :ivar file: forward-slash repo-relative path of the file holding
        the offending assignment, as written in source. matched against
        :func:`~threetears.enforcement.common.ast_helpers.relative_posix_path`
        of the violation's file.
    :ivar line: 1-based line number of the offending assignment. line
        matching is precise — a refactor that moves the assignment to
        a different line surfaces the entry as stale, which is the
        correct outcome.
    :ivar attr_name: name of the ``self.<attr>`` being assigned. always
        starts with an underscore because the walker only flags
        single-leading-underscore attributes.
    :ivar rationale: justification text. minimum
        :data:`_MIN_RATIONALE_LENGTH` characters; blanket phrases (e.g.
        ``"internal access"``, ``"tests need this"``) are rejected.
    """

    file: str
    line: int
    attr_name: str
    rationale: str

    def __post_init__(self) -> None:
        """validate ``rationale`` discipline at construction time.

        :raises AllowlistRationaleError: empty, blanket, or
            below-threshold rationale.
        """
        rationale = self.rationale.strip()
        if not rationale:
            raise AllowlistRationaleError(
                f"{self.file}:{self.line}:{self.attr_name}: rationale must be non-empty",
            )
        if len(rationale) < _MIN_RATIONALE_LENGTH:
            raise AllowlistRationaleError(
                f"{self.file}:{self.line}:{self.attr_name}: rationale "
                f"must be at least {_MIN_RATIONALE_LENGTH} characters; "
                f"got {len(rationale)}: {rationale!r}",
            )
        lower = rationale.lower()
        for phrase in _BLANKET_RATIONALE_PHRASES:
            if (
                lower == phrase
                or lower.startswith(phrase + " ")
                or lower.startswith(phrase + ".")
                or lower.startswith(phrase + ",")
            ):
                raise AllowlistRationaleError(
                    f"{self.file}:{self.line}:{self.attr_name}: rationale "
                    f"{rationale!r} matches blanket phrase {phrase!r}; "
                    f"rationales must be specific",
                )


@dataclass(frozen=True)
class DictStateConfig:
    """per-repo config for the dict-state-detection enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root (the
        directory containing its top-level ``pyproject.toml``).
    :ivar src_roots: optional explicit src-trees to scan. when
        ``None``, the runner calls
        :func:`threetears.enforcement.common.pyproject_discovery.discover_src_roots`
        so the walker sees every transitively-reachable path-dep src
        tree. set this to override discovery in tests or specialised
        harnesses.
    :ivar exemptions_path: path to
        ``_dict_state_detection_exemptions.txt``; ``None`` means "no
        exemptions file". retained for symmetry with sibling domains;
        in practice this domain leans on :attr:`allowlist` and
        :attr:`known_violations` for its escape hatches.
    :ivar mode_env_var: environment variable controlling strict vs
        report mode. defaults to ``DICT_STATE_ENFORCEMENT_MODE``.
    :ivar allowlist: tuple of :class:`DictStateAllowlistEntry`
        describing legitimately-ephemeral state — live connection
        objects, state machines, per-process counters. matched on
        ``(file, line, attr_name)``. each entry's ``rationale`` is
        validated at construction time (minimum length, blanket-phrase
        rejection). injected by the per-repo thin shell.
    :ivar known_violations: tuple of :class:`DictStateAllowlistEntry`
        describing violations being tracked for migration. same shape
        and matching as :attr:`allowlist`; the distinction is purely
        semantic (allowed forever vs. tracked for fix). reusing the
        same dataclass keeps the matching logic uniform.
    """

    repo_root: Path
    src_roots: tuple[Path, ...] | None = None
    exemptions_path: Path | None = None
    mode_env_var: str = "DICT_STATE_ENFORCEMENT_MODE"
    allowlist: tuple[DictStateAllowlistEntry, ...] = ()
    known_violations: tuple[DictStateAllowlistEntry, ...] = ()
