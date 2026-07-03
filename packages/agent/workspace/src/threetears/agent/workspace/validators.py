"""pattern-matched validator dispatch hook for workspace writes.

every write-class tool routes through :func:`dispatch_validators` between
the sandbox gate and the transactional write. validators declared in
``agent.yaml`` (``workspace.validators`` list of
:class:`ValidatorEntry`) match the write's ``relative_path`` against a
posix glob via :meth:`pathlib.PurePosixPath.full_match` (same anchored
semantics as the sandbox allow-list -- ``*`` is single-segment, ``**``
recurses) and, when a pattern matches, invoke the dotted callable
resolved lazily by :func:`_resolve_validator`.

design:

- **lazy resolution + module-level cache**: :func:`_resolve_validator`
  imports each dotted path at most once per process and caches the
  callable. first call pays import cost; subsequent calls are a dict
  lookup. the cache stores the *callable*, never the *result* -- every
  dispatch re-runs validation against fresh content.
- **content-or-doc convention**: for ``fs_*`` writes the validator
  receives the raw bytes about to land in L3; for ``doc_*`` writes the
  tool dumps the post-mutation tree to text and passes the bytes, so
  validators always see exactly what the file-system would see (same
  contract for both tool families). validators that want structural
  access re-parse the bytes.
- **error shape**: on failure validators raise
  :class:`WorkspaceValidationError` (preferred) or any other exception.
  :func:`dispatch_validators` re-raises ``WorkspaceValidationError``
  unchanged and wraps anything else (``pydantic.ValidationError``,
  ``ValueError``, ``yaml.YAMLError``, etc.) into a fresh
  ``WorkspaceValidationError`` carrying ``pattern``, ``validator_path``,
  and ``reason=str(e)``. this lets 3tears-side validators raise stock
  pydantic exceptions without importing anything from this package.
- **fail-fast**: multiple matching validators run in ``agent.yaml`` list
  order; the first failure aborts. subsequent validators do not run.
- **no side channels**: dispatch never touches the DB, never logs, and
  never swallows exceptions. it's a pure predicate over
  ``(relative_path, content)`` that raises on failure and returns
  ``None`` on pass.

the module is import-cheap: only ``importlib`` and ``pathlib`` are
imported at module load. the :class:`WorkspaceConfig` / :class:
`ValidatorEntry` types are referenced only via ``TYPE_CHECKING`` so a
bare ``from threetears.agent.workspace.validators import
WorkspaceValidationError`` costs no extra import graph.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

__all__ = [
    "WorkspaceValidationError",
    "dispatch_validators",
]

if TYPE_CHECKING:
    from threetears.agent.workspace.config import ValidatorEntry


class WorkspaceValidationError(ValueError):
    """raised when a workspace-write validator rejects content.

    subclass of :class:`ValueError` so callers that already handle
    value-errors at a boundary continue to catch it, while write-class
    tools catch the specific type and convert it to a structured
    :class:`ToolResult` carrying the pattern and reason for agent-side
    self-correction.

    :ivar pattern: fnmatch glob from the :class:`ValidatorEntry` that
        triggered this failure (e.g. ``"*/audience_settings.yaml"``)
    :ivar validator_path: dotted import path of the validator callable
        that rejected the content (e.g.
        ``"3tears.agents.audience_builder.schemas.audience_settings.validate_audience_settings"``)
    :ivar reason: human-readable rejection reason suitable for surfacing
        to the LLM verbatim
    """

    def __init__(self, pattern: str, validator_path: str, reason: str) -> None:
        """
        capture pattern, validator path, and reason for actionable errors.

        :param pattern: fnmatch glob from the rejecting ValidatorEntry
        :ptype pattern: str
        :param validator_path: dotted import path of rejecting validator
        :ptype validator_path: str
        :param reason: human-readable rejection reason
        :ptype reason: str
        :return: None
        :rtype: None
        """
        self.pattern = pattern
        self.validator_path = validator_path
        self.reason = reason
        super().__init__(f"validation failed for pattern {pattern!r} (via {validator_path}): {reason}")


_RESOLVED: dict[str, Callable[..., Any]] = {}


def _resolve_validator(dotted: str) -> Callable[..., Any]:
    """
    resolve dotted import path to a callable; cache the callable for reuse.

    splits on the final ``.`` to derive ``module_path`` and ``attr``,
    calls :func:`importlib.import_module` on the module path, then
    :func:`getattr` on the resulting module. results are cached in the
    module-level ``_RESOLVED`` dict so each dotted path imports at most
    once per process. on code change the agent must restart -- same
    constraint as ``tools:`` entries in agent.yaml today.

    only the *callable* is cached, never a validator's return value.

    :param dotted: dotted import path resolving to a callable
        (e.g. ``"pkg.module.function"``)
    :ptype dotted: str
    :return: resolved callable
    :rtype: Callable[..., Any]
    :raises ValueError: if ``dotted`` has no module component
        (no ``.`` separator, e.g. bare ``"function"``), or if the
        resolved attribute is not callable
    :raises ImportError: if ``importlib.import_module`` fails on the
        module path (propagated unchanged -- the agent.yaml is
        misconfigured and the operator must fix it)
    :raises AttributeError: if the module lacks the named attribute
        (propagated unchanged, same rationale)
    """
    result: Callable[..., Any]
    cached = _RESOLVED.get(dotted)
    if cached is not None:
        return cached
    module_path, _, attr = dotted.rpartition(".")
    if not module_path:
        raise ValueError(f"validator must be a dotted import path, got {dotted!r}")
    module = importlib.import_module(module_path)
    fn = getattr(module, attr)
    if not callable(fn):
        raise ValueError(f"validator {dotted!r} is not callable")
    _RESOLVED[dotted] = fn
    result = fn
    return result


def dispatch_validators(
    validators: list[ValidatorEntry],
    relative_path: str,
    content_or_doc: Any,
) -> None:
    """
    run every matching validator; raise on the first failure.

    matches ``relative_path`` against each ``entry.pattern`` via
    :meth:`PurePosixPath.full_match` (consistent with
    :class:`threetears.agent.workspace.sandbox.WorkspaceSandbox` and
    :class:`threetears.core.security.sandbox.PathSandbox`: ``*`` is
    single-segment and anchored; ``**`` recurses across segments). for
    every matching entry, resolves the dotted callable via
    :func:`_resolve_validator` and invokes it as
    ``fn(relative_path, content_or_doc)``. a validator signals rejection
    by raising an exception:

    - :class:`WorkspaceValidationError` is re-raised unchanged.
    - any other exception is wrapped into a fresh
      :class:`WorkspaceValidationError` carrying the rejecting entry's
      ``pattern``, ``validator_path``, and ``reason=str(exc)``.

    the first failure aborts; subsequent validators are not invoked so
    the LLM sees one actionable error, not a pile. on pass, returns
    ``None``.

    takes a ``list[ValidatorEntry]`` directly rather than the full
    :class:`WorkspaceConfig` so :func:`_write_file_atomic` can keep
    dependency flow explicit and tests can exercise dispatch without
    instantiating a whole config graph.

    :param validators: per-pattern validator entries from the workspace
        configuration; empty list is a no-op
    :ptype validators: list[ValidatorEntry]
    :param relative_path: workspace-relative path being written, matched
        against each ``entry.pattern`` via ``PurePosixPath.full_match``
    :ptype relative_path: str
    :param content_or_doc: bytes about to land on disk (per shard
        decision, both ``fs_*`` and ``doc_*`` pass post-dump bytes)
    :ptype content_or_doc: Any
    :return: None on pass
    :rtype: None
    :raises WorkspaceValidationError: on the first validator that
        rejects the content
    """
    path = PurePosixPath(relative_path)
    for entry in validators:
        # full_match (vs match) anchors the pattern end-to-end so
        # `config/*.yaml` does NOT match `deep/nested/config/x.yaml` and
        # authors who want recursion use `**/...` explicitly.
        if not path.full_match(entry.pattern):
            continue
        fn = _resolve_validator(entry.validator)
        try:
            fn(relative_path, content_or_doc)
        except WorkspaceValidationError:
            raise
        except Exception as exc:
            raise WorkspaceValidationError(
                pattern=entry.pattern,
                validator_path=entry.validator,
                reason=str(exc),
            ) from exc
