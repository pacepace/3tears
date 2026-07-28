"""configuration dataclass for single-return enforcement.

the single-return domain enforces one contract: a function's
business logic has at most ONE ``return``, with leading guard
clauses exempted. a guard clause is a plain ``if`` (no ``else`` /
``elif``) whose body is exactly one ``return``, appearing before any
non-guard statement. unlimited leading guards are allowed; after the
guards, the business logic returns at most once.

the rule is about readability of the exit path, so two knobs cover
the legitimate escape hatches without forking the walker:

- :attr:`excluded_function_names`: functions that idiomatically
  branch-return. defaults to the rich-comparison / truthiness /
  repr dunders, which the language itself pushes toward multiple
  returns.
- :attr:`exempt_files`: relative-posix-path -> rationale mapping.
  files listed here are skipped entirely.

nested scopes are charged to themselves, never to the enclosing
function -- a ``def`` or ``lambda`` inside a function body ends the
enclosing function's return accounting, because the module-level
walk visits that nested definition on its own. this is the one
correctness point where a naive ``ast.walk`` implementation gets it
wrong, and it is why the walker lives here rather than being
re-derived per repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["SingleReturnConfig"]


#: dunders the language idiomatically pushes toward multiple returns:
#: rich comparison, hashing, truthiness, and the two repr protocols.
_DEFAULT_EXCLUDED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__hash__",
        "__bool__",
        "__repr__",
        "__str__",
    }
)


@dataclass(frozen=True)
class SingleReturnConfig:
    """per-repo config for the single-return enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root (the
        directory containing its top-level ``pyproject.toml``).
    :ivar src_roots: optional explicit src-trees to scan. when
        ``None``, the runner calls
        :func:`threetears.enforcement.common.pyproject_discovery.discover_src_roots`
        so the walker sees every transitively-reachable path-dep src
        tree. set this to override discovery in tests or specialised
        harnesses, and to scan ONE component of a multi-component
        repo.
    :ivar exemptions_path: path to ``_single_return_exemptions.txt``;
        ``None`` means "no exemptions file".
    :ivar mode_env_var: environment variable controlling strict vs
        report mode. defaults to ``SINGLE_RETURN_ENFORCEMENT_MODE``.
    :ivar excluded_function_names: function names skipped entirely.
        defaults to the comparison / truthiness / repr dunders.
    :ivar exempt_files: relative-posix-path -> rationale mapping.
        files listed here are skipped before any AST work.
    """

    repo_root: Path
    src_roots: tuple[Path, ...] | None = None
    exemptions_path: Path | None = None
    mode_env_var: str = "SINGLE_RETURN_ENFORCEMENT_MODE"
    excluded_function_names: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_EXCLUDED_FUNCTIONS,
    )
    exempt_files: dict[str, str] = field(default_factory=dict)
