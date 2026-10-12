"""dependency-free helpers for the JSON Schema a tool advertises for its arguments.

:func:`self_contained_input_schema` inlines every ``$ref`` and collapses every optional union, so a
model or validator handed one schema per tool sees nested models as the objects they are.
:func:`declared_type` reads the one JSON type a property declares through optional unions and
references. This package depends on nothing, so every tool host -- a model adapter, a tool
framework, an MCP server -- can take it without inheriting another's dependency closure; purity is
enforced by the contract-purity check in the workspace's ``tests/enforcement/``.
"""

# Version derived from package metadata so the metadata is the single source of truth -- a release
# that bumps pyproject cannot leave a stale runtime ``__version__`` behind. The fallback keeps the
# import working from a source tree that was never installed. ``importlib.metadata`` is stdlib, so
# the dependency-free floor is untouched.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-tool-schema")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.tool_schema.self_contained import declared_type, self_contained_input_schema

__all__ = [
    "declared_type",
    "self_contained_input_schema",
]
