"""3tears-agent-knowledge: governed-knowledge retrieval + injection for LangGraph agents.

Homes the agent-side knowledge read stack -- the playbook-entry / concept
collections over the platform RBAC proxy, the ``KnowledgeIntegration`` wiring, and
the ``create_agent`` ``wrap_model_call`` injection middleware -- on top of the shared
three-scope merge authority in :mod:`threetears.knowledge` (core). This package
exists because the collections need :mod:`threetears.agent.acl`, which core (where
``threetears.knowledge`` lives) cannot depend on.
"""

from __future__ import annotations

# Version derived from pyproject.toml so the metadata is the single source of
# truth -- a future release that bumps pyproject without updating ``__init__.py``
# can't drift the runtime ``__version__``. The except guard handles the rare case
# where the package isn't installed via importlib.metadata (e.g. running directly
# from a checked-out source tree without ``uv sync``); the fallback keeps imports
# working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from typing import TYPE_CHECKING

try:
    __version__ = _version("3tears-agent-knowledge")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

# lazy public API (PEP 562). the package namespace no longer imports its
# implementation modules eagerly: importing this package (or any of its
# submodules) costs only this file, and each public attribute resolves its
# defining module on first access. the TYPE_CHECKING block carries the real
# imports so mypy and IDEs see the full statically-typed API; the _LAZY map is the
# runtime equivalent. the three-way agreement between __all__, _LAZY, and the
# TYPE_CHECKING block is pinned by the package's lazy-surface consistency test.
if TYPE_CHECKING:
    from threetears.agent.knowledge.collections import (
        ConceptCollection,
        PlaybookEntryCollection,
    )
    from threetears.agent.knowledge.entities import ConceptEntity, PlaybookEntryEntity
    from threetears.agent.knowledge.integration import (
        DraftView,
        KnowledgeIntegration,
        retrieve_concepts,
        retrieve_entries,
        scope_context_admitted_scopes,
    )
    from threetears.agent.knowledge.middleware import KnowledgeInjectionMiddleware

# public attribute -> (defining module, attribute name in that module)
_LAZY: dict[str, tuple[str, str]] = {
    "ConceptCollection": ("threetears.agent.knowledge.collections", "ConceptCollection"),
    "ConceptEntity": ("threetears.agent.knowledge.entities", "ConceptEntity"),
    "DraftView": ("threetears.agent.knowledge.integration", "DraftView"),
    "KnowledgeInjectionMiddleware": ("threetears.agent.knowledge.middleware", "KnowledgeInjectionMiddleware"),
    "KnowledgeIntegration": ("threetears.agent.knowledge.integration", "KnowledgeIntegration"),
    "PlaybookEntryCollection": ("threetears.agent.knowledge.collections", "PlaybookEntryCollection"),
    "PlaybookEntryEntity": ("threetears.agent.knowledge.entities", "PlaybookEntryEntity"),
    "retrieve_concepts": ("threetears.agent.knowledge.integration", "retrieve_concepts"),
    "retrieve_entries": ("threetears.agent.knowledge.integration", "retrieve_entries"),
    "scope_context_admitted_scopes": ("threetears.agent.knowledge.integration", "scope_context_admitted_scopes"),
}

__all__ = [
    "ConceptCollection",
    "ConceptEntity",
    "DraftView",
    "KnowledgeInjectionMiddleware",
    "KnowledgeIntegration",
    "PlaybookEntryCollection",
    "PlaybookEntryEntity",
    "retrieve_concepts",
    "retrieve_entries",
    "scope_context_admitted_scopes",
]


def __getattr__(name: str) -> object:
    """resolve a public attribute from its defining module on first access.

    :param name: attribute name being resolved
    :ptype name: str
    :return: the resolved attribute (also cached in module globals so
        ``__getattr__`` fires at most once per name)
    :rtype: object
    :raises AttributeError: when ``name`` is not part of the public API
    """
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr = entry
    value: object = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """include lazy attributes in ``dir()`` output.

    :return: sorted union of materialized globals and lazy names
    :rtype: list[str]
    """
    return sorted(set(globals()) | set(_LAZY))
