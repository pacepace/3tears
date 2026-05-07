"""tool group aliases and selector expansion.

agent.yaml ``access.tools`` patterns may include shorthand group names
that resolve to concrete tool name globs. this module owns the canonical
group -> pattern set and the :func:`expand_selectors` helper that the
SDK calls at config-load time to flatten aliases into glob patterns the
existing access-pattern matchers consume.

groups intentionally model the way platform users think about tools:

* ``standard`` -- cheap deterministic local utilities (calculator,
  current_date, dictionary, unit_converter, timezone_converter). every
  agent should have these on by default; they hit no network and need
  no API keys.
* ``web`` -- web_search, web_fetch. opt-in because they hit the network
  and may not be wanted for every agent.
* ``media`` -- analyze_media, parse_document, image_prep,
  image_generation. opt-in because they need provider API keys
  (analyze_media / image_generation) or heavy optional deps
  (parse_document needs PyMuPDF + OCR stack) and incur cost.
* ``workspace`` -- the full 19-tool ``threetears.workspace.*`` bundle.
  opt-in because it mutates filesystem state under ``bind_root``.
* ``workspace.fs`` / ``workspace.doc`` / ``workspace.lifecycle`` --
  sub-bundles for callers who want a slice of workspace without the
  whole thing.

selector expansion is monotonic: each selector either resolves to a
group's underlying patterns or passes through verbatim (for individual
tool names, custom Python class paths, and explicit globs the user
wrote). disable patterns from the structured form are merged in
:func:`expand_selectors` so consumers get one flat allow-list.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "GROUP_ALIASES",
    "STANDARD_TOOLS",
    "WEB_TOOLS",
    "MEDIA_TOOLS",
    "WORKSPACE_FS_TOOLS",
    "WORKSPACE_DOC_TOOLS",
    "WORKSPACE_LIFECYCLE_TOOLS",
    "WORKSPACE_TOOLS",
    "expand_selectors",
    "is_python_class_path",
]


# canonical builtin tool name sets. names match the ``mcp_name`` each
# builtin's ``TearsTool`` subclass returns. external pods register
# under the same names via the registry, so a pattern like ``calculator``
# matches whether the tool runs in-process or in a separate pod.

STANDARD_TOOLS: frozenset[str] = frozenset({
    "threetears.calculator",
    "threetears.current_date",
    "threetears.dictionary",
    "threetears.unit_converter",
    "threetears.timezone_converter",
})

WEB_TOOLS: frozenset[str] = frozenset({
    "threetears.web_search",
    "threetears.web_fetch",
})

# media tool name forms vary by registration path:
# * ``threetears.analyze_media`` -- TearsTool subclass under ``builtin/``
# * ``threetears.parse_document`` -- TearsTool subclass under
#   ``document/`` (separate dir because PyMuPDF / pdfminer / OCR
#   stack is an optional dependency; the tool registers only when
#   the deps are installed, see ``agent.tools.serve``)
# * ``threetears.image_prep`` -- TearsTool subclass under ``builtin/``;
#   preprocessing / resizing for vision-input images, NOT generation
# * ``image_generation`` -- StructuredTool factory path, registers
#   under the bare ``image_generation`` name (no ``threetears.``
#   prefix); creates new images via Anthropic / OpenAI / etc.
# the alias matches whatever name actually appears in the registry.
MEDIA_TOOLS: frozenset[str] = frozenset({
    "threetears.analyze_media",
    "threetears.parse_document",
    "threetears.image_prep",
    "image_generation",
})

# workspace tool names match the keys
# :func:`threetears.agent.workspace.tool_factories.build_workspace_tools`
# registers; they are namespaced under ``threetears.workspace.*`` on
# the wire but the short form (without the namespace prefix) is what
# selector matchers compare against.

WORKSPACE_FS_TOOLS: frozenset[str] = frozenset({
    "threetears.workspace.fs_read",
    "threetears.workspace.fs_write",
    "threetears.workspace.fs_edit",
    "threetears.workspace.fs_list",
})

WORKSPACE_DOC_TOOLS: frozenset[str] = frozenset({
    "threetears.workspace.doc_get",
    "threetears.workspace.doc_set",
    "threetears.workspace.doc_merge",
})

WORKSPACE_LIFECYCLE_TOOLS: frozenset[str] = frozenset({
    "threetears.workspace.create",
    "threetears.workspace.delete",
    "threetears.workspace.use",
    "threetears.workspace.list",
    "threetears.workspace.current",
    "threetears.workspace.refresh_from_disk",
    "threetears.workspace.flush_to_disk",
    "threetears.workspace.checkpoint",
    "threetears.workspace.history",
    "threetears.workspace.diff",
    "threetears.workspace.reset",
    "threetears.workspace.rollback_to",
})

WORKSPACE_TOOLS: frozenset[str] = (
    WORKSPACE_FS_TOOLS | WORKSPACE_DOC_TOOLS | WORKSPACE_LIFECYCLE_TOOLS
)


# group alias -> set of concrete tool names. keys are the bare words
# users write in ``access.tools``; values are the underlying tool names
# the resolver expands them into. dotted sub-aliases (``workspace.fs``)
# are siblings of the parent group rather than nested keys so the
# resolver can do a single dict lookup.

GROUP_ALIASES: dict[str, frozenset[str]] = {
    "standard": STANDARD_TOOLS,
    "web": WEB_TOOLS,
    "media": MEDIA_TOOLS,
    "workspace": WORKSPACE_TOOLS,
    "workspace.fs": WORKSPACE_FS_TOOLS,
    "workspace.doc": WORKSPACE_DOC_TOOLS,
    "workspace.lifecycle": WORKSPACE_LIFECYCLE_TOOLS,
}


def is_python_class_path(selector: str) -> bool:
    """heuristically identify a Python dotted class path in a selector list.

    a Python class path is a dotted name with at least two segments
    where the final segment starts with an uppercase letter (PEP 8
    class naming). used by the SDK tool loader to distinguish
    custom-tool-loader entries from access-filter patterns when both
    ride on the same agent.yaml field.

    intentionally conservative: a pattern like ``my_pkg.SearchTool``
    is treated as a class path; ``my_pkg.search_tool`` is not (lower-
    snake suggests a module-level glob, not a class). users who want
    to force class-path interpretation can prefix with ``python:``
    (handled by the SDK tool loader, not here).

    :param selector: bare selector string from ``access.tools`` /
        ``tools.enable``
    :ptype selector: str
    :return: ``True`` when the selector looks like a Python class path
        (multi-segment, last segment starts uppercase); ``False``
        otherwise (group alias, glob, single tool name, etc.)
    :rtype: bool
    """
    result = False
    if "." in selector and "*" not in selector and "?" not in selector:
        last_segment = selector.rsplit(".", 1)[-1]
        if last_segment and last_segment[0].isupper():
            result = True
    return result


def expand_selectors(
    enable: list[str],
    disable: list[str] | None = None,
) -> list[str]:
    """expand group aliases and apply disable filter; return flat pattern list.

    walks ``enable`` and:

    * passes through globs (``foo.*``, ``bar?``) verbatim for the
      caller's existing fnmatch-style matcher to consume
    * passes through individual concrete tool names verbatim
    * passes through Python class paths (per
      :func:`is_python_class_path`) verbatim so the SDK tool loader
      sees them on the same field
    * expands group aliases (``standard``, ``workspace.fs``, ...) to
      their concrete tool name set

    then removes any expanded name that matches a pattern in
    ``disable`` (literal name match or fnmatch glob). preserves
    declaration order on first occurrence; de-duplicates so the
    returned list is suitable for an ordered allow-list. caller is
    responsible for converting the returned list back into whatever
    matcher shape the consuming layer expects (a ``list[str]`` works
    for the existing access-pattern fnmatch path).

    :param enable: enable selectors from ``access.tools`` /
        ``tools.enable``. group aliases are expanded; everything else
        is passed through. empty list resolves to empty result
    :ptype enable: list[str]
    :param disable: optional disable patterns. each may be a literal
        tool name or an fnmatch glob; matched names are subtracted
        from the expanded set. globs not matching any expanded name
        are silently dropped (no error -- a glob may target tools the
        agent did not enable in the first place)
    :ptype disable: list[str] | None
    :return: flat ordered list of patterns + concrete names ready for
        the access matcher
    :rtype: list[str]
    """
    import fnmatch

    expanded: list[str] = []
    seen: set[str] = set()

    def _emit(name: str) -> None:
        if name not in seen:
            seen.add(name)
            expanded.append(name)

    for selector in enable:
        if selector in GROUP_ALIASES:
            for tool_name in sorted(GROUP_ALIASES[selector]):
                _emit(tool_name)
        else:
            _emit(selector)

    disable_patterns = list(disable) if disable else []
    if not disable_patterns:
        result = expanded
    else:
        result = []
        for name in expanded:
            kept = True
            for pattern in disable_patterns:
                if name == pattern or fnmatch.fnmatch(name, pattern):
                    kept = False
                    break
            if kept:
                result.append(name)
    return result


def _doc_round_trip_check() -> dict[str, Any]:
    """sanity-check helper used by the unit tests; not part of the public API.

    returns a dict snapshot of every group alias's expanded tool
    set so a single test can assert the alias map covers the
    canonical builtin sets without re-listing every name. exposed
    via ``_`` prefix because it is purely a test affordance --
    production code reads :data:`GROUP_ALIASES` directly.

    :return: snapshot map of group name -> sorted tool name list
    :rtype: dict[str, Any]
    """
    snapshot: dict[str, Any] = {}
    for group, tools in GROUP_ALIASES.items():
        snapshot[group] = sorted(tools)
    return snapshot
