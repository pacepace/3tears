"""YAML loading for files a HUMAN authored, where a silent parse is the hazard.

``yaml.safe_load`` resolves a duplicated mapping key by last-wins, silently. For a
machine-generated document that is harmless. For a file someone edits by hand it is
the worst available behaviour: a botched edit that duplicates a key parses cleanly,
every validation surface downstream reports success, and the value the author wrote
is simply gone. Nothing is available to notice it later, because by the time a model
validates the mapping it has already collapsed.

:func:`safe_load_authored` refuses that document instead, naming the key and both
source positions.

Placement note, because ``serialization.py`` states the opposite rule for FORMATS
("each format lives in its own package and self-registers"): this is not a format
handler, it is a parser-safety guard, and the packages that need it
(``3tears-datasources`` reading dataset definitions, ``3tears-scrape`` reading target
configs) share only ``3tears`` core. Three copies of one security-relevant guard in
a single monorepo rots; one copy plus a declared ``pyyaml`` dependency does not.

The equivalent guard in ``aibots_agents.devx.authored_file`` predates this one and
should be retired onto it rather than maintained in parallel -- it was written for
the same failure and found it in real authored knowledge files.
"""

from __future__ import annotations

from typing import Any

import yaml

__all__ = [
    "DuplicateKeyError",
    "safe_load_authored",
]


class DuplicateKeyError(yaml.constructor.ConstructorError):
    """A mapping in an authored document declared the same key twice.

    A subclass rather than the bare ``ConstructorError`` so a caller can tell this
    refusal from a syntax error and say something useful about it, without matching
    on message text.
    """


class _DuplicateKeyRefusingLoader(yaml.SafeLoader):
    """:class:`yaml.SafeLoader` that refuses a duplicated mapping key."""


def _construct_mapping_refusing_duplicates(
    loader: _DuplicateKeyRefusingLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """build a mapping, raising on any duplicated key.

    :param loader: the active loader instance
    :ptype loader: _DuplicateKeyRefusingLoader
    :param node: the mapping node being constructed
    :ptype node: yaml.nodes.MappingNode
    :param deep: construct nested objects eagerly (PyYAML's own contract)
    :ptype deep: bool
    :return: the constructed mapping
    :rtype: dict[Any, Any]
    :raises DuplicateKeyError: on a duplicate key, naming the key and both
        source positions
    """
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise DuplicateKeyError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r} (an earlier value for this key would be silently discarded)",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DuplicateKeyRefusingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_refusing_duplicates,
)


def safe_load_authored(stream: Any) -> Any:
    """parse an authored YAML document, refusing a duplicated mapping key.

    A drop-in replacement for :func:`yaml.safe_load` at any site reading a file a
    human maintains. Accepts the same argument shapes, including an open handle --
    worth passing rather than a string, because PyYAML's problem mark then names the
    real path beside its line and column instead of ``<unicode string>``.

    :param stream: YAML text, bytes, or an open file handle
    :ptype stream: Any
    :return: the parsed document, or ``None`` for an empty one
    :rtype: Any
    :raises DuplicateKeyError: a mapping declared the same key twice
    :raises yaml.YAMLError: the document is not well-formed YAML
    """
    return yaml.load(stream, Loader=_DuplicateKeyRefusingLoader)
