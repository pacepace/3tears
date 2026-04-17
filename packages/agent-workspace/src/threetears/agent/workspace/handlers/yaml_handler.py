"""YamlHandler: round-trip YAML format handler backed by ruamel.yaml.

implements the :class:`threetears.core.serialization.FormatHandler`
protocol for ``.yaml``/``.yml`` files, preserving comments, key order,
anchors, aliases, and quote styles across load/dump cycles so agent
edits to structured configuration files do not lose authorial intent.

path operations interpret expressions as jsonpath via
:mod:`jsonpath_ng.ext` so filter expressions and array slices work.

module import triggers registration with the core format-handler
registry via :func:`threetears.core.serialization.register_handler`.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

import jsonpath_ng.ext
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from threetears.core.serialization import register_handler


class YamlHandler:
    """round-trip YAML handler using ruamel.yaml.

    preserves comments, key order, anchors, aliases, and quote styles
    across load/dump cycles. constructs a single ``YAML(typ="rt")``
    instance per handler and reuses it for every operation.

    :cvar extensions: file extensions handler owns, in leading-dot form
    :ivar _yaml: configured ruamel.yaml round-trip instance
    """

    extensions: tuple[str, ...] = (".yaml", ".yml")

    def __init__(self) -> None:
        """initialize ruamel.yaml round-trip instance once per handler.

        preserve_quotes is enabled so source quote style survives dump.
        indent is set to 2-space mapping, 4-space sequence with 2-space
        offset to match the convention used in the bluelabs audience
        YAML fixtures.

        :return: None
        :rtype: None
        """
        self._yaml = YAML(typ="rt")
        self._yaml.preserve_quotes = True
        self._yaml.indent(mapping=2, sequence=4, offset=2)

    def load(self, text: str) -> Any:
        """parse yaml document body into round-trip tree.

        :param text: serialized yaml document body
        :ptype text: str
        :return: in-memory tree (CommentedMap, CommentedSeq, or scalar)
        :rtype: Any
        :raises ValueError: if text cannot be parsed as yaml
        """
        return self._yaml.load(StringIO(text))

    def dump(self, tree: Any) -> str:
        """serialize round-trip tree back to yaml text.

        :param tree: in-memory yaml tree to serialize
        :ptype tree: Any
        :return: serialized yaml document body
        :rtype: str
        :raises TypeError: if tree contains types handler cannot serialize
        """
        buf = StringIO()
        self._yaml.dump(tree, buf)
        return buf.getvalue()

    def get(self, tree: Any, path: str) -> Any:
        """resolve jsonpath expression against tree.

        single match returns value; multiple matches return list of
        values; no match returns None. invalid jsonpath propagates
        parser exception.

        :param tree: in-memory yaml tree
        :ptype tree: Any
        :param path: jsonpath expression (supports ext grammar: filters, slices)
        :ptype path: str
        :return: single value, list of values, or None
        :rtype: Any
        :raises jsonpath_ng.exceptions.JsonPathParserError: if path is invalid
        """
        expr = jsonpath_ng.ext.parse(path)
        matches = expr.find(tree)
        result: Any
        if not matches:
            result = None
        elif len(matches) == 1:
            result = matches[0].value
        else:
            result = [m.value for m in matches]
        return result

    def set(self, tree: Any, path: str, value: Any) -> Any:
        """assign value at jsonpath within tree, creating missing segments.

        uses ``update_or_create`` so intermediate mappings are constructed
        when absent. mutates tree in place and returns same tree.

        :param tree: in-memory yaml tree
        :ptype tree: Any
        :param path: jsonpath expression identifying target location
        :ptype path: str
        :param value: value to assign at target location
        :ptype value: Any
        :return: tree (same instance, mutated in place)
        :rtype: Any
        :raises jsonpath_ng.exceptions.JsonPathParserError: if path is invalid
        """
        expr = jsonpath_ng.ext.parse(path)
        expr.update_or_create(tree, value)
        return tree

    def merge(self, tree: Any, partial: dict[str, Any]) -> Any:
        """deep-merge partial mapping into tree.

        mapping-in-mapping recursively merges; scalars and lists replace
        wholesale. callers needing surgical list edits should use
        :meth:`set` with an indexed jsonpath instead.

        :param tree: in-memory yaml tree (mapping at root)
        :ptype tree: Any
        :param partial: partial mapping to merge into tree
        :ptype partial: dict[str, Any]
        :return: tree (same instance, mutated in place)
        :rtype: Any
        """
        self._deep_merge(tree, partial)
        return tree

    def _deep_merge(self, into: Any, src: Any) -> Any:
        """recursive helper for :meth:`merge`; mutates ``into`` in place.

        when both sides are mappings, recurse key-by-key; otherwise the
        caller replaces the slot with ``src``. returns the (possibly
        unchanged) ``into`` when mappings merge, or ``src`` when caller
        should substitute.

        :param into: destination slot (mapping, list, or scalar)
        :ptype into: Any
        :param src: source value to merge in
        :ptype src: Any
        :return: merged value for caller to store in parent slot
        :rtype: Any
        """
        result: Any
        if isinstance(into, (CommentedMap, dict)) and isinstance(src, dict):
            for key, value in src.items():
                if (
                    key in into
                    and isinstance(into[key], (CommentedMap, dict))
                    and isinstance(value, dict)
                ):
                    self._deep_merge(into[key], value)
                else:
                    into[key] = value
            result = into
        else:
            result = src
        return result


register_handler(YamlHandler())
