"""tests for threetears.datasources.namespace.

covers determinism of :func:`datasource_namespace_id` and the canonical
``datasources.<name>`` shape produced by
:func:`datasource_namespace_name`.
"""

from __future__ import annotations

from uuid import uuid4

from threetears.datasources.namespace import (
    DATASOURCE_NAMESPACE_TYPE,
    datasource_namespace_id,
    datasource_namespace_name,
)


class TestNamespaceConstants:
    """canonical type literal exposed for callers that key into it."""

    def test_namespace_type_value(self) -> None:
        assert DATASOURCE_NAMESPACE_TYPE == "datasource"


class TestDatasourceNamespaceId:
    """deterministic uuid5: same input -> same UUID, every time."""

    def test_deterministic(self) -> None:
        datasource_id = uuid4()
        first = datasource_namespace_id(datasource_id)
        second = datasource_namespace_id(datasource_id)
        assert first == second

    def test_different_inputs_different_outputs(self) -> None:
        a = datasource_namespace_id(uuid4())
        b = datasource_namespace_id(uuid4())
        assert a != b


class TestDatasourceNamespaceName:
    """canonical ``datasources.<name>`` shape; sanitization of dots."""

    def test_simple_name(self) -> None:
        assert datasource_namespace_name("central-reporting") == "datasources.central-reporting"

    def test_sanitizes_dots(self) -> None:
        # build_namespace_name replaces dots in the segment so the
        # canonical form keeps ``.`` as the prefix-separator alone.
        # any internal ``.`` becomes ``-``.
        assert datasource_namespace_name("my.ds") == "datasources.my-ds"
