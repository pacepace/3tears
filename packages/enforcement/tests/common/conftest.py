"""fixtures shared by the ``common/`` helper tests.

The call-parsing helper was duplicated byte-identically across two test modules the moment
the spelling helpers moved between them -- a small instance of exactly what those helpers
exist to stop.

It is a FIXTURE rather than an importable function on purpose: this repo has both a
top-level ``tests/`` package and ``packages/enforcement/tests/``, so ``from tests.common
import ...`` resolves differently depending on the rootdir pytest was invoked from. A
fixture has no such ambiguity.
"""

from __future__ import annotations

import ast
from collections.abc import Callable

import pytest


@pytest.fixture
def parse_call() -> Callable[[str], ast.Call]:
    """return a helper that parses one line of source into its single call node.

    :return: callable taking source text and returning the parsed call
    :rtype: Callable[[str], ast.Call]
    """

    def _parse(source: str) -> ast.Call:
        node = ast.parse(source).body[0]
        assert isinstance(node, ast.Expr), f"expected a bare expression, got {type(node).__name__}"
        assert isinstance(node.value, ast.Call), f"expected a call, got {type(node.value).__name__}"
        return node.value

    return _parse
