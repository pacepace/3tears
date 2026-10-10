"""Every measure the engine's catalogue declares has a producer, or is marked as a host's to produce (#573).

A descriptor in :data:`~threetears.evals.kernel.metrics.METRIC_DESCRIPTORS` is a promise to every host
that inherits it: a renderer shows it, a reader expects it. Six async-delivery measures were declared for
years with nothing in the package computing them, so they rendered, were never filled, and never failed.

The guard is the search that found them, made a test: a declared name is PRODUCED when engine code
outside the catalogue names it as code — a string literal that is not a docstring (a row key, a measure
name a summary is built under) or a field a model declares. A name that appears only in prose, or only in
the catalogue itself, is produced by nothing. The one exemption is
:data:`~threetears.evals.kernel.metrics.HOST_PRODUCED_MEASURES`: core measures a host's candidate kind
writes and the engine only reads.
"""

from __future__ import annotations

import ast
from pathlib import Path

import threetears.evals as evals_package
from threetears.evals.kernel import metrics
from threetears.evals.kernel.metrics import HOST_PRODUCED_MEASURES, METRIC_DESCRIPTORS

_SOURCE_ROOT = Path(evals_package.__file__).parent
_CATALOGUE = Path(metrics.__file__)


def _named_in_code(tree: ast.Module) -> set[str]:
    """The string literals (docstrings and bare string statements excluded) and model field names in ``tree``."""
    prose = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    }
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose:
            names.add(node.value)
        elif isinstance(node, ast.ClassDef):
            names.update(
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
            )
    return names


def _produced_names() -> set[str]:
    produced: set[str] = set()
    for path in _SOURCE_ROOT.rglob("*.py"):
        if path == _CATALOGUE:
            continue
        produced |= _named_in_code(ast.parse(path.read_text(encoding="utf-8")))
    return produced


def test_every_declared_core_measure_has_an_engine_producer_or_is_host_produced():
    produced = _produced_names()

    unproduced = sorted(
        name for name in METRIC_DESCRIPTORS if name not in produced and name not in HOST_PRODUCED_MEASURES
    )

    assert not unproduced, (
        "these core measures are declared and nothing in the engine produces them, so they render and are never "
        f"filled: {unproduced}. Produce each, remove its declaration, or (if a host's kind writes it) add it to "
        "HOST_PRODUCED_MEASURES"
    )


def test_every_host_produced_mark_names_a_declared_measure():
    """A stale exemption would hide nothing today and silently cover a future name of the same spelling."""
    assert HOST_PRODUCED_MEASURES <= set(METRIC_DESCRIPTORS)


def test_the_scan_does_not_count_prose_as_a_producer():
    """The guard's own premise: a name only in a docstring or comment is produced by nothing."""
    tree = ast.parse('"""async_delivery_p95_elapsed_ms"""\n# async_deliveries_substituted\nx = {"n_results": 1}\n')

    assert _named_in_code(tree) == {"n_results"}
