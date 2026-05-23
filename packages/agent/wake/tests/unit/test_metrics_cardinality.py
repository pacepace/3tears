"""Drift-guard for the agent-wake Prometheus + Loki observability surface.

Two contracts pinned here (PLACEMENT §1.15 + spec OBS-10):

1. **No unbounded-cardinality labels.** ``conversation_id``,
   ``user_id``, ``schedule_id``, ``subscription_id``, ``fire_id``,
   ``agent_id`` MUST NOT appear as Prometheus labels on any wake
   instrument. Time-series-database cardinality explodes on these
   columns; the test reads the declared labelnames off
   :data:`threetears.agent.wake.metrics.WAKE_LABEL_SETS` and asserts
   each set's disjoint with :data:`FORBIDDEN_LABEL_NAMES`.
2. **No ``task_prompt`` in Loki payloads.** The structured-log emit
   sites under ``packages/agent/wake/src/threetears/agent/wake/`` MUST
   NOT include the literal string ``"task_prompt"`` inside any
   ``extra_data={...}`` payload (conversation messages are the source
   of truth for what was said; PII risk otherwise). The walker does an
   AST scan over the source files and fails on any violation.

Both guards run on the static surface -- no DB, no Prometheus
registry, no event loop -- so a `pytest -x` boot catches drift before
any heavier test executes.
"""

from __future__ import annotations

import ast
from pathlib import Path

from threetears.agent.wake.metrics import (
    FORBIDDEN_LABEL_NAMES,
    WAKE_LABEL_SETS,
    WAKE_PROMETHEUS_NAMES,
)


_WAKE_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "threetears" / "agent" / "wake"


def test_forbidden_label_set_covers_all_unbounded_columns() -> None:
    """The forbidden set MUST include every unbounded column the schema uses.

    If a future shard adds another unbounded column (e.g. a new join
    key), it must be added to :data:`FORBIDDEN_LABEL_NAMES` so the
    walker catches accidental labelling.
    """
    expected = {
        "conversation_id",
        "user_id",
        "schedule_id",
        "subscription_id",
        "fire_id",
        "agent_id",
    }
    assert expected.issubset(FORBIDDEN_LABEL_NAMES)


def test_every_registered_instrument_appears_in_label_sets_table() -> None:
    """Every instrument name must declare its labelnames in WAKE_LABEL_SETS.

    Catches the failure mode where someone registers a new Counter /
    Histogram but forgets to thread it through the cardinality
    walker.
    """
    declared_in_table = set(WAKE_LABEL_SETS.keys())
    declared_in_names = set(WAKE_PROMETHEUS_NAMES)
    assert declared_in_names == declared_in_table, (
        f"WAKE_PROMETHEUS_NAMES vs WAKE_LABEL_SETS keys diverged: "
        f"only in names = {declared_in_names - declared_in_table}; "
        f"only in table = {declared_in_table - declared_in_names}"
    )


def test_no_wake_instrument_declares_forbidden_label() -> None:
    """No declared labelname may be a member of :data:`FORBIDDEN_LABEL_NAMES`.

    The single source of truth for declared labels is
    :data:`WAKE_LABEL_SETS`; emit sites read off that table via the
    typed enum helpers. A new instrument added without going through
    the table escapes this check, which is what the companion
    :func:`test_every_registered_instrument_appears_in_label_sets_table`
    catches.
    """
    violations: list[tuple[str, frozenset[str]]] = []
    for name, labels in WAKE_LABEL_SETS.items():
        bad = FORBIDDEN_LABEL_NAMES.intersection(labels)
        if bad:
            violations.append((name, frozenset(bad)))
    assert not violations, f"Wake instruments declared forbidden labels: {violations}"


def _wake_source_files() -> list[Path]:
    """Return every source file under the wake package (recursive)."""
    return sorted(p for p in _WAKE_SRC_ROOT.rglob("*.py") if p.is_file())


def _has_task_prompt_string_in_extra_data(tree: ast.AST) -> list[tuple[int, str]]:
    """Walk ``tree`` and return ``(lineno, snippet)`` pairs for each violation.

    A violation is a dict literal that contains a key node whose
    constant value is the literal string ``"task_prompt"``, where the
    enclosing call passes the dict as ``extra_data=...`` inside an
    ``extra={"extra_data": {...}}`` kwargs payload to a logger method
    (``log.*`` / ``logger.*`` / ``_logger.*``).

    Catches the case where a future Loki emit site PR re-introduces
    ``task_prompt`` content into the structured payload. The walker
    is intentionally syntactic (no value tracing) -- the cost of
    being slightly over-broad is one extra ``_logger.info`` rewrite;
    the cost of missing it is PII in Loki.
    """
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Look at kwargs named ``extra``; its value should be a dict
        # with one key ``"extra_data"`` whose value is the user dict
        # we audit.
        for kw in node.keywords:
            if kw.arg != "extra" or not isinstance(kw.value, ast.Dict):
                continue
            for k, v in zip(kw.value.keys, kw.value.values, strict=False):
                if not (isinstance(k, ast.Constant) and k.value == "extra_data"):
                    continue
                if not isinstance(v, ast.Dict):
                    continue
                for inner_k in v.keys:
                    if isinstance(inner_k, ast.Constant) and inner_k.value == "task_prompt":
                        violations.append((inner_k.lineno, "task_prompt key in extra_data"))
    return violations


def test_no_task_prompt_string_in_wake_log_extra_data() -> None:
    """No source file under wake/ may pass ``task_prompt`` as an extra_data key.

    Catches the PII drift class -- the spec body's anti-pattern (last
    bullet under OBS-12) plus PLACEMENT §1.4's rule that conversation
    messages are the canonical source of truth for assistant +
    schedule text.
    """
    offenders: dict[str, list[tuple[int, str]]] = {}
    for path in _wake_source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        violations = _has_task_prompt_string_in_extra_data(tree)
        if violations:
            offenders[str(path)] = violations
    assert not offenders, f"Loki emit sites may not include task_prompt in extra_data; violations: {offenders}"
