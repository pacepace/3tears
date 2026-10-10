"""Export: the score projection's flat rows, as CSV or JSON.

:func:`export_projection` serializes a projection in an :data:`EXPORT_FORMATS` format; :func:`export_records_csv`
writes the CSV with every formula-triggering cell neutralised (:func:`_csv_safe`).
"""

from __future__ import annotations

import csv
import io
import json
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import Field

from threetears.evals.schema.base import EvalBaseModel, VerbatimText
from threetears.evals.analysis.reporting import ProjectionExclusions, ScoreProjection, ScoreRecord

if TYPE_CHECKING:
    pass


#: The two on-demand serializations. Parquet is deferred: pyarrow is not a current
#: dependency and the dependency-manifest rule governs — CSV covers DuckDB/pandas
#: ingestion, which is the stated need. A format outside this set is refused rather
#: than defaulted, so a typo'd `format=jsom` is a visible error, not a silent CSV.
ExportFormat = Literal["csv", "json"]

#: :data:`ExportFormat`'s values, in the order a refusal lists them.
EXPORT_FORMATS: tuple[ExportFormat, ...] = get_args(ExportFormat)

# A score record's open maps — neither is a column of its own. `factors` carries the
# host's levers; `host_measures` carries a code-graded kind's own grade. Each flattens to
# one column per key it holds, so a bake-off's swept knob and a classifier's accuracy are
# both pivotable pandas columns rather than nested blobs.
_EXPORT_OPEN_MAPS = ("factors", "host_measures")

# Prefix on every flattened host-measure column. Present so a grade cannot be mistaken for
# a lever in a spreadsheet, and so a host measure whose name collides with a declared
# coordinate cannot emit a duplicate header. A COLON rather than a dot, deliberately: a
# host's levers are often dotted (`<kind>.<field>`), so a dotted grade column would read
# as a lever in a spreadsheet beside the lever columns `factors` flattens to.
_HOST_MEASURE_COLUMN_PREFIX = "host_measure:"

# The declared ScoreRecord columns for a CSV export, derived from the model so a
# coordinate added later becomes a column with no edit here — the export layer's
# version of the open-factor rule. Derived by subtracting the open maps rather
# than by listing the scalars, for the same reason.
_EXPORT_SCALAR_COLUMNS = tuple(name for name in ScoreRecord.model_fields if name not in _EXPORT_OPEN_MAPS)

# CSV-formula-injection guard. A spreadsheet treats a cell whose text
# begins with one of these as a formula, so a user-authored value — a subject
# name, a prompt id, a run-scoped config override flattened into a factor column —
# like ``=cmd|'/C calc'!A1`` would execute on open. The characters, per OWASP.
_CSV_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def _is_numeric(text: str) -> bool:
    """True when ``text`` is a plain number Python parses — safe to leave verbatim."""
    try:
        float(text)
        return True
    except ValueError:
        return False


def _csv_safe(value: str) -> str:
    """Force a spreadsheet-formula-triggering cell to text by prefixing a quote.

    A legitimate number (a negative delta, ``+1.5e3``) is left untouched: the
    export's contract is that numeric columns ingest to pandas as numbers, and
    ``'-0.5`` would corrupt that. A real formula payload never parses as a float,
    so number-gating neutralizes the attack surface without touching the data.
    """
    if value and value[0] in _CSV_FORMULA_TRIGGERS and not _is_numeric(value):
        return "'" + value
    return value


class ExportError(ValueError):
    """An export was requested in a format this surface cannot emit.

    Raised rather than defaulted to CSV: a caller who asked for ``parquet`` or
    mistyped ``jsom`` wants that request answered, not silently reinterpreted as
    a different format whose bytes they will then try to parse as the one they
    asked for.
    """


def export_records_csv(records: list[ScoreRecord] | list[dict[str, Any]]) -> str:
    """Serialize projection rows to CSV, one row per record, factors flattened to columns.

    The declared coordinates are fixed columns; every distinct open-factor key
    seen across the records becomes its own column (sorted), blank where a record
    did not carry it. That makes a swept lever a first-class pandas column — a
    kind's ``gm.difficulty`` sits beside ``model`` — rather than a nested blob a
    reader has to unpack, and it needs no per-key code, exactly as pivoting on
    such a key does.

    **A code-graded run's grade flattens the same way**, under
    :data:`_HOST_MEASURE_COLUMN_PREFIX` — ``host_measure:field_accuracy`` — so a kind
    that grades on two axes is two columns rather than a second export shape. The columns
    exist only when some record carries one, so an export of judge-graded runs alone is
    byte-identical to what it was before the grade had anywhere to go.

    Args:
        records: The projection rows, as :class:`ScoreRecord` instances or their
            JSON-safe dumps. Both are accepted because the service dumps the
            projection to JSON before the surfaces render it, while a direct
            caller has the models in hand.

    Returns:
        CSV text with a header row. A ``None`` value renders as an empty cell —
        never ``0`` or ``"None"`` — so an unmeasured observation is blank, not a
        fabricated zero, and ingests into pandas as ``NaN``. A cell whose text
        would trigger a spreadsheet formula (leading ``=``/``+``/``-``/``@``) is
        forced to text via :func:`_csv_safe`, except a legitimate number, which is
        left verbatim.
    """
    rows = [record.model_dump(mode="json") if isinstance(record, ScoreRecord) else record for record in records]
    factor_keys = sorted({key for row in rows for key in (row.get("factors") or {})})
    measure_keys = sorted({key for row in rows for key in (row.get("host_measures") or {})})
    header = list(_EXPORT_SCALAR_COLUMNS) + factor_keys + [_HOST_MEASURE_COLUMN_PREFIX + key for key in measure_keys]

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    for row in rows:
        factors = row.get("factors") or {}
        measures = row.get("host_measures") or {}
        cells = [
            "" if row.get(column) is None else _csv_safe(str(row.get(column))) for column in _EXPORT_SCALAR_COLUMNS
        ]
        cells += ["" if factors.get(key) is None else _csv_safe(str(factors[key])) for key in factor_keys]
        # Blank, never 0, for a measure this row did not take — the same rule `value` follows
        # one column over. A classifier that named no class reports no accuracy at all, and a
        # zero there would enter the mean as a wrong answer rather than as an absent one.
        cells += ["" if measures.get(key) is None else _csv_safe(str(measures[key])) for key in measure_keys]
        writer.writerow(cells)
    return buffer.getvalue()


def serialize_export(projection: ScoreProjection, *, fmt: str) -> str:
    """Serialize a projection to the requested export format, as a text body.

    The one serialization seam both surfaces share, so REST and MCP emit
    byte-identical exports. JSON carries the whole :class:`ScoreProjection` —
    ``records`` **and** ``exclusions`` — because an export that dropped the
    exclusion counts would let an all-excluded corpus read as an empty one, the
    same misreading the counts exist to prevent everywhere else. CSV is the flat
    rows alone; a reader who needs the exclusion accounting takes JSON.

    Args:
        projection: The rows to export, plus what the projection dropped.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The serialized body.

    Raises:
        ExportError: ``fmt`` is not one this surface can emit.
    """
    match export_format(fmt):
        case "csv":
            return export_records_csv(projection.records)
        case "json":
            return json.dumps(projection.model_dump(mode="json"))


def export_format(fmt: str) -> ExportFormat:
    """The export format a caller named, or the refusal naming the ones there are.

    Args:
        fmt: The format as the caller spelled it.

    Returns:
        It, as an :data:`ExportFormat`.

    Raises:
        ExportError: ``fmt`` is not one this surface can emit.
    """
    for known in EXPORT_FORMATS:
        if fmt == known:
            return known
    raise ExportError(f"unknown export format {fmt!r} — one of {', '.join(EXPORT_FORMATS)}")


class ScoreExport(EvalBaseModel):
    """A projection's rows serialized for analysis elsewhere, with the account a CSV body cannot carry.

    ``body`` is :func:`serialize_export`'s output byte for byte. The JSON form already holds the
    exclusions and the completeness disclosures; the CSV form is the flat rows alone, so the counts
    ride beside the body here — an export whose every result was excluded and one over an empty scope
    are both a header row, and only these fields tell them apart.
    """

    format: ExportFormat
    body: VerbatimText = Field(description="The export, exactly as serialized: CSV text or a JSON ScoreProjection.")
    n_records: int = Field(description="How many rows the export holds — one per projected observation.")
    exclusions: ProjectionExclusions
    #: ``run_id -> DEGRADED sentence`` for the exported runs that came up short of their matrix.
    completeness_disclosures: dict[str, str] = {}


def export_projection(projection: ScoreProjection, *, fmt: str) -> ScoreExport:
    """Serialize a projection in the named format, with the counts that qualify its rows.

    Args:
        projection: The rows to export, plus what the projection dropped.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The export.

    Raises:
        ExportError: ``fmt`` is not one this surface can emit.
    """
    known = export_format(fmt)
    return ScoreExport(
        format=known,
        body=serialize_export(projection, fmt=known),
        n_records=len(projection.records),
        exclusions=projection.exclusions,
        completeness_disclosures=projection.completeness_disclosures,
    )


__all__ = [
    "export_format",
    "EXPORT_FORMATS",
    "export_projection",
    "export_records_csv",
    "ExportError",
    "ExportFormat",
    "ScoreExport",
    "serialize_export",
]
