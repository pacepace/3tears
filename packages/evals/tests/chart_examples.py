"""One payload per chart type: the shared examples the intent, report and renderer suites all draw.

Data only. Each type's example is shaped after a real campaign rather than minimally, so the cross-type
rules — the intent policy, the report's chart blocks, every renderer's conformance — are exercised on
the combinations a minimal fixture would miss.
"""

from __future__ import annotations

from typing import Any

PAYLOAD = {
    "parts": [
        {"label": "confidence_met", "value": 19.0, "n": 9},
        {"label": "budget_exhausted", "value": 42.0, "n": 21},
        {"label": "tool_error", "value": 12.0, "n": 6},
        {"label": "no_new_sources", "value": 27.0, "n": 13},
    ],
    "unit": "%",
    "measure": "share of stops",
    "total": 100.0,
    "total_n": 49,
}

DISTRIBUTION = {
    "groups": [
        {
            "label": "model-b",
            "samples": [1200.0, 1450.0, 1310.0, 1600.0],
            "ci": {"low": 1250.0, "high": 1520.0, "mean": 1390.0, "variability": "across 5 runs", "level": 0.95},
            "n": 5,
        },
        {
            "label": "deepseek",
            "buckets": [{"range": "1000-1500", "count": 2}, {"range": "1500-2000", "count": 7}],
            "ci": {"low": 1400.0, "high": 1900.0, "mean": 1650.0, "variability": "across 5 runs", "level": 0.95},
            "n": 9,
        },
    ],
    "unit": "ms",
    "x_label": "pipeline_synthesis_ms",
}

NULL_RESULT = {
    "groups": [
        {
            "label": "timeout=4s",
            "ci": {"low": 0.71, "high": 0.85, "mean": 0.78, "variability": "across the 12 cases", "level": 0.95},
            "n": 12,
        },
        {
            "label": "timeout=8s",
            "ci": {"low": 0.74, "high": 0.88, "mean": 0.81, "variability": "across the 12 cases", "level": 0.95},
            "n": 12,
        },
    ],
    "metric": "mean_composite",
    "mechanism": "The batch never fills before the deadline at either setting.",
}

DELTA_TABLE = {
    "rows": [
        {
            "metric": "cost_usd",
            "a": 0.011,
            "b": 0.019,
            "unit": "usd",
            "delta": 0.008,
            "d_z": 1.2,
            "p": 0.004,
            "n": 24,
            # Stated, because the row carries a single `n` — 24 PAIRS. An
            # unpaired test over the same two arms has two sample sizes and no
            # pair count, so a row that means this one has to say so.
            "paired": True,
            "significant": True,
        },
        {"metric": "total_ms", "a": 16162.0, "b": 11040.0, "unit": "ms", "delta": -5122.0},
    ],
    "a_label": "model-b",
    "b_label": "deepseek",
}


#: The point plot — two contestants placed against each other, one of them beaten.
#:
#: The only entry below whose figure is not row-based, which is why several
#: geometry assertions state their subject as "row-based" rather than "every".
FRONTIER: dict = {
    "points": [
        {
            "label": "model-a-3.5-fast-lite",
            "cost": 0.0071,
            "quality": 0.2,
            "latency_ms": 31000.0,
            "dominance": "not_separated",
        },
        {
            "label": "model-b",
            "cost": 0.0174,
            "quality": 0.0,
            "latency_ms": 48700.0,
            "dominated": True,
            "dominance": "dominated",
        },
    ],
    "bar": 0.5,
    "cost_label": "Cost per run (USD)",
    "quality_label": "pass^k",
}


#: A four-configuration sweep over one categorical lever and two ordered ones.
#:
#: Shaped after the reference campaign rather than minimally: one lever whose levels
#: are names, two whose levels are numbers, and a level nobody set — which is the
#: combination that exercises both ink vocabularies and the absence sentinel in one
#: figure, and the one a minimal fixture would have missed.
SWEEP_RANKING = {
    "ranked": {"measure": "pass^k", "unit": None},
    "secondary": {"measure": "cost per run", "unit": "usd"},
    "rows": [
        {
            "config": {"model": "gpt-5", "fetch_concurrency": "8", "search_depth": "2"},
            "ranked_value": 0.72,
            "secondary_value": 0.0111,
            "n": 5,
        },
        {
            "config": {"model": "model-b", "fetch_concurrency": "4", "search_depth": "2"},
            "ranked_value": 0.61,
            "secondary_value": 0.0094,
            "n": 5,
        },
        {
            "config": {"model": "gpt-5", "fetch_concurrency": "2", "search_depth": "1"},
            "ranked_value": 0.55,
            "secondary_value": 0.0142,
            "n": 5,
        },
        {
            "config": {"model": "model-b", "fetch_concurrency": "1", "search_depth": "—"},
            "ranked_value": 0.5,
            "secondary_value": 0.0081,
            "n": 5,
        },
    ],
}


def _timeseries_ci(mean: float, half: float) -> dict:
    return {
        "low": mean - half,
        "high": mean + half,
        "mean": mean,
        "level": 0.95,
        "variability": "the cell's observations",
    }


#: One reading across four builds, two lines — the second missing a build, so the gap is in the
#: payload every cross-type rule walks.
TIMESERIES = {
    "metric": "total_ms",
    "unit": "ms",
    "basis": "release",
    "release_label": "app_version",
    "positions": ["0.9", "0.10", "0.11", "0.12"],
    "series": [
        {
            "label": "anthropic/model-a",
            "points": [
                {"position": "0.9", "ci": _timeseries_ci(1200.0, 90.0), "n": 12},
                {"position": "0.10", "ci": _timeseries_ci(1150.0, 80.0), "n": 12},
                {"position": "0.11", "ci": _timeseries_ci(980.0, 70.0), "n": 12},
                {"position": "0.12", "ci": _timeseries_ci(940.0, 60.0), "n": 12},
            ],
        },
        {
            "label": "anthropic/model-b",
            "points": [
                {"position": "0.9", "ci": _timeseries_ci(1500.0, 120.0), "n": 12},
                {"position": "0.11", "ci": _timeseries_ci(1320.0, 100.0), "n": 12},
                {"position": "0.12", "ci": _timeseries_ci(1290.0, 90.0), "n": 12},
            ],
        },
    ],
    "gaps": [{"series": "anthropic/model-b", "position": "0.10", "reason": "the cell was not measured there"}],
}


#: One payload per drawable type, for the checks that must hold across all of them.
EVERY_TYPE: dict[str, dict] = {
    "sweep_ranking": SWEEP_RANKING,
    "breakdown": PAYLOAD,
    "distribution": DISTRIBUTION,
    "null_result": NULL_RESULT,
    "delta_table": DELTA_TABLE,
    "frontier": FRONTIER,
    "attribution": {
        "end_to_end": {"measure": "total_ms", "delta": -31800.0, "a": 58300.0, "b": 26500.0, "n": 12},
        "subsystem": {"measure": "tool_ms", "delta": -3800.0, "n": 12},
        "unit": "ms",
        "contained_by": "total_ms",
        "unattributed_delta": -28000.0,
        "lever": "pipeline.pipeline_model",
        "a_label": "model-b",
        "b_label": "deepseek",
    },
    "timeseries": TIMESERIES,
}


def timeseries_ci(mean: float, half: float = 50.0) -> dict[str, Any]:
    """A 95% interval of ``half`` either side of ``mean``, over a cell's observations."""
    return {
        "low": mean - half,
        "high": mean + half,
        "mean": mean,
        "level": 0.95,
        "variability": "the cell's observations",
    }


def timeseries_payload(**update: Any) -> dict[str, Any]:
    """Two series over three UTC days, the second missing the middle one as a stated gap; ``update`` applied over it."""
    payload: dict[str, Any] = {
        "metric": "total_ms",
        "unit": "ms",
        "basis": "date",
        "positions": ["2026-03-14", "2026-03-15", "2026-03-16"],
        "series": [
            {
                "label": "narrow",
                "points": [
                    {"position": "2026-03-14", "ci": timeseries_ci(900.0), "n": 6},
                    {"position": "2026-03-15", "ci": timeseries_ci(990.0), "n": 6},
                    {"position": "2026-03-16", "ci": timeseries_ci(1080.0), "n": 6},
                ],
            },
            {
                "label": "wide",
                "points": [
                    {"position": "2026-03-14", "ci": timeseries_ci(1400.0), "n": 6},
                    {"position": "2026-03-16", "ci": timeseries_ci(1500.0), "n": 6},
                ],
            },
        ],
        "gaps": [{"series": "wide", "position": "2026-03-15", "reason": "the cell was not measured there"}],
    }
    payload.update(update)
    return payload
