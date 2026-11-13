"""The figure-key grammar as the analyst's reference section teaches it, whole or a page at a time, with its
examples rendered on a made-up input.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from threetears.evals.analysis.report.keys.families import _FAMILIES, _FAMILY_GROUPS, _grouped
from threetears.evals.analysis.report.keys.figure import PERCENTILE_NAMES, _FILTERS, _FILTER_MEANS, _UNIT_PHRASES
from threetears.evals.analysis.report.keys.resolve import format_figure, parse_key, render, resolve
from threetears.evals.analysis.report.keys.vocabulary import _Family
from threetears.evals.analysis.report_input import ReportInput

if TYPE_CHECKING:
    # A page's forms decide over the figures the keys read, so the module that decides them reads the keys' parts and
    # none of them reads it: every caller hands its page's forms in, and the type is named for the checker alone.
    from threetears.evals.analysis.report.forms import PageForms


# --- the grammar, as the analyst's reference section teaches it ----------------------------------------------------

#: The heading the figure-key section opens with.
GRAMMAR_HEADING = "Figure keys"


def _example_input() -> ReportInput:
    """A small made-up input the grammar's examples are rendered on, with a dotted arm id and a dotted dimension: one
    asked aspect and one other, with a measure on both, a pair reading, a floor, a bar, strata, a cell, criteria, a
    numeric setting, each arm's depth, a criterion set late, a second contrast family naming no aspect, a monthly
    volume, a frozen example beside the trials the pass rule failed, and the cases the two aspects share."""
    rate = {"base": (0.912, 897), "fast.v2": (0.874, 895)}
    return ReportInput.model_validate(
        {
            "schema": "report-input/1",
            "meta": {
                "campaign": "Example",
                "kind": "change",
                "question": "Is fast.v2 as good as base?",
                "volume_per_month": 1_000_000,
            },
            "arms": [
                {
                    "id": "base",
                    "name": "Baseline",
                    "role": "control",
                    "served_model": "vendor/base-7",
                    "settings": {"model": "vendor/base-7", "max_output": "64k", "retries": 2},
                    "reasoning_ratio": 0.62,
                },
                {
                    "id": "fast.v2",
                    "role": "challenger",
                    "served_model": "vendor/fast-2",
                    "settings": {"model": "vendor/fast-2"},
                    "reasoning_ratio": 0.41,
                },
            ],
            "aspects": [
                {
                    "id": "answers",
                    "asked": True,
                    "margin": 10,
                    "readings": [
                        {
                            "arm": "fast.v2",
                            "reading": "within_margin",
                            "change": -4.26,
                            "within_interval": [-7.1, -1.4],
                            "gain_interval": [-7.6, -0.9],
                            "gain_confidence": 0.975,
                            "cases": [300, 300],
                        }
                    ],
                    "depth": {
                        "base": {"cases": 300, "trials": 900, "repeats_per_case": [3, 3.0, 3]},
                        "fast.v2": {"cases": 300, "trials": 897, "repeats_per_case": [2, 2.99, 3]},
                    },
                },
                {
                    "id": "speed",
                    "readings": [
                        {"arm": "fast.v2", "reading": "improved", "change": 7.4, "within_interval": [3.6, 11.2]}
                    ],
                    "pairs": [
                        {
                            "arms": ["base", "fast.v2"],
                            "change": -7.4,
                            "interval": [-11.2, -3.6],
                            "confidence": 0.95,
                            "ahead": "fast.v2",
                        }
                    ],
                    "scoreboard": {
                        "base": {"ran": 300, "usable": 298, "passed": 240},
                        "fast.v2": {"ran": 300, "usable": 299, "passed": 262},
                    },
                },
            ],
            "floors": [
                {
                    "floor": "answers/rate_drop/*",
                    "aspect": "answers",
                    "arm": "fast.v2",
                    "status": "held",
                    "interval": [-7.1, -1.4],
                    "worst_drop": 7.1,
                    "margin": 10,
                }
            ],
            "bars": [
                {
                    "aspect": "speed",
                    "arm": "fast.v2",
                    "measure": "latency_ms",
                    "bar": 900,
                    "direction": "lower_is_better",
                    "status": "met",
                    "value": 640,
                    "interval": [610, 672],
                }
            ],
            "criteria": [
                {
                    "id": "p95",
                    "kind": "gate",
                    "measure": "latency_ms",
                    "statistic": "p95",
                    "direction": "lower_is_better",
                    "threshold": 1500,
                },
                {
                    "id": "cell_floor",
                    "kind": "floor",
                    "measure": "tone.kept",
                    "direction": "higher_is_better",
                    "scope": "each_cell",
                    "threshold": 0.85,
                },
            ],
            "scoreboard": {
                "base": {"passed": 765, "usable": 897, "reply_cost_usd": 0.001892},
                "fast.v2": {
                    "passed": 731,
                    "usable": 895,
                    "reply_cost_usd": 0.002427,
                    "unanswered": {
                        "candidate": {
                            "retried": 9,
                            "lost": 3,
                            "reasons": {"timeout": 2, "tool_choice_refused": 9},
                            "calls": 2685,
                            "share": 12 / 2697,
                        }
                    },
                },
            },
            "measures": {
                "latency_ms": {
                    "unit": "ms",
                    "higher_is_better": False,
                    "per_arm": {"base": {"mean": 820, "p95": 1310}, "fast.v2": {"mean": 655, "p95": 980}},
                    "by_aspect": {"speed": {"base": {"mean": 790}, "fast.v2": {"mean": 640, "ci": [610, 672]}}},
                }
            },
            "judged": {
                "tone.kept": {
                    "scale": "pass_fail",
                    "per_arm": {arm: {"mean": mean, "n": n} for arm, (mean, n) in rate.items()},
                }
            },
            "primary_judged": "tone.kept",
            "contrasts": [
                {
                    "arm": "fast.v2",
                    "measure": "tone.kept",
                    "kind": "judged",
                    "aspect": "answers",
                    "verdict": "regressed",
                    "delta": -0.038,
                    "interval": [-0.061, -0.015],
                },
                {
                    "arm": "fast.v2",
                    "measure": "latency_ms",
                    "kind": "measure",
                    "aspect": "speed",
                    "verdict": "improved",
                    "delta": -165,
                    "interval": [-190, -140],
                },
            ],
            "contrast_families": [
                {"aspect": "answers", "correction": "holm", "alpha": 0.05, "family_size": 1},
                {"correction": "none", "family_size": 1},
            ],
            "strata": [
                {
                    "id": "off_topic",
                    "aspect": "answers",
                    "n_cases": 40,
                    "per_arm": {"base": {"tone.kept": {"mean": 0.70}}, "fast.v2": {"tone.kept": {"mean": 0.55}}},
                },
                {
                    "id": "greeting",
                    "aspect": "answers",
                    "n_cases": 60,
                    "per_arm": {"base": {"tone.kept": {"mean": 0.95}}, "fast.v2": {"tone.kept": {"mean": 0.85}}},
                },
            ],
            "target_matrix": {
                "rows": {"id": "length", "levels": [{"id": "short", "name": "Short"}]},
                "columns": {"id": "lang", "levels": [{"id": "en", "name": "English"}]},
            },
            "cells": {
                "short|en": {
                    "row": "short",
                    "column": "en",
                    "n_cases": 50,
                    "per_arm": {"base": {"tone.kept": {"mean": 0.94}}, "fast.v2": {"tone.kept": {"mean": 0.9}}},
                }
            },
            "caveats": {
                "late_criteria": [{"criterion": "floor", "name": "answers/rate_drop/*", "set_at": "25 Oct 2026"}]
            },
            "examples": [{"trial": "T7", "case": "c-12", "aspect": "answers", "arm": "fast.v2", "outcome": "judged"}],
            "failed_trials": [{"aspect": "answers", "arm": "fast.v2", "failed": 61}],
            "case_overlap": [{"aspects": ["answers", "speed"], "shared": 300, "only_first": 0, "only_second": 0}],
        }
    )


#: One example per filter, each a key on :func:`_example_input`.
_FILTER_EXAMPLES: dict[str, str] = {
    "pct": "pass_rate.base|pct",
    "pts": "contrast.fast.v2.tone.kept.delta|pts",
    "abs": "reading.fast.v2.change|abs|pts",
    "signed": "criteria.p95.fast.v2.gap|signed",
    "usd": "cost_per_1k.base|usd",
    "int": "aspect.margin|int",
    "pct10": "strata.off_topic.loss_share|pct10",
    "interval": "reading.fast.v2.interval|interval",
    "usd_per_1k": "scoreboard.base.reply_cost_usd|usd_per_1k",
    "usd_whole": "cost_per_month.base|usd_whole",
    "words": "count.arms|words",
}


def _family_line(family: _Family, names: Mapping[str, str], forms: PageForms) -> str:
    """One family's line in the grammar: its form, what it reads, its statistics, what of it is derived, and its example
    rendered on the made-up input its forms carry."""
    statistic = family.statistic()
    listed = f" ({statistic.name}: {', '.join(statistic.options)})" if statistic else ""
    tag = (
        (f" {family.derived_note}" if family.derived_note else " Derived: descriptive, no interval.")
        if family.derived
        else ""
    )
    if family.derived_statistics:
        derived = ", ".join(f"`{name}`" for name in family.derived_statistics)
        tag += f" {derived} derived: descriptive, no interval."
    if family.example:
        shown = render(f"{{{family.example}}}", names, forms=forms)
        tag += f' `{{{family.example}}}` reads "{shown}".'
    return f"- `{family.form()}`: {family.what}{listed}.{tag}"  # one line: module constants


def _grammar_rules() -> list[str]:
    """The grammar's rules, after its heading and before its paths: how a key is written and how a path is matched."""
    return [
        "This section is written by the code that reads your keys, so it is current: where anything above describes "
        "figure keys differently, follow this section.",
        "Never type a number into prose. Where a sentence needs one, write a figure key: `{path|filter|filter}`, "
        "in single braces. (`{{…}}` is a different grammar and is refused here.) Write `{arm:<id>}` where a sentence "
        "names an arm: it prints the arm's short name and letter tag.",
        "A path is matched against the ids this input holds, never split at dots: write an id exactly as the input "
        "spells it, dots and spaces included (`reading.fast.v2.change` is arm `fast.v2`). A path that reads two ways, "
        "or names nothing the input holds, is refused and sent back naming what could come next. Where the input "
        "holds a figure on more than one aspect (a contrast, a stratum, a contrast family), the form without "
        "`aspects.<aspect>.` is refused and names the `aspects.<aspect>.` keys to write instead.",
    ]


def _grammar_filters(example: ReportInput, names: Mapping[str, str], forms: PageForms) -> list[str]:
    """The grammar's filters, each with its example rendered in the made-up input's forms, and what a key prints:
    everything after its paths."""
    lines = ["Filters (one spelling at most, plus `abs` or `signed`, and `interval`):"]
    for name in _FILTERS:
        key = parse_key(_FILTER_EXAMPLES[name])
        shown = format_figure(resolve(key.path, example), key.filters, forms=forms)
        means, example_key = _FILTER_MEANS[name], _FILTER_EXAMPLES[name]
        lines.append(f'- `{name}`: {means}. `{{{example_key}}}` reads "{shown}".')  # one line: module constants
    sentence = "{arm:fast.v2} is down {reading.fast.v2.change|abs|pts} points ({reading.fast.v2.interval|pts})."
    lines.append(f'For example, "{sentence}" reads "{render(sentence, names, forms=forms)}".')
    per_reply = "scoreboard.base.reply_cost_usd"
    lines.append(
        "A key prints its own unit (`%`, `$`, `ms`, `s`, or per 1,000 of what the input's cost counts, such as "
        "`per 1,000 conversations`); never write a unit after it. A figure "
        "in points alone prints its bare number, so the word `points` follows it, as above. Money per reply reads per "
        f"1,000 replies: write `{{{per_reply}|usd_per_1k}}`, which reads "
        f'"{format_figure(resolve(per_reply, example), ("usd_per_1k",), forms=forms)}" at any amount; a bare '
        f"`{{{per_reply}}}` "
        "reads so only under a cent, never as a fraction of a cent, and in plain dollars from a cent up."
    )
    lines.append(
        "Without a filter a rate reads as a percent, a change in points, money in dollars. A digit outside a key is "
        "refused unless it is part of a name the input holds: an arm's id or name, a served model, a setting held as "
        "text (such as `64k`; one held as a number is a key), an aspect, measure, dimension, stratum or task id or name, the campaign's name, "
        "a trial or case id. An id made only of digits is never such a name: write `{arm:1017}`, not `1017`. "
        f"Percentile names ({', '.join(f'`{name}`' for name in PERCENTILE_NAMES)}) and the package's unit phrases "
        "(`per 1,000` or `per 1,000,000` of what the input's costs count, such as `per 1,000 conversations`, and, "
        "for the scoreboard's money per reply (`reply_cost_usd`), "
        f"{', '.join(f'`{phrase}`' for phrase in _UNIT_PHRASES)}; each whole) are written as they are; any other "
        "number is a key."
    )
    return lines


#: The names the grammar's examples print each arm by.
_EXAMPLE_NAMES: Mapping[str, str] = {"base": "Baseline", "fast.v2": "fast.v2"}


def figure_key_grammar(*, forms_of: Callable[[ReportInput], PageForms]) -> str:
    """Write the section that teaches figure keys, from the parser's own families and filters: the whole grammar, as
    the analyst's reference pages hold it.

    Each family's form and statistics, each filter and every example come from the code that parses and renders
    them; the examples are rendered on a made-up input, so a change to a family, a filter or a spelling reaches the
    analyst with the code.

    :param forms_of: How a page's forms are built from a report input (:meth:`~threetears.evals.analysis.report.forms.
        PageForms.of`), which the examples are printed in on the made-up input: the keys read no page's forms of
        their own, since the forms read the keys' parts.
    :ptype forms_of: Callable[[ReportInput], PageForms]
    :return: The section, heading first.
    :rtype: str
    :raises UnresolvableReference: An example no longer resolves: the grammar changed under this function, and every
        plan would share the break, so it fails here.
    """
    example = _example_input()
    forms = forms_of(example)
    lines = [f"## {GRAMMAR_HEADING}", *_grammar_rules(), "Paths:"]
    lines += [_family_line(family, _EXAMPLE_NAMES, forms) for family in _FAMILIES]
    lines += _grammar_filters(example, _EXAMPLE_NAMES, forms)
    return "\n".join(lines)


# --- the grammar, as a reference read a page at a time -------------------------------------------------------------


def figure_key_index(topic: str, *, forms_of: Callable[[ReportInput], PageForms]) -> str:
    """The grammar as a reference's index: its rules and filters whole, and one line per group of path families, naming
    the page each group is read on and its forms.

    The grammar (:func:`figure_key_grammar`) runs past what one tool result may take of a reader's context, so a
    reference serves it in pages (:func:`figure_key_pages`); the index and the pages together hold every line of it.

    :param topic: What the reference is served as: each group's page is ``<topic>.<group>``.
    :ptype topic: str
    :param forms_of: How a page's forms are built from a report input (:meth:`~threetears.evals.analysis.report.forms.
        PageForms.of`), which the examples are printed in on the made-up input: the keys read no page's forms of
        their own, since the forms read the keys' parts.
    :ptype forms_of: Callable[[ReportInput], PageForms]
    :return: The index, heading first.
    :rtype: str
    :raises UnresolvableReference: An example no longer resolves, as :func:`figure_key_grammar` raises it.
    :raises ValueError: A family is in no group.
    """
    lines = [f"## {GRAMMAR_HEADING}", *_grammar_rules()]
    lines.append(
        f"Paths, by group: each group's forms, their statistics and an example are its page, topic='{topic}.<group>'."
    )
    summaries = {name: what for name, what, _ in _FAMILY_GROUPS}
    for group, families in _grouped().items():
        forms = ", ".join(f"`{family.form()}`" for family in families)
        lines.append(f"- {topic}.{group}: {summaries[group]}. {forms}.")
    example = _example_input()
    lines += _grammar_filters(example, _EXAMPLE_NAMES, forms_of(example))
    return "\n".join(lines)


def figure_key_pages(*, forms_of: Callable[[ReportInput], PageForms]) -> dict[str, str]:
    """Each group of the grammar's path families as a page: its forms, what each reads, its statistics and an example,
    written as the grammar writes them (:func:`figure_key_grammar`).

    :param forms_of: How a page's forms are built from a report input (:meth:`~threetears.evals.analysis.report.forms.
        PageForms.of`), which the examples are printed in on the made-up input: the keys read no page's forms of
        their own, since the forms read the keys' parts.
    :ptype forms_of: Callable[[ReportInput], PageForms]
    :return: Each page, by its group's name, in the order the index lists them (:func:`figure_key_index`).
    :rtype: dict[str, str]
    :raises UnresolvableReference: An example no longer resolves, as :func:`figure_key_grammar` raises it.
    :raises ValueError: A family is in no group.
    """
    example = _example_input()
    forms = forms_of(example)
    pages: dict[str, str] = {}
    for (group, families), (_, what, _) in zip(_grouped().items(), _FAMILY_GROUPS, strict=True):
        lines = [f"## {GRAMMAR_HEADING}: {group}", f"Paths of {what}:"]
        lines += [_family_line(family, _EXAMPLE_NAMES, forms) for family in families]
        pages[group] = "\n".join(lines)
    return pages


__all__ = [
    "figure_key_grammar",
    "figure_key_index",
    "figure_key_pages",
]
