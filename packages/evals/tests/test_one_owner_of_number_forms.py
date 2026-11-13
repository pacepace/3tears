"""One owner of number forms: every place a figure's form is decided is an owner, or is named, with its reason, as
right for good.

A number's form is its digits, sign, unit and interval as printed. Spelling one number is centralised already
(``kernel/numbers.py``); deciding which figures form a set, and how many digits the set reads at, is not: it happens
at hundreds of sites, at three different times, by several owners, so one measure prints two ways on one page ("85%"
in a table, "85.3%" in the chart head). Every decision is ``analysis/report/forms.py``'s, and this test holds it there:

* **The walk** reads every module of ``threetears/evals`` and the kit's ``kit.js``, and finds each site that decides
  a form: a digit-choosing format spec, ``round(``, a call to a spelling or set primitive outside the two owners
  (``kernel/numbers.py`` and ``analysis/report/forms.py``), and in ``kit.js`` the digit primitives and the named
  formats used outside the kit's owners of them (:data:`_KIT_OWNERS`), but a pick by a field of the chart's data,
  whose format Python decided. A package function that returns what a site produced is a primitive
  too, to the fixed point, so a call rerouted through a wrapper is still a site where it is called. That holds where
  the figure reaches the return through a helper that hands back what it is given (``with_unit`` hands
  ``number_in``'s figure to ``_joined``), by an argument, a nested call or a local: the helper is derived, not listed,
  and alone, handed a value nobody decided, it is no site. A primitive stored in a table or bound to a name is a site
  where it is stored, handed on as a parameter's default or returned by name, and a wrapper property is one where
  ``self`` reads it; a class named in an ``isinstance`` or ``issubclass`` test is read, not handed on, and is none.
  So is a public free function of ``forms.py`` (a decision relocated there): only ``PageForms``, asked by measure and
  role, is the migrated route.
* **The engine** (:data:`_ENGINE_TREES`): the trees whose imports, followed to their end, never reach
  ``threetears.evals.analysis`` reach no page's forms, and what they spell (a launch's refusal, a timeout's
  message) is no page figure. There a call to a primitive of ``kernel/numbers.py`` is the route, not a site, and a
  function returning one is no wrapper to a caller in the engine, though it is one to a caller above the line; a
  format spec, ``round()``, ``quantize()`` and digits from a ``log10`` stay sites there as anywhere. Above that line
  ``PageForms`` is the only route, through ``PageForms.off_page()`` for a surface no page prints.
* **The rule** fails on any site the walk finds that :data:`_PERMANENT` does not name at its function and form, on
  one more site there than an entry counts, and on an entry counting more sites than the walk finds.
* **What the wrapper rule cannot see** is a stated limit, not a hidden one: a figure reached through another object's
  method of a common name, a property read through anything but ``self``, or a method handed on by reference from
  anything but ``self``; lines appended then joined, a ``yield``, a tuple unpacked; a formatter a factory returns (a
  nested function, a lambda or a ``functools.partial``, as ``chart_svg._kit_format`` does), and a lambda table, each
  counted once where its site is written rather than at every call of it; or digits derived as ``floor(v + 0.5)``. A
  function's nested returns count as its own, which over-counts, the safe way.
* **The permanent list** (:data:`_PERMANENT`) names the sites whose fixed form is right for good, keyed by module,
  enclosing qualname and form, each with its count and its reason: text stored where a fingerprint holds it,
  arithmetic, CSS, SVG and the kit's own drawing, the helpers ``forms.py`` imports and so cannot route through, and
  messages no page prints. It has a ceiling (:data:`_PERMANENT_CEILING`): a new exemption is an owner's decision,
  made where the ceiling is.

It also holds the report trees to the stricter rule ``test_numbers.py`` held them to before this test existed:
under ``analysis/`` and ``ops/`` no format spec chooses digits except a share's ``%`` and a permanent site.
"""

from __future__ import annotations

import ast
import bisect
import functools
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
_PACKAGE = "threetears/evals"

#: The rule's spelling primitives, and the module that decides a page's forms. Nothing in either is a site: they are
#: where a form is meant to be decided. A tree without ``forms.py`` (a self-test's) walks as before it existed.
_RULE = "threetears/evals/kernel/numbers.py"
_FORMS = "threetears/evals/analysis/report/forms.py"
_OWNERS = (_RULE, _FORMS)

#: The kit's script. Its copy under ``docs/eval-report-kit/kit/`` is pinned byte-identical to this one
#: (``test_report_kit_pages``), so walking one walks both.
_KIT_JS = "threetears/evals/analysis/report/kit/kit.js"

#: The kit's owners of a drawn figure's form: its two named-format tables and the helpers that pick and apply a named
#: format (``kShort`` writes a table's digits, ``namedFmt`` picks a table entry by its name, ``kTickFmt`` and
#: ``kTickLabels`` an axis's). A site inside one is not a site; a call to one outside them still is, by its name.
_KIT_OWNERS = frozenset({"fmt", "K_FMTS", "kShort", "namedFmt", "kTickFmt", "kTickLabels"})

#: The kit's pickers of a named format. Called with a field of the chart's data (``namedFmt(d.fmt)``), a picker
#: applies the format Python decided and wrote into the data, where the walk reads it, so the call is no site; a
#: literal, a default or anything else it is handed is the kit deciding, and is one.
_KIT_PICKERS = ("namedFmt", "kTickFmt", "kTickLabels")

#: The set and precision helpers outside the rule that decide a figure's form, each a primitive by its name. A call to
#: one, or one passed as a function value, is a site. The rule's own public callables join them by derivation
#: (:func:`_rule_primitives`), so a primitive added to ``numbers.py`` is covered without an edit here.
#: ``number_in``, ``with_unit``, ``signed_with_unit`` and ``render_cell`` (a figure in its unit, or a values cell)
#: live where ``forms.py`` imports them, so they cannot route through it: each stays a primitive by name, so its
#: callers are sites even where its own body is named in :data:`_PERMANENT` (a permanent site wraps nothing, so
#: derivation alone would let them go).
#: ``figure_keys.held_means`` is not listed: it returns a measure's floats, and what reads them to classify a measure
#: (its scale, its unit, whether it is a rubric score) sets no digits.
#: ``figure_keys.format_figure`` is not listed: it chooses which spelling a key's filters name and refuses one that
#: does not suit, and every number it prints, an interval's precision and how its ends are joined are a
#: ``PageForms.key_*`` method's, so a call to it decides nothing. A primitive written back into it is a site where it
#: is written, as any is.
_SET_HELPERS = frozenset(
    {
        "bare_places",
        "dollars_text",
        "figure_places",
        "figure_text",
        "interval_level",
        "interval_words",
        "number_in",
        "render_cell",
        "signed_with_unit",
        "with_unit",
    }
)

#: Families named by prefix (``percent_*``, ``respell_*``, ``quantity*``), so a new member is a site too.
_SET_HELPER_PREFIXES = ("percent_", "respell_", "quantity")

#: Private helpers that decide a figure's places, keyed by the one module allowed to call them (a leading underscore
#: binds nothing outside its module). Matched by name alone they would also catch ``levers._places``, which lists
#: where an effect is read, and ``build._at``, which links a cell to a finding.
_PRIVATE_SET_HELPERS = frozenset(
    {
        ("threetears/evals/analysis/report/campaign/scorecard.py", "_one_precision"),
        ("threetears/evals/analysis/report/figure_keys.py", "_one_precision"),
        # The figure-key grammar's own modules, one row each, so the helper is held wherever the grammar's code is.
        *(
            (path.relative_to(_SOURCE_ROOT).as_posix(), "_one_precision")
            for path in sorted((_SOURCE_ROOT / "threetears/evals/analysis/report/keys").glob("*.py"))
        ),
        ("threetears/evals/harness/writer_eval.py", "_places"),
    }
)

#: The trees whose every number is shown to a reader, held to the stricter rule: the analysis package, what the
#: generator writes from, and the ops surfaces it renders through. A host's own renders (an operator console, a tool
#: surface) are the host's to walk on the same terms. A module sits here by itself, rather than its whole directory,
#: where the rest of that directory is not read as a report.
_SPELLING_TREES = ("threetears/evals/analysis", "threetears/evals/ops")

#: What sits inside :data:`_SPELLING_TREES` and is not held to the stricter rule (its sites are still named in
#: :data:`_PERMANENT`): ``report/svg.py`` spells only a glyph's drawing coordinates (to a tenth of a unit), every value
#: it shows a reader going through the rule.
_NOT_READ_AS_REPORTS = ("threetears/evals/analysis/report/svg.py",)

#: The engine: the trees from which no import, followed through the package to its end, reaches
#: ``threetears.evals.analysis`` (:class:`TestTheEngineRule` holds them to it), so no page's forms reach them and
#: what they spell is no page figure. ``actions``, ``testing`` and ``transports`` reach it through ``ops``, and
#: ``providers`` through ``testing``, so they are above the line. In them a call to a primitive of
#: ``kernel/numbers.py``, called, handed on or a parameter's default, is the route rather than a site, and a function
#: returning one is no wrapper to a caller there (it is one to a caller above the line); a format spec, ``round()``,
#: ``quantize()`` and digits from a ``log10`` stay sites. The list only narrows: a tree joins it by an owner's
#: decision, which raises :data:`_ENGINE_TREES_CEILING`.
_ENGINE_TREES = tuple(
    f"{_PACKAGE}/{tree}"
    for tree in (
        "gen",
        "kernel",
        "negotiation",
        "run",
        "schema",
        "storage",
    )
)
_ENGINE_TREES_CEILING = 6

#: The package the engine may not import.
_ANALYSIS = "threetears.evals.analysis"


def _in_engine(module: str) -> bool:
    """Whether ``module`` (relative to the package's ``src``) sits in a tree of :data:`_ENGINE_TREES`."""
    return any(module.startswith(f"{tree}/") for tree in _ENGINE_TREES)


#: Presentation types that choose how many digits a reader sees. ``d`` and ``s`` do not.
_NUMERIC_TYPES = frozenset("eEfFgGn%")

#: A share of a whole spelled as a percentage (``.0%``) is a different sentence from a measured value, "37% of it",
#: and carries its own fixed precision. The stricter rule passes it by presentation type; the walk still counts it,
#: because a page's share prints through ``forms.share``.
_PERCENTAGE_TYPE = "%"

#: The static page's chart drawer, and why its kit formats spell as they do.
_CHART_SVG = "threetears/evals/analysis/report/chart_svg.py"
_KIT_FORMAT = (
    "the kit's named format, restated from kit.js for the static page and pinned to number-format-cases.json: "
)

#: Why each permanent site's fixed form is right for good, one sentence per kind, shared by the entries of that kind.
_COVARIATE_RATIO = (
    "a stored covariate, the reasoning share of a reply's tokens, rounded as arithmetic before it is kept"
)
_PHASE_TIMINGS = "stored phase timings, summed in milliseconds and rounded as arithmetic before they are kept"
_JUDGE_SCORE = "a judge's criterion score normalised to its scale and rounded as arithmetic before it is kept"
_TOKEN_BUDGET = "a token count: a share of the provider's cap rounded to whole tokens, sent to the provider"
_NEGOTIATION_RATIO = (
    "a settings note's ratio; the negotiation package may not import the kernel, so it cannot spell through numbers"
)
_FINGERPRINT = "a float written at fixed significant figures into a fingerprint, which moves if its form does"
_SCHEMA_VALUE = (
    "a declared value echoed as written; the schema loads before the kernel and importing kernel.numbers from it "
    "cycles through schema.models"
)
_TOOL_LIMIT = "a character limit in a tool's description, grouped by the rule's own primitive; no page prints it"
_CATALOGUE_BOUND = "a bound echoed as declared in a model catalogue's load refusal; no page prints it"
_CONFORMANCE_TIMEOUT = "a timeout, as declared, in a job conformance check's message; no page prints it"
_TRANSPORT_TIMEOUT = "a timeout, as declared, in a transport's error to its caller; no page prints it"
_KIT_FLOAT = "float cleanup of an axis tick's position before it is placed, not a printed figure"
_KIT_PIXELS = "a chart's height in pixels, not a printed figure"
_KIT_COLOUR = "the share of a colour mix, not a printed figure"
_KIT_TOOLTIP_COUNT = "a heatmap tooltip's count of cases, a whole number grouped by thousands"
_KIT_DEFAULT = "the format a confusion chart reads in when the host's chart names none"
_WHOLE_CASES = "a count of cases recovered from a share of a whole count, rounded to whole cases"
_UNANSWERED_COUNT = "a count of unanswered cases in a figure key's line, grouped by the rule's own primitive"
_COST_UNIT = (
    "the money ruler, the one owner of a set of amounts' form; forms.py imports it, so it cannot route through forms, "
    "and its callers are the page's forms"
)
_QUANTITIES = (
    "a figure in its unit, spelled by the number rule; forms.py imports this module, so it cannot route through "
    "forms, and each helper stays a primitive by its name, so its callers are sites"
)
_LOOK = (
    "a drawn figure's form by its unit and the money ruler; forms.py imports this module, so it cannot route through "
    "forms, and the helper stays a primitive by its name, so its callers are sites"
)
_RESPELL = (
    "the chart texts' spelling of the kit's named formats, which forms.chart_texts delegates to so a chart's drawn "
    "figures read as the kit draws them"
)
_TABLE_CELL = "a values cell that holds words, a yes or no, or nothing; a number cell carries its decided text"
_HOST_CHART = "a chart a host built with no decided texts, spelled by the number rule so it still reads"
_STORED_READING = "a reason stored in a campaign's readings, which are fingerprinted, so its text may not move"
_WHOLE_COUNT = "a test that a count is whole, not a printed figure"
_CSS_WIDTH = "a table column's width in rem, written to the page's CSS, not a figure"
_CLASSIFIER = "a classifier: it reads a figure's unit to refuse mismatched axes and sets no digits"
_SVG_FIXED = _KIT_FORMAT + "a number at the format's fixed decimals, as toFixed writes it"
_SVG_USD = _KIT_FORMAT + "dollars at two significant figures under ten"
_SVG_TICKS = "axis arithmetic: where a tick falls and how fine the step is, as the kit's script places them"
_BUNDLE = "text stored in the fingerprinted analysis bundle, so changing its form moves every bundle's fingerprint"
_SNAPSHOT = "text stored in the report-input snapshot, which a stored report reads back unchanged"
_CENT_CEILING = "arithmetic: a budget in cents with its float error cleaned before it is rounded up to the cent"
_STORED_MEAN = "a stored mean latency, rounded as arithmetic before it is kept"
_COUNTS_FROM_SHARES = "arithmetic: a count recovered from a share of a whole count"
_WHOLE_SECONDS = "arithmetic: whole seconds split into hours, minutes and seconds"
_MARGIN_ARITHMETIC = "arithmetic: a margin in points with its float error cleaned, before it is declared"

_WHOLE_FROM_SHARE = "a whole count recovered from a share; sets no digits"
_INTERVAL_SENTENCE = (
    "the interval's sentence around the sizes the page's forms decided (forms.sizes); it chooses words, not digits"
)
_HEAT_COUNT = "a count in a heatmap row's table text, spelled by the rule's count primitive; charts sits below forms"
_AXIS = "axis arithmetic: tick positions and the kit's named axis format"
_KERNEL_TIMEOUT = "relays the kernel's timeout message verbatim; no page prints it"
_TABLE_CLASSIFIER = (
    "classifies a table column's unit and scale (and which money set it reads in); its printed digits are the page's "
    "forms'"
)

_SVG_STROKE = "an SVG stroke width and its opacity, which no reader reads as a number"

#: Sites whose fixed form is right for good, keyed as the walk keys a site, by (module, enclosing qualname, form),
#: each with how many sites of that form the function holds and why the fixed form is right there. A moved line keeps
#: its entry; one more site of the form in the function, or one in another function, fails, and so does one fewer
#: until the count is cut. A permanent site wraps nothing, so a helper whose every site is named here is no wrapper,
#: and its callers are sites only where it is a primitive by its name.
_PERMANENT: dict[tuple[str, str, str], tuple[int, str]] = {
    ("threetears/evals/actions/analyst.py", "<module>", "group_thousands()"): (1, _TOOL_LIMIT),
    ("threetears/evals/analysis/aspects.py", "_unread_without_control", "format_number()"): (1, _BUNDLE),
    ("threetears/evals/analysis/bundle/assemble.py", "_declared_words", "format_number()"): (3, _BUNDLE),
    ("threetears/evals/analysis/bundle/bars.py", "_frontier_bar.named", "format_number()"): (1, _BUNDLE),
    ("threetears/evals/analysis/bundle/comparisons.py", "_family_disclosure", "format_number()"): (3, _BUNDLE),
    ("threetears/evals/analysis/bundle/comparisons.py", "_guardrail_check", "format_number()"): (3, _BUNDLE),
    ("threetears/evals/analysis/bundle/coverage.py", "_within_level_dispersion", "format_number()"): (1, _BUNDLE),
    ("threetears/evals/analysis/bundle/schema.py", "AnalysisContextBundle", "format_number()"): (2, _BUNDLE),
    ("threetears/evals/analysis/depth.py", "precision", "round()"): (1, _COUNTS_FROM_SHARES),
    ("threetears/evals/analysis/latency_partition.py", "decompose_total_ms", "format_number()"): (1, _BUNDLE),
    ("threetears/evals/analysis/lenses/frontier.py", "_decide_boundary_pillar", "format_number()"): (1, _BUNDLE),
    ("threetears/evals/analysis/lenses/frontier.py", "_frontier_point", "round()"): (1, _STORED_MEAN),
    ("threetears/evals/analysis/measurement_windows.py", "format_window_gap", "round()"): (1, _WHOLE_SECONDS),
    ("threetears/evals/analysis/quantities.py", "_at_precision", "round()"): (1, _QUANTITIES),
    ("threetears/evals/analysis/quantities.py", "number_in", "format_number()"): (1, _QUANTITIES),
    ("threetears/evals/analysis/quantities.py", "render_cell", "format_number()"): (1, _QUANTITIES),
    ("threetears/evals/analysis/quantities.py", "signed_with_unit", "format_signed()"): (1, _QUANTITIES),
    ("threetears/evals/analysis/quantities.py", "with_unit", "number_in()"): (1, _QUANTITIES),
    ("threetears/evals/analysis/report/blocks.py", "ChartBlock._cost_drawn", "with_unit()"): (1, _HOST_CHART),
    ("threetears/evals/analysis/report/blocks.py", "ChartBlock._cost_quality_scatter", "with_unit()"): (3, _HOST_CHART),
    ("threetears/evals/analysis/report/blocks.py", "ChartBlock._outcome_bars", "with_unit()"): (1, _HOST_CHART),
    ("threetears/evals/analysis/report/blocks.py", "ChartBlock._values_table.said", "number_in()"): (1, _HOST_CHART),
    ("threetears/evals/analysis/report/blocks.py", "TableCell.figure", "render_cell()"): (1, _TABLE_CELL),
    ("threetears/evals/analysis/report/blocks.py", "interval_text", "number_in()"): (2, _HOST_CHART),
    ("threetears/evals/analysis/report/blocks.py", "interval_value_text", "number_in()"): (1, _HOST_CHART),
    ("threetears/evals/analysis/report/campaign/scorecard.py", "_rate_counts", "round()"): (1, _WHOLE_FROM_SHARE),
    ("threetears/evals/analysis/report/campaign/scorecard.py", "row_figures", "interval_words()"): (
        1,
        _INTERVAL_SENTENCE,
    ),
    ("threetears/evals/analysis/report/chart_svg.py", "<module>", "f-string spec"): (
        1,
        _KIT_FORMAT + "a whole amount, grouped",
    ),
    ("threetears/evals/analysis/report/chart_svg.py", "_count", "f-string spec"): (
        1,
        _KIT_FORMAT + "a tooltip's count, as toLocaleString writes it",
    ),
    ("threetears/evals/analysis/report/chart_svg.py", "_diff_heatmap", "round()"): (1, _KIT_COLOUR),
    ("threetears/evals/analysis/report/chart_svg.py", "_fixed", "quantize()"): (1, _SVG_FIXED),
    ("threetears/evals/analysis/report/chart_svg.py", "_grouped", "f-string spec"): (
        1,
        _KIT_FORMAT + "grouped dollars at the format's digits",
    ),
    ("threetears/evals/analysis/report/chart_svg.py", "_log_ticks.at", "ceil(log10)"): (1, _SVG_TICKS),
    ("threetears/evals/analysis/report/chart_svg.py", "_log_ticks.at", "floor(log10)"): (1, _SVG_TICKS),
    ("threetears/evals/analysis/report/chart_svg.py", "_scatter", "f-string spec"): (2, _SVG_STROKE),
    ("threetears/evals/analysis/report/chart_svg.py", "_scatter", "round()"): (1, _KIT_PIXELS),
    ("threetears/evals/analysis/report/chart_svg.py", "_swatch", "f-string spec"): (1, "an SVG opacity"),
    ("threetears/evals/analysis/report/chart_svg.py", "_texture", "f-string spec"): (1, "an SVG opacity"),
    ("threetears/evals/analysis/report/chart_svg.py", "_tick_labels", "round()"): (1, _SVG_TICKS),
    ("threetears/evals/analysis/report/chart_svg.py", "_ticks_raw", "floor(log10)"): (1, _SVG_TICKS),
    ("threetears/evals/analysis/report/chart_svg.py", "_ticks_raw", "round()"): (1, _SVG_TICKS),
    ("threetears/evals/analysis/report/chart_svg.py", "_usd", "f-string spec"): (
        1,
        _KIT_FORMAT + "whole dollars from $10",
    ),
    ("threetears/evals/analysis/report/chart_svg.py", "_usd", "floor(log10)"): (1, _SVG_USD),
    ("threetears/evals/analysis/report/chart_svg.py", "_usdk", "f-string spec"): (
        1,
        _KIT_FORMAT + "a whole amount, grouped",
    ),
    ("threetears/evals/analysis/report/charts.py", "_ends", "ends_places()"): (1, _RESPELL),
    ("threetears/evals/analysis/report/charts.py", "_formatted", "MoneyForm()"): (1, _RESPELL),
    ("threetears/evals/analysis/report/charts.py", "_formatted", "number_in()"): (1, _RESPELL),
    ("threetears/evals/analysis/report/charts.py", "_formatted", "signed_with_unit()"): (1, _RESPELL),
    ("threetears/evals/analysis/report/charts.py", "_formatted", "spell()"): (1, _RESPELL),
    ("threetears/evals/analysis/report/charts.py", "_formatted", "spell_at()"): (2, _RESPELL),
    ("threetears/evals/analysis/report/charts.py", "_formatted", "with_unit()"): (1, _RESPELL),
    ("threetears/evals/analysis/report/charts.py", "_heat_detail", "spell_count()"): (1, _HEAT_COUNT),
    ("threetears/evals/analysis/report/charts.py", "_respelled", "group_thousands()"): (1, _RESPELL),
    ("threetears/evals/analysis/report/charts.py", "_restates", "signed_with_unit()"): (1, _RESPELL),
    ("threetears/evals/analysis/report/charts.py", "_restates", "with_unit()"): (1, _RESPELL),
    ("threetears/evals/analysis/report/cost_unit.py", "MoneyRuler.for_changes", "for_changes()"): (1, _COST_UNIT),
    ("threetears/evals/analysis/report/cost_unit.py", "MoneyRuler.spell", "spell()"): (1, _COST_UNIT),
    ("threetears/evals/analysis/report/cost_unit.py", "MoneyRuler.spell_ends", "spell_ends()"): (1, _COST_UNIT),
    ("threetears/evals/analysis/report/cost_unit.py", "money_ruler", "money_form()"): (2, _COST_UNIT),
    ("threetears/evals/analysis/report/cost_unit.py", "per_count", "format_number()"): (1, _COST_UNIT),
    ("threetears/evals/analysis/report/cost_unit.py", "per_count", "group_thousands()"): (1, _COST_UNIT),
    ("threetears/evals/analysis/report/decision.py", "_held", "round()"): (2, _WHOLE_COUNT),
    ("threetears/evals/analysis/report/decision.py", "floor_readings", "round()"): (2, _WHOLE_COUNT),
    ("threetears/evals/analysis/report/html_parts.py", "_head_width", "round()"): (1, _CSS_WIDTH),
    ("threetears/evals/analysis/report/html_parts.py", "_id_rem", "round()"): (1, _CSS_WIDTH),
    ("threetears/evals/analysis/report/html_parts.py", "_key_rem", "round()"): (1, _CSS_WIDTH),
    ("threetears/evals/analysis/report/html_parts.py", "_lines_rem", "round()"): (1, _CSS_WIDTH),
    ("threetears/evals/analysis/report/input_build.py", "_level", "format_number()"): (1, _SNAPSHOT),
    ("threetears/evals/analysis/report/keys/prices.py", "_unanswered_figure", "group_thousands()"): (
        1,
        _UNANSWERED_COUNT,
    ),
    ("threetears/evals/analysis/report/keys/settle.py", "_assumed_cases", "round()"): (2, _WHOLE_CASES),
    ("threetears/evals/analysis/report/kit/kit.js", "kConfusion", "namedFmt"): (1, _KIT_DEFAULT),
    ("threetears/evals/analysis/report/kit/kit.js", "kDiffHeatmap", "Math.round()"): (2, _KIT_COLOUR),
    ("threetears/evals/analysis/report/kit/kit.js", "kHeatmap", "toLocaleString()"): (1, _KIT_TOOLTIP_COUNT),
    ("threetears/evals/analysis/report/kit/kit.js", "kLogTicks", "toPrecision()"): (1, _KIT_FLOAT),
    ("threetears/evals/analysis/report/kit/kit.js", "kScatter", "Math.round()"): (1, _KIT_PIXELS),
    ("threetears/evals/analysis/report/kit/kit.js", "ticksRaw", "toFixed()"): (1, _KIT_FLOAT),
    ("threetears/evals/analysis/report/parts/charting.py", "contrast_unit", "quantity()"): (2, _CLASSIFIER),
    ("threetears/evals/analysis/report/parts/charting.py", "contrast_unit_of", "quantity()"): (2, _CLASSIFIER),
    ("threetears/evals/analysis/report/parts/charting.py", "unit_refusal", "quantity()"): (1, _CLASSIFIER),
    ("threetears/evals/analysis/report/parts/look.py", "_money_text", "format_number()"): (1, _LOOK),
    ("threetears/evals/analysis/report/parts/look.py", "_money_text", "group_thousands()"): (1, _LOOK),
    ("threetears/evals/analysis/report/parts/look.py", "_money_text", "round()"): (1, _LOOK),
    ("threetears/evals/analysis/report/parts/look.py", "_money_text", "spell()"): (1, _LOOK),
    ("threetears/evals/analysis/report/parts/look.py", "_money_text", "spell_dollars()"): (1, _LOOK),
    ("threetears/evals/analysis/report/parts/look.py", "_round_up", "floor(log10)"): (1, _AXIS),
    ("threetears/evals/analysis/report/parts/look.py", "cost_axis_format", "ceil(log10)"): (1, _AXIS),
    ("threetears/evals/analysis/report/parts/look.py", "figure_text", "signed_with_unit()"): (1, _LOOK),
    ("threetears/evals/analysis/report/parts/look.py", "figure_text", "spell_at()"): (1, _LOOK),
    ("threetears/evals/analysis/report/parts/look.py", "figure_text", "with_unit()"): (1, _LOOK),
    ("threetears/evals/analysis/report/parts/look.py", "in_change_form", "for_changes()"): (1, _LOOK),
    ("threetears/evals/analysis/report/parts/look.py", "log_domain", "ceil(log10)"): (1, _AXIS),
    ("threetears/evals/analysis/report/parts/look.py", "log_domain", "floor(log10)"): (1, _AXIS),
    ("threetears/evals/analysis/report/parts/look.py", "log_ticks", "round()"): (2, _AXIS),
    ("threetears/evals/analysis/report/parts/look.py", "quantity_of", "quantity()"): (2, _LOOK),
    ("threetears/evals/analysis/report/parts/resolve_table.py", "_bar_reads", "quantity_of()"): (1, _TABLE_CLASSIFIER),
    ("threetears/evals/analysis/report/parts/resolve_table.py", "_changes_drawn", "quantity()"): (1, _TABLE_CLASSIFIER),
    ("threetears/evals/analysis/report/parts/resolve_table.py", "_changes_drawn", "quantity_of()"): (
        1,
        _TABLE_CLASSIFIER,
    ),
    ("threetears/evals/analysis/report/parts/resolve_table.py", "_column_quantity", "quantity_of()"): (
        1,
        _TABLE_CLASSIFIER,
    ),
    ("threetears/evals/analysis/report/parts/resolve_table.py", "_floors", "quantity()"): (1, _TABLE_CLASSIFIER),
    ("threetears/evals/analysis/report/said.py", "_stored_figure", "format_number()"): (3, _STORED_READING),
    ("threetears/evals/analysis/report/said.py", "_stored_figure", "round()"): (2, _STORED_READING),
    ("threetears/evals/analysis/report/svg.py", "tenth_px", "f-string spec"): (
        1,
        "an SVG coordinate, to a tenth of a px, which no reader reads as a number",
    ),
    ("threetears/evals/kernel/covariates.py", "_reasoning_ratio_of", "round()"): (1, _COVARIATE_RATIO),
    ("threetears/evals/kernel/covariates.py", "fold_phase_timings", "round()"): (1, _PHASE_TIMINGS),
    ("threetears/evals/negotiation/settings.py", "_budget_for_word", "round()"): (1, _TOKEN_BUDGET),
    ("threetears/evals/negotiation/settings.py", "_closest_reasoning", "f-string spec"): (2, _NEGOTIATION_RATIO),
    ("threetears/evals/ops/analysis.py", "_write_doc_analysis", "timed_out_detail()"): (1, _KERNEL_TIMEOUT),
    ("threetears/evals/ops/report_writer/tally.py", "ReportBudget.first_turn_refusal", "round()"): (1, _CENT_CEILING),
    ("threetears/evals/ops/sweeps.py", "_run_sweep", "timed_out_detail()"): (1, _KERNEL_TIMEOUT),
    ("threetears/evals/providers/catalogue.py", "Catalogue.load", "f-string spec"): (1, _CATALOGUE_BOUND),
    ("threetears/evals/quick/result.py", "declare", "round()"): (1, _MARGIN_ARITHMETIC),
    ("threetears/evals/run/judge.py", "_normalize_criteria_scores", "round()"): (1, _JUDGE_SCORE),
    ("threetears/evals/schema/hashing.py", "round_figures", "f-string spec"): (1, _FINGERPRINT),
    ("threetears/evals/schema/values.py", "SweepableValue._build", "f-string spec"): (1, _SCHEMA_VALUE),
    ("threetears/evals/testing/jobs_conformance.py", "CompletionGuard.after_the_call", "spell_declared()"): (
        1,
        _CONFORMANCE_TIMEOUT,
    ),
    ("threetears/evals/testing/jobs_conformance.py", "_happy_path_case.check", "spell_declared()"): (
        1,
        _CONFORMANCE_TIMEOUT,
    ),
    ("threetears/evals/transports/fastmcp/__init__.py", "CallTimedOut.__init__", "spell_declared()"): (
        1,
        _TRANSPORT_TIMEOUT,
    ),
    ("threetears/evals/transports/fastmcp/__init__.py", "CatalogueTool.run", "spell_declared()"): (
        1,
        _TRANSPORT_TIMEOUT,
    ),
}

#: How many entries :data:`_PERMANENT` holds, pinned: an entry routed through ``forms.py`` lowers it in the same
#: change, and a new exemption raises it only by an owner's decision, so no slack is ever refilled unseen.
_PERMANENT_CEILING = 118

#: The one-line remedy every failure of the rule ends with.
_REMEDY = (
    "route the figure through analysis/report/forms.py (PageForms; off_page() where no page prints it), or, in the "
    "engine trees, through a kernel/numbers.py primitive; naming it in _PERMANENT is an owner's decision, which "
    "raises _PERMANENT_CEILING"
)

#: A citation of a plan, a chunk, a review or a question that no reader of the clone can open: a reason states itself.
_CITES = re.compile(r"\bplan\b|rev-\d{8}T\d{6}Z|\bQ\d+\b|\bC\d{1,2}[a-g]?(-\d)?\b|\bchunks? \d+\b")


@dataclass(frozen=True)
class _Site:
    """One place a figure's form is decided.

    :ivar module: The file, relative to the package's ``src``.
    :ivar line: Its line, for the reader of a failure only: no key holds it.
    :ivar qualname: The enclosing definition (``Class.method``, ``outer.inner``; ``<module>`` or ``<top>`` outside one).
    :ivar form: How the form is decided: ``f-string spec``, ``round()``, ``format_number()``, ``toFixed()`` ...
    :ivar expression: The value spelled, for a spec; the call or reference itself otherwise.
    :ivar spec: The format spec, for a spec site; ``None`` for any other form.
    """

    module: str
    line: int
    qualname: str
    form: str
    expression: str
    spec: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.module, self.qualname, self.form)

    def shown(self) -> str:
        what = f"{{{self.expression}:{self.spec}}}" if self.spec is not None else self.expression
        return f"{self.module}:{self.line} ({self.qualname}, {self.form}): {what}"


# --- Python -------------------------------------------------------------------------------------------------------

_PRECISION = re.compile(r"\.(\d+|\{\})")
_PERCENT_DIRECTIVE = re.compile(r"%[-+ #0]*(\d+|\*)?(\.(\d+|\*))?[eEfFgG]")
_BRACE_FIELD_SPEC = re.compile(r"\{[^{}]*:([^{}]*)\}")


def _chooses_digits(spec: str) -> bool:
    """Whether a format spec decides how many digits a number is written with.

    A precision (fixed or nested), a numeric presentation type, a thousands separator, or a spec
    built at run time, which cannot be judged here and so is flagged.
    """
    if "{}" in spec or _PRECISION.search(spec) or "," in spec or "_" in spec:
        return True
    return bool(spec) and spec[-1] in _NUMERIC_TYPES


def _joined_spec(spec: ast.JoinedStr) -> str:
    return "".join(part.value if isinstance(part, ast.Constant) else "{}" for part in spec.values)


def _rule_primitives() -> tuple[frozenset[str], frozenset[str]]:
    """The rule's public callables: its top-level functions and classes, and apart, the public methods of those classes.

    A method's name (``spell``, ``places``) is also an ordinary local's, so it is a site only as a method call.
    """
    tree = ast.parse((_SOURCE_ROOT / _RULE).read_text(encoding="utf-8"))
    names: set[str] = set()
    methods: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            names.add(node.name)
            if isinstance(node, ast.ClassDef):
                methods |= {
                    member.name
                    for member in node.body
                    if isinstance(member, ast.FunctionDef) and not member.name.startswith("_")
                }
    return frozenset(names), frozenset(methods)


_RULE_CALLABLES, _RULE_METHODS = _rule_primitives()


def _forms_primitives(root: Path = _SOURCE_ROOT) -> frozenset[str]:
    """The public module-level functions of ``forms.py``: a decision relocated there and called as a free function
    is still decided where it is called, so its caller is a site. Only :class:`PageForms`, built once per page and
    asked by measure and role, is the migrated route: its classes (the page's forms and the value types it hands out)
    and their methods are not primitives. None before ``forms.py`` exists.
    """
    path = root / _FORMS
    if not path.exists():
        return frozenset()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
    )


_FORMS_CALLABLES = _forms_primitives()

_BASE_PRIMITIVES = _RULE_CALLABLES | _SET_HELPERS | _FORMS_CALLABLES

#: Every method name of a builtin value. A wrapper method named like one (``format``, ``count``) is still a site
#: through ``self``, but an attribute call by that name alone is not taken to reach it.
_BUILTIN_METHODS = frozenset(name for kind in (str, bytes, list, dict, set, tuple, int, float) for name in dir(kind))


#: ``math.floor`` / ``math.ceil`` over a ``log10`` is digits derived from a magnitude (``2 - floor(log10(x))``
#: places); the ``log10`` alone is also a log axis's position, so it is a site only inside a floor or a ceiling.
_DIGITS_FROM_MAGNITUDE = frozenset({"floor", "ceil"})

#: Calls that hand a spelled figure on as text: a string method on it, or a builtin over it. What a function returns
#: through one of these is still the figure's form, so the function is a wrapper. A method is matched as an attribute
#: call and a builtin as a bare name, so ``dataclasses.replace(cell, text=...)`` is not read as ``str.replace``.
_TEXT_METHODS = frozenset(
    {
        "capitalize",
        "center",
        "join",
        "ljust",
        "lower",
        "lstrip",
        "removeprefix",
        "removesuffix",
        "replace",
        "rjust",
        "rstrip",
        "strip",
        "title",
        "upper",
    }
)
_TEXT_BUILTINS = frozenset({"abs", "int", "list", "max", "min", "sorted", "str", "tuple"})


def _dotted(module: str) -> str:
    """``threetears/evals/a/b.py`` as ``threetears.evals.a.b``; a package's ``__init__.py`` as the package."""
    parts = module.removesuffix(".py").split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


@dataclass(frozen=True)
class _Signature:
    """A function's parameters as a call binds them, to tell which parameter an argument reaches.

    :ivar positional: The parameters an argument binds by place, less a method's ``self`` or ``cls``.
    :ivar keyword_only: The parameters an argument binds by name only.
    :ivar star: The ``*args`` parameter, or ``None``.
    :ivar double_star: The ``**kwargs`` parameter, or ``None``.
    """

    positional: tuple[str, ...]
    keyword_only: tuple[str, ...]
    star: str | None
    double_star: str | None

    @classmethod
    def of(cls, function: ast.FunctionDef | ast.AsyncFunctionDef, *, method: bool) -> _Signature:
        arguments = function.args
        positional = tuple(argument.arg for argument in (*arguments.posonlyargs, *arguments.args))
        static = any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in function.decorator_list)
        return cls(
            positional[1:] if method and not static else positional,
            tuple(argument.arg for argument in arguments.kwonlyargs),
            arguments.vararg.arg if arguments.vararg else None,
            arguments.kwarg.arg if arguments.kwarg else None,
        )

    @property
    def names(self) -> tuple[str, ...]:
        extra = tuple(name for name in (self.star, self.double_star) if name is not None)
        return (*self.positional, *self.keyword_only, *extra)

    def bound(self, call: ast.Call) -> Iterator[tuple[ast.expr, frozenset[str]]]:
        """Each argument of ``call`` with the parameters it may bind: one where the source says which, every one it
        could reach where an unpacked ``*`` or ``**`` hides it."""
        by_place = (*self.positional, *([self.star] if self.star else []))
        for place, argument in enumerate(call.args):
            if isinstance(argument, ast.Starred):
                yield argument, frozenset(by_place[place:])
            elif place < len(self.positional):
                yield argument, frozenset({self.positional[place]})
            elif self.star is not None:
                yield argument, frozenset({self.star})
        by_name = (*self.positional, *self.keyword_only)
        for keyword in call.keywords:
            if keyword.arg is None:
                yield keyword.value, frozenset((*by_name, *([self.double_star] if self.double_star else [])))
            elif keyword.arg in by_name:
                yield keyword.value, frozenset({keyword.arg})
            elif self.double_star is not None:
                yield keyword.value, frozenset({self.double_star})


#: The nodes a site can be: a call, a formatted field, and a ``%`` over a literal.
_SITE_NODES = (ast.Call, ast.FormattedValue, ast.Interpolation, ast.BinOp)


def _scoped_nodes(tree: ast.AST, scope: tuple[str, ...] = ()) -> Iterator[tuple[int, str]]:
    """Each node of ``tree`` a site can be, by its ``id``, with the qualname the walk keys a site at it by: the
    enclosing definitions' names joined (``Class.method.inner``), ``<module>`` outside one."""
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # As the walk keys them: a definition's decorators and defaults sit in its own scope.
            yield from _scoped_nodes(child, (*scope, child.name))
        else:
            if isinstance(child, _SITE_NODES):
                yield id(child), ".".join(scope) or "<module>"
            yield from _scoped_nodes(child, scope)


class _Index:
    """The walked modules' definitions and imports, to tell which function a call reaches.

    A call is resolved the way Python resolves it, as far as the source says: a name to its own module's definition
    or to the definition its import names (through any re-export), a module alias's attribute to that module's
    definition, ``self.name`` to the method of the module's own class. An attribute call on anything else is taken to
    reach a wrapper method only when every method of that name in the package is one.
    """

    def __init__(self, trees: dict[str, ast.Module]) -> None:
        self.trees = trees
        self.by_dotted = {_dotted(module): module for module in trees}
        self.defined: dict[str, set[str]] = {}
        self.imports: dict[str, dict[str, tuple[str, str | None]]] = {}
        #: Each method name, with ``(module, "Class.name")`` of every method that bears it.
        self.methods: dict[str, set[tuple[str, str]]] = {}
        #: Each function's parameters, by its key (``(module, name)``, or ``(module, "Class.name")`` for a method);
        #: nested functions of one name in one module share a key, as they share it everywhere in the walk.
        self.signatures: dict[tuple[str, str], list[_Signature]] = {}
        #: The methods read as attributes (a property, a cached one), by their key: reading one runs it.
        self.properties: set[tuple[str, str]] = set()
        #: Each module's every node that may be a site, by its ``id``, with the qualname of the definition it sits in,
        #: as a site is keyed: what :data:`_PERMANENT` names a permanent site by.
        self.scopes: dict[str, dict[int, str]] = {}
        for module, tree in trees.items():
            self.scopes[module] = dict(_scoped_nodes(tree))
            self.defined[module] = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            self.imports[module] = dict(self._imports(module, tree))
            for function, reached, cls in _definitions(tree):
                key = _key(module, function, reached, cls)
                if reached == "method":
                    self.methods.setdefault(function.name, set()).add(key)
                    if any(_reads_as_attribute(decorator) for decorator in function.decorator_list):
                        self.properties.add(key)
                self.signatures.setdefault(key, []).append(_Signature.of(function, method=reached == "method"))

    def permanent(self, module: str, node: ast.AST, form: str) -> bool:
        """Whether a site of ``form`` at ``node`` is one :data:`_PERMANENT` names, by its function: so it wraps nothing.
        The count is the rule's to hold, not the derivation's."""
        return (module, self.scopes.get(module, {}).get(id(node), "<module>"), form) in _PERMANENT

    def _imports(self, module: str, tree: ast.Module) -> Iterator[tuple[str, tuple[str, str | None]]]:
        package = _dotted(module).split(".") if module.endswith("__init__.py") else _dotted(module).split(".")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = ".".join([*package[: len(package) - node.level + 1], *([node.module] if node.module else [])])
                base = base if node.level else (node.module or "")
                for alias in node.names:
                    target = f"{base}.{alias.name}"
                    yield alias.asname or alias.name, (target, None) if target in self.by_dotted else (base, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        yield alias.asname, (alias.name, None)

    def resolve(self, module: str, name: str, depth: int = 0) -> tuple[str, str] | None:
        """The ``(module, name)`` of the definition ``name`` reaches in ``module``, or ``None`` outside the walk."""
        if name in self.defined.get(module, ()):
            return (module, name)
        target, attribute = self.imports.get(module, {}).get(name, ("", None))
        if attribute is None or target not in self.by_dotted or depth > 8:
            return None
        return self.resolve(self.by_dotted[target], attribute, depth + 1)

    def module_alias(self, module: str, name: str) -> str | None:
        target, attribute = self.imports.get(module, {}).get(name, ("", ""))
        return self.by_dotted.get(target) if attribute is None else None


@dataclass(frozen=True)
class _Lookup:
    """What a call names to be a site: the rule's primitives and the set helpers, and the wrappers derived so far.

    :ivar index: The walked modules' definitions and imports.
    :ivar functions: ``(module, name)`` of every module-level or nested function found to be a wrapper.
    :ivar methods: ``(module, "Class.name")`` of every method found to be a wrapper.
    :ivar primitives: The names a call is a site by: the rule's, the set helpers and ``forms.py``'s public functions.
    :ivar carriers: Each function, by its key, that returns what some of its parameters were given, with those
        parameters: a call that hands one of them a form returns that form, so it carries a site to its caller.
    :ivar engine_route: Whether a ``numbers`` primitive is the route inside :data:`_ENGINE_TREES`: so for a call made
        there, and not for one made above the line, where an engine function returning a primitive's text is a
        wrapper like any other.
    """

    index: _Index
    functions: frozenset[tuple[str, str]] = frozenset()
    methods: frozenset[tuple[str, str]] = frozenset()
    primitives: frozenset[str] = _BASE_PRIMITIVES
    carriers: Mapping[tuple[str, str], frozenset[str]] = field(default_factory=dict)
    engine_route: bool = True

    def callees(self, func: ast.expr, where: tuple[str, str]) -> list[tuple[str, str]]:
        """The keys of the package functions a call of ``func`` reaches, resolved as :meth:`named` resolves a wrapper.

        An attribute call on anything but ``self``, ``cls`` or a module alias is taken to reach a method only when
        every method of that name in the package carries a parameter, so a common name reaches nothing.
        """
        module, cls = where
        signatures = self.index.signatures
        if isinstance(func, ast.Name):
            if (module, func.id) in signatures:
                return [(module, func.id)]
            resolved = self.index.resolve(module, func.id)
            return [resolved] if resolved in signatures else []
        if not isinstance(func, ast.Attribute):
            return []
        name, owner = func.attr, func.value
        if isinstance(owner, ast.Name):
            if owner.id in ("self", "cls") and (module, f"{cls}.{name}") in signatures:
                return [(module, f"{cls}.{name}")]
            alias = self.index.module_alias(module, owner.id)
            if alias is not None:
                resolved = self.index.resolve(alias, name)
                return [resolved] if resolved in signatures else []
        defined = self.index.methods.get(name, set())
        if defined and name not in _BUILTIN_METHODS and all(key in self.carriers for key in defined):
            return sorted(defined)
        return []

    def read(self, node: ast.expr, where: tuple[str, str]) -> str | None:
        """The wrapper ``node`` runs by reading it as an attribute (``self.figure``, a property that returns a form),
        or ``None``. Only through ``self`` or ``cls``: by name alone, ``.figure`` or ``.text`` is every dataclass's
        field too."""
        if not (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)):
            return None
        if not (isinstance(node.value, ast.Name) and node.value.id in ("self", "cls")):
            return None
        key = (where[0], f"{where[1]}.{node.attr}")
        return node.attr if key in self.methods and key in self.index.properties else None

    def carried(self, call: ast.Call, where: tuple[str, str]) -> list[ast.expr]:
        """The arguments of ``call`` whose value the function it reaches returns."""
        found = []
        for key in self.callees(call.func, where):
            carrying = self.carriers.get(key, frozenset())
            for signature in self.index.signatures[key] if carrying else ():
                found += [argument for argument, bound in signature.bound(call) if bound & carrying]
        return found

    def named(self, node: ast.expr, where: tuple[str, str], *, as_value: bool = False) -> str | None:
        """The primitive ``node`` names, called (or, with ``as_value``, handed on as a function value), or ``None``.

        A family prefix or a method name is also an ordinary local's name (``quantity``, ``places``), so a value
        handed on is matched by exact name only, and a method only as an attribute call. ``where`` is the call's
        module and enclosing class (empty outside one), which is what ``self`` names. In the engine
        (:data:`_ENGINE_TREES`) the rule's own callables and methods are the route and name nothing.
        """
        module, cls = where
        engine = self.engine_route and _in_engine(module)
        if isinstance(node, ast.Name):
            name = node.id
            if (module, name) in _PRIVATE_SET_HELPERS or (module, name) in self.functions:
                return name
            if self.index.resolve(module, name) in self.functions:
                return name
        elif isinstance(node, ast.Attribute):
            name = node.attr
            if not as_value and name in _RULE_METHODS and not engine:
                return name
            owner = node.value
            if isinstance(owner, ast.Name):
                if owner.id in ("self", "cls") and (module, f"{cls}.{name}") in self.methods:
                    return name
                alias = self.index.module_alias(module, owner.id)
                if alias is not None and self.index.resolve(alias, name) in self.functions:
                    return name
            defined = self.index.methods.get(name, set())
            if not as_value and defined and name not in _BUILTIN_METHODS and defined <= self.methods:
                return name
        else:
            return None
        if name.startswith("_") or (engine and name in _RULE_CALLABLES):
            return None
        if name in self.primitives or (not as_value and name.startswith(_SET_HELPER_PREFIXES)):
            return name
        return None


def _call_form(node: ast.Call, where: tuple[str, str], lookup: _Lookup) -> str | None:
    """The form a call decides by itself (a format spec aside), or ``None``."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == "round":
        return "round()"
    if isinstance(func, ast.Attribute) and func.attr == "quantize":
        return "quantize()"
    if (
        isinstance(func, ast.Attribute)
        and func.attr in _DIGITS_FROM_MAGNITUDE
        and any(
            isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "log10"
            for argument in node.args
            for inner in ast.walk(argument)
        )
    ):
        return f"{func.attr}(log10)"
    name = lookup.named(func, where)
    return None if name is None else f"{name}()"


def _spec_sites(node: ast.AST) -> Iterator[tuple[str, str, str]]:
    """Every ``(form, expression, spec)`` the node itself spells with a digit-choosing spec.

    Covers the ways Python formats a number: an f-string or t-string field, the ``format`` builtin, ``str.format``
    on a literal, and ``%`` on a literal.
    """
    if isinstance(node, (ast.FormattedValue, ast.Interpolation)) and isinstance(node.format_spec, ast.JoinedStr):
        spec = _joined_spec(node.format_spec)
        if _chooses_digits(spec):
            form = "f-string spec" if isinstance(node, ast.FormattedValue) else "t-string spec"
            yield form, ast.unparse(node.value), spec
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            for match in _PERCENT_DIRECTIVE.finditer(node.left.value):
                yield "% spec", ast.unparse(node.right), match.group(0)
    elif isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "format" and len(node.args) == 2:
            spec_node = node.args[1]
            spec = spec_node.value if isinstance(spec_node, ast.Constant) and isinstance(spec_node.value, str) else "{}"
            if _chooses_digits(spec):
                yield "format() spec", ast.unparse(node.args[0]), spec
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "format"
            and isinstance(func.value, ast.Constant)
            and isinstance(func.value.value, str)
        ):
            for spec in _BRACE_FIELD_SPEC.findall(func.value.value):
                if _chooses_digits(spec):
                    yield "str.format() spec", ast.unparse(node), spec


class _PythonForms(ast.NodeVisitor):
    """Collects every site in one module, each under its enclosing definition."""

    def __init__(self, module: str, lookup: _Lookup) -> None:
        self.module = module
        self.lookup = lookup
        self.scope: list[str] = []
        self.classes: list[str] = []
        self.sites: list[_Site] = []
        #: The collections an ``isinstance`` or ``issubclass`` test names its classes in, whose elements it reads.
        self.class_tests: set[int] = set()
        #: The functions each enclosing definition builds inside itself: one it returns is a factory's formatter.
        self.built: list[frozenset[str]] = []
        #: The names each enclosing scope binds itself (its parameters, what it assigns or loops over).
        self.bound: list[frozenset[str]] = []

    @property
    def where(self) -> tuple[str, str]:
        return (self.module, self.classes[-1] if self.classes else "")

    def _add(self, node: ast.AST, form: str, expression: str, spec: str | None = None) -> None:
        qualname = ".".join(self.scope) or "<module>"
        self.sites.append(_Site(self.module, getattr(node, "lineno", 0), qualname, form, expression, spec))

    def _scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.scope.append(node.name)
        if isinstance(node, ast.ClassDef):
            self.classes.append(node.name)
        self.built.append(
            frozenset()
            if isinstance(node, ast.ClassDef)
            else frozenset(
                inner.name
                for inner in ast.walk(node)
                if inner is not node and isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        )
        self.bound.append(_bound_here(node))
        self.generic_visit(node)
        self.bound.pop()
        self.built.pop()
        if isinstance(node, ast.ClassDef):
            self.classes.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node)

    def visit_Module(self, node: ast.Module) -> None:
        self.bound.append(_bound_here(node))
        self.generic_visit(node)
        self.bound.pop()

    def _reads_a_local(self, value: ast.expr) -> bool:
        """Whether ``value`` is, or is a field read of, a name its scope binds itself: a local named like a primitive
        (``confidence=interval_level``, ``family.interval_level``) holds a value, not the primitive. ``self`` and
        ``cls`` are left to the wrapper methods :meth:`_Lookup.named` resolves through them."""
        root = value
        while isinstance(root, ast.Attribute):
            root = root.value
        return (
            isinstance(root, ast.Name)
            and root.id not in ("self", "cls")
            and bool(self.bound)
            and (root.id in self.bound[-1])
        )

    def _returns_what_it_built(self, node: ast.AST) -> bool:
        """Whether ``node`` returns, by name, a function its enclosing definition built: a factory's formatter, whose
        site is counted where it is written, not again where it is handed back."""
        returned = node.value if isinstance(node, ast.Return) else None
        return isinstance(returned, ast.Name) and bool(self.built) and returned.id in self.built[-1]

    def generic_visit(self, node: ast.AST) -> None:
        for form, expression, spec in _spec_sites(node):
            self._add(node, form, expression, spec)
        if isinstance(node, ast.Call):
            if (form := _call_form(node, self.where, self.lookup)) is not None:
                self._add(node, form, ast.unparse(node))
        elif (read := self.lookup.read(node, self.where)) is not None:  # a property is called by reading it
            self._add(node, f"{read}()", ast.unparse(node))
        # A primitive handed on as a function value (``map(format_number, ...)``, ``key=spell_at``, a table of
        # formatters, a name bound to one, a parameter's default, a return) decides where it is applied, which the
        # walk cannot follow, so it is counted where it is handed on, which is here.
        if _tests_a_class(node):
            assert isinstance(node, ast.Call)
            self.class_tests |= {id(names) for argument in node.args[1:] for names in _class_tuples(argument)}
        for value in [] if id(node) in self.class_tests or self._returns_what_it_built(node) else _handed_values(node):
            if self._reads_a_local(value):
                continue
            if (handed := self.lookup.named(value, self.where, as_value=True)) is not None:
                self._add(value, f"{handed}()", ast.unparse(value))
        super().generic_visit(node)


def _bound_here(scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> frozenset[str]:
    """The names ``scope`` binds itself: a function's parameters, and every name it assigns, loops over or binds with
    ``as``, read without entering a nested function or class, which binds its own. An import binds the module's name
    for what it imports, so it is not counted."""
    names: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = scope.args
        names |= {argument.arg for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
        names |= {argument.arg for argument in (arguments.vararg, arguments.kwarg) if argument is not None}
    pending: list[ast.AST] = list(scope.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            pending.extend(ast.iter_child_nodes(node))
    return frozenset(names)


#: The builtins whose arguments name a class to test against, read rather than handed on.
_CLASS_TESTS = frozenset({"isinstance", "issubclass"})


def _tests_a_class(node: ast.AST) -> bool:
    """Whether ``node`` is an ``isinstance`` or ``issubclass`` call: its arguments are read, never handed on."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CLASS_TESTS


def _class_tuples(node: ast.expr) -> Iterator[ast.Tuple]:
    """The tuples, nested to any depth, that a class test's second argument names its classes in."""
    if isinstance(node, ast.Tuple):
        yield node
        for element in node.elts:
            yield from _class_tuples(element)


def _handed_values(node: ast.AST) -> list[ast.expr]:
    """The expressions ``node`` hands on as values: a call's arguments and a parameter's default, and, by bare name,
    a dict's values, a collection's elements, what an assignment binds and what a ``return`` returns.

    Outside a call and a default only a bare name counts: an attribute there (``family.interval_level``) is far
    likelier a field read than a method handed on, where a default, evaluated once as the function is defined, reads
    no instance. An annotation, or the class an ``isinstance`` or ``issubclass`` test names, reads a class, not a
    formatter, and is none of these. A function returning a formatter it built is a factory, whose site is counted
    where it is written; one returning a primitive by its name hands that primitive on, and is a site where it does.
    """
    if _tests_a_class(node):
        return []
    if isinstance(node, ast.Call):
        return [*node.args, *(keyword.value for keyword in node.keywords)]
    if isinstance(node, ast.arguments):
        defaults = [*node.defaults, *(default for default in node.kw_defaults if default is not None)]
        return [default for default in defaults if isinstance(default, (ast.Name, ast.Attribute))]
    if isinstance(node, ast.Dict):
        values: list[ast.expr | None] = list(node.values)
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = list(node.elts)
    elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return)):
        values = [node.value]
    else:
        values = []
    return [value for value in values if isinstance(value, ast.Name)]


def _spelled(node: ast.AST | None, where: tuple[str, str], lookup: _Lookup, spelled: set[str]) -> bool:
    """Whether ``node``'s value is a figure's form: a site's result, or text built from one.

    Only what carries the text on counts (a string built from it, a branch, a tuple, a string method or a builtin
    over it). An object built around it (a table cell, a page) does not, and neither do lines joined into a document:
    were they to, every builder that ever placed a figure would be a wrapper, and so would every caller of one, which
    over this package is most of it (3,357 of its 5,716 functions, measured 2026-11-08).
    """
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id in spelled
    module = where[0]
    if any(not lookup.index.permanent(module, node, form) for form, _expression, _spec in _spec_sites(node)):
        return True
    if isinstance(node, ast.JoinedStr):
        return any(_spelled(part, where, lookup, spelled) for part in node.values)
    if isinstance(node, ast.FormattedValue):
        return _spelled(node.value, where, lookup, spelled)
    if isinstance(node, ast.BinOp):
        return _spelled(node.left, where, lookup, spelled) or _spelled(node.right, where, lookup, spelled)
    if isinstance(node, ast.UnaryOp):
        return _spelled(node.operand, where, lookup, spelled)
    if isinstance(node, ast.IfExp):
        return _spelled(node.body, where, lookup, spelled) or _spelled(node.orelse, where, lookup, spelled)
    if isinstance(node, ast.BoolOp):
        return any(_spelled(value, where, lookup, spelled) for value in node.values)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_spelled(element, where, lookup, spelled) for element in node.elts)
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return _spelled(node.elt, where, lookup, spelled)
    if isinstance(node, (ast.Starred, ast.Subscript, ast.Await)):
        return _spelled(node.value, where, lookup, spelled)
    if isinstance(node, ast.Attribute):
        return lookup.read(node, where) is not None
    if isinstance(node, ast.Call):
        form = _call_form(node, where, lookup)
        if form is not None and not lookup.index.permanent(module, node, form):
            return True
        func = node.func
        if isinstance(func, ast.Name) and func.id in _TEXT_BUILTINS:
            carried = [*node.args, *(keyword.value for keyword in node.keywords)]
            return any(_spelled(part, where, lookup, spelled) for part in carried)
        if isinstance(func, ast.Attribute) and func.attr in _TEXT_METHODS:
            joins_lines = isinstance(func.value, ast.Constant) and "\n" in str(func.value.value)
            carried = [func.value, *node.args]
            return not joins_lines and any(_spelled(part, where, lookup, spelled) for part in carried)
        # A package helper that returns what it was handed (a unit joined on, a sign put in front) hands a form it
        # was given back: the call is the form where its argument is.
        return any(_spelled(argument, where, lookup, spelled) for argument in lookup.carried(node, where))
    return False


def _reads_as_attribute(decorator: ast.expr) -> bool:
    """Whether ``decorator`` makes a method one read as an attribute: ``property``, ``cached_property``."""
    name = decorator.attr if isinstance(decorator, ast.Attribute) else getattr(decorator, "id", "")
    return name in ("property", "cached_property")


def _key(module: str, function: ast.FunctionDef | ast.AsyncFunctionDef, reached: str, cls: str) -> tuple[str, str]:
    """A function's key in the walk: ``(module, "Class.name")`` for a method, ``(module, name)`` for any other."""
    return (module, f"{cls}.{function.name}") if reached == "method" else (module, function.name)


def _definitions(tree: ast.Module) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str, str]]:
    """Every function in ``tree``, how a call reaches it (``module`` level, a ``method``, or ``nested``), and the
    class whose ``self`` it reads (empty outside one)."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, "module", ""
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield member, "method", node.name
    for top in tree.body:
        for node in ast.walk(top):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cls = top.name if isinstance(top, ast.ClassDef) else ""
                for inner in ast.walk(node):
                    if inner is not node and isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        yield inner, "nested", cls


def _returns_a_form(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    where: tuple[str, str],
    lookup: _Lookup,
    given: frozenset[str] = frozenset(),
) -> bool:
    """Whether what ``function`` returns is a figure's form, directly or through a local it spelled.

    ``given`` names parameters taken to hold a form already, which asks whether the function hands back what it was
    given rather than whether it decides a form itself.
    """
    spelled: set[str] = set(given)
    body = list(ast.walk(function))
    for _ in range(2):  # twice, so a local spelled from one assigned below it in the source is still seen
        for node in body:
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if _spelled(node.value, where, lookup, spelled):
                    spelled |= {target.id for target in targets if isinstance(target, ast.Name)}
    return any(isinstance(node, ast.Return) and _spelled(node.value, where, lookup, spelled) for node in body)


def _reached_names(function: ast.FunctionDef | ast.AsyncFunctionDef, module: str, index: _Index) -> frozenset[str]:
    """The names a function's calls reach: each as written, and each import alias as the name it imports."""
    names = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            names.add(node.id)
            resolved = index.resolve(module, node.id)
            if resolved is not None:
                names.add(resolved[1])
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return frozenset(names)


def _carried_parameters(
    function: ast.FunctionDef | ast.AsyncFunctionDef, where: tuple[str, str], lookup: _Lookup, parameters: Iterable[str]
) -> frozenset[str]:
    """The parameters whose value ``function`` returns, each found alone so a parameter is never credited with what
    another one carries; read one by one only where all of them together carry anything."""
    parameters = frozenset(parameters)
    if not parameters or not _returns_a_form(function, where, lookup, parameters):
        return frozenset()
    return frozenset(name for name in parameters if _returns_a_form(function, where, lookup, frozenset({name})))


def _derive_wrappers(
    index: _Index, primitives: frozenset[str] = _BASE_PRIMITIVES, *, engine_route: bool = True
) -> _Lookup:
    """Every function over the walked modules that returns a form, and every one that returns what it was handed,
    to the fixed point.

    A helper that returns what a site produced is a primitive itself, and so is one that returns what such a helper
    produced, so a call rerouted through a wrapper is still a site where it is called. A helper that only hands back
    what it was given (``_joined(text, unit)``) decides nothing, so a call to it is no site; but a function that hands
    it a form, through an argument, a nested call or a local, returns that form and is a wrapper. The owners are not
    walked, so routing through them stays the remedy, and a permanent site wraps nothing. After the first pass only a
    function that names a wrapper or a carrier found in the pass before can change, so only those are read again.
    """
    candidates = [
        (
            function,
            module,
            cls,
            reached == "method",
            _key(module, function, reached, cls),
            _reached_names(function, module, index),
            _Signature.of(function, method=reached == "method").names,
        )
        for module, tree in index.trees.items()
        for function, reached, cls in _definitions(tree)
        if not function.name.startswith("__")
    ]
    lookup = _Lookup(index, primitives=primitives, engine_route=engine_route)
    fresh: frozenset[str] | None = None
    while True:
        wrappers: dict[tuple[str, str], bool] = {}
        carriers: dict[tuple[str, str], frozenset[str]] = {}
        for function, module, cls, method, key, names, parameters in candidates:
            if key in lookup.functions or key in lookup.methods or (fresh is not None and not names & fresh):
                continue
            if _returns_a_form(function, (module, cls), lookup):
                wrappers[key] = method
                continue
            known = lookup.carriers.get(key, frozenset()) | carriers.get(key, frozenset())
            carrying = _carried_parameters(function, (module, cls), lookup, parameters)
            if carrying - known:
                carriers[key] = known | carrying
        if not wrappers and not carriers:
            return lookup
        fresh = frozenset(name.rsplit(".", 1)[-1] for _module, name in (*wrappers, *carriers))
        lookup = _Lookup(
            index,
            lookup.functions | {key for key, method in wrappers.items() if not method},
            lookup.methods | {key for key, method in wrappers.items() if method},
            primitives,
            {**lookup.carriers, **carriers},
            engine_route,
        )


def _walk_python(sources: dict[str, str], primitives: frozenset[str] = _BASE_PRIMITIVES) -> list[_Site]:
    """Every site in ``sources`` (module to text), with the wrappers derived over them as primitives.

    Derived twice where the engine is walked: a call made inside :data:`_ENGINE_TREES` is read with ``numbers`` as
    the route, and one made above the line with every engine function that returns a primitive's text a wrapper.
    """
    index = _Index({module: ast.parse(text) for module, text in sources.items()})
    above = _derive_wrappers(index, primitives, engine_route=False)
    engine = _derive_wrappers(index, primitives) if any(_in_engine(module) for module in index.trees) else above
    sites = []
    for module, tree in index.trees.items():
        visitor = _PythonForms(module, engine if _in_engine(module) else above)
        visitor.visit(tree)
        sites.extend(visitor.sites)
    return sorted(sites, key=lambda site: (site.module, site.line, site.form, site.expression))


def _python_sites(
    source: str, module: str = "sample.py", *, primitives: frozenset[str] = _BASE_PRIMITIVES
) -> list[_Site]:
    return _walk_python({module: source}, primitives)


# --- kit.js -------------------------------------------------------------------------------------------------------

_JS_WORD = re.compile(r"[A-Za-z0-9_$]+")
#: After one of these a ``/`` opens a regular expression; after anything else it divides.
_REGEX_AFTER = frozenset("(,=:[!&|?{};+-*%<>~^")
_REGEX_AFTER_WORDS = frozenset(
    {"return", "typeof", "case", "in", "of", "new", "delete", "void", "throw", "else", "do", "yield", "await"}
)
_JS_TOKEN = re.compile(r"[A-Za-z_$][\w$]*|[()\[\]{};\n]")
_JS_DECLARES = frozenset({"function", "const", "let", "var", "class"})
_JS_DECLARED = re.compile(r"(?:\bfunction|\bconst|\blet|\bvar|\bclass)\s*$")


def _js_named(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w$.]){re.escape(name)}(?![\w$])")


#: Each kit.js form: the digit primitives (anything ``Intl`` offers among them), and the named formats picked or
#: applied by name. A call that applies a format the chart's data names (``f(v)``, ``opt.fmt(v)``) is not a site:
#: the data's ``fmt`` is decided in Python, where the walk sees it, and the pick is the ``namedFmt`` call.
_JS_FORMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("toFixed()", re.compile(r"\.\s*toFixed\s*\(")),
    ("toPrecision()", re.compile(r"\.\s*toPrecision\s*\(")),
    ("toLocaleString()", re.compile(r"\.\s*toLocaleString\s*\(")),
    ("Intl.", re.compile(r"(?<![\w$.])Intl\s*\.")),
    ("Math.round()", re.compile(r"(?<![\w$.])Math\s*\.\s*round\s*\(")),
    ("fmt.", re.compile(r"(?<![\w$.])fmt\s*\.")),
    ("K_FMTS", _js_named("K_FMTS")),
    ("namedFmt", _js_named("namedFmt")),
    ("kTickFmt", _js_named("kTickFmt")),
    ("kTickLabels", _js_named("kTickLabels")),
    ("kShort", _js_named("kShort")),
)


def _quoted_end(source: str, start: int) -> int:
    quote, at = source[start], start + 1
    while at < len(source):
        if source[at] == "\\":
            at += 2
        elif source[at] in (quote, "\n"):
            return at + 1 if source[at] == quote else at
        else:
            at += 1
    return len(source)


def _template_text_end(source: str, start: int) -> tuple[int, bool]:
    """Where a template literal's text run from ``start`` ends, and whether a ``${`` (rather than the close) ended it."""
    at = start
    while at < len(source):
        if source[at] == "\\":
            at += 2
        elif source[at] == "`":
            return at, False
        elif source.startswith("${", at):
            return at, True
        else:
            at += 1
    return len(source), False


def _regex_end(source: str, start: int) -> tuple[int, int] | None:
    """The end of a regex literal's body and of its flags, or ``None`` where the ``/`` at ``start`` opens none."""
    at, in_class = start + 1, False
    while at < len(source):
        char = source[at]
        if char == "\\":
            at += 2
            continue
        if char == "\n":
            return None
        if in_class:
            in_class = char != "]"
        elif char == "[":
            in_class = True
        elif char == "/":
            flags = at + 1
            while flags < len(source) and source[flags].isalpha():
                flags += 1
            return at, flags
        at += 1
    return None


def _code_only(source: str) -> str:
    """``source`` with every comment, string, template text and regex body blanked, offsets and newlines kept.

    A template's ``${...}`` expressions stay, since they are code; what is blanked can hold no site.
    """
    chars = list(source)

    def blank(start: int, end: int) -> None:
        for at in range(start, end):
            if chars[at] != "\n":
                chars[at] = " "

    at, last, templates = 0, "", []
    while at < len(source):
        char = source[at]
        if char.isspace():
            at += 1
            continue
        if source.startswith("//", at):
            end = source.find("\n", at)
            end = len(source) if end < 0 else end
            blank(at, end)
            at = end
            continue
        if source.startswith("/*", at):
            end = source.find("*/", at + 2)
            end = len(source) if end < 0 else end + 2
            blank(at, end)
            at = end
            continue
        if char in "'\"":
            end = _quoted_end(source, at)
            blank(at + 1, end - 1)
            at, last = end, "'"
            continue
        if char == "`" or (char == "}" and templates and templates[-1] == 0):
            if char == "}":
                templates.pop()
            end, opened = _template_text_end(source, at + 1)
            blank(at + 1, end)
            if opened:
                templates.append(0)
                at, last = end + 2, "("
            else:
                at, last = end + 1, "'"
            continue
        if char == "/" and (last == "" or last in _REGEX_AFTER or last in _REGEX_AFTER_WORDS):
            ends = _regex_end(source, at)
            if ends is not None:
                blank(at + 1, ends[0])
                at, last = ends[1], "'"
                continue
        word = _JS_WORD.match(source, at)
        if word:
            at, last = word.end(), word.group()
            continue
        if templates and char in "{}":
            templates[-1] += 1 if char == "{" else -1
        at, last = at + 1, char
    return "".join(chars)


def _js_statements(code: str) -> tuple[list[int], list[str]]:
    """The start offset and name of each top-level statement: its declared name, or ``<top>`` for any other."""
    offsets, names = [0], ["<top>"]
    depth, at_start, declaring = 0, True, -1
    for token in _JS_TOKEN.finditer(code):
        text = token.group()
        if text in ("(", "[", "{"):
            depth += 1
        elif text in (")", "]", "}"):
            depth -= 1
            at_start = at_start or (depth == 0 and text == "}")
        elif depth:
            continue
        elif text in (";", "\n"):
            at_start = True
        elif declaring >= 0:
            offsets.append(declaring)
            names.append(text)
            declaring = -1
        elif at_start and text != "async":
            at_start = False
            if text in _JS_DECLARES:
                declaring = token.start()
            else:
                offsets.append(token.start())
                names.append("<top>")
    return offsets, names


def _balanced_end(code: str, opening: int) -> int:
    depth = 0
    for at in range(opening, len(code)):
        if code[at] in "([{":
            depth += 1
        elif code[at] in ")]}":
            depth -= 1
            if depth == 0:
                return at + 1
    return len(code)


def _js_expression(source: str, code: str, start: int, end: int) -> str:
    """The site's text: the match, on through any member chain, and through the call or index that follows it."""
    if code[end - 1] == "(":
        end = _balanced_end(code, end - 1)
    else:
        while end < len(code) and (code[end].isalnum() or code[end] in "_$."):
            end += 1
        if end < len(code) and code[end] in "([":
            end = _balanced_end(code, end)
    return " ".join(source[start:end].split())


_JS_EXPRESSION_ARROW = re.compile(r"(?:const|let|var)\s+[\w$]+\s*=\s*(?:async\s*)?(?:\([^()]*\)|[\w$]+)\s*=>\s*(?!\{)")


def _js_returned(code: str, start: int, end: int) -> list[tuple[int, int]]:
    """The spans of the top-level statement ``code[start:end]`` whose value it returns.

    An expression-bodied arrow (``const f=v=>...``) returns the whole of itself; anything else returns what each
    ``return`` runs to, its ``;`` or the close of the block it sits in.
    """
    if _JS_EXPRESSION_ARROW.match(code, start):
        return [(start, end)]
    spans = []
    for found in re.finditer(r"(?<![\w$.])return(?![\w$])", code[start:end]):
        depth, at = 0, start + found.end()
        while at < end and not (depth == 0 and code[at] == ";") and depth >= 0:
            depth += 1 if code[at] in "([{" else -1 if code[at] in ")]}" else 0
            at += 1
        spans.append((start + found.end(), at))
    return spans


#: A picker's call whose one argument, or whose first, is a format field of the chart's data as every drawer of the
#: kit names it: ``d``, the chart's data, and its ``fmt`` or a ``<role>_fmt`` (``d.fmt``, ``d.cost_fmt``). A field
#: of any other object (a constant table's ``K_DEF.fmt``) is the kit's own choice, and a site.
_JS_PICK_BY_FIELD = re.compile(r"\s*\(\s*d\s*\.\s*(?:[A-Za-z_$][\w$]*_)?fmt\s*[,)]")


def _js_found(code: str, forms: Iterable[tuple[str, re.Pattern[str]]]) -> Iterator[tuple[str, re.Match[str]]]:
    """Each match of ``forms`` in ``code`` but a declaration's name and a pick by a data field
    (:data:`_KIT_PICKERS`)."""
    for form, pattern in forms:
        for match in pattern.finditer(code):
            if _JS_DECLARED.search(code, 0, match.start()):
                continue
            if form in _KIT_PICKERS and _JS_PICK_BY_FIELD.match(code, match.end()):
                continue
            yield form, match


def _js_sites(source: str, module: str = "sample.js") -> list[_Site]:
    """Every site in a kit script, with each top-level function that returns one a form of its own.

    A function whose returned value holds a site (``const kAt=v=>v.toFixed(1)``) is a wrapper, and a call to it is a
    site, to the fixed point, so a figure rerouted through a new helper is still counted where it is called. A
    function that only holds a site in its body (``ticksRaw`` cleaning a float with ``toFixed(10)``) is not.
    """
    code = _code_only(source)
    offsets, names = _js_statements(code)
    bounds = [*offsets[1:], len(code)]
    spans = {}
    for start, end, name in zip(offsets, bounds, names, strict=True):
        if name != "<top>" and name not in _KIT_OWNERS:
            spans.setdefault(name, []).extend(_js_returned(code, start, end))
    named = {form for form, _pattern in _JS_FORMS}
    forms = list(_JS_FORMS)
    while True:
        returning = set()
        for _form, match in _js_found(code, forms):
            name = names[bisect.bisect_right(offsets, match.start()) - 1]
            if name not in named and any(low <= match.start() < high for low, high in spans.get(name, ())):
                returning.add(name)
        if not returning:
            break
        forms.extend((f"{name}()", re.compile(rf"(?<![\w$.]){re.escape(name)}\s*\(")) for name in sorted(returning))
        named |= returning
    sites = []
    for form, match in _js_found(code, forms):
        qualname = names[bisect.bisect_right(offsets, match.start()) - 1]
        if qualname in _KIT_OWNERS:
            continue
        line = code.count("\n", 0, match.start()) + 1
        sites.append(_Site(module, line, qualname, form, _js_expression(source, code, match.start(), match.end())))
    return sorted(sites, key=lambda site: (site.line, site.form, site.expression))


# --- The walk and the rule ----------------------------------------------------------------------------------------


def _walked_modules(root: Path = _SOURCE_ROOT) -> list[Path]:
    """Every Python module of the package under ``root`` but the owners."""
    owners = {root / owner for owner in _OWNERS}
    return [path for path in sorted((root / _PACKAGE).rglob("*.py")) if path not in owners]


def _modules_under(trees: Iterable[str], root: Path = _SOURCE_ROOT) -> list[Path]:
    """Every module an entry of ``trees`` names under ``root`` (each ``.py`` under a directory, or the file itself),
    less the owners and the ones :data:`_NOT_READ_AS_REPORTS` sets aside."""
    modules = []
    for tree_root in trees:
        named = root / tree_root
        assert named.is_dir() or (named.is_file() and named.suffix == ".py"), tree_root
        modules.extend(sorted(named.rglob("*.py")) if named.is_dir() else [named])
    excluded = [root / entry for entry in (*_NOT_READ_AS_REPORTS, *_OWNERS)]
    return [module for module in modules if not any(module == ex or ex in module.parents for ex in excluded)]


@functools.cache
def _all_sites() -> tuple[_Site, ...]:
    """Every site of the package and the kit, walked once: deriving the wrappers reads the whole package."""
    sources = {
        path.relative_to(_SOURCE_ROOT).as_posix(): path.read_text(encoding="utf-8") for path in _walked_modules()
    }
    kit = _js_sites((_SOURCE_ROOT / _KIT_JS).read_text(encoding="utf-8"), _KIT_JS)
    return (*_walk_python(sources), *kit)


def _unnamed(sites: Iterable[_Site]) -> list[_Site]:
    """Every site of a key :data:`_PERMANENT` does not name, or names at fewer sites than the walk finds there."""
    sites = list(sites)
    live = _tally(sites)
    over = {key for key, count in live.items() if count > _PERMANENT.get(key, (0, ""))[0]}
    return [site for site in sites if site.key in over]


def _outlived(sites: Iterable[_Site]) -> dict[tuple[str, str, str], tuple[int, int]]:
    """Each :data:`_PERMANENT` entry naming more sites than the walk finds there: its count, then the walk's."""
    live = _tally(sites)
    return {key: (count, live.get(key, 0)) for key, (count, _reason) in _PERMANENT.items() if live.get(key, 0) < count}


def _tally(sites: Iterable[_Site]) -> Counter[tuple[str, str, str]]:
    return Counter(site.key for site in sites)


class TestTheRule:
    """Every site the walk finds is an owner's or named in :data:`_PERMANENT`, and every entry there names a site."""

    def test_no_site_is_outside_the_owners_and_the_permanent_list(self):
        sites = _unnamed(_all_sites())
        shown = [site.shown() for site in sites]
        assert not sites, f"number-form sites outside the owners: {_REMEDY}.\n  " + "\n  ".join(shown)

    def test_every_permanent_entry_still_names_its_count_of_sites(self):
        """An exemption a site walked out of, or holding more than it names, is one nobody decided: cut its count."""
        assert _outlived(_all_sites()) == {}

    def test_every_permanent_reason_is_stated_and_cites_nothing_a_reader_cannot_open(self):
        uncounted = [key for key, (count, _reason) in _PERMANENT.items() if count < 1]
        unstated = [key for key, (_count, reason) in _PERMANENT.items() if not reason.strip()]
        citing = {key: found.group() for key, (_n, reason) in _PERMANENT.items() if (found := _CITES.search(reason))}
        assert uncounted == []
        assert unstated == []
        assert citing == {}

    def test_the_permanent_list_is_pinned_and_the_engine_has_a_ceiling(self):
        assert len(_PERMANENT) == _PERMANENT_CEILING
        assert len(_ENGINE_TREES) <= _ENGINE_TREES_CEILING

    def test_a_reason_that_cites_a_plan_or_an_id_is_caught(self):
        for reason in ("named at C6", "owner ruling Q3", "see the plan", "rev-20261110T135620Z", "chunk 6", "C10b"):
            assert _CITES.search(reason), reason
        assert not _CITES.search("a CSS rem width, planned by the table's columns; Q-sort C-5")

    def test_the_walk_reads_every_module_but_the_owners_and_the_kit(self):
        """Derived from the tree, so a module added anywhere in the package is walked the day it lands."""
        walked = {path.relative_to(_SOURCE_ROOT).as_posix() for path in _walked_modules()}
        every = {path.relative_to(_SOURCE_ROOT).as_posix() for path in (_SOURCE_ROOT / _PACKAGE).rglob("*.py")}
        assert walked == every - set(_OWNERS)
        assert _RULE in every
        assert {site.module for site in _all_sites()} >= {_KIT_JS, _CHART_SVG}

    def test_a_permanent_entry_holds_its_function_and_its_count(self, monkeypatch):
        """A moved line, or one rewritten in place, keeps its entry; one more of the form in the function, or the same
        expression in another function, fails; and an entry the function no longer fills outlives its site."""
        monkeypatch.setitem(_PERMANENT, ("sample.py", "g", "round()"), (1, "a self-test's permanent site"))
        moved = _python_sites("\n\n# a comment\ndef g(v):\n\n    return round(v, 1)\n")
        rewritten = _python_sites("def g(v):\n    return round(v, 2)\n")
        one_more = _python_sites("def g(v):\n    return round(v, 1), round(v, 1)\n")
        elsewhere = _python_sites("def g(v):\n    return round(v, 1)\n\ndef h(v):\n    return round(v, 1)\n")
        gone = _python_sites("def g(v):\n    return v\n")
        assert _unnamed(moved) == _unnamed(rewritten) == []
        assert [(site.qualname, site.expression) for site in _unnamed(one_more)] == [("g", "round(v, 1)")] * 2
        assert [(site.qualname, site.expression) for site in _unnamed(elsewhere)] == [("h", "round(v, 1)")]
        assert ("sample.py", "g", "round()") not in _outlived(moved)
        assert _outlived(gone)[("sample.py", "g", "round()")] == (1, 0)


class TestTheReportTreesKeepTheStricterRule:
    """Under ``analysis/`` and ``ops/`` no format spec chooses digits but a share's ``%`` and a permanent site.

    Derived from the source rather than from a function name: a second ``def format_number`` is one way to grow a
    second rule, and an inline ``f"{value:.4g}"`` is the commoner one, the shape that printed a latency ``1.235e+04``
    in a results pivot beside an interval spelling the same magnitude ``12346``. The engine's validation messages
    outside these trees are not read as reports, so :data:`_PERMANENT` names them instead.
    """

    def test_a_module_named_by_itself_is_walked(self):
        """A single-module entry is parsed, not skipped as a tree with nothing under it.

        Asked of the walk's module list rather than of the spellings it found: once a module routes
        every number through the rule it yields no spelling, and "no spelling" would then read the
        same whether the module was clean or never opened.
        """
        single = "threetears/evals/analysis/reporting.py"
        assert [path.relative_to(_SOURCE_ROOT).as_posix() for path in _modules_under((single,))] == [single]
        walked = {path.relative_to(_SOURCE_ROOT).as_posix() for path in _modules_under(_SPELLING_TREES)}
        assert single in walked
        # And what _NOT_READ_AS_REPORTS names is inside a walked tree yet not walked.
        assert not {
            m for m in walked if m in ("threetears/evals/kernel/numbers.py", "threetears/evals/analysis/report/svg.py")
        }

    def test_every_spelling_goes_through_the_one_rule_or_is_named_as_permanent(self):
        report_modules = {path.relative_to(_SOURCE_ROOT).as_posix() for path in _modules_under(_SPELLING_TREES)}
        offenders = [
            site.shown()
            for site in _all_sites()
            if site.module in report_modules
            if site.spec is not None and not site.spec.endswith(_PERCENTAGE_TYPE) and site.key not in _PERMANENT
        ]
        assert not offenders, (
            "these spell a number with their own precision; route them through format_number / format_signed, "
            "or name a deliberate fixed spelling in _PERMANENT with its reason:\n  " + "\n  ".join(offenders)
        )


def _specs(source: str) -> list[tuple[str, str | None]]:
    return [(site.expression, site.spec) for site in _python_sites(source) if site.spec is not None]


def _forms(sites: Iterable[_Site]) -> list[tuple[str, str, str]]:
    return [(site.qualname, site.form, site.expression) for site in sites]


class TestTheWalkSeesEveryPythonForm:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ('f"{value:.4g}"', ("value", ".4g")),
            ('f"{delta:+.3g}"', ("delta", "+.3g")),
            ('f"{latency:.0f} ms"', ("latency", ".0f")),
            ('f"{bound:g}"', ("bound", "g")),
            ('f"{count:,}"', ("count", ",")),
            ('f"{share:.0%}"', ("share", ".0%")),
            ('f"{value:.{places}f}"', ("value", ".{}f")),
            ('format(value, ".3f")', ("value", ".3f")),
            ("format(value, spec)", ("value", "{}")),
            ('"{:.2e}".format(value)', ("'{:.2e}'.format(value)", ".2e")),
            ('"%.3f" % value', ("value", "%.3f")),
            ('t"{value:.2f}"', ("value", ".2f")),
        ],
    )
    def test_the_walk_sees_every_way_python_spells_a_number(self, source, expected):
        assert _specs(source) == [expected]

    def test_a_t_string_field_is_its_own_form(self):
        assert [site.form for site in _python_sites('t"{value:.2f}"')] == ["t-string spec"]

    @pytest.mark.parametrize(
        "source", ['f"{name}"', 'f"{name!r}"', 'f"{count:d}"', 'f"{label:>10}"', 'f"{format_number(v)}"']
    )
    def test_the_walk_leaves_what_chooses_no_digits(self, source):
        assert _specs(source) == []

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("round(v, 2)", ("<module>", "round()", "round(v, 2)")),
            ("format_number(v)", ("<module>", "format_number()", "format_number(v)")),
            ("numbers.format_signed(v)", ("<module>", "format_signed()", "numbers.format_signed(v)")),
            ("spell_count(n)", ("<module>", "spell_count()", "spell_count(n)")),
            ("MoneyForm(decimals=2)", ("<module>", "MoneyForm()", "MoneyForm(decimals=2)")),
            ("form.spell(v)", ("<module>", "spell()", "form.spell(v)")),
            ("set_places(values)", ("<module>", "set_places()", "set_places(values)")),
            ("figure_places(key)", ("<module>", "figure_places()", "figure_places(key)")),
            ("charts.respell_bare(t, 1)", ("<module>", "respell_bare()", "charts.respell_bare(t, 1)")),
            ("percent_columns(rows)", ("<module>", "percent_columns()", "percent_columns(rows)")),
            ("look.quantity_of(u, v)", ("<module>", "quantity_of()", "look.quantity_of(u, v)")),
            ("number_in(v, '%')", ("<module>", "number_in()", "number_in(v, '%')")),
            ("quantities.with_unit(v, 'ms')", ("<module>", "with_unit()", "quantities.with_unit(v, 'ms')")),
            ("signed_with_unit(d, 'pts')", ("<module>", "signed_with_unit()", "signed_with_unit(d, 'pts')")),
            ("list(map(render_cell, row))", ("<module>", "render_cell()", "render_cell")),
            ("list(map(format_number, xs))", ("<module>", "format_number()", "format_number")),
            ("sorted(xs, key=numbers.spell_at)", ("<module>", "spell_at()", "numbers.spell_at")),
            ("Decimal(v).quantize(step)", ("<module>", "quantize()", "Decimal(v).quantize(step)")),
            ("2 - math.floor(math.log10(v))", ("<module>", "floor(log10)", "math.floor(math.log10(v))")),
            ("math.ceil(-math.log10(top / 5))", ("<module>", "ceil(log10)", "math.ceil(-math.log10(top / 5))")),
        ],
    )
    def test_the_walk_sees_rounding_and_every_primitive(self, source, expected):
        assert _forms(_python_sites(source)) == [expected]

    def test_a_site_is_keyed_by_its_enclosing_definition(self):
        source = "class Look:\n    def text(self, v):\n        def inner():\n            return round(v)\n        return inner\n"
        assert _forms(_python_sites(source)) == [("Look.text.inner", "round()", "round(v)")]

    def test_a_private_helper_is_a_site_only_in_the_module_that_decides_with_it(self):
        deciding = "threetears/evals/harness/writer_eval.py"
        assert _forms(_python_sites("_places(vs)", deciding)) == [("<module>", "_places()", "_places(vs)")]
        assert _python_sites("_places(vs)", "threetears/evals/analysis/report/build.py") == []

    @pytest.mark.parametrize(
        "source",
        [
            "np.round(v)",
            "len(xs)",
            "places(v)",
            "round_to(v, places)",
            "f(quantity, spell)",
            "f(self.respelled)",
            "math.log10(v)",
            "math.floor(v / step)",
            # A measure's held means are floats: what reads them to classify a measure sets no digits.
            "held_means(report_input, key)",
        ],
    )
    def test_the_walk_leaves_what_decides_no_form(self, source):
        assert _python_sites(source) == []


def _count(sites: Iterable[_Site]) -> int:
    return sum(_tally(sites).values())


class TestAWrapperIsAPrimitive:
    """A function that returns what a site produced is a primitive itself, so a call rerouted through it is a site."""

    def test_a_helper_that_wraps_a_site_makes_its_callers_sites(self):
        source = (
            "def bare_at(v, p):\n    return spell_at(v, p)\n\n"
            "def label(v):\n    return f'{bare_at(v, 1)} pts'\n\n"
            "def row(v):\n    return label(v)\n"
        )
        assert _forms(_python_sites(source)) == [
            ("bare_at", "spell_at()", "spell_at(v, p)"),
            ("label", "bare_at()", "bare_at(v, 1)"),
            ("row", "label()", "label(v)"),
        ]

    def test_rerouting_a_call_through_a_wrapper_does_not_shrink_the_count(self):
        """``spell_at`` moved into ``bare_at``: every caller keeps its site, and the wrapper holds one more."""
        direct = "def a(v):\n    return spell_at(v, 1)\n\ndef b(v):\n    return spell_at(v, 2)\n"
        rerouted = (
            "def bare_at(v, p):\n    return spell_at(v, p)\n\n"
            "def a(v):\n    return bare_at(v, 1)\n\ndef b(v):\n    return bare_at(v, 2)\n"
        )
        assert _count(_python_sites(direct)) == 2
        assert _count(_python_sites(rerouted)) == 3
        assert _tally(_python_sites(rerouted))[("sample.py", "a", "bare_at()")] == 1

    def test_a_wrapper_in_another_module_is_reached_through_its_import(self):
        sources = {
            "threetears/evals/a.py": "def bare_at(v, p):\n    text = spell_at(v, p)\n    return text.strip()\n",
            "threetears/evals/b.py": "from threetears.evals.a import bare_at as at\n\ndef f(v):\n    return at(v, 1)\n",
            "threetears/evals/c.py": "from threetears.evals import a\n\ndef g(v):\n    return a.bare_at(v, 1)\n",
            "threetears/evals/d.py": "from .a import bare_at\n\ndef h(v):\n    return bare_at(v, 1)\n",
            "threetears/evals/e.py": "def bare_at(v, p):\n    return v\n\ndef k(v):\n    return bare_at(v, 1)\n",
        }
        called = {(site.module, site.qualname, site.form) for site in _walk_python(sources)}
        assert called == {
            ("threetears/evals/a.py", "bare_at", "spell_at()"),
            ("threetears/evals/b.py", "f", "at()"),
            ("threetears/evals/c.py", "g", "bare_at()"),
            ("threetears/evals/d.py", "h", "bare_at()"),
        }

    def test_a_wrapper_method_is_reached_through_self_and_by_a_name_no_other_method_holds(self):
        source = (
            "class Ruler:\n"
            "    def figure(self, v):\n        return self.form.spell(v)\n"
            "    def head(self, v):\n        return self.figure(v)\n"
            "    def render(self, v):\n        return self.figure(v) + ' ms'\n\n"
            "class Page:\n"
            "    def render(self):\n        return Page()\n\n"
            "def show(ruler, page, v):\n    return ruler.figure(v), ruler.render(v), page.render()\n"
        )
        assert _forms(_python_sites(source)) == [
            ("Ruler.figure", "spell()", "self.form.spell(v)"),
            ("Ruler.head", "figure()", "self.figure(v)"),
            ("Ruler.render", "figure()", "self.figure(v)"),
            # ``render`` is also Page's, which returns no form, so ``ruler.render(v)`` is not taken to reach Ruler's.
            ("show", "figure()", "ruler.figure(v)"),
        ]

    def test_a_builder_around_a_figure_or_a_document_of_lines_is_not_a_wrapper(self):
        """Were it one, so would every page builder and every caller of one be (most of this package)."""
        source = (
            "def cell(v):\n    return TableCell(text=spell_at(v, 1))\n\n"
            "def page(v):\n    return '\\n'.join([spell_at(v, 1), 'end'])\n\n"
            "def table(v):\n    return [cell(v), page(v)]\n"
        )
        assert [(site.qualname, site.form) for site in _python_sites(source)] == [
            ("cell", "spell_at()"),
            ("page", "spell_at()"),
        ]

    def test_a_permanent_site_inside_a_helper_leaves_its_callers_unlisted(self, monkeypatch):
        """An SVG opacity is permanent, so the helper that writes it wraps nothing and drawing through it is no site."""
        monkeypatch.setitem(_PERMANENT, (_CHART_SVG, "_faded", "f-string spec"), (1, "an SVG opacity"))
        source = (
            "def _faded(opacity):\n    return f'{opacity:g}'\n\n"
            "def _swatch(opacity):\n    return f'<rect opacity=\"{_faded(opacity)}\"/>'\n"
        )
        sites = _python_sites(source, _CHART_SVG)
        assert [(site.qualname, site.form) for site in sites] == [("_faded", "f-string spec")]
        assert _unnamed(sites) == []

    def test_a_relocated_decision_called_as_a_free_function_is_still_a_site(self, tmp_path):
        """A decision moved into ``forms.py`` and called from a surface is decided where it is called, so a module
        that reroutes through one never reaches zero entries by the move alone. Every public free function there is
        such a primitive, and nothing private, nor a class or its methods, is."""
        forms = tmp_path / _FORMS
        forms.parent.mkdir(parents=True)
        forms.write_text(
            "def relocated(v):\n    return v\n\n"
            "def _kept_inside(v):\n    return v\n\n"
            "class PageForms:\n    def level(self, v):\n        return v\n",
            encoding="utf-8",
        )
        assert _forms_primitives(tmp_path) == frozenset({"relocated"})
        live = ast.parse((_SOURCE_ROOT / _FORMS).read_text(encoding="utf-8"))
        assert _FORMS_CALLABLES == {
            node.name
            for node in live.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
        }
        assert _FORMS_CALLABLES <= _BASE_PRIMITIVES
        assert _python_sites("relocated(cells)") == []
        primitives = _BASE_PRIMITIVES | _forms_primitives(tmp_path)
        assert _forms(_python_sites("relocated(cells)", primitives=primitives)) == [
            ("<module>", "relocated()", "relocated(cells)")
        ]
        assert _forms(_python_sites("forms.relocated(cells)", primitives=primitives)) == [
            ("<module>", "relocated()", "forms.relocated(cells)")
        ]
        assert _python_sites("page.level(v)", primitives=primitives) == []
        assert _python_sites("_kept_inside(v)", primitives=primitives) == []

    def test_a_page_s_forms_asked_by_measure_and_role_are_the_migrated_route(self):
        """``PageForms``, its constructors, its methods and the values it hands out decide nothing at the call."""
        assert not _FORMS_CALLABLES & {"PageForms", "MeasureForm", "Printed", "of"}
        source = (
            "page = PageForms.of(report_input)\n"
            "codes = PageForms.of_campaign(readings, reports)\n"
            "a = page.level(key, v).text\n"
            "b = page.change(key, v, kind=kind).digits\n"
            "c = page.interval(key, lo, hi, change=True).text\n"
            "d = page.measure(key).places\n"
        )
        assert _python_sites(source) == []

    def test_forms_primitives_are_none_before_forms_exists(self, tmp_path):
        assert _forms_primitives(tmp_path) == frozenset()

    def test_an_owner_is_not_walked_so_routing_through_it_stays_the_remedy(self, tmp_path):
        """``forms.py`` is an owner before it exists: the walk and the report trees both pass over it."""
        for module in (_FORMS, _RULE, "threetears/evals/analysis/report/figures.py"):
            (tmp_path / module).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / module).write_text("x = round(1.5)\n", encoding="utf-8")
        walked = [path.relative_to(tmp_path).as_posix() for path in _walked_modules(tmp_path)]
        under = [path.relative_to(tmp_path).as_posix() for path in _modules_under(_SPELLING_TREES[:1], tmp_path)]
        assert walked == under == ["threetears/evals/analysis/report/figures.py"]
        assert _FORMS in _OWNERS


#: A helper that decides nothing and hands back the text it was given, joined to its unit: ``quantities._joined``'s
#: shape, which ``with_unit`` hands ``number_in``'s figure.
_JOINED = "def _joined(text, unit):\n    return text if not unit else f'{text} {unit}'\n\n"


class TestASiteHandedThroughAHelperIsStillASite:
    """A function that hands a site's figure to a helper returning what it was given returns that figure, so it is a
    wrapper and its callers are sites; a helper handed a value nobody decided stays no site, and neither do its
    callers."""

    @pytest.mark.parametrize(
        ("body", "inner"),
        [
            # Through an argument: with_unit's shape, number_in's figure handed straight to _joined.
            ("    return _joined(spell_at(v, 1), unit)\n", ("figure", "spell_at()", "spell_at(v, 1)")),
            # Through a nested call: the figure passes two helpers that each hand it back.
            (
                "    return _joined(_trimmed(spell_at(v, 1)), unit)\n",
                ("figure", "spell_at()", "spell_at(v, 1)"),
            ),
            # Through a local the helper builds before handing it on.
            (
                "    text = spell_at(v, 1)\n    return _joined(text, unit=unit)\n",
                ("figure", "spell_at()", "spell_at(v, 1)"),
            ),
        ],
        ids=["argument", "nested-call", "local"],
    )
    def test_a_figure_handed_to_a_helper_makes_the_hander_a_wrapper(self, body, inner):
        source = (
            _JOINED
            + "def _trimmed(text):\n    return text.strip()\n\n"
            + f"def figure(v, unit):\n{body}\n"
            + "def head(v):\n    return f'{figure(v, \"ms\")} mean'\n\n"
            + "def row(v):\n    return head(v)\n"
        )
        assert _forms(_python_sites(source)) == [
            inner,
            ("head", "figure()", "figure(v, 'ms')"),
            ("row", "head()", "head(v)"),
        ]

    def test_a_value_only_passed_through_decides_nothing(self):
        """The helper and every caller handing it a raw value or a name are no site; only the call that hands it a
        figure is one, beside them, so the helper itself is never taken for a primitive. Nor is a figure handed to a
        parameter the helper does not return."""
        source = (
            _JOINED
            + "def _first(text, other):\n    return text\n\n"
            + "def label(name, unit):\n    return _joined(name, unit)\n\n"
            + "def header(name):\n    return label(name, 'ms')\n\n"
            + "def aside(name, v):\n    return _first(name, spell_at(v, 1))\n\n"
            + "def checked(v):\n    return aside('n', v)\n\n"
            + "def figure(v):\n    return _joined(spell_at(v, 1), 'ms')\n\n"
            + "def head(v):\n    return figure(v)\n"
        )
        assert _forms(_python_sites(source)) == [
            ("aside", "spell_at()", "spell_at(v, 1)"),
            ("figure", "spell_at()", "spell_at(v, 1)"),
            ("head", "figure()", "figure(v)"),
        ]

    def test_a_method_handing_a_figure_to_its_own_helper_is_a_wrapper(self):
        source = (
            "class Ruler:\n"
            "    def _joined(self, text, *, unit):\n        return f'{text} {unit}'\n"
            "    def figure(self, v):\n        return self._joined(spell_at(v, 1), unit='ms')\n"
            "    def head(self, v):\n        return self.figure(v)\n"
        )
        assert _forms(_python_sites(source)) == [
            ("Ruler.figure", "spell_at()", "spell_at(v, 1)"),
            ("Ruler.head", "figure()", "self.figure(v)"),
        ]

    def test_a_helper_named_a_primitive_keeps_its_callers_when_its_body_is_permanent(self, monkeypatch):
        """A permanent site wraps nothing, so a helper whose body is named permanently would lose its callers by
        derivation alone; one listed by name keeps them."""
        quantities = "threetears/evals/analysis/quantities.py"
        for helper in ("number_in", "tenths"):
            monkeypatch.setitem(_PERMANENT, (quantities, helper, "f-string spec"), (1, "a self-test's permanent body"))
        sources = {
            quantities: "def number_in(v, unit):\n    return f'{v:.1f}'\n\ndef tenths(v):\n    return f'{v:.1f}'\n",
            "threetears/evals/caller.py": (
                "from threetears.evals.analysis.quantities import number_in, tenths\n\n"
                "def level(v):\n    return number_in(v, 'ms')\n\n"
                "def other(v):\n    return tenths(v)\n"
            ),
        }
        ratcheted = _unnamed(_walk_python(sources))
        assert [(site.module, site.qualname, site.form) for site in ratcheted] == [
            ("threetears/evals/caller.py", "level", "number_in()")
        ]

    def test_the_package_s_own_with_unit_is_a_primitive(self):
        """``with_unit`` hands ``number_in``'s figure to ``_joined``, so every call to it is a site, by its name and by
        derivation alike."""
        quantities = "threetears/evals/analysis/quantities.py"
        sources = {
            quantities: (_SOURCE_ROOT / quantities).read_text(encoding="utf-8"),
            "threetears/evals/caller.py": (
                "from threetears.evals.analysis.quantities import signed_with_unit, with_unit\n\n"
                "def level(v):\n    return with_unit(100 * v, '%')\n\n"
                "def change(v):\n    return signed_with_unit(v, 'pts')\n"
            ),
        }
        called = {(site.qualname, site.form) for site in _walk_python(sources) if site.module.endswith("caller.py")}
        assert called == {("level", "with_unit()"), ("change", "signed_with_unit()")}


class TestAPrimitiveHandedOnAsAValueIsASite:
    """A primitive stored to be applied later decides where it is applied, which the walk cannot follow, so it is a
    site where it is handed on."""

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("FORMATS = {'num': format_number, 'pct': percent}\n", ("<module>", "format_number()", "format_number")),
            ("FORMATS = [format_number, str]\n", ("<module>", "format_number()", "format_number")),
            ("spell = format_number\n", ("<module>", "format_number()", "format_number")),
        ],
        ids=["dict", "list", "bound-name"],
    )
    def test_a_table_or_a_name_of_a_primitive_is_a_site(self, source, expected):
        assert _forms(_python_sites(source)) == [expected]

    def test_an_annotation_a_class_test_or_a_field_read_is_not(self):
        source = (
            "def f(form: MoneyForm, family, kind) -> MoneyForm:\n"
            "    assert isinstance(form, MoneyForm) and not issubclass(kind, (str, (MoneyForm, int)))\n"
            "    level = family.interval_level\n"
            "    return {'level': family.interval_level, 'form': form}\n"
        )
        assert _python_sites(source) == []

    def test_a_local_named_like_a_primitive_or_a_field_read_of_one_is_not(self):
        """A parameter, an assignment or a loop's name is a value: handing it on, or a field read through it, hands
        on no primitive. The primitive by the same name, where nothing local shadows it, still is."""
        source = (
            "def f(rows, interval_level):\n"
            "    for family in rows:\n"
            "        g(confidence=interval_level, level=family.interval_level)\n"
            "    spell = format_number\n"
            "    return h(spell)\n\n"
            "def k(verdict):\n"
            "    return g(verdict.interval_level, numbers.interval_level)\n\n"
            "def m(v):\n"
            "    return g(key=interval_level)\n"
        )
        assert _forms(_python_sites(source)) == [
            ("f", "format_number()", "format_number"),
            ("k", "interval_level()", "numbers.interval_level"),
            ("m", "interval_level()", "interval_level"),
        ]

    def test_a_class_test_still_sees_a_site_it_is_handed(self):
        """Only the class it names is read: a figure spelled inside the test, or a primitive handed to a call there,
        is a site as anywhere."""
        source = "ok = isinstance(format_number(v), str) and isinstance(apply(spell_at), (MoneyForm,))\n"
        assert _forms(_python_sites(source)) == [
            ("<module>", "format_number()", "format_number(v)"),
            ("<module>", "spell_at()", "spell_at"),
        ]

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("def f(v, spell=format_number):\n    return spell(v)\n", ("f", "format_number()", "format_number")),
            ("def f(v, *, spell=spell_at):\n    return spell(v, 1)\n", ("f", "spell_at()", "spell_at")),
            (
                "def f(v, spell=numbers.format_number):\n    return spell(v)\n",
                ("f", "format_number()", "numbers.format_number"),
            ),
            ("f = lambda v, spell=format_number: spell(v)\n", ("<module>", "format_number()", "format_number")),
        ],
        ids=["positional", "keyword-only", "through-a-module", "lambda"],
    )
    def test_a_primitive_as_a_parameter_s_default_is_a_site(self, source, expected):
        assert _forms(_python_sites(source)) == [expected]

    def test_a_primitive_returned_by_its_name_is_a_site_and_its_returner_is_a_factory(self):
        """The returner hands the primitive on, so its own call decides nothing and is no site."""
        source = "def pick(kind):\n    return format_number\n\ndef use(v):\n    return pick('n')(v)\n"
        assert _forms(_python_sites(source)) == [("pick", "format_number()", "format_number")]

    def test_a_default_or_a_return_of_anything_else_is_not(self):
        source = "def f(v, places=PLACES, unit=config.unit):\n    return places\n"
        assert _python_sites(source) == []

    def test_a_property_that_returns_a_form_is_a_site_where_self_reads_it(self):
        source = (
            "class Cell:\n"
            "    @property\n    def figure(self):\n        return spell_at(self.value, 1)\n"
            "    @property\n    def text(self):\n        said = self.figure\n        return said\n"
        )
        assert _forms(_python_sites(source)) == [
            ("Cell.figure", "spell_at()", "spell_at(self.value, 1)"),
            ("Cell.text", "figure()", "self.figure"),
        ]


def _imported(module: str, tree: ast.Module) -> Iterator[str]:
    """Every module ``tree`` imports, absolute, wherever the import sits (a function, a ``TYPE_CHECKING`` block), and
    each name a ``from`` import takes as the module it may be; ``importlib.import_module`` of a literal too."""
    package = _dotted(module).split(".") if module.endswith("__init__.py") else _dotted(module).split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            named = [node.module] if node.module else []
            base = ".".join([*package[: len(package) - node.level + 1], *named]) if node.level else ".".join(named)
            yield base
            yield from (f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
            == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            yield node.args[0].value


def _is_analysis(dotted: str) -> bool:
    return dotted == _ANALYSIS or dotted.startswith(f"{_ANALYSIS}.")


def _imports_analysis(module: str, source: str) -> list[str]:
    """What ``module`` imports from ``threetears.evals.analysis`` directly."""
    return sorted({name for name in _imported(module, ast.parse(source)) if _is_analysis(name)})


def _import_graph(root: Path = _SOURCE_ROOT) -> dict[str, frozenset[str]]:
    """Each module of the package under ``root``, dotted, with the modules of the package it imports (a package
    imported, or one a name is taken from, is its ``__init__``)."""
    paths = {_dotted(path.relative_to(root).as_posix()): path for path in sorted((root / _PACKAGE).rglob("*.py"))}
    return {
        dotted: frozenset(
            name
            for name in _imported(path.relative_to(root).as_posix(), ast.parse(path.read_text(encoding="utf-8")))
            if name in paths
        )
        for dotted, path in paths.items()
    }


def _route_to_analysis(graph: Mapping[str, Iterable[str]], start: str) -> list[str]:
    """The chain of imports by which ``start`` reaches the analysis package, shortest first found; empty where none
    does, followed to the fixed point."""
    came_from: dict[str, str | None] = {start: None}
    queue = [start]
    while queue:
        module = queue.pop(0)
        if _is_analysis(module):
            chain = [module]
            while (before := came_from[chain[-1]]) is not None:
                chain.append(before)
            return chain[::-1]
        for imported in sorted(graph.get(module, ())):
            if imported not in came_from:
                came_from[imported] = module
                queue.append(imported)
    return []


class TestTheEngineRule:
    """In the trees whose imports never reach the analysis package a ``numbers`` primitive is the route; anything else
    that decides a form there is a site, as anywhere."""

    def test_no_engine_module_reaches_the_analysis_package(self):
        """The rule holds only while no page's forms are reachable there: an engine module whose imports, followed
        through the package to the fixed point, reach the analysis package is above the line, and its spellings
        route through ``PageForms`` like any other. A tree joins the engine whole or not at all."""
        graph = _import_graph()
        engine = [module for module in graph if _in_engine(f"{module.replace('.', '/')}/")]
        assert len(engine) > 100  # every tree is read, not only the first
        routes = {module: route for module in engine if (route := _route_to_analysis(graph, module))}
        assert routes == {}, "\n".join(" -> ".join(route) for route in routes.values())

    def test_a_route_to_the_analysis_package_is_followed_through_every_import(self):
        graph = {
            "threetears.evals.run.launch": ["threetears.evals.testing"],
            "threetears.evals.testing": ["threetears.evals.testing.jobs"],
            "threetears.evals.testing.jobs": ["threetears.evals.ops"],
            "threetears.evals.ops": ["threetears.evals.analysis.report.forms"],
            "threetears.evals.kernel.numbers": ["threetears.evals.kernel"],
            "threetears.evals.kernel": ["threetears.evals.kernel.numbers"],
        }
        assert _route_to_analysis(graph, "threetears.evals.run.launch") == [
            "threetears.evals.run.launch",
            "threetears.evals.testing",
            "threetears.evals.testing.jobs",
            "threetears.evals.ops",
            "threetears.evals.analysis.report.forms",
        ]
        assert _route_to_analysis(graph, "threetears.evals.kernel.numbers") == []

    def test_the_import_graph_reads_packages_and_names_taken_from_them(self, tmp_path):
        files = {
            "threetears/evals/run/__init__.py": "from threetears.evals.run.launch import go\n",
            "threetears/evals/run/launch.py": "from ..testing import Recorded\n\ndef go():\n    import json\n",
            "threetears/evals/testing/__init__.py": "from .jobs import Recorded\n",
            "threetears/evals/testing/jobs.py": "from threetears.evals import ops\n",
            "threetears/evals/ops.py": "x = importlib.import_module('threetears.evals.analysis')\n",
            "threetears/evals/analysis/__init__.py": "",
        }
        for name, text in files.items():
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text(text, encoding="utf-8")
        graph = _import_graph(tmp_path)
        assert graph["threetears.evals.run.launch"] == {"threetears.evals.testing"}
        assert graph["threetears.evals.testing.jobs"] == {"threetears.evals.ops"}
        assert _route_to_analysis(graph, "threetears.evals.run") == [
            "threetears.evals.run",
            "threetears.evals.run.launch",
            "threetears.evals.testing",
            "threetears.evals.testing.jobs",
            "threetears.evals.ops",
            "threetears.evals.analysis",
        ]

    def test_the_engine_scope_only_narrows(self):
        """Six trees at most, each a top-level package of its own: a tree joins only by an owner's decision, which
        raises the ceiling in the same change."""
        assert len(_ENGINE_TREES) <= _ENGINE_TREES_CEILING == 6
        assert len(set(_ENGINE_TREES)) == len(_ENGINE_TREES)
        for tree in _ENGINE_TREES:
            assert (_SOURCE_ROOT / tree).is_dir(), tree
            assert tree.startswith(f"{_PACKAGE}/") and tree.count("/") == _PACKAGE.count("/") + 1, tree
        assert _in_engine("threetears/evals/run/launch.py")
        assert not _in_engine("threetears/evals/runner.py")
        assert not _in_engine("threetears/evals/analysis/report/forms.py")

    @pytest.mark.parametrize(
        "source",
        [
            "from threetears.evals.analysis.report.forms import PageForms\n",
            "from threetears.evals import analysis\n",
            "import threetears.evals.analysis.quantities as q\n",
            "from ..analysis import quantities\n",
            "def f():\n    from threetears.evals.analysis import aspects\n",
            "if TYPE_CHECKING:\n    from threetears.evals.analysis.report import forms\n",
            "m = importlib.import_module('threetears.evals.analysis.quantities')\n",
        ],
    )
    def test_every_way_of_importing_the_analysis_package_is_seen(self, source):
        assert _imports_analysis("threetears/evals/run/launch.py", source) != []

    @pytest.mark.parametrize(
        "source",
        [
            "from threetears.evals.kernel.numbers import format_number\n",
            "from threetears.evals.ops import analyst\n",
            "from .analysis_measures import BarAdjudication\n",
            "import threetears.evals.analysis_tools\n",
        ],
    )
    def test_what_is_not_the_analysis_package_is_not_taken_for_it(self, source):
        assert _imports_analysis("threetears/evals/kernel/surface.py", source) == []

    def test_a_numbers_primitive_is_the_route_in_the_engine_and_a_site_above_it(self):
        """Called, handed on, a default or a method of the rule's: none is a site in the engine, and a function
        returning one is no wrapper there. The same text above the line is a site at every one."""
        body = (
            "from threetears.evals.kernel.numbers import format_number\n\n"
            "def refusal(v):\n    return f'{format_number(v)} over the cap'\n\n"
            "def message(v):\n    return refusal(v)\n\n"
            "SPELL = {'n': format_number}\n\n"
            "def ends(form, v, spell=format_number):\n    return form.spell(v)\n"
        )
        engine = "threetears/evals/run/launch.py"
        above = "threetears/evals/ops/launch_words.py"
        sites = _walk_python({engine: body, above: body})
        assert {site.module for site in sites} == {above}
        assert [(site.qualname, site.form) for site in sites] == [
            ("refusal", "format_number()"),
            ("message", "refusal()"),
            ("<module>", "format_number()"),
            ("ends", "format_number()"),
            ("ends", "spell()"),
        ]

    def test_a_spec_round_quantize_and_log10_stay_sites_in_the_engine(self):
        source = (
            "def waited(s):\n    return f'{s:.1f} s'\n\n"
            "def message(s):\n    return 'after ' + waited(s)\n\n"
            "def places(v, step):\n"
            "    return round(v, 2), Decimal(v).quantize(step), math.floor(math.log10(v))\n"
        )
        assert _forms(_python_sites(source, "threetears/evals/kernel/job_timeout.py")) == [
            ("waited", "f-string spec", "s"),
            ("message", "waited()", "waited(s)"),
            ("places", "floor(log10)", "math.floor(math.log10(v))"),
            ("places", "quantize()", "Decimal(v).quantize(step)"),
            ("places", "round()", "round(v, 2)"),
        ]

    def test_a_page_module_calling_an_engine_helper_is_a_site(self):
        """``numbers`` is the route only for a call made in the engine: an engine helper that returns a primitive's
        text is a wrapper to a page module, so a page cannot borrow the engine's route to decide a figure. Inside the
        engine, its own callers stay no site."""
        sources = {
            "threetears/evals/run/budget.py": ("def cap_words(v):\n    return f'a cap of {format_number(v)}'\n"),
            "threetears/evals/analysis/report/notes.py": (
                "from threetears.evals.run.budget import cap_words\n\ndef note(v):\n    return cap_words(v)\n"
            ),
            "threetears/evals/run/launch.py": (
                "from threetears.evals.run.budget import cap_words\n\n"
                "def refusal(v):\n    return 'over ' + cap_words(v)\n"
            ),
        }
        assert [(site.module, site.qualname, site.form) for site in _walk_python(sources)] == [
            ("threetears/evals/analysis/report/notes.py", "note", "cap_words()")
        ]


class TestTheWalkSeesEveryKitForm:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("function a(v){return v.toFixed(2);}", ("a", "toFixed()", ".toFixed(2)")),
            ("function a(v){return v.toPrecision(12);}", ("a", "toPrecision()", ".toPrecision(12)")),
            ('function a(v){return v.toLocaleString("en-US");}', ("a", "toLocaleString()", '.toLocaleString("en-US")')),
            (
                'function a(v){return new Intl.NumberFormat("en-US").format(v);}',
                ("a", "Intl.", 'Intl.NumberFormat("en-US")'),
            ),
            ("function a(v){return Math.round(v);}", ("a", "Math.round()", "Math.round(v)")),
            ("function a(v){return fmt.usd(v);}", ("a", "fmt.", "fmt.usd(v)")),
            ("function a(v){return K_FMTS.pct(v,1);}", ("a", "K_FMTS", "K_FMTS.pct(v,1)")),
            ('function a(d){return namedFmt(d.fmt||"pct:1");}', ("a", "namedFmt", 'namedFmt(d.fmt||"pct:1")')),
            ('function a(){return kTickFmt("pct");}', ("a", "kTickFmt", 'kTickFmt("pct")')),
            ('function a(t){return kTickLabels("pct",t);}', ("a", "kTickLabels", 'kTickLabels("pct",t)')),
            ("function a(spec){return namedFmt(spec);}", ("a", "namedFmt", "namedFmt(spec)")),
            ("function a(v){return kShort(v,1);}", ("a", "kShort", "kShort(v,1)")),
        ],
    )
    def test_the_walk_sees_every_way_the_kit_spells_a_number(self, source, expected):
        assert _forms(_js_sites(source)) == [expected]

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("const b=v=>v.toFixed(1);", "b"),
            ("let c;const b=v=>v.toFixed(1);", "b"),
            ("async function b(v){return v.toFixed(1);}", "b"),
            ("function b(v){const f=x=>x.toFixed(1);return f(v);}", "b"),
            ("document.title=String(v.toFixed(1));", "<top>"),
            ("function a(){}\nif(on)a(v.toFixed(1));", "<top>"),
            ("function b(v){return `${v.toFixed(1)} ms`;}", "b"),
            ("function b(v){return `${`${v.toFixed(1)}`}` + `${{a:1}.a}`;}", "b"),
        ],
    )
    def test_a_kit_site_is_keyed_by_its_top_level_statement(self, source, expected):
        assert [site.qualname for site in _js_sites(source)] == [expected]

    @pytest.mark.parametrize(
        "source",
        [
            'const s="v.toFixed(2)";',
            "const s='Math.round(v)';",
            "const s=`v.toFixed(2) ${x}`;",
            "// v.toFixed(2)\nconst a=1;",
            "/* Intl.NumberFormat */ const a=1;",
            "const r=/toFixed\\(\\d\\)/g;",
            "const r=/[/]toFixed(/;",
            "function a(d,v){return opt.fmt(v)+d.fmt+f(v);}",
            "const fmt={pct:(v,d=1)=>v.toFixed(d)+'%'};",
            "const K_FMTS={int:v=>Math.round(v).toLocaleString('en-US'),pct:(v,d)=>kShort(v,d)};",
            "function namedFmt(spec){return spec;}",
            "const kShort=(v,d)=>v;",
            "const K_STEP_FMTS=new Set(['num']);",
            # The owners' own bodies: a table's digits, a pick by name, an axis's ticks.
            "const kShort=(v,d)=>d==null?String(+(+v).toFixed(2)):(+v).toFixed(d);",
            "function namedFmt(spec){const f=K_FMTS[spec];return v=>f(v,1);}",
            "const kTickFmt=spec=>namedFmt(spec.replace(/:\\d$/,''));",
            "function kTickLabels(spec,ticks){return ticks.map(t=>Math.round(t)+kTickFmt(spec)(t));}",
        ],
    )
    def test_the_walk_leaves_what_the_kit_does_not_decide(self, source):
        assert _js_sites(source) == []

    @pytest.mark.parametrize(
        "source",
        [
            "function a(d){return namedFmt(d.fmt);}",
            "function a(d){const f=d.cost_fmt?namedFmt(d.cost_fmt):null;return f;}",
            "function a(d){return kTickFmt( d.fmt );}",
            "function a(d,tv){return kTickLabels(d.fmt,tv);}",
            "function a(d){return namedFmt(d.x_fmt)+namedFmt(d.latency_fmt);}",
        ],
    )
    def test_a_pick_by_a_data_field_is_not_a_site(self, source):
        """The data's format is decided in Python, which the walk reads; the kit only applies it."""
        assert _js_sites(source) == []

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ('function a(d){return namedFmt(d.fmt||"pct:1");}', 'namedFmt(d.fmt||"pct:1")'),
            ('function a(d){return namedFmt(d.on?"pct:1":d.fmt);}', 'namedFmt(d.on?"pct:1":d.fmt)'),
            ('function a(){return namedFmt("usd:0");}', 'namedFmt("usd:0")'),
            ("function a(fmts){return namedFmt(fmts[0]);}", "namedFmt(fmts[0])"),
            ("function a(){return namedFmt(K_DEF.fmt);}", "namedFmt(K_DEF.fmt)"),
            ("function a(o){return namedFmt(o.chart.x_fmt);}", "namedFmt(o.chart.x_fmt)"),
            ("function a(d){return namedFmt(d.label);}", "namedFmt(d.label)"),
        ],
    )
    def test_a_pick_the_kit_decides_is_a_site(self, source, expected):
        """A literal, a default beside the data's field, or anything but the field itself is the kit's choice."""
        assert _forms(_js_sites(source)) == [("a", "namedFmt", expected)]

    def test_a_call_to_an_owner_outside_the_owners_is_still_a_site(self):
        source = "const kShort=(v,d)=>(+v).toFixed(d);\nfunction a(v){return kShort(v,1)+' ms';}\n"
        assert _forms(_js_sites(source)) == [("a", "kShort", "kShort(v,1)")]

    def test_a_function_returning_a_kit_site_makes_its_callers_sites(self):
        source = (
            "const kAt=v=>v.toFixed(1);\n"
            "function label(v){return kAt(v)+' ms';}\n"
            "function draw(v){el.textContent=label(v);return 1;}\n"
            "function clean(v){const t=+v.toFixed(10);return t;}\n"
            "function axis(v){return clean(v);}\n"
        )
        assert _forms(_js_sites(source)) == [
            ("kAt", "toFixed()", ".toFixed(1)"),
            ("label", "kAt()", "kAt(v)"),
            ("draw", "label()", "label(v)"),
            # A site held in a body and not returned (a float cleaned before placing a tick) wraps nothing.
            ("clean", "toFixed()", ".toFixed(10)"),
        ]

    def test_a_division_is_not_read_as_a_regex(self):
        source = "function a(v,w){return v/2+w/3+v.toFixed(1);}"
        assert _forms(_js_sites(source)) == [("a", "toFixed()", ".toFixed(1)")]
