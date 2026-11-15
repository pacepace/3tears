"""Figure keys: a report document's prose names a figure by its place in the report input, and code writes the number.

The analyst never types a figure into a sentence (GENERATOR.md §6.2). It writes a **figure key** —
``{<path>|<filter>|<filter>}`` — and :func:`render` replaces each with the value read from the
:class:`~threetears.evals.analysis.report_input.ReportInput`, spelled by the package's number rule
(:mod:`threetears.evals.kernel.numbers`, :mod:`threetears.evals.analysis.quantities`). ``{arm:<id>}`` prints an
arm's short name with its letter tag.

**Single braces only.** A double brace (``{{…}}``, the retired figure reference's grammar) is refused, never read
as a key.

**A path is matched, never split.** Ids carry dots (an arm ``qwen3.8-flash``, a dimension
``tone.kept``) and spaces, so a path is matched against a closed vocabulary built from the input: a family keyword,
then the ids that family takes (arms, aspects, measures, dimensions, strata, read from the input), then a statistic
from a fixed set. Every complete reading of the path is found. One is the figure. Two are an ambiguity, refused
naming both. None is refused naming the valid continuations at the point the path stopped matching. Each refusal is
an :class:`~threetears.evals.analysis.errors.UnresolvableReference`, so the analyst, told why, can correct it.

**Derived values are descriptive.** ``cost.uplift.<arm>``, ``cost.uplift_range``, every ``loss_share``, every
``flip_margin``, ``weakest.*``, a slice's ``vs_control``, a floor's ``held_above`` and ``broken_at`` and a criterion
figure's ``gap`` are computed from the input's point figures. They carry no interval and no test, the grammar
says so, and their :class:`Figure` is marked ``derived``. Everything else is copied from the input.

**A trade is priced too.** ``price.<from>.<to>.<gain>.<field>`` prices a move between two options on one gain
axis (:func:`gain_axis`): the gain, signed toward better, the cost gap per 1,000 of what the cost counts
(:func:`cost_basis`), and the price per point
gained, whose low and high ends are read off the gain's interval where a contrast states one (an end the interval
leaves open is refused as unbounded, never written as a number). ``cost_per_month.<arm>`` and the ``monthly_`` price
fields read the same at the declared ``meta.volume_per_month``. All are derived and marked so.

**A cost some trials did not report.** ``cost.unreported.<arm>.<field>`` reads the page's cost
(``cost_per_1k.<arm>``) where some of the arm's trials report no cost: ``missing`` counts them (the trials the arm's
scoreboard says ran, less the cost measure's observations), ``free`` is the cost per 1,000 had each of them cost
nothing (the mean times observations over trials run: the least the arm's mean over every trial can be), and
``at_p95`` had each cost the arm's own 95th percentile, a what-if, not a ceiling: the snapshot holds no highest
cost, so nothing caps what a missing trial could have cost. Its inputs are the measure the page's cost is read from,
overall on the arm (its mean, observations ``n`` and ``p95``), and the arm's ``scoreboard.<arm>.ran``. It reads each
observation as one trial's, so it refuses, as unstated, a cost the page reads from the scoreboard or blends over
tasks (no observations are counted), an arm whose measure or scoreboard leaves ``n`` or ``ran`` unstated, and a
measure with more observations than trials run, whose observations are not one per trial. All derived.

**Two arms apart, and the cases that would settle a reading.** ``spread.<arm>.<measure>.<where>.*`` reads an arm's
spread on a figure, ``compare.<arm>.<arm>.<measure>.<where>.*`` the difference between two arms with its interval (a
contrast's own where one pairs them, else independent), and ``settle.*`` the cases a design needs to separate an arm
from a declared criterion's threshold, a hypothetical line (``settle.hypothetical.<line>.``, the line a number in the
key), or another arm (:mod:`~threetears.evals.analysis.separation`): one-sided against a line, with the two-sided
size beside it, and two-sided between arms, each ``level`` a text that says which. A size counts cases, each run as
often as today, with the trials it takes beside it. ``settle.margin.`` and ``settle.floor.`` settle a candidate's
change from the control against minus an aspect's margin or a floor's, on the reading's own interval. Derived, read
off stated spreads: a normal approximation, never a test the pipeline ran. ``assumed.<cases>.`` (or
``assumed.share.``) before a spread, compare or settle key reads it as if each slice whose figure states no count held
that many cases (or the arm's own distinct cases times the slice's traffic share): a what-if the key names, refused overall and
where the figures state their count.

**An overall that is the plain mean of its cells.** ``overall_interval.<measure>.<arm>.*`` reads the interval on an
arm's overall figure that states none, where that figure is, by the numbers the input states, the unweighted mean of
one measure's target-matrix cells and every cell states its own spread
(:mod:`~threetears.evals.analysis.cell_mean`): its standard error the root of the cells' squared standard errors over
their count, at the cells' stated level. A criterion figure's ``low`` and ``high`` read the same interval there. All
derived, and refused, as unstated with why, wherever the figure does not qualify.

**What else a sentence counts or compares.** ``count.examples`` and ``count.failures`` (per aspect and arm) say how
many trials the snapshot froze as examples and how many trials the run failed, for any cause;
``overlap.<aspect>.<aspect>.`` whether two aspects ran the same cases; ``confusion.<axis>.<arm>.<expected>.<predicted>``
one cell of a classifier's confusion; ``cost.gap.<arm>.<arm>.`` two arms' costs apart, per 1,000, per month and as a
multiple. A count the input
does not hold (an uncounted input counts no failures, a bare one holds no case ids) is refused as unstated, never read
as none.

**Calls that went unanswered, beside the cost.** ``calls.unanswered.<arm>`` (and ``aspects.<aspect>.`` before it)
reads how many of an arm's calls went unanswered, retried and lost together, summed over every usage role that counted
them, with ``.retried``, ``.lost``, ``.calls`` (the answered ones), ``.share`` (of every call sent), ``.reasons`` and
``.uncounted`` (trials whose calls counted none) beside it. Copied where one role counted them; a share over several
roles is computed and marked derived. An arm no role counted for is no key at all: not counted is never none.

**How a trial passes.** ``aspects.<aspect>.pass_rule`` reads the label of the aspect's pass rule, which names every
check that fails a trial besides the goal checks (``Never topped up is false``), ``.pass_threshold`` the level a judged
dimension must reach, ``.no_criterion`` the trials that passed with nothing to fail them, ``.graded_by`` judged or
checked, and ``.description`` the rule in a sentence. An input that states no rule refuses them as unstated.

**What a contrast counted.** ``contrast.<arm>.<measure>.n_cases`` reads the cases the contrast's test read, one mean
per case, ``.control_n_cases`` the control's where they differ, ``.paired`` the word ``paired`` or ``unpaired``, and
``.pairing_r`` the arms' correlation across the paired cases. An input that states none refuses them as unstated.

**A figure held on more than one aspect is named by its aspect.** Every figure that sits on an aspect has an
``aspects.<aspect>.`` form. Where the input holds a contrast's arm and measure, a stratum id or a contrast family on
more than one aspect, the unqualified form is refused, naming the qualified key for each aspect, rather than reading
one of them.

**Digits only inside keys (D3).** :func:`literal_numbers` finds every digit that report prose types outside a key,
outside a name the input holds, outside a percentile's name (``p95``) and outside one of the package's own unit
phrases (``per 1,000 replies``, and ``per 1,000 conversations`` where the input's costs count conversations). A
number a reader sees is code's.

**The grammar is taught by its own code.** :func:`figure_key_grammar` writes the analyst's reference section from the
parser's own families, statistics and filters, with examples rendered on a synthetic input, so the section can never
teach a form this parser refuses. A reader that takes the grammar a page at a time reads the same lines:
:func:`figure_key_index` (the rules and filters, and a line per group of families) and :func:`figure_key_pages` (each
group's forms).

The grammar's code lives in :mod:`threetears.evals.analysis.report.keys`, one module per part; this module names
what hosts import from it.
"""

from __future__ import annotations

from threetears.evals.analysis.report.keys.figure import (
    FILTER_SPELLINGS,
    Figure,
    FigureKey,
    PERCENTILE_NAMES,
    written_arm,
)
from threetears.evals.analysis.report.keys.grammar import figure_key_grammar, figure_key_index, figure_key_pages
from threetears.evals.analysis.report.keys.levers import arm_settings, lever_of
from threetears.evals.analysis.report.keys.names import (
    NamedKind,
    input_check_name,
    page_arm_names,
    printed_arm_name,
    reader_name,
    reader_names,
)
from threetears.evals.analysis.report.keys.prices import (
    POOL_WEIGHT_WORDS,
    PriceFrontier,
    SLICE_COST_WORDS,
    SliceKind,
    gain_axis,
    pooled_figures,
    price_frontier,
)
from threetears.evals.analysis.report.keys.reads import (
    aspect_costs_per_thousand,
    cost_basis,
    cost_in_measure_unit,
    cost_ruler,
    criterion_ruler,
)
from threetears.evals.analysis.report.keys.resolve import (
    KeySubjects,
    cost_arms,
    figure_filters,
    figure_unit,
    format_figure,
    held_figure,
    held_kind,
    is_figure_key,
    is_rubric_score,
    key_form,
    key_subjects,
    keys_in,
    literal_numbers,
    parse_key,
    plus_signed,
    render,
    resolve,
)
from threetears.evals.analysis.report.keys.separation import correction_words, pairing_contrast
from threetears.evals.analysis.report.keys.settle import assumable, criterion_wheres, passed_aspects, separation_wheres
from threetears.evals.analysis.report.keys.vocabulary import joint_guarantee, joint_level, on_lead_aspect
from threetears.evals.analysis.report.keys.weakest import WEAKEST_PREFIX
from threetears.evals.analysis.report_margins import flip_margin

__all__ = [
    "aspect_costs_per_thousand",
    "cost_in_measure_unit",
    "held_figure",
    "cost_arms",
    "cost_ruler",
    "FILTER_SPELLINGS",
    "PERCENTILE_NAMES",
    "WEAKEST_PREFIX",
    "Figure",
    "FigureKey",
    "PriceFrontier",
    "KeySubjects",
    "NamedKind",
    "arm_settings",
    "assumable",
    "criterion_wheres",
    "POOL_WEIGHT_WORDS",
    "SLICE_COST_WORDS",
    "SliceKind",
    "figure_key_grammar",
    "figure_key_index",
    "figure_filters",
    "figure_unit",
    "figure_key_pages",
    "cost_basis",
    "criterion_ruler",
    "flip_margin",
    "format_figure",
    "held_kind",
    "input_check_name",
    "gain_axis",
    "is_figure_key",
    "is_rubric_score",
    "joint_guarantee",
    "joint_level",
    "key_form",
    "key_subjects",
    "keys_in",
    "lever_of",
    "literal_numbers",
    "parse_key",
    "pooled_figures",
    "price_frontier",
    "page_arm_names",
    "plus_signed",
    "printed_arm_name",
    "written_arm",
    "reader_name",
    "reader_names",
    "render",
    "resolve",
    "separation_wheres",
    "on_lead_aspect",
    "pairing_contrast",
    "passed_aspects",
    "correction_words",
]
