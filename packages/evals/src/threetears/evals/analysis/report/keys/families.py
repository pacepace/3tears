"""Every family a key's path is matched against, in the one order the refusals, the grammar and its pages read
them in, and the groups the grammar's pages show them in.
"""

from __future__ import annotations

from threetears.evals.analysis.report.keys.counts import (
    _BAR_FIELDS,
    _CELL_FIELDS,
    _COUNTED,
    _FLOOR_DERIVED,
    _FLOOR_FIELDS,
    _OVERLAP_FIELDS,
    _PAIR_STATISTICS,
    _bar,
    _bar_arms,
    _bar_aspects,
    _bar_measures,
    _campaign_count,
    _cell_arms,
    _cell_field,
    _cell_figure,
    _cell_keys,
    _confusion,
    _confusion_arms,
    _confusion_axes,
    _confusion_labels,
    _criterion_ids,
    _example_arms,
    _example_aspects,
    _example_strata,
    _factor_ids,
    _failed_count,
    _failed_trials_count,
    _failing_count,
    _floor,
    _floor_arms,
    _floor_aspects,
    _floor_keys,
    _gate_ids,
    _kept_count,
    _levels_count,
    _other_aspects,
    _overlap,
    _pair,
    _pair_firsts,
    _pair_seconds,
    _passing_count,
    _places_count,
    _unknown_count,
)
from threetears.evals.analysis.report.keys.levers import (
    _INTERACTION_FIELDS,
    _LEVER,
    _LEVER_BASIS,
    _LEVER_FIELDS,
    _LEVER_NOTE,
    _OVERALL_INTERVAL_NOTE,
    _PLACE,
    _POOLED_NOTE,
    _PRICE_NOTE,
    _SECOND_MOVE,
    _interaction,
    _lever,
)
from threetears.evals.analysis.report.keys.prices import (
    _COST_GAP_FIELDS,
    _OVERALL_INTERVAL_FIELDS,
    _POOLED_FIELDS,
    _PRICE_FIELDS,
    _SLICE_PRICE_FIELDS,
    _UNANSWERED_FIELDS,
    _UNANSWERED_RULE,
    _UNREPORTED_FIELDS,
    _aspect_unanswered,
    _aspect_unanswered_arms,
    _cost_gap,
    _cost_gap_seconds,
    _cost_per_month,
    _frontier_count,
    _gain_arms,
    _gain_axes,
    _gap_compares,
    _means_of,
    _overall_interval,
    _pooled,
    _pooled_arms,
    _pooled_keys,
    _price,
    _slice_arms,
    _slice_compares,
    _slice_price,
    _slice_seconds,
    _steps_count,
    _unanswered,
    _unanswered_arms,
    _unreported,
    _unspread_arms,
    _unspread_keys,
)
from threetears.evals.analysis.report.keys.reads import (
    _ARM_FIELDS,
    _ASPECT_FIELDS,
    _A_REPLY_COST,
    _CONTRAST_STATISTICS,
    _CRITERION_DERIVED,
    _CRITERION_FIELDS,
    _CRITERION_FIGURE_FIELDS,
    _DEPTH_FIELDS,
    _FAILURES_RULE,
    _FAMILY_FIELDS,
    _JUDGE_FIELDS,
    _MEASURE_STATISTICS,
    _META_FIELDS,
    _READING_LEDGER,
    _READING_STATISTICS,
    _SCOREBOARD_FAILED,
    _SCOREBOARD_STATISTICS,
    _SLICE_DERIVED,
    _SLICE_STATISTICS,
    _STRATUM_FIELDS,
    _TASK_FIELDS,
    _a_bar_s_level,
    _a_floor_s_rate,
    _arm_depth,
    _asked_aspect_field,
    _asked_reading,
    _aspect_board_arms,
    _aspect_contrast,
    _aspect_contrast_arms,
    _aspect_contrasted,
    _aspect_depth,
    _aspect_family,
    _aspect_flip_margin,
    _aspect_judged,
    _aspect_judged_arms,
    _aspect_judged_dimensions,
    _aspect_measure,
    _aspect_measure_arms,
    _aspect_measures,
    _aspect_scoreboard,
    _cases,
    _contrast,
    _contrasted,
    _control_field,
    _cost_per_1k,
    _costs_per_thousand,
    _criterion_arm,
    _criterion_arm_at,
    _criterion_arms,
    _criterion_field,
    _criterion_places,
    _depth_arms,
    _family,
    _family_places,
    _flip_margin,
    _indexed_family,
    _judge,
    _judged,
    _judged_arms,
    _judged_field,
    _judged_stat,
    _late_criterion_set_at,
    _late_names,
    _late_set_at,
    _latency,
    _loss_keys,
    _loss_share,
    _measure,
    _measure_arms,
    _meta,
    _n_candidates,
    _named_aspect_field,
    _named_reading,
    _pass_rate,
    _passing,
    _reasoning_arms,
    _reasoning_ratio,
    _repeats,
    _scoreboard,
    _setting,
    _setting_arms,
    _setting_names,
    _strata_resolver,
    _stratum_arms,
    _stratum_field,
    _stratum_figure,
    _stratum_keys,
    _task_arms,
    _task_field,
    _task_figure,
    _task_measures,
    _tasks,
    _the_page_s_cost,
    _the_pass_rate,
    _uplift,
    _uplift_range,
)
from threetears.evals.analysis.report.keys.separation import _asked_reading_arms, _aspects, _named_reading_arms
from threetears.evals.analysis.report.keys.settle import (
    _ARM,
    _ASPECT_STRATUM,
    _ASSUMED_FAMILIES,
    _CANDIDATE,
    _CELL,
    _ON_ASPECT,
    _SEPARATION_FAMILIES,
    _STRATUM,
    _family_of,
    _keyed,
)
from threetears.evals.analysis.report.keys.vocabulary import _Family, _Ids, _Statistic, _Word, _candidates
from threetears.evals.analysis.report.keys.weakest import _WEAKEST_FIELDS, _weakest


#: Every path family, in the order the grammar teaches them.
_FAMILIES: tuple[_Family, ...] = (
    _family_of(_Word("meta"), _Statistic(_META_FIELDS, "<field>"), what="the campaign's own words", resolve=_meta),
    _family_of(_Word("control"), _Statistic(_ARM_FIELDS, "<field>"), what="the control arm", resolve=_control_field),
    _family_of(_Word("n_candidates"), what="how many candidates ran", resolve=_n_candidates),
    _family_of(
        _Word("aspect"),
        _Statistic(_ASPECT_FIELDS, "<field>"),
        what="the one asked aspect",
        resolve=_asked_aspect_field,
        rate=_the_pass_rate,
    ),
    _family_of(
        _Word("aspects"),
        _Ids("<aspect>", _aspects),
        _Statistic(_ASPECT_FIELDS, "<field>"),
        what="any aspect, by id",
        resolve=_named_aspect_field,
        rate=_the_pass_rate,
    ),
    _family_of(
        _Word("reading"),
        _Ids("<arm>", _asked_reading_arms),
        _Statistic(_READING_STATISTICS),
        what="an arm's reading on the asked aspect: its change from the control, in points",
        resolve=_asked_reading,
        ledger_as=_READING_LEDGER,
        rate=_the_pass_rate,
    ),
    _family_of(
        _Word("aspects"),
        _Ids("<aspect>", _aspects),
        _Word("reading"),
        _Ids("<arm>", _named_reading_arms),
        _Statistic(_READING_STATISTICS),
        what="an arm's reading on any aspect",
        resolve=_named_reading,
        ledger_as=_READING_LEDGER,
        rate=_the_pass_rate,
    ),
    _family_of(
        _Word("contrast"),
        _keyed("<arm>", lambda ri: dict.fromkeys(c.arm for c in ri.contrasts)),
        _Ids("<measure>", _contrasted),
        _Statistic(_CONTRAST_STATISTICS),
        what="a corrected contrast of an arm against the control on a measure or judged dimension",
        resolve=_contrast,
        compares=lambda ri, values: _means_of(ri, values[1]),
    ),
    _family_of(
        _Word("judged"),
        _keyed("<dimension>", lambda ri: ri.judged),
        _Ids("<arm>", _judged_arms),
        what="an arm's mean on a judged dimension",
        resolve=_judged,
    ),
    _family_of(
        _Word("judged"),
        _keyed("<dimension>", lambda ri: ri.judged),
        _Statistic(("label", *_JUDGE_FIELDS), "<field>"),
        what="a judged dimension's judge",
        resolve=_judged_field,
    ),
    _family_of(
        _Word("pass_rate"),
        _keyed("<arm>", _passing),
        what="an arm's pass rate: trials passed over trials usable",
        resolve=_pass_rate,
    ),
    _family_of(
        _Word("cost_per_1k"),
        _keyed("<arm>", _costs_per_thousand),
        what="an arm's cost per 1,000 of what the cost counts (`per 1,000 conversations` where the cost measure's "
        "label counts conversations)",
        resolve=_cost_per_1k,
    ),
    _family_of(
        _Word("measure"),
        _keyed("<measure>", lambda ri: ri.measures),
        _Ids("<arm>", _measure_arms),
        _Statistic(_MEASURE_STATISTICS),
        what="an arm's figure on a measure",
        resolve=_measure,
    ),
    _family_of(
        _Word("scoreboard"),
        _keyed("<arm>", lambda ri: ri.scoreboard),
        _Statistic(_SCOREBOARD_STATISTICS),
        what="how an arm's trials came out. " + _SCOREBOARD_FAILED,
        resolve=_scoreboard,
        ledger_as=_A_REPLY_COST,
    ),
    _family_of(
        _Word("calls.unanswered"),
        _keyed("<arm>", _unanswered_arms),
        what="how many of an arm's calls went unanswered, retried and lost together, summed over every role that "
        "counted them: beside its cost, never in it (the cost counts the answered calls); not counted, and refused, "
        "where a role counted only one kind. " + _UNANSWERED_RULE,
        resolve=_unanswered,
        example="calls.unanswered.fast.v2",
    ),
    _family_of(
        _Word("calls.unanswered"),
        _keyed("<arm>", _unanswered_arms),
        _Statistic(_UNANSWERED_FIELDS, "<field>"),
        what="the same, by field: `retried` (a forced tool choice refused, then answered unforced in the same call: "
        "attempts and latency, never a failure), `lost` (sent and never answered, so the turn ended in error: failures "
        "the arm's trials already count), `calls` the answered calls beside them, `share` retried and lost as a share "
        "of every call sent, `reasons` the reasons counted, in words, `uncounted` the trials whose calls counted none",
        resolve=_unanswered,
        example="calls.unanswered.fast.v2.share|pct",
    ),
    _family_of(
        _Word("reasoning_ratio"),
        _Ids("<arm>", _reasoning_arms),
        what="the share of an arm's output spent reasoning",
        resolve=_reasoning_ratio,
    ),
    _family_of(_Word("cases"), what="the cases the asked aspect ran", resolve=_cases),
    _family_of(_Word("repeats"), what="the repeats each case ran", resolve=_repeats),
    _family_of(_Word("late_criteria"), _Word("set_at"), what="when the late criteria were set", resolve=_late_set_at),
    _family_of(_Word("family"), _Statistic(_FAMILY_FIELDS, "<field>"), what="the one contrast family", resolve=_family),
    _family_of(
        _Word("judge"),
        _Statistic(_JUDGE_FIELDS, "<field>"),
        what="the judge, where every judged dimension shares it",
        resolve=_judge,
    ),
    _family_of(_Word("caveats"), _Word("latency"), what="why latency is not compared", resolve=_latency),
    _family_of(
        _Word("task"),
        _Ids("<task>", _tasks),
        _Statistic(_TASK_FIELDS, "<field>"),
        what="a task",
        resolve=_task_field,
    ),
    _family_of(
        _Word("task"),
        _Ids("<task>", _tasks),
        _Ids("<arm>", _task_arms),
        _Ids("<measure>", _task_measures),
        _Statistic(_SLICE_STATISTICS),
        what="an arm's figure on a measure or judged dimension within one task; `vs_control` is its mean minus the "
        "control's there",
        resolve=_task_figure,
        derived_statistics=_SLICE_DERIVED,
    ),
    _family_of(
        _Word("criteria"),
        _keyed("<criterion>", lambda ri: [criterion.id for criterion in ri.criteria]),
        _Statistic(_CRITERION_FIELDS, "<field>"),
        what="a declared criterion: its threshold in its measure's units, the measure, statistic and direction, and "
        "the aspect whose figures it reads (its own, else the lead aspect)",
        resolve=_criterion_field,
    ),
    _family_of(
        *_STRATUM,
        _Statistic(_STRATUM_FIELDS, "<field>"),
        what="a stratum",
        resolve=_strata_resolver(_stratum_field, qualified=False),
    ),
    _family_of(
        _Word("cost.uplift"),
        _CANDIDATE,
        what="how much more a candidate costs than the control, as a share of the control's cost",
        resolve=_uplift,
        derived=True,
    ),
    _family_of(
        _Word("cost.uplift_range"),
        what="the least and the most any candidate's cost uplift is",
        resolve=_uplift_range,
        derived=True,
    ),
    _family_of(
        *_STRATUM,
        _Word("loss_share"),
        what="the least share of any candidate's loss on the primary judged dimension that the stratum carries",
        resolve=_strata_resolver(_loss_share, qualified=False),
        derived=True,
    ),
    _family_of(
        *_STRATUM,
        _Word("loss_share"),
        _Ids("<candidate>", _candidates),
        what="the share of one candidate's loss the stratum carries",
        resolve=_strata_resolver(_loss_share, qualified=False),
        derived=True,
    ),
    _family_of(
        _Word("weakest"),
        _Statistic(_WEAKEST_FIELDS, "<field>"),
        what=(
            "the weakest margin: the figure nearest the line it is read against (a criterion, an aspect's margin or a "
            "floor), its arm, what it is read against, the figure, the line, and the gap between them"
        ),
        resolve=_weakest,
        derived=True,
    ),
    _family_of(
        _Word("flip_margin"),
        _Ids("<candidate>", _asked_reading_arms),
        what="the margin above which a candidate's asked reading holds: its interval's worst drop",
        resolve=_flip_margin,
        rate=_the_pass_rate,
        derived=True,
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("contrast"),
        _Ids("<arm>", _aspect_contrast_arms),
        _Ids("<measure>", _aspect_contrasted),
        _Statistic(_CONTRAST_STATISTICS),
        what="a corrected contrast on one aspect, where `contrast.` reads an arm and measure the input holds on more "
        "than one",
        resolve=_aspect_contrast,
        example="aspects.answers.contrast.fast.v2.tone.kept.delta|pts",
        compares=lambda ri, values: _means_of(ri, values[2]),
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("measure"),
        _Ids("<measure>", _aspect_measures),
        _Ids("<arm>", _aspect_measure_arms),
        _Statistic(_MEASURE_STATISTICS),
        what="an arm's figure on a measure on one aspect",
        resolve=_aspect_measure,
        example="aspects.speed.measure.latency_ms.fast.v2.mean",
    ),
    _family_of(
        _Word("judged"),
        _keyed("<dimension>", lambda ri: ri.judged),
        _Ids("<arm>", _judged_arms),
        _Statistic(_MEASURE_STATISTICS),
        what="one statistic of an arm's figure on a judged dimension",
        resolve=_judged_stat,
        example="judged.tone.kept.base.n",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("judged"),
        _Ids("<dimension>", _aspect_judged_dimensions),
        _Ids("<arm>", _aspect_judged_arms),
        _Statistic(_MEASURE_STATISTICS),
        what="an arm's figure on a judged dimension on one aspect",
        resolve=_aspect_judged,
        example="aspects.answers.judged.tone.kept.fast.v2.mean|pct",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("scoreboard"),
        _Ids("<arm>", _aspect_board_arms),
        _Statistic(_SCOREBOARD_STATISTICS),
        what="how an arm's trials came out on one aspect. " + _SCOREBOARD_FAILED,
        resolve=_aspect_scoreboard,
        example="aspects.speed.scoreboard.fast.v2.passed",
        ledger_as=_A_REPLY_COST,
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("calls.unanswered"),
        _Ids("<arm>", _aspect_unanswered_arms),
        what="how many of an arm's calls went unanswered on one aspect, retried and lost together",
        resolve=_aspect_unanswered,
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("calls.unanswered"),
        _Ids("<arm>", _aspect_unanswered_arms),
        _Statistic(_UNANSWERED_FIELDS, "<field>"),
        what="the same, by field, as `calls.unanswered.<arm>.<field>` reads it",
        resolve=_aspect_unanswered,
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("pair"),
        _Ids("<arm>", _pair_firsts),
        _Ids("<arm>", _pair_seconds),
        _Statistic(_PAIR_STATISTICS),
        what="a pair reading on one aspect: the first arm minus the second, in points, as the input orders them; "
        "`ahead` is the arm shown ahead",
        resolve=_pair,
        rate=_the_pass_rate,
        example="aspects.speed.pair.base.fast.v2.change|pts",
    ),
    _family_of(
        _Word("aspects"),
        _Ids("<aspect>", _floor_aspects),
        _Word("floor"),
        _Ids("<floor>", _floor_keys),
        _Ids("<arm>", _floor_arms),
        _Statistic(_FLOOR_FIELDS, "<field>"),
        what="a floor read on one candidate, on the aspect the read names (one the input may hold no aspect record "
        "for): `held_above` is the margin above which it holds (minus the interval's low end), `broken_at` the margin "
        "at or below which it breaks (minus the high end)",
        resolve=_floor,
        rate=_a_floor_s_rate,
        derived_statistics=_FLOOR_DERIVED,
        example="aspects.answers.floor.answers/rate_drop/*.fast.v2.held_above",
    ),
    _family_of(
        _Word("aspects"),
        _Ids("<aspect>", _bar_aspects),
        _Word("bar"),
        _Ids("<measure>", _bar_measures),
        _Ids("<arm>", _bar_arms),
        _Statistic(_BAR_FIELDS, "<field>"),
        what="an arm against an absolute bar, in the measure's units, on the aspect the read names (one the input may "
        "hold no aspect record for)",
        resolve=_bar,
        example="aspects.speed.bar.latency_ms.fast.v2.value",
        ledger_as=_a_bar_s_level,
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("family"),
        _Statistic(_FAMILY_FIELDS, "<field>"),
        what="the contrast family correcting one aspect's contrasts, where the input holds more than one",
        resolve=_aspect_family,
        example="aspects.answers.family.correction",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("flip_margin"),
        _Ids("<candidate>", _named_reading_arms),
        what="the margin above which a candidate's reading on one aspect holds: its interval's worst drop",
        resolve=_aspect_flip_margin,
        rate=_the_pass_rate,
        derived=True,
        example="aspects.answers.flip_margin.fast.v2",
    ),
    _family_of(
        *_STRATUM,
        _Ids("<arm>", _stratum_arms(qualified=False)),
        _Ids("<measure>", _stratum_keys(qualified=False)),
        _Statistic(_SLICE_STATISTICS),
        what="an arm's figure on a measure or judged dimension within one stratum; `vs_control` is its mean minus the "
        "control's there",
        resolve=_strata_resolver(_stratum_figure, qualified=False),
        derived_statistics=_SLICE_DERIVED,
        example="strata.off_topic.fast.v2.tone.kept.vs_control|pts",
    ),
    _family_of(
        *_STRATUM,
        _Word("loss_share"),
        _Ids("<candidate>", _candidates),
        _Ids("<measure>", _loss_keys(qualified=False)),
        what="the share of one candidate's loss on any measure or judged dimension the stratum carries, a loss "
        "running the way the measure is worse",
        resolve=_strata_resolver(_loss_share, qualified=False),
        derived=True,
        example="strata.off_topic.loss_share.fast.v2.tone.kept|pct",
    ),
    _family_of(
        *_ASPECT_STRATUM,
        _Statistic(_STRATUM_FIELDS, "<field>"),
        what="a stratum of one aspect's cases, where the input holds its id on more than one aspect",
        resolve=_strata_resolver(_stratum_field, qualified=True),
        example="aspects.answers.strata.off_topic.n_cases",
    ),
    _family_of(
        *_ASPECT_STRATUM,
        _Ids("<arm>", _stratum_arms(qualified=True)),
        _Ids("<measure>", _stratum_keys(qualified=True)),
        _Statistic(_SLICE_STATISTICS),
        what="an arm's figure within a stratum of one aspect's cases",
        resolve=_strata_resolver(_stratum_figure, qualified=True),
        derived_statistics=_SLICE_DERIVED,
        example="aspects.answers.strata.off_topic.fast.v2.tone.kept.mean|pct",
    ),
    _family_of(
        *_ASPECT_STRATUM,
        _Word("loss_share"),
        what="the least share of any candidate's loss on the primary judged dimension that a stratum of one aspect "
        "carries",
        resolve=_strata_resolver(_loss_share, qualified=True),
        derived=True,
    ),
    _family_of(
        *_ASPECT_STRATUM,
        _Word("loss_share"),
        _Ids("<candidate>", _candidates),
        what="the share of one candidate's loss a stratum of one aspect carries",
        resolve=_strata_resolver(_loss_share, qualified=True),
        derived=True,
    ),
    _family_of(
        *_ASPECT_STRATUM,
        _Word("loss_share"),
        _Ids("<candidate>", _candidates),
        _Ids("<measure>", _loss_keys(qualified=True)),
        what="the share of one candidate's loss on any measure that a stratum of one aspect carries",
        resolve=_strata_resolver(_loss_share, qualified=True),
        derived=True,
    ),
    _family_of(
        _Word("cells"),
        _CELL,
        _Statistic(_CELL_FIELDS, "<field>"),
        what="a target-matrix cell, written by its row and column level ids joined by a dot (`<row>.<column>`)",
        resolve=_cell_field,
        example="cells.short.en.name",
    ),
    _family_of(
        _Word("cells"),
        _CELL,
        _Ids("<arm>", _cell_arms),
        _Ids("<measure>", _cell_keys),
        _Statistic(_SLICE_STATISTICS),
        what="an arm's figure on a measure or judged dimension within one cell; `vs_control` is its mean minus the "
        "control's there",
        resolve=_cell_figure,
        derived_statistics=_SLICE_DERIVED,
        example="cells.short.en.fast.v2.tone.kept.mean|pct",
    ),
    _family_of(
        _Word("criteria"),
        _keyed("<criterion>", lambda ri: [criterion.id for criterion in ri.criteria]),
        _Ids("<arm>", _criterion_arms),
        _Statistic(_CRITERION_FIGURE_FIELDS, "<field>"),
        what="the one figure a criterion reads for an arm, in its measure's units; `gap` is how far it clears the "
        "threshold, signed toward passing, in points for a rate; `low` and `high` are derived where the input states "
        "no interval and `overall_interval` derives one",
        resolve=_criterion_arm,
        derived_statistics=_CRITERION_DERIVED,
        example="criteria.p95.fast.v2.gap",
    ),
    _family_of(
        _Word("criteria"),
        _keyed("<criterion>", lambda ri: [criterion.id for criterion in ri.criteria]),
        _Ids("<arm>", _criterion_arms),
        _Ids("<place>", _criterion_places),
        _Statistic(_CRITERION_FIGURE_FIELDS, "<field>"),
        what="a criterion's figure for an arm at one place it reads: a task by id, a cell as `<row>.<column>`",
        resolve=_criterion_arm_at,
        derived_statistics=_CRITERION_DERIVED,
        example="criteria.cell_floor.fast.v2.short.en.value|pct",
    ),
    _family_of(
        _Word("settings"),
        _Ids("<arm>", _setting_arms),
        _Ids("<setting>", _setting_names),
        what="one of an arm's settings as the input holds it: a number where it holds a number, else its text",
        resolve=_setting,
        example="settings.base.retries",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("cases"),
        what="the cases one aspect ran: the control's, else the count every arm agrees on",
        resolve=lambda report_input, values, path: _aspect_depth(report_input, (values[0], "cases"), path),
        example="aspects.answers.cases",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("repeats"),
        what="the mean repeats each case of one aspect ran: the control's, else the figure every arm agrees on",
        resolve=lambda report_input, values, path: _aspect_depth(report_input, (values[0], "repeats"), path),
        example="aspects.answers.repeats",
    ),
    _family_of(
        *_ON_ASPECT,
        _Word("depth"),
        _Ids("<arm>", _depth_arms),
        _Statistic(_DEPTH_FIELDS, "<field>"),
        what="how deeply one arm ran one aspect: its distinct cases, its trials, the mean repeats a case got",
        resolve=_arm_depth,
        example="aspects.answers.depth.fast.v2.repeats",
    ),
    _family_of(
        _Word("late_criteria"),
        _Ids("<late>", _late_names),
        _Word("set_at"),
        what="when one criterion was set after the results were seen, named as its caveat id names it after `late:` "
        "(what was set, then `:` and which one where it names one)",
        resolve=_late_criterion_set_at,
        example="late_criteria.floor:answers/rate_drop/*.set_at",
    ),
    _family_of(
        _Word("families"),
        _Ids("<family>", _family_places),
        _Statistic(_FAMILY_FIELDS, "<field>"),
        what="any contrast family by its place in the input's families, counting from 0: the key for one that names "
        "no aspect, or shares its aspect with another",
        resolve=_indexed_family,
        example="families.1.correction",
    ),
    _family_of(
        _Word("price"),
        _keyed("<from>", _costs_per_thousand),
        _Ids("<to>", _gain_arms),
        _Ids("<measure>", _gain_axes),
        _Statistic(_PRICE_FIELDS, "<field>"),
        what="a move from one option to another priced on one gain axis (a measure with a direction, or a judged "
        "dimension): `gain` is how much better the second reads, toward better; `cost_gap` what it costs more per "
        "1,000 of what the cost counts; `per_point` the cost gap per point gained (per unit off a points scale), its "
        "`_low` and `_high` ends read off `gain_interval`, the contrast's interval where the move is between the "
        "control and a candidate; an end the interval leaves open is refused as unbounded; `monthly_` fields at the "
        "declared monthly volume. Cost is read as exact",
        resolve=_price,
        derived=True,
        derived_note=_PRICE_NOTE,
        example="price.base.fast.v2.latency_ms.monthly_per_point",
        compares=lambda ri, values: _gap_compares(ri, _costs_per_thousand(ri), values[-1]),
    ),
    _family_of(
        _Word("task"),
        _Ids("<task>", _tasks),
        _Word("price"),
        _Ids("<from>", _slice_arms("task")),
        _Ids("<to>", _slice_seconds("task")),
        _Ids("<measure>", _gain_axes),
        _Statistic(_SLICE_PRICE_FIELDS, "<field>"),
        what="a move priced within one task, on the arms' figures there: the task's own cost where it states one for "
        "every option it places, else each option's overall cost, as `cost_basis` says; its interval is unpaired, the "
        "two options' own half-widths there combined as independent",
        resolve=_slice_price("task"),
        derived=True,
        derived_note=_PRICE_NOTE,
        compares=_slice_compares("task"),
    ),
    _family_of(
        *_STRATUM,
        _Word("price"),
        _Ids("<from>", _slice_arms("strata")),
        _Ids("<to>", _slice_seconds("strata")),
        _Ids("<measure>", _gain_axes),
        _Statistic(_SLICE_PRICE_FIELDS, "<field>"),
        what="a move priced within one stratum, as within a task",
        resolve=_slice_price("strata"),
        derived=True,
        derived_note=_PRICE_NOTE,
        compares=_slice_compares("strata"),
    ),
    _family_of(
        _Word("cells"),
        _CELL,
        _Word("price"),
        _Ids("<from>", _slice_arms("cells")),
        _Ids("<to>", _slice_seconds("cells")),
        _Ids("<measure>", _gain_axes),
        _Statistic(_SLICE_PRICE_FIELDS, "<field>"),
        what="a move priced within one target-matrix cell, as within a task",
        resolve=_slice_price("cells"),
        derived=True,
        derived_note=_PRICE_NOTE,
        compares=_slice_compares("cells"),
    ),
    _family_of(
        _Word("pooled"),
        _Ids("<measure>", _pooled_keys),
        _Ids("<arm>", _pooled_arms),
        _Statistic(_POOLED_FIELDS, "<field>"),
        what="an arm's figure pooled over the slices holding a measure or judged dimension (its tasks, else its "
        "strata of one stratification, else its cells): their means weighted by traffic share where the input states "
        "one for every slice, else equally, as `weights` says; `half_width` combines the slices' as independent; "
        "`slices` counts them. A price reads it where no overall figure is stated",
        resolve=_pooled,
        derived=True,
        derived_note=_POOLED_NOTE,
        example="pooled.tone.kept.fast.v2.weights",
    ),
    _family_of(
        _Word("overall_interval"),
        _Ids("<measure>", _unspread_keys),
        _Ids("<arm>", _unspread_arms),
        _Statistic(_OVERALL_INTERVAL_FIELDS, "<field>"),
        what="the interval on an arm's overall figure that states none, where that figure is the plain mean of one "
        "measure's target-matrix cells and every cell states a spread: `se` is the root of the cells' squared "
        "standard errors over their count, `interval` the figure plus and minus it at the cells' stated `level`, `of` "
        "the measure averaged, `cells` how many, `weights` whether a case-weighted mean of them fits it too (the "
        "interval then reads the wider standard error), `basis` how it was built. Refused where the figure is no plain mean "
        "of its cells, a cell states no spread, or the levels differ. A gate on the figure reads this interval, and "
        "`criteria.<criterion>.<arm>.low` and `.high` read its ends",
        resolve=_overall_interval,
        derived=True,
        derived_note=_OVERALL_INTERVAL_NOTE,
    ),
    _family_of(
        _Word("cost_per_month"),
        _keyed("<arm>", _costs_per_thousand),
        what="an arm's cost per month at the declared monthly volume",
        resolve=_cost_per_month,
        derived=True,
        example="cost_per_month.base",
    ),
    *_SEPARATION_FAMILIES,
    *_ASSUMED_FAMILIES,
    _family_of(
        *_LEVER,
        _Statistic(_LEVER_FIELDS, "<field>"),
        what="a lever's effect: one setting moved between two arms whose settings differ in it alone, `<to>` minus "
        "`<from>` on a measure or judged dimension (`effect`: a pairing contrast's difference, else the two means'), "
        "overall. " + _LEVER_BASIS,
        resolve=_lever,
        derived=True,
        derived_note=_LEVER_NOTE,
    ),
    _family_of(
        *_LEVER,
        _PLACE,
        _Statistic(_LEVER_FIELDS, "<field>"),
        what="a lever's effect within one task, stratum or cell (`<row>.<column>`). " + _LEVER_BASIS,
        resolve=_lever,
        derived=True,
        derived_note=_LEVER_NOTE,
    ),
    _family_of(
        *_LEVER,
        *_SECOND_MOVE,
        _Statistic(_INTERACTION_FIELDS, "<field>"),
        what="an interaction: the same move made where one other setting differs, its effect minus the first move's "
        "(`difference`), overall. " + _LEVER_BASIS,
        resolve=_interaction,
        derived=True,
        derived_note=_LEVER_NOTE,
    ),
    _family_of(
        *_LEVER,
        _PLACE,
        *_SECOND_MOVE,
        _Statistic(_INTERACTION_FIELDS, "<field>"),
        what="an interaction within one task, stratum or cell. " + _LEVER_BASIS,
        resolve=_interaction,
        derived=True,
        derived_note=_LEVER_NOTE,
    ),
    _family_of(
        _Word("count"),
        _Statistic(_COUNTED, "<what>"),
        what="how many the campaign holds: arms (every option, the control among them), aspects, tasks, strata, "
        "cells, configs, gates (gates and guardrails), criteria; `surviving` is how many options survive every gate "
        "and guardrail, borderline counting as passing; `undecided` how many no gate figure fails while a gate has no "
        "figure for them somewhere: neither passed nor removed; `pairs` how many pairs of arms its pair readings read "
        "apart, `arm_pairs` how many pairs of arms there are; `examples` the failed trials the snapshot froze as "
        "examples, `failures` every trial the pass rule failed, for any cause, the scoreboard's `failed` (refused "
        "where the snapshot does not count them: not known)",
        resolve=_campaign_count,
        derived_statistics=("surviving", "undecided"),
        example="count.arms|words",
    ),
    _family_of(
        _Word("count.passing"),
        _keyed("<criterion>", _gate_ids),
        what="how many options clear one gate or guardrail at every place it reads (a borderline figure clears it)",
        resolve=_passing_count,
        derived=True,
        example="count.passing.p95",
    ),
    _family_of(
        _Word("count.places"),
        _keyed("<criterion>", _criterion_ids),
        what="how many places a criterion reads an arm's figure at: one overall, else each task or cell it reads in",
        resolve=_places_count,
        example="count.places.cell_floor",
    ),
    _family_of(
        _Word("count.failing"),
        _keyed("<criterion>", _criterion_ids),
        _ARM,
        what="how many of a criterion's places an arm misses: a figure on the wrong side of the threshold (a place "
        "with no figure is unknown, counted by `count.unknown`)",
        resolve=_failing_count,
        derived=True,
        example="count.failing.cell_floor.fast.v2|words",
    ),
    _family_of(
        _Word("count.unknown"),
        _keyed("<criterion>", _criterion_ids),
        _ARM,
        what="how many of a criterion's places hold no figure for an arm: unknown there, neither passed nor missed",
        resolve=_unknown_count,
        derived=True,
        example="count.unknown.p95.fast.v2|words",
    ),
    _family_of(
        _Word("count.failed"),
        _ARM,
        what="how many gates and guardrails an arm misses on a figure somewhere they read",
        resolve=_failed_count,
        derived=True,
        example="count.failed.fast.v2",
    ),
    _family_of(
        _Word("count.frontier"),
        _Ids("<measure>", _gain_axes),
        what="how many options are best at some price per point on a gain axis: the price frontier's options",
        resolve=_frontier_count,
        derived=True,
        example="count.frontier.tone.kept|words",
    ),
    _family_of(
        _Word("count.steps"),
        _Ids("<measure>", _gain_axes),
        what="how many steps the price frontier on a gain axis takes: the prices at which the best option moves",
        resolve=_steps_count,
        derived=True,
        example="count.steps.tone.kept",
    ),
    _family_of(
        _Word("count.levels"),
        _keyed("<factor>", _factor_ids),
        what="how many levels one factor of a factorial grid holds",
        resolve=_levels_count,
    ),
    _family_of(
        _Word("count.examples"),
        _keyed("<aspect>", _example_aspects),
        _Ids("<arm>", _example_arms),
        what="how many of one arm's trials on one aspect the snapshot froze as examples (`count.examples` counts every "
        "one)",
        resolve=_kept_count,
        example="count.examples.answers.fast.v2",
    ),
    _family_of(
        _Word("count.examples"),
        _keyed("<aspect>", _example_aspects),
        _Ids("<arm>", _example_arms),
        _Ids("<stratum>", _example_strata),
        what="the same within one stratum or task",
        resolve=_kept_count,
    ),
    _family_of(
        _Word("count.failures"),
        _keyed("<aspect>", _example_aspects),
        _Ids("<arm>", _example_arms),
        what="how many of one arm's trials on one aspect failed (`count.failures` counts every one); not known, and "
        "refused, where the snapshot does not count them. " + _FAILURES_RULE,
        resolve=_failed_trials_count,
        example="count.failures.answers.fast.v2",
    ),
    _family_of(
        _Word("count.failures"),
        _keyed("<aspect>", _example_aspects),
        _Ids("<arm>", _example_arms),
        _Ids("<stratum>", _example_strata),
        what="the same within one stratum or task",
        resolve=_failed_trials_count,
    ),
    _family_of(
        _Word("overlap"),
        _keyed("<aspect>", lambda ri: [aspect.id for aspect in ri.aspects]),
        _Ids("<aspect>", _other_aspects),
        _Statistic(_OVERLAP_FIELDS, "<field>"),
        what="whether two aspects ran the same cases, by the cases' stored ids: `same` in words (the same cases, "
        "partly, different, or not known where the snapshot does not hold their case ids), `shared` the cases both "
        "ran, `only_first` and `only_second` the cases only one ran; the counts are refused where not known",
        resolve=_overlap,
        example="overlap.answers.speed.same",
    ),
    _family_of(
        _Word("confusion"),
        _keyed("<axis>", _confusion_axes),
        _Ids("<arm>", _confusion_arms),
        _Ids("<expected>", _confusion_labels),
        _Ids("<predicted>", _confusion_labels),
        what="one cell of an arm's confusion on a label axis: the share of the expected label's cases predicted as the "
        "predicted label, a percent of its row (or a count, where the axis counts observations)",
        resolve=_confusion,
    ),
    _family_of(
        _Word("cost.gap"),
        _keyed("<arm>", _costs_per_thousand),
        _Ids("<arm>", _cost_gap_seconds),
        _Statistic(_COST_GAP_FIELDS, "<field>"),
        what="two arms' costs apart, the first's minus the second's: `per_1k` per 1,000 of what the cost counts, "
        "`monthly` per month at the declared monthly volume, `multiple` the first's cost over the second's, read with its "
        "unit under every filter (`21 times`, `2.4 times`: whole from ten up, one decimal below)",
        resolve=_cost_gap,
        derived=True,
        example="cost.gap.fast.v2.base.multiple",
        compares=lambda ri, values: _gap_compares(ri, _costs_per_thousand(ri), values[-1]),
    ),
    _family_of(
        _Word("cost.unreported"),
        _keyed("<arm>", _costs_per_thousand),
        _Statistic(_UNREPORTED_FIELDS, "<field>"),
        what="an arm's cost where some of its trials report none: `missing` how many (trials run less the cost's "
        "observations), `free` the cost per 1,000 had each cost nothing (the least the mean over every trial can be), "
        "`at_p95` had each cost the arm's own 95th percentile (a what-if, not a ceiling: the snapshot holds no "
        "highest cost); refused where the cost is not read off one measure's observations, one per trial",
        resolve=_unreported,
        derived=True,
        ledger_as=_the_page_s_cost("free", "at_p95"),
    ),
)


#: The groups the grammar's path families are paged in, in the order the index lists them: each group's name, what its
#: figures are of, and the subjects of the families it holds (:func:`_subject`). Every family's subject is in one.
_FAMILY_GROUPS: tuple[tuple[str, str, frozenset[str]], ...] = (
    (
        "campaign",
        "the campaign, its arms, aspects and settings, how deeply each arm ran, and whether two aspects ran the same "
        "cases",
        frozenset(
            {
                "meta",
                "control",
                "n_candidates",
                "aspect",
                "aspects",
                "settings",
                "cases",
                "repeats",
                "depth",
                "late_criteria",
                "caveats",
                "overlap",
            }
        ),
    ),
    (
        "readings",
        "an arm's reading against the control on an aspect, pair readings and flip margins",
        frozenset({"reading", "pair", "flip_margin"}),
    ),
    (
        "contrasts",
        "corrected contrasts against the control, and the families correcting them",
        frozenset({"contrast", "family", "families"}),
    ),
    (
        "measures",
        "an arm's figures on a measure or judged dimension, its pass rate and scoreboard, and the judge",
        frozenset(
            {"measure", "judged", "judge", "pass_rate", "scoreboard", "reasoning_ratio", "pooled", "overall_interval"}
        ),
    ),
    (
        "slices",
        "figures within one task, stratum or target-matrix cell, the share of a loss each carries, and where a "
        "classifier's predictions of each label went",
        frozenset({"task", "strata", "cells", "confusion"}),
    ),
    (
        "gates",
        "declared criteria, floors and bars, and the weakest margin",
        frozenset({"criteria", "floor", "bar", "weakest"}),
    ),
    (
        "cost",
        "cost per 1,000 of what the cost counts and per month, cost uplift, a cost some trials did not report, the calls "
        "that went unanswered beside it, and a move between options priced per point",
        frozenset({"cost_per_1k", "cost", "price", "cost_per_month", "calls"}),
    ),
    (
        "separation",
        "an arm's spread on a figure, two arms' difference with its interval, and the cases that would settle a "
        "reading against a line or two arms apart",
        frozenset({"spread", "compare", "settle"}),
    ),
    (
        "assumed",
        "a spread, comparison or settle key read as if each slice whose figure states no count held an assumed count: "
        "a what-if the key names",
        frozenset({"assumed"}),
    ),
    (
        "levers",
        "the effect of one setting moved between arms that differ in it alone, and how it changes with another setting",
        frozenset({"lever"}),
    ),
    (
        "counts",
        "how many of a thing a sentence counts: arms, slices, gates, the options passing each, the places one misses, "
        "the examples kept and the trials the pass rule failed",
        frozenset({"count"}),
    ),
)


def _subject(family: _Family) -> str:
    """What a family's figures are of: its first word, or the word after ``aspects.<aspect>.`` in a qualified form, up
    to any dot a word is written with (``cost.uplift`` is of ``cost``)."""
    (first, *rest) = family.slots
    qualified = (
        isinstance(first, _Word) and first.spelling == "aspects" and len(rest) > 1 and isinstance(rest[1], _Word)
    )
    word = rest[1] if qualified else first
    return word.spelling.split(".")[0] if isinstance(word, _Word) else word.name


def _grouped() -> dict[str, list[_Family]]:
    """Every family, by the group it is paged in, groups and families each in the grammar's order.

    :raises ValueError: A family's subject is in no group: a family was added without a page to be read on.
    """
    grouped: dict[str, list[_Family]] = {name: [] for name, _, _ in _FAMILY_GROUPS}
    for family in _FAMILIES:
        group = next((name for name, _, subjects in _FAMILY_GROUPS if _subject(family) in subjects), None)
        if group is None:
            raise ValueError(f"figure key family `{family.form()}` is in no group of _FAMILY_GROUPS: place it in one")
        grouped[group].append(family)
    return grouped


__all__: list[str] = []
