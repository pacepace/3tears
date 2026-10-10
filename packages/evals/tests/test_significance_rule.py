"""The server half of two significance rules the browser kit restates.

`threetears.evals.analysis.reporting.significance_read` decides whether a significance flag may be
reported at all, and a finding's `delta_table` payload decides what an effect size is CALLED. The
browser kit that renders the same stored rows restates both rules in TypeScript; the pins between
the two languages live with that kit, in the host that ships it. What stays here is this side of
each rule, asserted as the facts the kit's copy is pinned against: change one of these and the
cross-language pin is the next thing to fail.
"""

from __future__ import annotations

import inspect

from threetears.evals.analysis.significance import (
    NOT_SIGNIFICANT_LABEL,
    NOT_TESTED_LABEL,
    PAIRED_EFFECT_LABEL,
    SIGNIFICANT_LABEL,
    UNPAIRED_EFFECT_LABEL,
    significance_read,
)
from threetears.evals.analysis.viz import chart_intent
from threetears.evals.analysis.viz.payloads import DeltaRow


def _effect_cell(row: DeltaRow) -> str:
    """The effect column the compiled delta table carries for one row, through the public intent builder."""
    [compiled] = chart_intent("delta_table", {"rows": [row.model_dump()]}).rows
    return str(compiled["effect"])


def test_p_or_d_z_is_the_test_ran_predicate():
    """`p is not None or <effect size> is not None` -- never `and`, never `n`."""
    assert significance_read(significant=True, p=0.01) != NOT_TESTED_LABEL
    assert significance_read(significant=True, effect=1.4) != NOT_TESTED_LABEL
    assert significance_read(significant=True) == NOT_TESTED_LABEL


def test_a_sample_size_is_not_a_test():
    """`significance_read` takes no ``n``, so a sample size can never become evidence of a test."""
    assert "n" not in inspect.signature(significance_read).parameters


def test_the_three_labels_are_three_distinct_words():
    """The vocabulary the kit spells word for word; three labels, never two that collide."""
    assert len({SIGNIFICANT_LABEL, NOT_SIGNIFICANT_LABEL, NOT_TESTED_LABEL}) == 3


def test_a_row_that_does_not_state_its_pairing_is_unpaired():
    """One stored dict, two renderers: absence has to mean unpaired, and the server renders it so.

    The payload default is asserted as the fact it is, and then the rendering is exercised, because
    a default nothing reads would agree with the kit while the compiler went on printing the paired
    name.
    """
    assert DeltaRow.model_fields["paired"].default is False

    unstated = DeltaRow(metric="cost_usd", a=0.011, b=0.019, d_z=1.2, p=0.004, n=24, significant=True)

    assert f"{UNPAIRED_EFFECT_LABEL}=1.2" in _effect_cell(unstated)


def test_a_row_that_states_its_pairing_is_named_for_the_test_it_states():
    """The default is the weaker claim, not a ceiling -- a stated pairing still earns d_z.

    Pinned beside the default so a fix that hard-codes the unpaired label -- the same defect this
    pair replaced, one direction over -- fails here.
    """
    stated = DeltaRow(metric="cost_usd", a=0.011, b=0.019, d_z=1.2, p=0.004, n=24, significant=True, paired=True)

    assert f"{PAIRED_EFFECT_LABEL}=1.2" in _effect_cell(stated)
