"""The server half of two significance rules the browser kit restates.

`threetears.evals.analysis.significance.significance_read` decides whether a significance flag may be
reported at all, and `format_significance` decides what an effect size is CALLED. The
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
    format_significance,
    significance_read,
)


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


def test_the_effect_size_is_named_for_the_test_that_ran():
    """Unpaired is the weaker claim and the name a caller gets unless it states the pairing; a stated one earns d_z.

    Pinned both ways so a fix that hard-codes either label fails here.
    """
    unpaired = format_significance(significant=True, paired=False, p=0.004, effect=1.2, n=24)
    paired = format_significance(significant=True, paired=True, p=0.004, effect=1.2, n=24)
    assert f"{UNPAIRED_EFFECT_LABEL}=1.2" in unpaired and f"{PAIRED_EFFECT_LABEL}=1.2" in paired
