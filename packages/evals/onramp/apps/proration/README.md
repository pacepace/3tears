# proration

We are moving subscription billing off our payment provider's built-in proration and onto our own, so that the
invoice preview in the app shows the same numbers the invoice will. `proration.py` works out the two lines a
mid-cycle plan change puts on the next invoice: `prorate(old_plan, new_plan, anchor_day, changed_on)` returns an
`Adjustment` with the credit for the old plan's unused days and the charge for the new plan over the same days, each
in pence. How a cycle is laid out, which days count, and how each line is rounded (to the nearest penny, halves up)
are in the module docstring. The plans and their monthly prices are in `PLANS`.

`data/invoices.jsonl` is a sample of the plan changes the provider invoiced this summer and autumn, newest first, as
its export lists them: the old and new plan, the subscription's anchor day, the day of the change, and the two lines
the provider put on the invoice (`credit_pence`, `charge_pence`). The provider's rules are the ones in the docstring,
and ours should give the same two lines for every change.

Tests: `python -m unittest discover tests` from this directory.
