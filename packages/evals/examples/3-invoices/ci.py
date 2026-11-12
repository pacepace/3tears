"""Rung 3d: the gate in CI. ``python ci.py`` runs guardrail.py's eval, writes the report for the reviewer, and exits 4
when the gate fails: a regression, a breach, or a guardrail not shown held (as this one is not), else 0."""

import asyncio

from guardrail import invoices

if __name__ == "__main__":
    result = asyncio.run(invoices.run())
    print(result)  # the verdict line, then each arm's readings; every delta and its interval is in the report
    result.write("report")  # report.html for the reviewer, beside report.md, campaign.json and the bundle
    raise SystemExit(4 if result.gate().failed else 0)
