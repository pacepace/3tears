# warranty

The warranty check for the repairs desk. When a product comes in, the technician logs its faults, and
`decide(claim)` says what the warranty does about it: `out_of_warranty`, `refused`, `not_covered`, or `accepted`
with the faults to repair. `Claim.from_row` reads a claim as the desk's system exports it (`claim_id`, `category`,
`bought`, an optional `registered`, `claimed`, and `faults` in the order logged); `decide_file(path)` does a whole
export.

The rules are in the module docstring of `claims.py`. In short: 12 months from purchase, 24 for a large appliance,
and 12 more if the product was registered within 30 days of purchase. Faults are covered (repaired), cosmetic (a
scratch or a dent: not repaired, but the claim's other faults still are), or excluding (liquid, a drop, someone
else's repair: the whole claim is refused).

`data/decided.jsonl` is the autumn's claims as the senior technician decided them by hand while the desk ran without
this script; `desk` on each line is her decision (`outcome` and `repair`). The script should reach the same decision
on every claim.

Tests: `python -m unittest discover tests` from this directory.
