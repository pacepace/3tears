# shipeta

The delivery estimate on our booking page. `estimate(booked_at, origin, destination)` takes the time a parcel is
booked and its lane (depot codes: `LDN`, `MAN`, `BHM`, `LDS`, `BRS`, `GLA` in Great Britain; `DUB`, `PAR`, `AMS`,
`BRU` abroad) and returns when it should reach the customer. The steps a parcel goes through, and the figures for
each, are in the module docstring of `eta.py`; the lanes are in `lanes.json`. Lanes abroad wait at customs for the
number of days the lane table gives (`customs_days`), on top of the drive.

An estimate is never exact: a delivery round takes the morning or the afternoon, and the estimate is the middle of
it. What we promise customers is that the parcel arrives within 3 hours either side of the estimate, so an estimate
is good when the actual delivery time is within 3 hours of it, and the measure we watch is the share of shipments
whose estimate was good.

`data/delivered.jsonl` is September's delivered shipments from the tracking system, in booking order: the
consignment number, when it was booked (`booked_at`), its lane, and when the driver scanned it as delivered
(`delivered_at`). All times are UK time.

Tests: `python -m unittest discover tests` from this directory.
