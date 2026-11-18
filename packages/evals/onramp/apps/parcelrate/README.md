# parcelrate

Postage quotes for the dispatch desk (`rates.py`). `quote(weight_kg, zone, signed=False)` prices a parcel from our
courier's rate card, `data/rate_card.json`, and returns a `Quote` with the band it fell in and the price in pence.
`quote_many(parcels)` prices a batch. Zones are `domestic`, `europe` and `world`.

The courier bills us once a month. `data/courier_invoice.jsonl` is February's invoice so far: each consignment as the
courier weighed it, where it went, whether it was signed for, and what they charged (`charged_pence`). The desk quotes
customers before the parcel goes, so our quote for a consignment should come to what the courier then charges for it.

Tests: `python -m unittest discover tests` from this directory.
