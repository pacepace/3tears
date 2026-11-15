# holds

The holds desk service (`service.py`): `handle(request)` answers a request about the books patrons asked the library
to keep, as a `Reply`. Holds live in `data/holds.json`. Requests the desk has seen, with the holds that should read
as expired, are in `tests/fixtures/requests.jsonl`.
