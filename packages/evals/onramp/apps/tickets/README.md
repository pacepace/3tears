# tickets

Ticket status for our new support desk (`desk.py`). A ticket's status is worked out by replaying its history:
`replay(ticket_id, events)` applies the events in time order and returns a `Ticket` with its `status`, how many times
the customer reopened it (`reopens`), and whether a follow-up ticket was started (`follow_up`). The rules are in the
module docstring; they are the old helpdesk's rules, written down.

We are switching off the old helpdesk at the end of the month. `data/export.jsonl` is its export of February's
tickets: each ticket's events, and the status and reopen count the old desk showed for it (`old_desk`). Replaying a
ticket's events should land where the old desk had it.

Tests: `python -m pytest tests` (or `python -m unittest discover tests`) from this directory.
