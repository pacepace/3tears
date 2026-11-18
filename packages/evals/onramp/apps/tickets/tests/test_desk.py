import unittest

from desk import Ticket, apply, replay


def events(*kinds):
    return [{"at": f"2026-02-03T{9 + i:02d}:00", "event": kind} for i, kind in enumerate(kinds)]


class ReplayTests(unittest.TestCase):
    def test_a_new_ticket(self):
        self.assertEqual(replay("T-1", []).status, "new")

    def test_the_usual_conversation(self):
        ticket = replay("T-2", events("agent_reply", "customer_reply", "agent_reply", "solve", "close"))
        self.assertEqual((ticket.status, ticket.reopens, ticket.follow_up), ("closed", 0, False))

    def test_waiting_on_each_side(self):
        self.assertEqual(replay("T-3", events("agent_reply")).status, "pending")
        self.assertEqual(replay("T-4", events("agent_reply", "customer_reply")).status, "open")

    def test_notes_never_move_a_ticket(self):
        self.assertEqual(replay("T-5", events("note", "agent_reply", "note")).status, "pending")

    def test_events_are_applied_in_time_order(self):
        shuffled = list(reversed(events("agent_reply", "customer_reply", "solve")))
        self.assertEqual(replay("T-6", shuffled).status, "solved")

    def test_a_closed_ticket_stays_closed_and_starts_a_follow_up(self):
        ticket = replay("T-7", events("solve", "close", "customer_reply", "agent_reply"))
        self.assertEqual((ticket.status, ticket.follow_up), ("closed", True))

    def test_close_only_applies_to_solved_tickets(self):
        self.assertEqual(replay("T-8", events("agent_reply", "close")).status, "pending")

    def test_an_unknown_event_is_refused(self):
        with self.assertRaises(ValueError):
            apply(Ticket(id="T-9"), "escalate")


if __name__ == "__main__":
    unittest.main()
