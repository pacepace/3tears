import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from claims import Claim, Decision, add_months, decide, decide_file, in_warranty, warranty_months


def claim(faults, category="laptop", bought="2026-01-15", registered=None, claimed="2026-09-01"):
    return Claim(
        "c1",
        category,
        date.fromisoformat(bought),
        date.fromisoformat(registered) if registered else None,
        date.fromisoformat(claimed),
        tuple(faults),
    )


class PeriodTests(unittest.TestCase):
    def test_months(self):
        self.assertEqual(add_months(date(2025, 1, 31), 1), date(2025, 2, 28))
        self.assertEqual(add_months(date(2025, 11, 5), 14), date(2027, 1, 5))

    def test_the_periods(self):
        self.assertEqual(warranty_months(claim(["no_power"])), 12)
        self.assertEqual(warranty_months(claim(["no_power"], category="oven")), 24)
        self.assertEqual(warranty_months(claim(["no_power"], registered="2026-02-14")), 24)
        self.assertEqual(warranty_months(claim(["no_power"], registered="2026-02-15")), 12)

    def test_the_last_day(self):
        self.assertTrue(in_warranty(claim(["no_power"], bought="2025-09-02", claimed="2026-09-01")))
        self.assertFalse(in_warranty(claim(["no_power"], bought="2025-09-01", claimed="2026-09-01")))


class DecisionTests(unittest.TestCase):
    def test_accepted(self):
        self.assertEqual(
            decide(claim(["battery_failure", "screen_failure"])),
            Decision("accepted", ("battery_failure", "screen_failure")),
        )

    def test_out_of_warranty(self):
        self.assertEqual(decide(claim(["no_power"], bought="2025-03-01")), Decision("out_of_warranty"))

    def test_an_excluding_fault_refuses_the_claim(self):
        self.assertEqual(decide(claim(["screen_failure", "liquid_damage"])), Decision("refused"))

    def test_a_scratch_is_not_repaired(self):
        self.assertEqual(decide(claim(["scratch"])), Decision("not_covered"))
        self.assertEqual(decide(claim(["no_power", "dent"])), Decision("accepted", ("no_power",)))

    def test_an_unknown_fault(self):
        with self.assertRaises(ValueError):
            decide(claim(["haunted"]))

    def test_an_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claims.jsonl"
            row = {"claim_id": "a", "category": "dishwasher", "bought": "2025-02-01", "claimed": "2026-09-01",
                   "faults": ["pump_failure"]}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(decide_file(path), {"a": Decision("accepted", ("pump_failure",))})


if __name__ == "__main__":
    unittest.main()
