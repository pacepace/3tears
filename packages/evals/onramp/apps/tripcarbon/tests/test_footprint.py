import json
import tempfile
import unittest
from pathlib import Path

from footprint import FACTORS, Leg, Trip, footprint_file, footprint_kg, leg_kg


class LegTests(unittest.TestCase):
    def test_a_car_leg(self):
        self.assertAlmostEqual(leg_kg(Leg("car", 100)), 100 * FACTORS["car"])

    def test_a_business_class_flight_weighs_more(self):
        economy = leg_kg(Leg("flight_long_haul", 5000))
        self.assertAlmostEqual(leg_kg(Leg("flight_long_haul", 5000, "business")), 2.9 * economy)

    def test_the_cabin_is_only_for_flights(self):
        self.assertEqual(leg_kg(Leg("coach", 100, "business")), leg_kg(Leg("coach", 100)))

    def test_an_unknown_mode(self):
        with self.assertRaises(ValueError):
            leg_kg(Leg("hovercraft", 10))


class TripTests(unittest.TestCase):
    def test_a_flight_and_a_hotel(self):
        trip = Trip("t1", (Leg("flight_short_haul", 371), Leg("flight_short_haul", 371)), (("NL", 2),))
        self.assertEqual(footprint_kg(trip), 135.86)

    def test_a_drive(self):
        self.assertEqual(footprint_kg(Trip("t2", (Leg("electric_car", 80), Leg("electric_car", 80)))), 7.51)

    def test_an_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trips.jsonl"
            rows = [
                {"trip_id": "a", "legs": [{"mode": "taxi", "km": 10}]},
                {"trip_id": "b", "legs": [{"mode": "flight_domestic", "km": 500, "cabin": "economy"}],
                 "hotel_nights": {"GB": 1}},
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            self.assertEqual(footprint_file(path), {"a": 1.49, "b": 133.34})


if __name__ == "__main__":
    unittest.main()
