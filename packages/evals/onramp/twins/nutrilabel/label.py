"""Front-of-pack nutrition labels, from a product's spec sheet.

A spec sheet gives the product's ``form`` (``food``, whose values are per 100 g, or ``drink``, per 100 ml), its
shelf ``category`` (``bakery``, ``soft drinks``, ...), and per-100 values: ``energy_kj``, and ``fat``, ``saturates``,
``sugars`` and ``salt`` in grams. The label is one line:

    Per 100g: Energy 1046kJ/250kcal | Fat 3.0g LOW | Saturates 1.3g LOW | Sugars 22.6g HIGH | Salt 0.80g MED

- kcal is kJ divided by 4.184, to the nearest whole number;
- fat, saturates and sugars are shown to one decimal place, salt to two;
- each of the four gets a colour band from the thresholds for the product's form: LOW at or below the first, HIGH
  above the second, MED between. Drinks are judged per 100 ml against their own, lower, thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

KJ_PER_KCAL = 4.184
NUTRIENTS = (("fat", "Fat", 1), ("saturates", "Saturates", 1), ("sugars", "Sugars", 1), ("salt", "Salt", 2))
#: (LOW at or below, HIGH above) for each nutrient, per 100 g of food.
FOOD = {"fat": (3.0, 17.5), "saturates": (1.5, 5.0), "sugars": (5.0, 22.5), "salt": (0.3, 1.5)}
#: The same, per 100 ml of a drink.
DRINK = {"fat": (1.5, 8.75), "saturates": (0.75, 2.5), "sugars": (2.5, 11.25), "salt": (0.3, 0.75)}
THRESHOLDS = {"food": FOOD, "drink": DRINK}
PER = {"food": "100g", "drink": "100ml"}


def band(value: float, low: float, high: float) -> str:
    if value <= low:
        return "LOW"
    return "HIGH" if value > high else "MED"


def label(spec: dict) -> str:
    """The label line for one product's spec sheet."""
    if spec["form"] not in PER:
        raise ValueError(f"{spec.get('sku', 'a product')}: form must be food or drink, not {spec['form']!r}")
    limits = THRESHOLDS.get(spec["form"], FOOD)
    kcal = round(spec["energy_kj"] / KJ_PER_KCAL)
    parts = [f"Energy {spec['energy_kj']}kJ/{kcal}kcal"]
    for key, name, places in NUTRIENTS:
        value = spec[key]
        parts.append(f"{name} {value:.{places}f}g {band(value, *limits[key])}")
    return f"Per {PER[spec['form']]}: " + " | ".join(parts)


def labels_file(path: str | Path) -> dict[str, str]:
    """Every product's label in a file of spec sheets (one JSON sheet per line, with a ``sku``), by SKU."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["sku"]: label(row) for row in rows}
