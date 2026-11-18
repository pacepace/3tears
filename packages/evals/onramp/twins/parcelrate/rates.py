"""Postage quotes for the dispatch desk: what a parcel costs to send, by its weight and where it is going.

``quote(weight_kg, zone)`` prices one parcel as a :class:`Quote`, from the courier's rate card in
``data/rate_card.json``. Each zone on the card lists its weight bands from lightest to heaviest; a band covers every
weight up to and including its limit, and a parcel pays the first band that covers it. ``signed=True`` adds the
card's signed-for surcharge. A parcel heavier than a zone's last band is refused with :class:`QuoteError`.

Weights are read as decimals, so ``"2.0"``, ``"2"`` and ``2`` are the same parcel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

RATE_CARD = Path(__file__).with_name("data") / "rate_card.json"


class QuoteError(ValueError):
    """A parcel the card cannot price: an unknown zone, a weight that is not a positive number, or one too heavy."""


@dataclass(frozen=True)
class Quote:
    zone: str
    weight_kg: Decimal
    band: str
    pence: int

    @property
    def price(self) -> str:
        """The price as the desk writes it on the docket: ``£4.95``."""
        return f"£{self.pence // 100}.{self.pence % 100:02d}"


def load_card(path: Path = RATE_CARD) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _weight(value: object) -> Decimal:
    try:
        weight = Decimal(str(value).strip())
    except InvalidOperation:
        raise QuoteError(f"weight {value!r} is not a number") from None
    if not weight.is_finite() or weight <= 0:
        raise QuoteError(f"weight {value!r} is not a positive number of kilograms")
    return weight


def quote(weight_kg: object, zone: str, *, signed: bool = False, card: dict[str, Any] | None = None) -> Quote:
    """Price one parcel from the rate card (the shipped card unless ``card`` is given)."""
    card = card if card is not None else load_card()
    if zone not in card["zones"]:
        raise QuoteError(f"no zone {zone!r} on the rate card")
    weight = _weight(weight_kg)
    for band in card["zones"][zone]:
        if weight <= Decimal(band["up_to_kg"]):
            surcharge = card["signed_surcharge_pence"] if signed else 0
            return Quote(zone=zone, weight_kg=weight, band=band["name"], pence=band["pence"] + surcharge)
    raise QuoteError(f"{weight} kg is over the heaviest {zone} band")


def quote_many(parcels: list[dict[str, Any]]) -> list[Quote]:
    """Price a batch of parcels, each ``{"weight_kg": ..., "zone": ..., "signed": ...}``, with one read of the card."""
    card = load_card()
    return [quote(p["weight_kg"], p["zone"], signed=bool(p.get("signed", False)), card=card) for p in parcels]
