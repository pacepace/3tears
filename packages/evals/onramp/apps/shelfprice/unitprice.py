"""Unit prices for the shelf-edge labels: what a product costs per kilogram, per litre or per item.

:func:`unit_price` takes a product's price in pounds (``"2.49"``) and its pack size as the supplier's feed writes it,
and gives the price of one unit as a :class:`UnitPrice`. A pack size is:

- a quantity and a unit: ``500g``, ``1.5kg``, ``75cl``, ``330 ml``, ``2l``. Grams and kilograms are priced per
  kilogram; millilitres, centilitres and litres per litre;
- a multipack: ``6 x 330ml`` or ``4x1.5l`` is six (or four) of the size after the ``x``, priced on the total;
- a count: ``12 pack``, ``6 each`` or a bare ``4``, priced per item.

Some of our suppliers write a decimal comma, as continental suppliers do: ``1,5 kg`` is one and a half kilograms, and
``0,75 l`` three quarters of a litre. No feed writes a thousands separator in a size.

The unit price is in pence, rounded half up to a whole penny, and :attr:`UnitPrice.label` writes it as the shelf
label does: ``£3.32/kg``, ``£1.10/l``, ``£0.42 each``. :func:`best_value` picks the cheapest by unit price from
products sold by the same unit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

AUDIT = Path(__file__).with_name("data") / "shelf_audit.jsonl"

#: Each unit a size may be in: the unit it is priced per, and how many of that unit one of it is.
UNITS = {
    "g": ("kg", Decimal("0.001")),
    "kg": ("kg", Decimal(1)),
    "ml": ("l", Decimal("0.001")),
    "cl": ("l", Decimal("0.01")),
    "l": ("l", Decimal(1)),
    "pack": ("each", Decimal(1)),
    "each": ("each", Decimal(1)),
    "": ("each", Decimal(1)),
}
_SIZE = re.compile(r"^\s*(?:(\d+)\s*[x×]\s*)?(\d+(?:[.,]\d+)?)\s*([a-z]*)\s*$", re.IGNORECASE)


class SizeError(ValueError):
    """A pack size or price this module cannot read."""


@dataclass(frozen=True)
class UnitPrice:
    pence: int
    per: str

    @property
    def label(self) -> str:
        """The unit price as the shelf label writes it: ``£3.32/kg``, ``£0.42 each``."""
        amount = f"£{self.pence // 100}.{self.pence % 100:02d}"
        return f"{amount} each" if self.per == "each" else f"{amount}/{self.per}"


def _number(text: str) -> Decimal:
    return Decimal(text.replace(",", ""))


def pack_quantity(size: str) -> tuple[Decimal, str]:
    """How much of its unit a pack holds, and that unit: ``"6 x 330ml"`` is ``(Decimal("1.980"), "l")``."""
    match = _SIZE.match(size)
    if match is None or match[3].lower() not in UNITS:
        raise SizeError(f"cannot read the pack size {size!r}")
    per, factor = UNITS[match[3].lower()]
    count = int(match[1]) if match[1] else 1
    quantity = count * _number(match[2]) * factor
    if quantity <= 0:
        raise SizeError(f"the pack size {size!r} holds nothing")
    return quantity, per


def unit_price(price: str, size: str) -> UnitPrice:
    """The price of one kilogram, litre or item of a product at ``price`` pounds in packs of ``size``."""
    try:
        pounds = Decimal(price.strip().lstrip("£"))
    except InvalidOperation:
        raise SizeError(f"cannot read the price {price!r}") from None
    quantity, per = pack_quantity(size)
    pence = (pounds * 100 / quantity).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return UnitPrice(pence=int(pence), per=per)


def best_value(products: list[dict[str, str]]) -> dict[str, str]:
    """The product with the lowest unit price, from products ``{"name", "price", "size"}`` sold by the same unit."""
    prices = [unit_price(product["price"], product["size"]) for product in products]
    if len({price.per for price in prices}) != 1:
        raise SizeError("products sold by different units cannot be compared")
    return min(zip(products, prices), key=lambda pair: pair[1].pence)[0]


def load_audit(path: Path = AUDIT) -> list[dict[str, str]]:
    """The products in a shelf audit file, one JSON object a line."""
    return [json.loads(text) for text in path.read_text(encoding="utf-8").splitlines() if text.strip()]
