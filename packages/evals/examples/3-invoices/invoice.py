"""What every rung-3 file reads an invoice into, and the two prompts it compares; no eval here, so none runs twice."""

from typing import Literal

from pydantic import BaseModel


class Invoice(BaseModel):
    number: str
    currency: Literal["USD", "EUR", "GBP", "JPY", "CHF", "CAD"]
    due: str
    total: float | None


V1 = "Extract the invoice number, currency, due date (YYYY-MM-DD) and total due from this invoice:\n\n{text}"
V2 = V1 + (
    "\n\nThe currency is the one the total is charged in, never one quoted for reference. Read dates in the "
    "issuer's own order (09/03 is 9 March in Europe, 3 September in the US). Undo OCR slips (O for 0, l for 1). "
    "Leave total null unless the invoice prints the amount due: never add up the lines yourself."
)
