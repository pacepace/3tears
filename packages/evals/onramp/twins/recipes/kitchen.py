"""Scale a recipe to a number of servings, and tidy its units as a cook would write them.

    >>> Kitchen().scale(recipe, servings=8)

A recipe is a dict: ``name``, ``serves`` and ``ingredients``, each ingredient an ``item``, a ``qty`` and a ``unit``
(``tsp``, ``tbsp``, ``cup``, ``g``, ``ml`` or ``each``). Scaling multiplies every quantity, then moves spoon and
cup measures to the largest unit that keeps the number at one or more: 3 teaspoons are a tablespoon, 16
tablespoons a cup. Grams, millilitres and counted items are only multiplied.
"""

from __future__ import annotations

from fractions import Fraction

TSP_PER_TBSP = 3
TBSP_PER_CUP = 16

#: Each volume unit in teaspoons.
_IN_TSP = {"tsp": Fraction(1), "tbsp": Fraction(TSP_PER_TBSP), "cup": Fraction(TSP_PER_TBSP * TBSP_PER_CUP)}


class Kitchen:
    """The scaling service. ``rounding`` is how many decimals a quantity keeps."""

    def __init__(self, rounding: int = 2) -> None:
        self.rounding = rounding

    def scale(self, recipe: dict, servings: int) -> dict:
        """The recipe for ``servings`` people: every quantity multiplied, spoon and cup measures tidied."""
        if servings < 1:
            raise ValueError(f"servings must be at least 1, not {servings}")
        factor = Fraction(servings, recipe["serves"])
        out = []
        for ingredient in recipe["ingredients"]:
            qty = Fraction(str(ingredient["qty"])) * factor
            unit = ingredient["unit"]
            if unit in _IN_TSP:
                qty, unit = self._tidy(qty * _IN_TSP[unit])
            out.append({"item": ingredient["item"], "qty": round(float(qty), self.rounding), "unit": unit})
        return {"name": recipe["name"], "serves": servings, "ingredients": out}

    def _tidy(self, tsp: Fraction) -> tuple[Fraction, str]:
        """A volume in teaspoons, in the largest unit that keeps it at one or more."""
        for unit in ("cup", "tbsp"):
            if tsp >= _IN_TSP[unit]:
                return tsp / _IN_TSP[unit], unit
        return tsp, "tsp"


def main() -> None:
    """``python kitchen.py RECIPE.json SERVINGS``: print the scaled recipe."""
    import json
    import sys

    recipe = json.loads(open(sys.argv[1], encoding="utf-8").read())
    print(json.dumps(Kitchen().scale(recipe, int(sys.argv[2])), indent=2))


if __name__ == "__main__":
    main()
