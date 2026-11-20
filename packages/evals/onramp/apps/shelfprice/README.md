# shelfprice

Unit prices for the shelf-edge labels in our two shops (`unitprice.py`). The law wants a grocery label to show the
price per kilogram, per litre or per item next to the pack price, so shoppers can compare. `unit_price(price, size)`
works it out from the pack price in pounds and the pack size as the supplier's feed writes it, and its `label` is the
text the label printer prints (`£3.32/kg`, `£1.10/l`, `£0.42 each`). `pack_quantity(size)` reads a pack size,
`best_value(products)` picks the cheapest per unit, and `load_audit()` reads the audit file.

The sizes we take and how they are priced are in the module docstring. Our feeds come from British and continental
suppliers, and the continental ones write a decimal comma: `0,75 l` is 750 ml, and `1,5 kg` is a kilo and a half.

After a trading standards visit last month the shop manager worked out the unit price of every line on the audit list
by hand, from the pack price and size, and checked it against the scales and the measuring jug where she could.
`data/shelf_audit.jsonl` is that list, with the label she signed off for each (`audited_label`). Our labels should
match hers exactly.

Tests: `python -m unittest discover tests` from this directory.
