# postal

Address labels for the packing bench (`address.py`). Our checkout joins a customer's address fields into one line,
separated by commas, and that is what reaches the warehouse: `flat 2, 14 mill ln, leeds, ls6 2ab`. `label_lines(raw)`
turns it into the lines the courier's label printer takes, and `label(raw)` joins them for printing. `load_addresses()`
reads the addresses from an orders file.

The label rules are in the module docstring: capitals, no full stops, a flat or unit on a line of its own first (an
apartment is printed as a flat), each other part on its own line with an abbreviated street type spelt out, and the
postcode last with its space in the right place. So `Flat 2, 14 Mill Ln, Leeds, LS6 2AB` prints as

```
FLAT 2
14 MILL LANE
LEEDS
LS6 2AB
```

Until this week the packers keyed every address into the courier's own booking site, which checks it against the
Royal Mail address file and prints the label. `data/order_addresses.jsonl` is last week's orders: the address as the
checkout gave it, and `courier_label`, the lines the courier's site printed for it. Our labels should match the
courier's line for line.

Tests: `python -m unittest discover tests` from this directory.
