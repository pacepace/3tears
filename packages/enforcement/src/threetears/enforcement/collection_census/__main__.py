"""``python -m threetears.enforcement.collection_census [--framework] <src root> ...``: print the census as JSON."""

from __future__ import annotations

import json
import sys

from threetears.enforcement.collection_census.census import main

json.dump(main(sys.argv[1:]), sys.stdout)
