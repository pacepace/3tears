"""Provider-agnostic web and media search for the 3tears family.

The buildable authority for this package is ``docs/search-spec.md`` (decisions
D1-D28); requirement IDs cited in docstrings (``SR-*``, ``G*``, ``P*``) are
defined in ``docs/search-requirements.md``.

The public lingua franca lives in :mod:`threetears.search.contracts` -- the
leaf within the leaf. This top-level ``__init__`` deliberately imports nothing:
importing ``threetears.search.contracts`` executes this module first, and the
contracts module is required to pull nothing beyond stdlib, pydantic, and
``3tears-media-contracts`` (search-spec.md section 2).
"""

from __future__ import annotations
