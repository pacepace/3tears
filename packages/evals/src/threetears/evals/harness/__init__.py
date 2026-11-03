"""The writer harness: the report writer measured as the thing under test, over the quick path's own store.

Three modules build it, each for one part of a writer eval:

- :mod:`~threetears.evals.harness.writer_host`: a host that launches report writer arms over a store, beside the
  quick host's runs in it;
- :mod:`~threetears.evals.harness.writer_turns`: an arm's writer models, replayed from one recording per case, or
  recorded once;
- :mod:`~threetears.evals.harness.writer_cases`: a writer case frozen into a portable file, read back and minted
  into any scope.

This package sits above ``quick``: it composes the quick path with the surfaces above the engine (``ops``,
``actions``) and the replay format (``testing``), and nothing in the package imports it. It may import ``schema``,
``kernel``, ``run``, ``analysis``, ``storage``, ``testing``, ``quick``, ``ops``, ``actions`` and itself, and
``providers`` only as ``quick`` does, inside the function that builds a ``Model``, so importing it loads no client.

Nothing is exported yet: each name joins ``__all__`` with its first caller outside this package, the writer eval, so
the public-API caller gate (``tests/enforcement/test_public_api_has_a_caller.py``) never reads one as test-only.
"""

from __future__ import annotations

__all__: list[str] = []
