"""The statistics certificate's product shards, which the testament pool runs as one request of the ``simulation``
suite: through ``Eval.run`` or the engine's operator path, each shard a few hundred newcomers or comparisons at most.

Each shard records its counts as metrics named by the certification run's key (``stats_certificate.metric_name``), and
the certification command (``packages/evals/scripts/certify_stats.py``) reads them back into the certificate. A shard
that holds an exact claim (the restatement read for read as the product; a drained watcher or engine comparison read as
its first draws) fails at the first newcomer or comparison that breaks it, saying where.

Every shard is ``long_simulation``, so the unit suite leaves them out, and each is an xdist group of its own, so the
pool runs them side by side even before it has a recorded duration for any. Run them through the command, never alone:
a shard on its own certifies nothing. ``docs/testing.md`` says when to re-certify.
"""

from __future__ import annotations

from typing import Any

import pytest

from packages.evals.tests.stats_certificate import Shard, metric_name, pool_shards, run_key, run_pool


def _param(shard: Shard) -> Any:
    return pytest.param(
        shard,
        id=shard.id,
        marks=(pytest.mark.long_simulation, pytest.mark.xdist_group(f"simcert-{shard.id}")),
    )


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("shard", [_param(shard) for shard in pool_shards()])
async def test_certified_simulation_shard(shard: Shard, record_property: Any) -> None:
    """One product shard, its counts recorded for the certification that sent it."""
    counts = await run_pool(shard)
    key = run_key()
    for name, value in counts.items():
        record_property(metric_name(key, name), value)
