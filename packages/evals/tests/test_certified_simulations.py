"""Every shard of the statistics certificate, which the testament pool runs as one request of the ``simulation`` suite:
the restated shards (pure arithmetic over seeded answers) and the product shards (through ``Eval.run`` or the engine's
operator path), each a few hundred newcomers or comparisons, or a few thousand restated, at most.

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

from packages.evals.tests.stats_certificate import (
    SHARDS,
    Shard,
    metric_bucket,
    metric_name,
    recorded_fields,
    run,
    run_key,
)


def _param(shard: Shard) -> Any:
    return pytest.param(
        shard,
        id=shard.id,
        marks=(pytest.mark.long_simulation, pytest.mark.xdist_group(f"simcert-{shard.id}")),
    )


#: One parameter per shard: the certification sends every shard to the pool.
PARAMS = [_param(shard) for shard in SHARDS]


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("shard", PARAMS)
async def test_certified_simulation_shard(shard: Shard, record_property: Any) -> None:
    """One shard, its counts recorded for the certification that sent it."""
    counts = await run(shard)
    key = run_key()
    fields = recorded_fields(shard)
    assert set(fields) <= set(counts), f"{shard.id} counted {sorted(counts)}, short of what its claims read {fields}"
    for name in fields:
        record_property(metric_name(key, metric_bucket(shard), name), counts[name])
