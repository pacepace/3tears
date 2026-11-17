"""Certify the evals reading's error-rate claims under the null, and write ``packages/evals/tests/fixtures/stats_certificate.json``.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed. Run it from the
repository's root whenever ``test_stats_certificate.py`` says the certificate is stale:

    uv run python packages/evals/scripts/certify_stats.py

It runs every shard of every claim (``packages/evals/tests/stats_certificate.py``): the restated shards here, side by
side in processes, and the product shards on the testament pool as one request of the ``simulation`` suite
(``test_certified_simulations.py``), whose counts it reads back from the metrics each shard records under this run's
key. Then it writes the certificate (every claim, its count, its Clopper-Pearson bounds, its seeds and the digest of the
files that decide it) and exits 1 if any claim does not clear its target. Commit the certificate it writes.

    # Run the pool's shards here too, when the pool can't take them (slower: an hour or more):
    uv run python packages/evals/scripts/certify_stats.py --local
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH", "")]))

from packages.evals.tests import stats_certificate as cert  # noqa: E402

_SHARD_FILE = "packages/evals/tests/test_certified_simulations.py"
_DROPPED = ("connection_lost", "coordinator_unreachable")


def _pool(key: str) -> tuple[dict[str, dict[str, int]], str]:
    """Run every product shard on the pool as one request and read each one's counts back. Returns them by shard id,
    and the request's id."""
    command = [
        "testament", "run", "--no-fallback", "--no-cache", "--suite", "simulation", "--json=summary",
        "--", _SHARD_FILE, "-q",
    ]  # fmt: skip
    for attempt in range(1, 5):
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        said = done.stdout + done.stderr
        if not any(word in said for word in _DROPPED):
            break
        print(f"pool: the connection dropped (attempt {attempt}); trying again in a minute", flush=True)
        time.sleep(60)
    try:
        envelope = json.loads(done.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"pool: no result envelope (exit {done.returncode}):\n{said[-4000:]}") from None
    report = envelope.get("report") or {}
    failed = [item for item in report.get("items", []) if item.get("outcome") != "passed"]
    if envelope.get("verdict") != "passed" or failed:
        lines = [f"  {item.get('id')}: {item.get('outcome')} {item.get('message', '')[:2000]}" for item in failed]
        raise SystemExit(f"pool: the shards did not all pass ({envelope.get('verdict')}):\n" + "\n".join(lines))
    by_item: dict[str, dict[str, int]] = {}
    for name in cert.FIELDS:
        asked = subprocess.run(
            ["testament", "metrics", "--name", cert.metric_name(key, name), "--since", "2d", "--limit", "1000",
             "--json"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )  # fmt: skip
        for series in json.loads(asked.stdout)["result"]["series"]:
            by_item.setdefault(series["item_id"], {})[name] = int(series["days"][-1]["last"])
    results = {}
    for shard in cert.pool_shards():
        item = f"{_SHARD_FILE}::test_certified_simulation_shard[{shard.id}]"
        if item not in by_item:
            raise SystemExit(f"pool: shard {shard.id} recorded no counts under key {key}")
        results[shard.id] = by_item[item]
    return results, str(envelope.get("request_id", ""))


def _here(shards: list[cert.Shard], workers: int) -> dict[str, dict[str, int]]:
    """Run ``shards`` here, side by side in ``workers`` processes."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {shard.id: pool.submit(cert.run_shard, shard) for shard in shards}
        return {shard_id: future.result() for shard_id, future in futures.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="run the pool's shards here too")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="processes for the shards run here")
    parser.add_argument("--out", type=Path, default=cert.CERTIFICATE, help="where to write the certificate")
    options = parser.parse_args(argv)
    started = time.monotonic()
    key = secrets.token_hex(6)
    about: dict[str, Any] = {}
    cert.RUN_KEY.write_text(key + "\n")
    try:
        if options.local:
            results = _here(cert.pool_shards() + cert.local_shards(), options.workers)
            about["pool_request"] = None
        else:
            with ThreadPoolExecutor(max_workers=1) as background:
                pooled = background.submit(_pool, key)
                results = _here(cert.local_shards(), options.workers)
                print(f"restated shards done at {time.monotonic() - started:.0f}s; waiting on the pool", flush=True)
                pool_results, request = pooled.result()
            results |= pool_results
            about["pool_request"] = request
    finally:
        cert.RUN_KEY.unlink(missing_ok=True)
    about["wall_s"] = round(time.monotonic() - started)
    document = cert.certificate(results, **about)
    cert.write(document, options.out)
    failing = [claim for claim in document["claims"] if not claim["holds"]]
    for claim in document["claims"]:
        mark = "holds" if claim["holds"] else "FAILS"
        low, high = claim["bounds"]
        print(
            f"{mark}  {claim['id']}: {claim['count']}/{claim['replicates']} "
            f"(95% {low:.4f} to {high:.4f}; {claim['test']} {claim['target']:.5g})"
        )
    print(f"certificate {document['digest'][:16]} written to {options.out} in {about['wall_s']}s")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
