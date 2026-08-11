"""Shared conformance suite and declared test doubles (SR-O5).

Provider conformance is one suite every adapter passes, not a per-adapter
test file that happens to check similar things: the second adapter is where
"similar things" starts meaning "a different set of things", and the whole
point is that a consumer can swap providers and keep its guarantees.

Shipped in ``src`` rather than under ``tests`` deliberately. It has to be
importable by a *consumer's* test suite -- a host injecting its own transport
wants the same five pins run against its own wiring -- and a test tree is not
importable from outside its package.

Nothing here imports pytest. The suite is a mixin of ``async def test_*``
methods over plain assertions, so it collects under whatever runner the
consumer already has and adds no test-framework dependency to a package
whose install weight is a requirement (SR-L6).
"""

from __future__ import annotations

from threetears.search.testing.conformance import ProviderConformanceCase, ProviderConformanceSuite
from threetears.search.testing.fakes import (
    FakeBudgetPort,
    FakeRateLimiterPort,
    ScriptedTransport,
    TransportScript,
)

__all__ = [
    "FakeBudgetPort",
    "FakeRateLimiterPort",
    "ProviderConformanceCase",
    "ProviderConformanceSuite",
    "ScriptedTransport",
    "TransportScript",
]
