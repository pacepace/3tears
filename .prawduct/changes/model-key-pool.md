---
schema: 1
type: feature
scope: model-key-pool
date: 2026-11-26
title: Model key pool
---
A chat model can run on a pool of 1..n keys for one provider, each inside an operator-set budget, with
sticky rotation: the pool stays on its active key until that key is exhausted, revoked, forbidden,
rate-limited, tripped or low on budget, then moves it, durably, for every process sharing the pool.
Requirements: `docs/model-key-pool-requirements.md`. (Narrative completed at chunk 08.)

## Consumer note
To be written at chunk 08. It will include the Q2 release note: `threetears_llm_cost_usd_total` now
prices cache reads and writes at their own rates, so it drops for cache-heavy Anthropic traffic.
