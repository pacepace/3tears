# 3tears-media-contracts

`threetears.media.contracts` -- dependency-free media capability contracts
shared by providers and tools.

## Problem

Media *providers* (e.g. `3tears-models`, generating or transcribing media)
and media *consumers* (e.g. `3tears-agent-tools`, using that media) don't
need each other's full dependency closure -- just a shared interface. Coupling
them directly means a tool package pulls in every provider's SDK dependency
just to accept media it produces.

## What it does

- Zero-dependency `Protocol` and dataclass interfaces describing media
  capabilities.

## Design philosophy

Exists purely to decouple providers from consumers. A package that only
needs to accept or produce media in a contract-compatible shape never has
to import a specific provider's package, and never inherits that package's
dependency footprint.

## When to adopt

Any package that needs to produce or accept media without depending on the
full provider or consumer package on the other side.

## Composes with

- [`models`](models.md) -- a typical implementer, on the provider side (for
  image providers; the Whisper transcription provider does not use this
  contract).
- [`agent-tools`](agent-tools.md) -- a typical consumer, on the tool side.
- [`object-store`](object-store.md) -- implements the `ObjectStore`
  protocol defined here.
- [`backup`](backup.md), [`langgraph`](langgraph.md) -- also depend on this
  package directly.

## Install

```bash
pip install 3tears-media-contracts
```
