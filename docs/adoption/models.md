# 3tears-models

`threetears.models` -- LangChain-native model adapters. Anthropic, OpenAI,
OpenRouter, VoyageAI, Whisper, and image providers, with capability
metadata, circuit breakers, error translation, and unified usage tracking.

## Problem

Every LLM provider has its own client, its own error shapes, and its own
usage-reporting format. An app that talks to more than one provider ends up
with per-provider glue code duplicated across every call site, and no
single place to see usage or apply a circuit breaker when a provider is
degraded.

## What it does

- LangChain-native factories for chat and embedding models across
  Anthropic, OpenAI, OpenRouter, VoyageAI, Whisper, and image providers.
- Capability metadata per model (context window, supported modalities,
  etc.).
- Circuit breakers so a degraded provider doesn't take down every caller.
- Error translation to a consistent shape across providers.
- Unified usage tracking, wired in automatically rather than per call site.

## Design philosophy

One factory call replaces per-provider client construction and per-provider
error handling. Usage tracking, circuit breaking, and error translation are
wired into that single call rather than left as something every consumer
must remember to add. The premise is simplification at the call site: an
app that switches providers, or adds a second one, changes a factory
argument, not its error-handling code.

## When to adopt

Any app calling more than one LLM provider, or any app that wants circuit
breaking and unified usage tracking without building it per provider.

## Composes with

- [`media-contracts`](media-contracts.md) -- image providers implement
  this contract so consumers don't need a provider-specific type. The
  Whisper transcription provider does *not* use this contract -- it
  returns its own `TranscriptionResult` shape.
- [`observe`](observe.md) -- usage tracking surfaces through structured
  logging/tracing.

There is no code-level dependency from `langgraph` back to this package --
`langgraph`'s serialization layer is deliberately adapter-agnostic and
imports no provider-specific classes. Don't assume the two are pre-wired;
an agent built with `langgraph` still constructs its model via this
package's factories itself.

## Install

```bash
pip install 3tears-models
# provider/capability extras are opt-in, e.g.:
pip install "3tears-models[voyageai]"
pip install "3tears-models[whisper]"
pip install "3tears-models[image]"
```
