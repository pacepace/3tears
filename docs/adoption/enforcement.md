# 3tears-enforcement

`threetears.enforcement` -- static-analysis enforcement scanners and shared
test utilities. Naming conventions, schema agreement, datetime-aware
auditing.

## Problem

Architectural rules that live only in a reviewer's head, or in a style
guide nobody rereads, drift as a codebase grows -- especially under
AI-assisted development, where code volume outpaces manual review capacity.
Vendoring the same enforcement test files across repos and syncing them by
hand is its own maintenance burden and inevitably falls out of sync.

## What it does

- Shared AST-based static-analysis scanners covering roughly ten
  architectural domains: cache primitive contracts, underscore-access
  discipline, no bare print/logging, exception-swallow discipline, and
  more.
- Used directly by roughly half of this repo's own root-level
  `tests/enforcement/` suite, and by the packages that define their own
  `tests/enforcement/` directory -- not universally wired into every
  package's tests.

## Design philosophy

The central operating premise: if an architectural rule can be stated
objectively, it can be enforced automatically instead of relying on a human
to catch it in review. This package centralizes scanner *logic* across the
3tears ecosystem, replacing the earlier anti-pattern of vendoring identical
enforcement test files per repo and syncing them manually. Per-repo
configuration -- allowlists, exemptions -- stays local, where it belongs;
only the scanning logic itself is shared. It is not itself dependency-free:
some of its walkers (e.g. migration safety) import `threetears.core`
directly to inspect real migration objects, not just source text.

This is the concrete implementation of the "Enforcement Tests" pillar of
Enforcement-Accelerated Development, the methodology 3tears itself is built
with (see the root `README.md`, "Developed with EAD").

## When to adopt

Any 3tears-derived codebase, or any Python codebase that wants the same
class of AST-enforced architectural invariants without hand-rolling
scanners from scratch.

## Composes with

- [`core`](core.md) -- some scanners inspect real `threetears.core`
  migration objects rather than source text alone.

## Install

```bash
pip install 3tears-enforcement
# most scanners import pytest at module level; install the test extra:
pip install "3tears-enforcement[test]"
```
