# 3tears-interfaces

`threetears.interfaces` -- the family's shared Protocols and vocabulary, in one
package that depends on nothing.

## Problem

Two packages that must agree on a type should not have to depend on each
other to share it. Media *providers* (`3tears-models`, `3tears-object-store`)
and media *consumers* (`3tears-agent-tools`) need one interface, not each
other's closures: coupling them directly makes a tool package pull in every
provider's SDK just to accept the media it produces. Authentication
(`3tears-iam`) and authorization (`3tears-agent-acl`) guard the same sensitive
actions from two sides, and sharing that type by having one import the other
made a service that only signs people in install the RBAC evaluator and its
database stack, just to name an action.

## What it does

One module per domain:

| Module | Holds |
|---|---|
| `threetears.interfaces.access` | `ImpersonationCategory`, the fixed set of six sensitive actions agent-acl deny-lists under impersonation and iam's `stepup` requires a recent sign-in for |
| `threetears.interfaces.media` | The media capability Protocols (`VisionProvider`, `ReferenceVisionProvider`, `TextProvider`, `TranscriptionProvider`, `ImageGenerationBackend`, `MediaStorage`, `ObjectStore`), the value types that cross them (`MediaInfo`, `GeneratedImage`, `ObjectHandle`, `ObjectListing`, `MediaSizeLimitExceeded`, `MediaFacets`), the extraction-status and generation-metadata constants, and the object-key builder `build_object_key` |

```python
from threetears.interfaces.access import ImpersonationCategory
from threetears.interfaces.media import GeneratedImage, ImageGenerationBackend, ObjectStore
```

## The rule

Protocols, the value types and enums that cross them, and pure helpers over
those types. No implementations, no third-party imports, no dependencies.
`tests/enforcement/` fails any module-top import in the package that is not
the stdlib or `threetears.interfaces` itself, so implementing or accepting an
interface never costs a package another feature package's closure.

A value only one side uses stays with that side. A new domain gets its own
module here when two packages that should not depend on each other must agree
on it.

## When to adopt

Any package that implements or accepts a family Protocol, or names a shared
value, without needing the package on the other side: an image backend, an
object store, a policy table, an audit consumer, an admin surface.

## Composes with

- [`models`](models.md) -- implements the media Protocols on the provider side
  (image providers; the Whisper transcription provider does not use them).
- [`agent-tools`](agent-tools.md) -- accepts them on the tool side.
- [`object-store`](object-store.md) -- implements `ObjectStore`.
- [`iam`](iam.md) -- the step-up check, which reads `ImpersonationCategory`.
- [`agent-acl`](agent-acl.md) -- the impersonation deny-list, which re-exports
  `ImpersonationCategory` as `threetears.agent.acl.ImpersonationCategory` (the
  same class).
- [`backup`](backup.md), [`geo`](geo.md), [`langgraph`](langgraph.md),
  `search` -- also depend on this package directly.

## Install

```bash
pip install 3tears-interfaces
```
