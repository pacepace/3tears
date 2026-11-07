# 3tears-interfaces

The 3tears family's shared Protocols and vocabulary, in one package that
depends on nothing.

**The rule.** This package holds Protocols, the value types and enums that
cross them, and pure helpers over those types. No implementations, no
third-party imports, no dependencies. A provider can implement a Protocol, and
a consumer can accept one, without inheriting any feature package's dependency
closure. The workspace's `tests/enforcement/` fails any module-top import here
that is not the stdlib or `threetears.interfaces` itself.

One module per domain:

| Module | What it holds |
|---|---|
| `threetears.interfaces.access` | The identity and authorization vocabulary `3tears-iam` and `3tears-agent-acl` both name |
| `threetears.interfaces.media` | Media capability Protocols, the value types that cross them, and the object-key helpers |

## access

```python
from threetears.interfaces.access import ImpersonationCategory
```

`ImpersonationCategory` is the fixed set of six sensitive actions: credential,
passkey and MFA management; account deletion; API-key mint or rotate for the
target; email change; RBAC grant changes; audit editing. agent-acl deny-lists
all six under impersonation, and iam's `stepup` requires a recent sign-in for
each. iam proves who a caller is and agent-acl decides what they may do;
neither depends on the other, so the values both must agree on live here.

## media

```python
from threetears.interfaces.media import GeneratedImage, ImageGenerationBackend
```

The Protocols media providers (`3tears-models`, `3tears-object-store`)
implement and media consumers (`3tears-agent-tools`) accept: `VisionProvider`,
`ReferenceVisionProvider`, `TextProvider`, `TranscriptionProvider`,
`ImageGenerationBackend`, `MediaStorage` and `ObjectStore`. The value types
that cross them: `MediaInfo`, `GeneratedImage`, `ObjectHandle`,
`ObjectListing` and `MediaSizeLimitExceeded`. Plus the extraction-status and
generation-metadata constants, and the object-key helpers `build_object_key`
and `sanitize_segment`.

`MediaFacets` carries the additive carrier facets: rights status, pixel
dimensions, and whether a locator points at the file itself or at the page
containing it. The vocabulary grows by addition only, and a consumer that
does not recognise a facet ignores it rather than failing, so a new carrier
ships without a coordinated release.

```python
from threetears.interfaces.media import LOCATOR_KIND_DIRECT_FILE, MediaFacets
```

## Install

```bash
pip install 3tears-interfaces
```

## Versioning policy

`3tears-interfaces` versions in lockstep with the rest of the 3tears
monorepo: every package shares one version, tracking the framework git tag.
