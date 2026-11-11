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
| `threetears.interfaces.media` | Media capability Protocols, the value types that cross them, and the object-key builder |
| `threetears.interfaces.keyed` | The keyed helpers' vocabulary and pure builders: each keyed read and write's SQL text and binds, and the connection Protocol they run on |

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
generation-metadata constants, and the object-key builder `build_object_key`.

`MediaFacets` carries the additive carrier facets: rights status, pixel
dimensions, and whether a locator points at the file itself or at the page
containing it. The vocabulary grows by addition only, and a consumer that
does not recognise a facet ignores it rather than failing, so a new carrier
ships without a coordinated release.

```python
from threetears.interfaces.media import LOCATOR_KIND_DIRECT_FILE, MediaFacets
```

## keyed

```python
from threetears.interfaces.keyed import KeyBatch, read_keyed_statements, read_row_statement
```

The half of 3tears' keyed helpers that needs no connection. The vocabulary
(`AnyOf`, `Fragment`, `Selected`, `Lock`, `KeyBatch`, `Sort`, `Match`, and the
value types the runners answer) and the builders that write each shape's
statement: a point read (`read_row_statement`), a keyed batch
(`read_keyed_statements`), a listing's page and a listing read to its end
(`keyset_page_statement`, `keyset_rows_statement`), a keyed write
(`update_keyed_statements`, `delete_keyed_statements`) and an expiry sweep
(`delete_before_statement`). Each answers SQL text with `$n` placeholders and
its binds. `threetears.core.keyed` runs them and re-exports every name. A
package that may not depend on core runs a builder's statement on its own
`KeyedConnection`, or on SQLite through `sqlite_numbered`.

## Install

```bash
pip install 3tears-interfaces
```

## Versioning policy

`3tears-interfaces` versions in lockstep with the rest of the 3tears
monorepo: every package shares one version, tracking the framework git tag.
