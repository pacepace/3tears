# 3tears-media-contracts

Dependency-free media capability contracts shared between 3tears media
*providers* (`3tears-models`) and media *consumers* (`3tears-agent-tools`).

This package contains only pure interface types: `typing.Protocol`
classes and stdlib dataclasses. It has **zero dependencies** by design,
so a provider library can implement (or a consumer can accept) these
contracts without inheriting any feature package's dependency closure.

```python
from threetears.media.contracts import GeneratedImage, ImageGenerationBackend
```

`MediaFacets` carries the additive carrier facets -- rights status, pixel
dimensions, and whether a locator points at the file itself or at the page
containing it. The vocabulary grows by addition only, and a consumer that
does not recognise a facet ignores it rather than failing, so a new carrier
ships without a coordinated release.

```python
from threetears.media.contracts import LOCATOR_KIND_DIRECT_FILE, MediaFacets
```

The legacy import path `threetears.agent.tools.protocols` remains a
re-export shim for installed `3tears-agent-tools` consumers.
