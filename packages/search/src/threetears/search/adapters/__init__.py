"""Provider adapters -- one provider's API each, over the injected transport.

Adapters ship in the base package rather than behind extras: they are pure
logic over an injected
:class:`~threetears.search.contracts.transport.SearchTransport` and weigh
nothing (D24). Extras carry *weight*, and an adapter has none -- it opens no
client, imports no HTTP library, and reads no environment.

Importing an adapter module registers its capability declaration
(:func:`threetears.search.contracts.register_capabilities`), following the
``3tears-models`` precedent: a consumer can then ask what SearXNG can
express without constructing one, which would need a base URL and a
transport it may not have yet.

This ``__init__`` imports nothing. Adapters are chosen by name -- a host
that speaks to SearXNG should not pay to import Tavily's declaration, and a
package-level fan-in would make that impossible.
"""

from __future__ import annotations
