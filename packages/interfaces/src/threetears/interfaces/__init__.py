"""The 3tears family's shared Protocols and vocabulary, in one package that depends on nothing.

One module per domain:

- :mod:`threetears.interfaces.access` -- the identity and authorization vocabulary
  ``3tears-iam`` (who a caller is) and ``3tears-agent-acl`` (what they may do) both name.
- :mod:`threetears.interfaces.media` -- the media capability Protocols shared by media
  providers (``3tears-models``, ``3tears-object-store``) and consumers (``3tears-agent-tools``).

The rule: Protocols, the value types and enums that cross them, and pure helpers over those
types. No implementations, no third-party imports, no dependencies. A package that speaks
the vocabulary or implements a Protocol never inherits another feature package's closure by
doing so. The workspace's ``tests/enforcement/`` enforces it: every module-top import here
must be the stdlib or ``threetears.interfaces`` itself.
"""

# Version derived from package metadata so the metadata is the single source of truth: a
# release that bumps pyproject cannot drift the runtime ``__version__``. ``importlib.metadata``
# is stdlib, so the dependency-free floor is untouched.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-interfaces")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"
