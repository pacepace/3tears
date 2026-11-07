"""Dependency-free identity and authorization vocabulary.

The values authentication (``3tears-iam``) and authorization (``3tears-agent-acl``) must
agree on, declared once. Neither package depends on the other: iam proves who a caller is
and agent-acl decides what they may do, and a consumer taking one should not install the
other's closure to share a vocabulary. This package depends on nothing; purity is enforced
by the contract-purity check in the workspace's ``tests/enforcement/``.

It holds only what both sides name. A value one side uses alone stays with that side.
"""

# Version derived from package metadata so the metadata is the single source of truth: a
# release that bumps pyproject cannot drift the runtime ``__version__``. ``importlib.metadata``
# is stdlib, so the dependency-free floor is untouched.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-access-contracts")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.access.contracts.categories import ImpersonationCategory

__all__ = ["ImpersonationCategory"]
