"""nats-wrapper-usage enforcement domain — direct ``nats`` import walker.

every production module (and, when configured, every test module)
must route ``nats-py`` access through the
``threetears.nats.NatsClient`` wrapper. direct ``import nats`` /
``from nats import ...`` / ``from nats.X import ...`` lines bypass
the wrapper and break the abstraction it exists to provide. flagged
at the import boundary, the contract is cheap to audit and easy to
fix.

per-repo configuration goes through :class:`NatsWrapperConfig`;
:func:`run_nats_enforcement` is the pytest-friendly entry point that
orchestrates the production / tests walkers, applies the whole-file
exemption list, emits the report, and fails in strict mode.
"""

from threetears.enforcement.nats_wrapper_usage.config import (
    NatsWrapperConfig,
)
from threetears.enforcement.nats_wrapper_usage.runner import (
    run_nats_enforcement,
)
from threetears.enforcement.nats_wrapper_usage.walkers import (
    find_direct_nats_imports,
    find_test_nats_imports,
    is_forbidden_module,
)

__all__ = [
    "NatsWrapperConfig",
    "find_direct_nats_imports",
    "find_test_nats_imports",
    "is_forbidden_module",
    "run_nats_enforcement",
]
