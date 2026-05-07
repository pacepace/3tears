"""Top-level conftest for the 3tears workspace.

pytest 8+ requires ``pytest_plugins`` declarations to live in the
top-level conftest at the rootdir. cross-package + per-package runs
both resolve their rootdir to the workspace root (the parent
``pyproject.toml`` at this directory), so this file is the canonical
home for plugin registration.

re-exports the testcontainer + nats fixtures from
:mod:`threetears.core.testing.fixtures` (test-harness-task-01) so
every package's integration suite picks them up without redeclaring
``pytest_plugins`` in a nested conftest.

also installs a warnings filter for the upstream
``LangChainPendingDeprecationWarning`` raised by
``langgraph.checkpoint.serde.jsonplus``'s module-level
``LC_REVIVER = Reviver()``: that ``Reviver`` is constructed without
``allowed_objects`` because the module is library code we do not own,
and the warning is fired at import time -- before the
``[tool.pytest.ini_options].filterwarnings`` setting is applied. our
own ``threetears.langgraph.serde.UUIDSafeSerializer`` constructs a
``JsonPlusSerializer()`` (no ``Reviver`` kwargs are exposed on that
surface), so there is nothing for us to fix at the call site; revisit
when langchain-core flips the default in a future release.
"""

from __future__ import annotations

import warnings

# Importing langchain_core re-enables LangChain's own deprecation
# warnings via ``surface_langchain_deprecation_warnings()``; that
# function prepends a ``default``-action filter for
# ``LangChainPendingDeprecationWarning`` which trumps a generic
# ``PendingDeprecationWarning`` ignore. Pulling the exact subclass
# here lets the filter target the same class langchain_core's surfacer
# uses, so it actually wins.
from langchain_core._api.deprecation import (  # noqa: E402
    LangChainDeprecationWarning,
    LangChainPendingDeprecationWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r"The default value of .allowed_objects. will change in a future version",
    category=LangChainPendingDeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The default value of .allowed_objects. will change in a future version",
    category=LangChainDeprecationWarning,
)

pytest_plugins = ["threetears.core.testing.fixtures"]
