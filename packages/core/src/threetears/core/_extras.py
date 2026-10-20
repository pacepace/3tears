"""import a core module, naming the extra to install when one of its dependencies is missing.

``cryptography`` and ``pyjwt`` are the ``security`` extra and ``httpx`` the ``http`` extra: the
signed-token and encryption modules of :mod:`threetears.core.security`, the NATS L3 proxy that signs
its door requests with them, and the traced HTTP client. Everything else in core imports without
them. The lazy package surfaces resolve through :func:`import_extras_module`, so a caller reaching
one of those names on an install without the extra is told which extra to add instead of only
which module was missing.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["import_extras_module"]

#: the top-level module of each distribution an extra provides, and the extra it comes from
_EXTRA_FOR_MODULE: dict[str, str] = {
    "cryptography": "security",
    "jwt": "security",
    "httpx": "http",
}


def import_extras_module(module_name: str) -> ModuleType:
    """import *module_name*, or say which extra its missing dependency comes from.

    :param module_name: the dotted module to import
    :ptype module_name: str
    :return: the module
    :rtype: ModuleType
    :raises ImportError: naming the extra, when the module needs one that is not installed
    :raises ModuleNotFoundError: unchanged, for any other missing module
    """
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        extra = _EXTRA_FOR_MODULE.get(missing)
        if extra is None:
            raise
        raise ImportError(
            f"{module_name} needs {missing}, which comes with the {extra} extra: install 3tears[{extra}]",
        ) from exc
