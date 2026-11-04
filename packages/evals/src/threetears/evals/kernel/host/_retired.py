"""The host contract's names the retired writers left, kept for one minor after they retired.

A host declared its tone register with :class:`StyleProfile` and :data:`ToneRegister`, and binds them at import, so each
stays importable from :mod:`threetears.evals.kernel.host` and warns on every read
(:func:`~threetears.evals.kernel.retired.served_retired`). Nothing reads either. The next minor removes them.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from threetears.evals.kernel.host.style import StyleProfile, ToneRegister
from threetears.evals.kernel.retired import HOST_STYLE_RETIRED, Retired

#: Each retired name the host contract still serves, with why it retired.
RETIRED: Mapping[str, Retired] = MappingProxyType(
    {
        "StyleProfile": Retired(StyleProfile, HOST_STYLE_RETIRED),
        "ToneRegister": Retired(ToneRegister, HOST_STYLE_RETIRED),
    }
)

__all__ = ["RETIRED", "StyleProfile", "ToneRegister"]
