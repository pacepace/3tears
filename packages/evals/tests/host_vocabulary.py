"""The host nouns the engine's vocabulary canaries refuse, shared by every suite that reads them.

One list, because three suites match on it: the shared-contract name scan
(``test_no_host_names_in_shared_contract.py``), the prose ceiling over the engine
(``test_host_vocabulary_ceiling.py``) and the reporting suite. Two copies of a word list drift,
and a word missing from one copy is a silent exemption rather than a smaller check.
"""

from __future__ import annotations

__all__ = ["HOST_NOUNS"]

#: Nouns that belong to a host and not to an eval engine.
HOST_NOUNS: tuple[str, ...] = (
    "persona",
    "discord",
    "music",
    "dj",
    "universe",
    "segue",
    # The research tool is the second archetype of host coupling. A ratchet is only evidence about
    # the nouns it holds, so a missing noun is not a smaller ratchet, it is a silent exemption — and
    # modules declared shared contract had carried research names for as long as it was missing.
    "research",
    "tavily",
    "openrouter",
    # A host's world may model a room — who is present, who arrives, the audience — and eval core names
    # none of it: listeners, arrivals and presence are a host's vocabulary and its seed templates', never
    # the engine's. A generic seam a host fills is the only way this may reach the shared contract.
    "listener",
    "arrival",
    "presence",
    "audience",
)
