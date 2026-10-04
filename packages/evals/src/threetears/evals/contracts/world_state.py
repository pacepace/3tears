"""WorldState — the stateful eval substrate.

A :class:`WorldState` is a serializable container that holds the seeded world one in-flight
scenario runs against. A host seeds each value into its candidate through its world registry's own
handles, so what the candidate perceives is the SAME world; it mirrors every action the candidate
took that succeeded into it as a recorded call; and the goal-state DSL reads the final state to
score the run.

**A namespace sub-key is a declared world dimension or it is not seedable.** This container will
hold any key; what a run may put in one is the registry's answer, and a seed naming something no
dimension declares is refused before anything is applied. Accepting it would leave a value the
candidate's tools never received for the goal judge to read anyway — the candidate/judge divergence
the seeding step exists to prevent.

**A seed is data, never a reference.** Every value is the namespace's literal initial state as the
template states it. Nothing here resolves a value against the subject: a world that depends on what
the subject carries is the host's to build, through the seams its kind owns, and the engine never
learns what a subject carries.

The contract is pure data (no LLM clients, no I/O), and nothing a host's tools are built from needs to
import it: a host seeds its tools from outside them, taking a namespace as a plain ``dict[str, Any]``.

Namespaces
----------

State is keyed by tool name. Sub-keys are tool-specific. For example, a calendar tool might store::

    state.namespace("calendar") == {
        "events": [...],            # scenario-provided events
        "calls": [{"action": ..., "params": ...}, ...],
    }

Every action the candidate takes that succeeded is recorded under the ``calls`` sub-key, and on the
cross-tool :attr:`WorldState.global_calls` ledger, so goal-state checks can assert call ordering
and counts. The host records them from what its tools actually returned, so the audit trail
describes what the candidate did through the real tool, not what a simulation reported.

Serialization
-------------

``WorldState.serialize()`` returns a JSON-safe dict; ``deserialize`` round-trips it. Mutations of
the namespace dict reference (via :meth:`namespace`) persist into the state automatically — callers
must not assume the returned dict is a copy.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from threetears.evals.contracts.models import WorldSeed


class WorldState(BaseModel):
    """Per-scenario seeded world the candidate runs against.

    Built from the template's ``world_seed`` (:func:`init_world`); the host mirrors
    each action the candidate took that succeeded into it, and the goal-state DSL reads the
    final state at scenario end.

    The state is JSON-serializable via :meth:`serialize` /
    :meth:`deserialize` — tools must store only primitive types,
    dicts, and lists.
    """

    model_config = ConfigDict(arbitrary_types_allowed=False, extra="forbid")

    namespaces: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Tool-name → mutable per-tool state dict.",
    )

    # Cross-tool call ledger — appended in order across every tool's
    # recorded actions. Used by the goal-state DSL's ordering
    # predicates (called_before / called_after / call_count / last_call_was)
    # to reason about action sequencing across tools, which per-namespace
    # call lists cannot express on their own.
    global_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{tool, action, params}, ...] in recorded order across all tools.",
    )

    def namespace(self, tool_name: str) -> dict[str, Any]:
        """Return the mutable per-tool state dict, lazily creating it.

        The returned dict is a reference — callers mutate it directly
        and the changes persist on the state instance.
        """
        return self.namespaces.setdefault(tool_name, {})

    def record_call(self, tool_name: str, action: str, params: dict[str, Any]) -> None:
        """Append the call to per-tool and global ledgers for goal-state inspection.

        Two writes:

        * ``state.namespace(tool).calls`` — per-tool trace; existing
          consumers (and their tests) read this for tool-local assertions.
        * ``state.global_calls`` — cross-tool ledger preserving recorded
          order across every tool's recorded actions. Goal-state
          DSL predicates (``called_before``, ``call_count``,
          ``last_call_was``) read this — per-tool calls can't express
          cross-tool ordering.

        Both writes copy ``params`` so the caller can mutate the original
        dict after the call without affecting the recorded value.
        """
        ns = self.namespace(tool_name)
        params_copy = dict(params)
        ns.setdefault("calls", []).append({"action": action, "params": params_copy})
        self.global_calls.append({"tool": tool_name, "action": action, "params": params_copy})

    def serialize(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation of the state.

        The warning set is not included — it's a per-run flag, not
        scenario data.
        """
        return self.model_dump()

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> WorldState:
        """Round-trip from a serialized dict back into an instance."""
        return cls.model_validate(data)


def init_world(seed: WorldSeed) -> WorldState:
    """Initialize a :class:`WorldState` from a template's seed.

    Each key in ``seed.namespaces`` becomes a namespace, its value deep-copied so the state can
    be mutated without touching the seed:

    * A dict is the namespace's initial state, verbatim.
    * Anything else — a list, a string, a scalar — is promoted into ``{"value": <it>}``, so every
      namespace is dict-shaped: :meth:`WorldState.namespace` and
      :meth:`WorldState.record_call` both require it.

    Args:
        seed: The template's :class:`~threetears.evals.contracts.models.WorldSeed`.

    Returns:
        A fresh :class:`WorldState` with namespaces populated. No calls recorded — those are
        recorded as the candidate acts.
    """
    state = WorldState()
    for tool_name, raw_value in seed.namespaces.items():
        value = copy.deepcopy(raw_value)
        state.namespaces[tool_name] = value if isinstance(value, dict) else {"value": value}
    return state


__all__ = [
    "WorldState",
    "init_world",
]
