"""In-memory tool context manager with pluggable persistence."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


class ToolContextManager:
    """Manages conversation context: variables, tool results, media slots, and workflows.

    All state is held in memory.  Consuming applications can persist via event
    hooks or by subclassing.  No database dependency.
    """

    def __init__(
        self,
        conversation_id: str,
        user_id: str,
        *,
        var_limit: int = 50,
        var_max_chars: int = 50_000,
    ) -> None:
        self.conversation_id = conversation_id
        self.user_id = user_id
        self._var_limit = var_limit
        self._var_max_chars = var_max_chars

        # In-memory stores
        self._variables: dict[str, dict[str, Any]] = {}
        self._tool_results: list[dict[str, Any]] = []
        self._media_slots: dict[str, dict[str, Any]] = {}
        self._workflow: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------

    def set_variable(self, key: str, value: str, value_type: str = "string") -> str:
        """Set or update a variable.  Returns the context_id."""
        if key not in self._variables and len(self._variables) >= self._var_limit:
            raise ValueError(
                f"Variable limit reached ({self._var_limit}). Delete unused variables before adding new ones."
            )
        # Truncate if needed
        if len(value) > self._var_max_chars:
            value = value[: self._var_max_chars]

        existing = self._variables.get(key)
        context_id = existing["context_id"] if existing else str(uuid.uuid4())

        self._variables[key] = {
            "context_id": context_id,
            "key": key,
            "value": value,
            "value_type": value_type,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        return context_id

    def get_variable(self, key: str) -> dict[str, Any] | None:
        """Get a variable by key, or ``None`` if not found."""
        return self._variables.get(key)

    def get_all_variables(self) -> list[dict[str, Any]]:
        """Return all variables as a list of dicts."""
        return list(self._variables.values())

    def delete_variable(self, key: str) -> bool:
        """Delete a variable by key.  Returns ``True`` if it existed."""
        return self._variables.pop(key, None) is not None

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    def save_tool_result(
        self,
        tool_name: str,
        result: str,
        *,
        context_type: str = "tool_result",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a tool result.  Returns the generated context_id."""
        context_id = str(uuid.uuid4())
        self._tool_results.append(
            {
                "context_id": context_id,
                "context_type": context_type,
                "key": tool_name,
                "value": result,
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return context_id

    def get_context_item(self, context_id: str) -> dict[str, Any] | None:
        """Retrieve a context item (variable or tool result) by id."""
        for item in self._tool_results:
            if item["context_id"] == context_id:
                return item
        for var in self._variables.values():
            if var["context_id"] == context_id:
                return var
        return None

    # ------------------------------------------------------------------
    # Media slots
    # ------------------------------------------------------------------

    def register_media(self, slot_name: str, **kwargs: Any) -> None:
        """Register a media slot."""
        self._media_slots[slot_name] = {
            "slot_name": slot_name,
            **kwargs,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_slots(self) -> dict[str, dict[str, Any]]:
        """Return all registered media slots."""
        return dict(self._media_slots)

    def build_media_context(self) -> str | None:
        """Format media slots into a prompt string, or ``None`` if empty."""
        if not self._media_slots:
            return None
        lines = ["[Active Media Slots]"]
        for name, slot in self._media_slots.items():
            lines.append(f"- {name}: {slot}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    def declare_workflow(self, plan: str, steps: list[str]) -> dict[str, Any]:
        """Declare a new workflow with a plan and steps."""
        self._workflow = {
            "plan": plan,
            "steps": steps,
            "current_step": 0,
            "status": "active",
            "declared_at": datetime.now(timezone.utc).isoformat(),
        }
        return self._workflow

    def advance_workflow_step(self) -> dict[str, Any] | None:
        """Advance to the next workflow step.  Returns updated state or ``None``."""
        if self._workflow is None or self._workflow["status"] != "active":
            return None
        self._workflow["current_step"] += 1
        if self._workflow["current_step"] >= len(self._workflow["steps"]):
            self._workflow["status"] = "completed"
        return self._workflow

    def complete_workflow(self) -> dict[str, Any] | None:
        """Mark the workflow as completed.  Returns final state or ``None``."""
        if self._workflow is None:
            return None
        self._workflow["status"] = "completed"
        return self._workflow

    @property
    def has_active_workflow(self) -> bool:
        """Whether there is an active workflow."""
        return self._workflow is not None and self._workflow["status"] == "active"

    @property
    def workflow_state(self) -> dict[str, Any] | None:
        """Current workflow state, or ``None``."""
        return self._workflow

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_conversation_context(self) -> str | None:
        """Format variables and tool results into a prompt string.

        Returns ``None`` if there is no context to include.
        """
        sections: list[str] = []

        if self._variables:
            lines = ["[Conversation Variables]"]
            for var in self._variables.values():
                lines.append(f"- {var['key']} ({var['value_type']}): {var['value']}")
            sections.append("\n".join(lines))

        if self._tool_results:
            lines = ["[Tool Results]"]
            for item in self._tool_results:
                lines.append(f"- [{item['context_id']}] {item['key']}: {item['value']}")
            sections.append("\n".join(lines))

        if not sections:
            return None
        return "\n\n".join(sections)

    @property
    def has_context(self) -> bool:
        """Whether there is any context (variables, tool results, or media)."""
        return bool(self._variables or self._tool_results or self._media_slots)
