"""Active-skill renderer + per-turn composition (shard-03).

Two public functions plus one dataclass implement the canonical
per-turn composition from the skills placement spec section
1.10:

- :func:`compose_turn_context` -- given the consumer's base system
  prompt + base tool surface and an optional active skill, produce a
  :class:`ComposedTurnContext` for the LLM call.
- :func:`render_skill_body_block` -- format a skill's prose body as a
  labeled markdown block (also used by ``skill_invoke``'s tool-result
  payload).

Both are **pure** (no DB, no I/O, no logging at INFO). The consumer
(the personality node) is responsible for:

- Building ``base_system_prompt`` (identity / persona, memories,
  per-conversation overrides) -- BUT NOT per-user additions (NSFW,
  jailbreak). Per-user additions are layered AFTER this function
  returns; the renderer never sees them and never decides about them.
- Building ``base_tool_names`` -- already filtered by
  ``tool_eligible=True`` AND ACL.
- Wiring ``acl_permits`` as a closure over the actor's identity.

The asymmetry the spec demands:

- ``tool_eligible`` is checked by the *consumer* for the BASE set; the
  renderer does NOT re-check it for ``tool_additions``. The active
  skill is the visibility gate for additions.
- ACL is checked by the renderer for ``tool_additions`` (a skill cannot
  bypass authorization). ``tool_restrictions`` does not consult ACL
  (removing a tool from the surface is always permitted).

Single skill per turn (PLACEMENT section 1.3) -- no multi-skill
blending, no truncation, no dedupe. The signature takes a single
optional :class:`AgentSkillEntity`, not a list.

Spec ref: ``docs/agent-skills/shard-03-skills-block-renderer.md``
requirements SK-18 .. SK-23.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from uuid import UUID

from threetears.agent.skills.entities import AgentSkillEntity

__all__ = [
    "ComposedTurnContext",
    "compose_turn_context",
    "render_skill_body_block",
]


@dataclass(frozen=True)
class ComposedTurnContext:
    """Per-turn composed prompt + tool surface for an LLM call.

    Returned by :func:`compose_turn_context`. Immutable so callers can
    pass the value through layers without worrying about downstream
    mutation.

    :ivar system_prompt: the composed system prompt. Equals
        ``base_system_prompt`` when no skill is active or when an
        ``'additive'``-mode skill has no body; otherwise reflects the
        skill body composition per the skills placement spec
        section 1.10. Per-user
        additions (NSFW, jailbreak) are NOT applied -- the consumer
        layers them on top after this function returns.
    :ivar available_tool_names: sorted list of tool ``mcp_name`` values
        the LLM may call this turn. ``base_tool_names`` plus
        ACL-permitted entries from ``active_skill.tool_additions``
        minus entries in ``active_skill.tool_restrictions``. Sorted
        for deterministic test assertions and prompt-cache stability.
    :ivar active_skill_id: the skill UUID echoed back for logging /
        metrics. ``None`` when no skill was active.
    """

    system_prompt: str
    available_tool_names: list[str]
    active_skill_id: UUID | None


def render_skill_body_block(skill: AgentSkillEntity) -> str:
    """Format a skill's prose body as a labeled markdown block.

    Used by:

    - :func:`compose_turn_context` for ``'additive'`` mode (appended to
      the consumer's base prompt) and for the ``'replace'`` substitution.
    - ``skill_invoke`` (shard-02) when delivering the active skill mid
      user-turn -- the tool-result payload reuses this exact rendering
      so the LLM sees the same block whether the skill arrived via a
      wake attachment or an explicit invoke.

    Block shape (when ``skill.body`` is non-empty)::

        ## Skill: <name>
        <tags: [tag1, tag2]>

        <body markdown>

    The ``<tags: ...>`` line is omitted when ``skill.tags`` is empty.

    Returns the empty string when ``skill.body`` is ``None`` (a pure
    tool-composition skill with no prose body). Callers decide what to
    do with an empty result -- ``'additive'`` mode appends nothing,
    ``'replace'`` mode falls back to ``skill.summary``.

    :param skill: the active skill whose body should be rendered
    :ptype skill: AgentSkillEntity
    :return: the labeled markdown block, or ``""`` if no body
    :rtype: str
    """
    body = skill.body
    if not body:
        return ""

    header_lines = [f"## Skill: {skill.name}"]
    tags = skill.tags
    if tags:
        tag_list = ", ".join(tags)
        header_lines.append(f"<tags: [{tag_list}]>")
    header = "\n".join(header_lines)
    return f"{header}\n\n{body}"


def compose_turn_context(
    active_skill: AgentSkillEntity | None,
    base_system_prompt: str,
    base_tool_names: Sequence[str],
    *,
    acl_permits: Callable[[str], bool],
) -> ComposedTurnContext:
    """Compose the turn's system prompt + tool surface per PLACEMENT 1.10.

    The canonical per-turn composition step. The caller is responsible
    for:

    - Building ``base_system_prompt`` -- consumer's identity / persona
      prompt (already including memories / per-conversation overrides)
      but NOT per-user additions like NSFW or jailbreak. The consumer
      layers those AFTER this function returns.
    - Building ``base_tool_names`` -- the tool surface the LLM would see
      with no skill loaded. Already filtered by ``tool_eligible=True``
      AND ACL.
    - Providing ``acl_permits`` -- a closure over the actor's identity
      that returns ``True`` iff the actor may call the named tool.

    Composition rules:

    - ``active_skill is None`` -> base prompt + base tools unchanged.
    - ``prompt_mode == 'additive'`` -> ``base_system_prompt`` followed
      by the rendered skill body block (per
      :func:`render_skill_body_block`). When the body is empty, the
      base prompt is returned unchanged.
    - ``prompt_mode == 'replace'`` -> the rendered skill body block
      substitutes the base prompt entirely. When the body is empty,
      falls back to ``active_skill.summary`` so the LLM has *some*
      identity. Per-user additions still apply on top (consumer's
      responsibility).
    - For each ``name`` in ``active_skill.tool_additions``: add to the
      surface iff ``acl_permits(name)`` returns ``True``.
      ``tool_eligible`` is NOT re-checked here -- the skill is the
      visibility gate.
    - For each ``name`` in ``active_skill.tool_restrictions``: remove
      from the surface. ACL is NOT consulted (subtractive operations
      are always permitted; you can always remove a tool from your own
      surface). ``discard`` semantics -- referencing a tool that was
      not in ``base_tool_names`` is a no-op, not an error.

    Pure function. No DB. No I/O. No ``bump_use_count`` -- the consumer
    handles use-count + invocation recording as a side effect outside
    the renderer.

    :param active_skill: the skill loaded for this turn, or ``None``
    :ptype active_skill: AgentSkillEntity | None
    :param base_system_prompt: the consumer's base system prompt
    :ptype base_system_prompt: str
    :param base_tool_names: tool ``mcp_name`` entries already filtered
        by ``tool_eligible=True`` AND ACL
    :ptype base_tool_names: Sequence[str]
    :param acl_permits: closure returning ``True`` iff the actor may
        call the named tool
    :ptype acl_permits: Callable[[str], bool]
    :return: composed prompt + composed tool surface + echo of the
        active skill id
    :rtype: ComposedTurnContext
    """
    available: set[str] = set(base_tool_names)

    if active_skill is None:
        return ComposedTurnContext(
            system_prompt=base_system_prompt,
            available_tool_names=sorted(available),
            active_skill_id=None,
        )

    # Compose tool surface. Additions consult ACL; restrictions do not.
    for name in active_skill.tool_additions:
        if acl_permits(name):
            available.add(name)
    for name in active_skill.tool_restrictions:
        available.discard(name)

    # Compose system prompt per prompt_mode.
    body_block = render_skill_body_block(active_skill)
    if active_skill.prompt_mode == "replace":
        # Replace-mode + no body falls back to the skill summary so the
        # LLM still has identity guidance. ``skill_create`` SHOULD warn
        # on this combination (shard-02 Implementation note 2) but
        # cannot block -- enforce graceful fallback here.
        system_prompt = body_block or active_skill.summary
    else:
        # Additive mode. Empty body -> no prompt change (tool
        # composition only).
        if body_block:
            system_prompt = f"{base_system_prompt}\n\n{body_block}"
        else:
            system_prompt = base_system_prompt

    return ComposedTurnContext(
        system_prompt=system_prompt,
        available_tool_names=sorted(available),
        active_skill_id=active_skill.skill_id,
    )
