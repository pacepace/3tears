"""Literal types + enum constants for agent-skills.

These mirror the CHECK constraints on the L3 tables: changing the
allowed values here requires a matching migration.

Why ``Literal`` and not ``Enum``: callers (the rest of 3tears, plus
external consumers) pass these values through tool input schemas + JSON
boundaries where strings round-trip cleanly. ``Literal`` keeps the
runtime payload a plain ``str`` so JSON encoding doesn't need a custom
serializer and mypy still pins valid value sets at every call site.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "InvocationSource",
    "OutcomeSource",
    "PromptMode",
    "SkillKind",
    "SkillKindFilter",
    "SkillOutcome",
    "SkillSource",
]


# ``prompt_mode`` column on ``agent_skills``: governs how the skill's
# ``body`` interacts with the consumer's base system prompt when the
# skill is active for a turn. Pure tool-composition skills (no body)
# still carry ``prompt_mode='additive'`` -- the renderer in shard 03
# treats a NULL body under either mode as "no prose addition".
PromptMode = Literal["additive", "replace"]


# ``source`` column on ``agent_skills``. Only ``'manual'`` exists in
# v1; the original ``'distilled'`` value tracked
# ``skill_create_from_range`` output which was dropped per PLACEMENT
# §1.4. Kept as a typed enum so re-introducing distillation later is
# a Literal-update only (no migration required because the column is
# free-text by CHECK absence).
SkillSource = Literal["manual"]


# ``invocation_source`` column on ``agent_skill_invocations``. ``'wake'``
# means the row was created when a wake schedule's ``skill_id`` resolved
# at fire time; ``'invoke'`` means the agent called ``skill_invoke``
# mid-user-turn. CHECK-pinned in the L3 schema.
InvocationSource = Literal["wake", "invoke"]


# ``outcome`` column on ``agent_skill_invocations``. NULL until
# something attributes an outcome to the invocation; today the only
# writer is the ``skill_report_outcome`` tool (see ``tools.py``), which
# replaced the retired ``[SUCCESS]``/``[FAILED]`` text-marker hook.
# CHECK-pinned in the L3 schema.
SkillOutcome = Literal["success", "failure"]


# ``kind`` discriminator on the skills REST surface. ``'prose'`` is an
# ``agent_skills`` row; ``'tool'`` is synthesized by the router from the
# tool registry and is never table-backed. Not a database column -- the
# table holds prose skills only -- so this is a wire-shape vocabulary,
# not a CHECK mirror.
SkillKind = Literal["prose", "tool"]


# ``kind_filter`` on the catalog-listing tool: :data:`SkillKind` widened
# with the no-filter sentinel. Kept as its own alias rather than an
# inline union so the sentinel is named once; a value added to
# :data:`SkillKind` must be added here too, which the enforcement gate
# in ``tests/enforcement/test_vocabulary_single_source.py`` does not
# catch and a reviewer must.
SkillKindFilter = Literal["all", "prose", "tool"]


# ``outcome_source`` column on ``agent_skill_invocations``. Records the
# provenance of the populated ``outcome`` value:
#
# - ``'agent_marker'`` -- DEPRECATED: was parsed from the assistant's
#   response text (the synchronous post-LLM hook path). Superseded by
#   ``'agent_tool'``; kept for historical rows only, no longer written.
# - ``'agent_tool'`` -- the agent self-reported via the
#   ``skill_report_outcome`` tool call. Never enters the visible
#   response stream (leak-proof by construction, unlike the retired
#   text-marker path).
# - ``'user_feedback'`` -- attributed to user-driven feedback (reserved
#   for future enhancement; not populated in v1 but typed here so the
#   column can carry the value without a follow-up migration).
#
# No CHECK constraint backs this column (``tables.py`` -- plain
# ``Text()``), so adding a value is Literal-only; no migration required.
OutcomeSource = Literal["agent_marker", "agent_tool", "user_feedback"]
