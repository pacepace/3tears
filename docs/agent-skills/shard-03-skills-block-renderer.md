# agent-skills / shard-03: Active-skill renderer + per-turn composition

> **Renumbered:** was shard-04 in the prior redesign; shard-02 (classifier) is DELETED per `metallm/docs/skills/PLACEMENT.md` §1.2.

## Objective

Land the **per-turn composition** functions: given an active skill (or `None`) and the consumer's base system prompt + base tool surface, produce the turn's composed system prompt + composed tool surface per PLACEMENT §1.10. Pure logic, no DB, no I/O. Two functions + a `ComposedTurnContext` dataclass. Replaces the old "multi-skill block renderer with truncation" shape from the prior redesign — under the new model (PLACEMENT §1.3), at most ONE skill is active per turn, so there's nothing to dedupe and no truncation algorithm to write.

## Locked design decisions (canonical source: `metallm/docs/skills/PLACEMENT.md`)

| Topic | Locked answer | PLACEMENT ref |
|---|---|---|
| Skills per turn | At most ONE active skill. No multi-skill blending. | §1.3 |
| Prompt mode | `'additive'` (default) appends body to base; `'replace'` substitutes base entirely. Per-user additions (NSFW, jailbreak) layered on top BY THE CONSUMER in either mode. | §1.1 |
| Tool composition | Base tools + skill.tool_additions − skill.tool_restrictions. ACL gates additions (skill cannot grant new capability). `tool_eligible` NOT re-checked for additions (skill is the visibility gate). | §1.10 |
| Returns | A `ComposedTurnContext` dataclass with `system_prompt: str` and `available_tool_names: list[str]`. | |

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| SK-18 | New module `threetears.agent.skills.rendering` with:<br>• `compose_turn_context(active_skill, base_system_prompt, base_tool_names, *, acl_permits) -> ComposedTurnContext` — the canonical composition function from PLACEMENT §1.10.<br>• `render_skill_body_block(skill) -> str` — formats a skill's prose body as a labeled markdown block (used by `'additive'` mode and by `skill_invoke` tool result; documented contract). | P0 |
| SK-19 | `compose_turn_context` MUST honor `prompt_mode`:<br>• `'additive'` → `system_prompt = base + "\n\n" + render_skill_body_block(active_skill)`<br>• `'replace'` → `system_prompt = render_skill_body_block(active_skill)` (or just `skill.body` directly with no header — see Implementation note 2)<br>• `active_skill is None` → `system_prompt = base`; `available_tool_names = base_tool_names` | P0 |
| SK-20 | `compose_turn_context` MUST apply tool composition:<br>• Start from `base_tool_names` (already filtered for `tool_eligible=True` AND ACL by the consumer).<br>• For each name in `skill.tool_additions`: if `acl_permits(name)` returns True, add to set (`tool_eligible` NOT checked — skill is the gate).<br>• For each name in `skill.tool_restrictions`: remove from set.<br>• Return as sorted list for deterministic output. | P0 |
| SK-21 | `acl_permits: Callable[[str], bool]` is the consumer-supplied callable. Tests use a stub; production wires through `3tears-registry.RbacEvaluatorAuthorizer`. | P0 |
| SK-22 | Per-user additions (NSFW, jailbreak, etc.) are NOT applied by this module. The consumer's system-prompt-assembly code applies them after `compose_turn_context` returns. Documented in the docstring. | P0 |
| SK-23 | Pure function discipline: no DB, no I/O, no `bump_use_count`, no logging at INFO. Logging at DEBUG OK (caller-provided logger pattern). | P0 |

---

## Public API

```python
# threetears.agent.skills.rendering

from dataclasses import dataclass
from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class ComposedTurnContext:
    system_prompt: str                          # composed prompt (base + skill body OR skill body only OR base if no skill)
    available_tool_names: list[str]             # sorted list of tool mcp_names for this turn
    active_skill_id: UUID | None                # echo back — convenience for logging / metrics


def compose_turn_context(
    active_skill: AgentSkillEntity | None,
    base_system_prompt: str,
    base_tool_names: Sequence[str],
    *,
    acl_permits: Callable[[str], bool],
) -> ComposedTurnContext:
    """Compose the turn's system prompt + tool surface per PLACEMENT §1.10.

    The caller is responsible for:
    - Building `base_system_prompt` (consumer's identity / persona prompt, already
      including memories / per-conversation overrides — but NOT per-user additions
      like NSFW or jailbreak; those are layered AFTER this function returns).
    - Building `base_tool_names` (already filtered by tool_eligible=True AND ACL).
    - Providing `acl_permits` (closure over the actor's identity).

    Returns ComposedTurnContext for the LLM call.
    """


def render_skill_body_block(skill: AgentSkillEntity) -> str:
    """Format a skill's body as a labeled markdown block.

    Used by:
    - compose_turn_context() for 'additive' mode (appended to base prompt).
    - skill_invoke tool result (shard-02) when delivering the active skill mid-turn.

    Returns empty string if skill.body is None (a pure tool-composition skill
    with no prose body).
    """
```

---

## Block format

For a skill with a prose body:

```
## Skill: <name>
<tags: [tag1, tag2]>

<body markdown>
```

The `<tags: ...>` line is omitted when `skill.tags` is empty.

For a pure tool-composition skill (body is None, only tool_additions/restrictions set): `render_skill_body_block` returns `""` (the empty string). The consumer typically skips the markdown block in this case and just lets the tool surface speak for itself — but a short auto-summary block is OK too:

```
## Active skill: <name>
<summary>
```

(Implementation choice — see Implementation note 4.)

---

## Composition logic (the function body)

```python
def compose_turn_context(active_skill, base_system_prompt, base_tool_names, *, acl_permits):
    available = set(base_tool_names)

    if active_skill is None:
        return ComposedTurnContext(
            system_prompt=base_system_prompt,
            available_tool_names=sorted(available),
            active_skill_id=None,
        )

    # Compose tool surface
    for name in active_skill.tool_additions:
        if acl_permits(name):
            available.add(name)
    for name in active_skill.tool_restrictions:
        available.discard(name)

    # Compose system prompt
    if active_skill.prompt_mode == 'replace':
        system_prompt = render_skill_body_block(active_skill) or active_skill.summary
    else:  # 'additive'
        body_block = render_skill_body_block(active_skill)
        if body_block:
            system_prompt = base_system_prompt + "\n\n" + body_block
        else:
            # additive mode + no body → no prompt change (tool composition only)
            system_prompt = base_system_prompt

    return ComposedTurnContext(
        system_prompt=system_prompt,
        available_tool_names=sorted(available),
        active_skill_id=active_skill.skill_id,
    )
```

The asymmetry per PLACEMENT §1.10:
- `tool_eligible` is checked for the BASE set (the caller already did this).
- `tool_eligible` is NOT checked for `tool_additions` (skill is the visibility gate).
- ACL is checked for both (skill cannot bypass ACL — `acl_permits` decides).

**⚠️ Critical contract for the renderer and its callers:**

The caller passes `base_tool_names` ALREADY FILTERED by `tool_eligible=True` AND ACL. The renderer assumes this and does NOT re-check `tool_eligible` for `tool_additions`. A "helpful" caller that pre-filters `tool_additions` by `tool_eligible` BEFORE passing them to the renderer (e.g. by intersecting `tool_additions` with the eligible-tools list) would BREAK the "code-skill without sandbox" pattern — `tool_eligible=False, skill_eligible=True` tools would never reach the available set even when surfaced by a loaded skill.

**Implementation rule:** the caller (metallm skills shard-03 / `apply_wake_skill`) MUST pass `tool_additions` through to `compose_turn_context` UN-FILTERED. Only ACL is checked, inside `compose_turn_context`, via the `acl_permits` callable. Double-filtering is a regression that defeats the entire eligibility-flags design.

---

## Patterns to follow

- Pure-function discipline: no DB, no I/O, no `bump_use_count` here. The consumer (metallm's `personality_node`) handles use-count + invocation recording.
- Test pattern: table-driven over `(active_skill, base_prompt, base_tools, acl_stub, expected_prompt, expected_tools)`.
- The `acl_permits` Protocol is a single callable returning bool — no need for a class. Mock is trivial.

---

## Files to create

```
packages/agent/skills/src/threetears/agent/skills/
└── rendering.py                                      # ComposedTurnContext + compose_turn_context + render_skill_body_block

packages/agent/skills/tests/
└── unit/
    └── test_rendering.py                            # exhaustive: no skill / additive prose / replace prose / pure tool-composition / additive+replace mode handling / acl-denies-addition / restriction-not-in-base / tool_eligible-false addition (gated by skill)
```

---

## Implementation notes

1. **`render_skill_body_block` returns `""` when `skill.body is None`.** The caller decides what to do (additive mode just doesn't append; replace mode falls back to `skill.summary` per the composition logic shown above).

2. **Replace mode + no body.** A pure tool-composition skill (body None) attached as `prompt_mode='replace'` is unusual but valid. The fallback: substitute the base prompt with just the skill's `summary` (a one-line replacement). Document this; the agent might want explicit guidance even in tool-only skills, so authors who use replace mode should populate body too. Validation in `skill_create` (shard-02) could WARN on this combination but not block it.

3. **Tag rendering.** `<tags: [tag1, tag2]>` only when `skill.tags` is non-empty.

4. **Block header for tool-only skills.** Open question (medium-low stakes): when a skill has `body=None` but is loaded, should `render_skill_body_block` return `""` (renderer does nothing) or a minimal `## Active skill: <name>\n<summary>` block (renderer announces the skill's existence)? Recommend: return `""`, let the consumer's wake-awareness block (long_running shard 06) announce the active skill separately. Keeps the renderer focused on body-rendering. Document this.

5. **No truncation.** The OLD shard had a truncation algorithm for multi-skill blocks. Under the new model (one skill per turn, max body 32KB enforced at skill_create), the system prompt's growth from a single skill is bounded. The consumer's overall context-window management is a separate concern (already handled by `services/context_budget.py` post-personality_cleanup).

6. **No use_count bump.** Side effects belong to the consumer. The renderer is pure.

7. **`acl_permits` Protocol.** A `Callable[[str], bool]` taking a tool name and returning True/False. The consumer wires this through `3tears-registry.RbacEvaluatorAuthorizer.is_authorized(actor, tool, "tool.call")` — but the renderer doesn't care; it just calls the callable. Tests mock with `lambda name: name not in {"forbidden_tool"}`.

8. **UTF-8 awareness.** Not relevant under single-skill model (no size capping), but `body` may contain non-ASCII. Python strings handle this transparently — no special handling needed.

---

## Anti-patterns

- DO NOT call `bump_use_count` or any other side effect from the renderer. Pure function.
- DO NOT layer per-user additions (NSFW, jailbreak) here. Those are consumer-side and applied AFTER this function returns.
- DO NOT add a cache-breakpoint parameter. Cache control is the consumer's concern.
- DO NOT mix skill rendering with memory rendering. Separate concerns; each has its own header.
- DO NOT special-case `prompt_mode='replace'` in a way that drops per-user additions. The renderer produces the BASE-or-replaced prompt; the consumer is responsible for layering per-user additions on either result.
- DO NOT re-add multi-skill truncation logic. One skill per turn (PLACEMENT §1.3).
- DO NOT check `tool_eligible` for `tool_additions`. The skill is the gate (PLACEMENT §1.10).

---

## Success criteria

- [ ] `compose_turn_context(None, base, tools, acl_permits=stub)` returns the base prompt + base tools unchanged.
- [ ] Additive mode with prose body appends correctly; tags rendered when present.
- [ ] Replace mode substitutes the system prompt entirely; per-user additions are NOT applied (consumer's job).
- [ ] `tool_additions` add to the surface; `tool_restrictions` remove.
- [ ] ACL-denied additions are silently dropped (no error — the renderer is a composition tool, not an authorization site).
- [ ] `tool_eligible=False` tools in `tool_additions` ARE added (skill is the gate; consumer's base set already filtered them out).
- [ ] Pure-tool-composition skill (body None) doesn't mutate the prompt in additive mode.
- [ ] Pure-tool-composition skill in replace mode substitutes with `summary` fallback.
- [ ] Linting + mypy strict clean.

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
./scripts/test.sh agent-skills
./scripts/lint.sh agent-skills
./scripts/typecheck.sh agent-skills
```

---

## Enforcement test suggestions

- Drift guard: `compose_turn_context` is the ONLY producer of composed `system_prompt + tool_surface` for skill-loaded turns. Grep across consumers (metallm) to confirm no in-line composition logic.
- Drift guard: NO DB / I/O imports in `rendering.py` (AST check).
- Drift guard: `render_skill_body_block` is the ONLY producer of `## Skill: <name>` headers. Cross-consumer grep.
