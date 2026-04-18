"""Default prompts for memory extraction, worthiness gating, and resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_EXTRACTION_PROMPT",
    "DEFAULT_RESOLUTION_PROMPT",
    "DEFAULT_WORTHINESS_PROMPT",
    "ExtractionPrompts",
]

DEFAULT_EXTRACTION_PROMPT = """Extract durable facts from this conversation turn. This includes facts about the user, but also facts about the system, infrastructure decisions, established norms, and the working relationship.

Return a JSON array. Each item has:
- "type": one of "preference", "fact", "decision", "topical_context", "relational_context"
- "content": 1-2 dense sentences. Specific names, tools, technologies, dates. No filler.

Types:
- "preference": Likes, dislikes, working style, aesthetic taste
- "fact": Biographical details, relationships, skills, location, job
- "decision": A choice they made and why (both parts required)
- "topical_context": A specific ongoing project or goal with concrete details
- "relational_context": System/infrastructure decisions, established workflow norms, technical constraints discovered together, operational facts about how the system is configured or why

Rules:
- SPECIFIC over general — proper nouns, versions, tool names
- FACTUAL over interpretive — what they said, not what it implies about them
- DENSE over verbose — every word must earn its place
- DURABLE over ephemeral — useful in a new conversation next month, not just today
- At most 1-3 memories per turn. Most turns produce 0-1.

DO NOT extract:
- Session-specific debugging steps or troubleshooting details
- Personality interpretation ("The user seems to prefer...", "They appear frustrated...")
- Generic questions the user asked (that's LLM knowledge, not user info)
- Vague summaries ("Working on a project", "Discussing technical topics")
- Media descriptions (images, documents) — stored separately
- Greetings, small talk, tool usage commands
- Assistant personality traits or self-concept — identity is defined by configuration, not memory

Bad examples (with reasons):
- "The user is working on debugging a WebSocket issue where messages aren't being delivered correctly, involving NATS JetStream configuration" — session-specific debugging, won't help next month
- "The user seems to be a detail-oriented developer who values thorough testing" — personality interpretation, not a fact
- "Working on a Python project with FastAPI" — too vague, no project name or distinguishing details
- "The user prefers clean code" — universal truism, not a real preference
- "The assistant is helpful and thorough" — self-concept, not an operational fact

Good examples:
- "Building MetaLLM, a self-hosted LLM orchestrator: Python/FastAPI, LangGraph, pgvector, multi-provider support."
- "Daughter named Lily, starting kindergarten fall 2026."
- "Prefers PostgreSQL over MongoDB for new projects — cites JSONB, mature tooling, pgvector."
- "Lives in Seattle. Works remotely as a senior backend engineer at Acme Corp."
- "Switched from OpenAI text-embedding-3-large to VoyageAI voyage-4 (1024 dims) for embeddings — political and technical reasons."
- "pgvector HNSW index has a hard 2000-dimension ceiling; embedding dims set to 1024 to stay within limit."
- "Memory system was intentionally cleared and rebuilt from scratch during the embedding provider migration."

If nothing durable is worth remembering, return [].
Return ONLY the JSON array, no other text.

User message: {user_message}

Assistant response: {assistant_response}"""

DEFAULT_WORTHINESS_PROMPT = """Does this turn reveal a durable fact that would be useful in a completely different conversation? This includes facts about the user, but also system/infrastructure decisions, technical constraints, or established norms.

YES: proper names, biographical details, specific preferences with reasons, concrete project details, relationships, decisions with rationale, system configuration changes, infrastructure decisions and their reasoning, discovered technical constraints

NO: debugging/troubleshooting steps, session-specific technical details, generic questions, tool usage, greetings, small talk, things the user is asking about (LLM knowledge, not user info), media descriptions (stored separately)

If the answer is only "maybe", return false. Bias toward rejection.

User message: {user_message}

Assistant response (first 500 chars): {assistant_response_preview}

Return JSON: {{"worthy": true, "reason": "..."}} or {{"worthy": false}}"""

DEFAULT_RESOLUTION_PROMPT = """You are a memory manager. First check candidates against EACH OTHER for thematic overlap, then compare against existing memories.

One conversation should not produce multiple memories about the same theme. If two candidates cover the same topic, NOOP the weaker one.

Actions:
- "ADD": Genuinely new information not in any existing memory or other candidate.
- "UPDATE": Refines or corrects an existing memory. Provide memory_id and updated content. Updated content must stay 1-2 sentences — do not merge into a paragraph.
- "DELETE": New info makes an existing memory factually wrong. Provide memory_id.
- "NOOP": Already captured by an existing memory OR overlaps with another candidate being ADDed. Skip it.

Return a JSON array with one decision per candidate:
- "index": candidate index (0-based)
- "action": one of "ADD", "UPDATE", "DELETE", "NOOP"
- "memory_id": (required for UPDATE and DELETE) existing memory ID
- "content": (required for UPDATE) new merged content (1-2 sentences max)
- "type": (required for UPDATE) memory type

Rules:
- Prefer NOOP over ADD when the information is essentially already known
- Prefer UPDATE over ADD+DELETE when a memory just needs refinement
- Only use DELETE when new information makes an old memory factually wrong
- Return ONLY the JSON array, no other text

{candidates_section}"""


@dataclass
class ExtractionPrompts:
    """Customizable prompts for the memory extraction pipeline."""

    extraction: str = field(default=DEFAULT_EXTRACTION_PROMPT)
    worthiness: str = field(default=DEFAULT_WORTHINESS_PROMPT)
    resolution: str = field(default=DEFAULT_RESOLUTION_PROMPT)
