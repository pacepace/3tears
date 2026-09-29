"""Default prompts for memory extraction, worthiness gating, and resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_CONSOLIDATION_PROMPT",
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
- "Building a self-hosted LLM orchestrator: Python/FastAPI, LangGraph, pgvector, multi-provider support."
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

DEFAULT_WORTHINESS_PROMPT = """Does this turn hold a lasting fact worth knowing in a different conversation? It can be about the person, about the assistant and its work, or about a decision, a system or a rule that was agreed.

Yes: names, life details, preferences with their reasons, project details, relationships, decisions and why they were made, changes to systems or setup, limits that were found.

No: debugging steps, details that only matter in this session, general questions, tool use, greetings, small talk, general knowledge the person asked about, descriptions of media (those are kept elsewhere).

If it is only a maybe, answer false.

User message: {user_message}

The assistant's reply (its first 500 characters): {assistant_response_preview}

Return JSON: {{"worthy": true, "reason": "..."}} or {{"worthy": false}}"""

DEFAULT_RESOLUTION_PROMPT = """Decide what to do with each candidate memory. Compare the candidates with each other first, then with the existing memories.

One conversation gives one memory per topic. If two candidates cover the same topic, NOOP the one that says less.

Actions:
- "ADD": it says something no existing memory and no other candidate says.
- "UPDATE": it corrects or adds to an existing memory. Give memory_id and the new content, 1-2 sentences, not a paragraph. Write the new content in the existing memory's voice: if it says "I", keep "I"; keep names as names.
- "DELETE": it shows an existing memory is wrong. Give memory_id.
- "NOOP": an existing memory or another candidate already says it.

When an existing memory needs a correction, UPDATE it. Do not ADD a new one and DELETE the old one.

Return a JSON array with one decision per candidate:
- "index": the candidate's number, from 0
- "action": one of "ADD", "UPDATE", "DELETE", "NOOP"
- "memory_id": (UPDATE and DELETE) the existing memory's id
- "content": (UPDATE) the new content, 1-2 sentences
- "type": (UPDATE) the memory type

Return ONLY the JSON array, no other text.

{candidates_section}"""


DEFAULT_CONSOLIDATION_PROMPT = """You are consolidating a cluster of closely-related memories into a single gist.

These memories all cover the same theme. Synthesize them into ONE dense memory that captures the shared, durable truth — subsuming the details without losing any specific fact that still matters.

Rules:
- 1-2 dense sentences. Specific names, tools, technologies, dates. No filler.
- Preserve the specifics that appear across the sources; drop only genuine redundancy.
- FACTUAL over interpretive — state what is known, not what it implies.
- Do NOT invent facts absent from the sources.

Return ONLY JSON, no other text:
{{"gist": "the consolidated 1-2 sentence memory", "rationale": "one short sentence on why these merged"}}

Memories to consolidate:
{sources_section}"""


@dataclass
class ExtractionPrompts:
    """Customizable prompts for the memory extraction + consolidation pipeline."""

    extraction: str = field(default=DEFAULT_EXTRACTION_PROMPT)
    worthiness: str = field(default=DEFAULT_WORTHINESS_PROMPT)
    resolution: str = field(default=DEFAULT_RESOLUTION_PROMPT)
    consolidation: str = field(default=DEFAULT_CONSOLIDATION_PROMPT)
