"""canonical current-model pins -- the single source of truth.

every place the platform needs "the current chat / fast / large / embedding
model" reads a constant from HERE instead of hardcoding a model-id string.
source defaults (hub bootstrap, agent config defaults, the ``3tears init``
scaffold), and tests, all import these. a model rev is therefore exactly two
edits in this package: bump the constant below, and add/replace the matching
:class:`~threetears.models.capabilities.ModelCapabilities` entry in the
provider module -- nothing in the hub, the SDK, or any test changes.

this is a PIN, not "always latest". production runs known, tested models; the
constants change only when a human deliberately adopts a new pin. the
``scripts/check-model-currency.py`` job (in the hub repo) queries each
provider's live ``models.list`` endpoint and flags when a newer model exists
or a pin was retired upstream -- so drift is detected automatically and
surfaced as a one-line change, never discovered by accident.

invariants (enforced by ``tests/enforcement`` in the consuming repos):
- every MODEL-ID constant below MUST be present in the capabilities registry;
- raw model-id string literals are banned outside this module + the provider
  capability tables (the registry) + a small allowlist.

not every constant here is a model id. :data:`DEFAULT_MAX_TOKENS` is an
invocation default, so the registry invariant does not apply to it -- it is
here because it is the same KIND of thing: one value the whole platform reads
instead of each side carrying its own copy to drift.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_FAST_MODEL",
    "DEFAULT_LARGE_MODEL",
    "DEFAULT_MAX_TOKENS",
    "CURRENT_ANTHROPIC_CHAT_MODELS",
    "CURRENT_VOYAGEAI_EMBEDDING_MODELS",
]

#: maximum-capability chat model (deep reasoning, highest cost).
DEFAULT_LARGE_MODEL = "claude-opus-4-8"

#: balanced default chat model (quality vs. cost). matches the seeded lineup
#: (docker/seed/cluster.yaml) so a scaffolded agent's default resolves to a model
#: the gateway actually carries.
DEFAULT_CHAT_MODEL = "claude-sonnet-5"

#: cheap, fast utility chat model -- the default for scaffolded agents, the
#: schema-interview agent, and per-turn helpers (adversarial review, summaries).
DEFAULT_FAST_MODEL = "claude-haiku-4-5-20251001"

#: default embedding model (knowledge retrieval, agent memory).
DEFAULT_EMBEDDING_MODEL = "voyage-4"

#: default per-response output-token CAP for a chat completion. a ceiling, not a
#: target: room left unused costs nothing, while a cap set too low truncates the
#: answer itself. 4096 was that mistake -- a datasource agent is required to echo
#: the SQL it ran, so a long query plus any preamble could consume the budget
#: before the figure was ever stated (an eval case scored 0.20 with the right
#: number sitting in the tool result and absent from the prose).
#:
#: this is the ONE knob. every default that means "the output cap" reads it:
#: ``LLMConfig.max_tokens`` on both the hub and SDK sides. the transport and
#: chat-model classes deliberately have NO default -- every producer sets the
#: field explicitly, so a second number there could only ever drift out of step
#: with this one.
DEFAULT_MAX_TOKENS = 16384

#: the current Anthropic chat lineup the platform pins, largest to smallest.
#: bootstrap + the currency check read this; do not hardcode the ids elsewhere.
CURRENT_ANTHROPIC_CHAT_MODELS = (
    DEFAULT_LARGE_MODEL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_FAST_MODEL,
)

# OpenAI is intentionally NOT pinned. the platform runs Anthropic (chat) +
# VoyageAI (embedding); there is no OpenAI key in the stack, so the current
# OpenAI lineup cannot be verified live against ``models.list`` -- and pinning
# an unverified model id is the exact failure this module exists to kill. the
# capabilities registry keeps reference entries for OpenAI models that exist,
# but the platform does not seed or default to them. to adopt OpenAI: provision
# a key, verify the current ids live, then add a pin here.

#: the current VoyageAI embedding lineup the platform pins.
CURRENT_VOYAGEAI_EMBEDDING_MODELS = (
    DEFAULT_EMBEDDING_MODEL,
    "voyage-4-large",
)
