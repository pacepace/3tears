"""The package's own prompts, declared once: each id as a constant, and the catalogue a host registers them from.

The engine steers three LLM calls of its own with a prompt — the analysis generator, the capability
proposer and the boundary proposer — and each prompt is store-master: a host seeds an empty slot of its
prompt registry from the shipped default, and an operator tunes the stored copy with no deploy. So a
host needs two things from the engine for each one: the id its registry keys the slot by, and where
the shipped default lives. :data:`SEED_PROMPTS` is both, as
:class:`~threetears.evals.kernel.prompts.seed.SeedPrompt` declarations.

**Each id is spelled once, here.** A seed constant names its own id (a template's ``template_type``)
by the constant below rather than by a literal, and so does every message that names the slot, so an
id the catalogue declares and the seed it points at cannot disagree; ``tests/test_prompt_catalogue.py``
resolves every entry by import and holds each to it.

**Nothing here imports a seed.** The catalogue states where each default lives — a module path, a
source path and a constant name — and a host's registry imports it from there. That keeps the
kernel free of the packages that own the prompts (``analysis`` and ``gen``), which depend on the
kernel and not the other way round, and it is the direction :class:`SeedPrompt` exists for: a prompt
that moves inside the engine moves its own entry here, and no host edit follows.

**A host registers its own prompts through the same type.** Its catalogue is a tuple of
:class:`SeedPrompt` beside this one, and its registry seeds from the concatenation; the engine's ids
are prefixed ``eval_`` so a host's own need not avoid any other name.
"""

from __future__ import annotations

from threetears.evals.kernel.prompts.seed import KIND_TEMPLATE, KIND_TEXT, SeedPrompt

#: The analysis generator's system prompt — a single string, seeded into a shared-prompt registry. The
#: registry key :class:`~threetears.evals.ops.AnalysisGeneration` resolves the generator's prompt under,
#: and the subject a reporter run measures.
ANALYSIS_GEN_PROMPT = "eval_analysis_gen"

#: The capability-rubric proposer's prompt — a sectioned template, seeded into a template registry.
PROPOSER_PROMPT = "eval_proposer"

#: The adversarial boundary proposer's prompt — a sectioned template, seeded into a template registry.
BOUNDARY_GEN_PROMPT = "eval_boundary_gen"

#: Where the package's source files sit in the repository, for each entry's ``definition_path`` — the
#: file a promotion writes a tuned preset back into.
_SOURCE_ROOT = "packages/evals/src/threetears/evals"

#: Every prompt the engine ships a default for, in the order a registry seeds them. A host seeds an empty
#: slot from each entry's :attr:`~SeedPrompt.seed_ref` and never overwrites a stored one.
SEED_PROMPTS: tuple[SeedPrompt, ...] = (
    SeedPrompt(
        kind=KIND_TEXT,
        prompt_type=ANALYSIS_GEN_PROMPT,
        seed_module="threetears.evals.analysis.gen_prompt",
        definition_path=f"{_SOURCE_ROOT}/analysis/gen_prompt.py",
        constant="EVAL_ANALYSIS_GEN_DEFAULT",
    ),
    SeedPrompt(
        kind=KIND_TEMPLATE,
        prompt_type=PROPOSER_PROMPT,
        seed_module="threetears.evals.gen.prompts.proposer",
        definition_path=f"{_SOURCE_ROOT}/gen/prompts/proposer.py",
        constant="EVAL_PROPOSER_TEMPLATE_DEFAULT",
    ),
    SeedPrompt(
        kind=KIND_TEMPLATE,
        prompt_type=BOUNDARY_GEN_PROMPT,
        seed_module="threetears.evals.gen.prompts.boundary_gen",
        definition_path=f"{_SOURCE_ROOT}/gen/prompts/boundary_gen.py",
        constant="EVAL_BOUNDARY_GEN_TEMPLATE_DEFAULT",
    ),
)


__all__ = [
    "ANALYSIS_GEN_PROMPT",
    "BOUNDARY_GEN_PROMPT",
    "PROPOSER_PROMPT",
    "SEED_PROMPTS",
]
