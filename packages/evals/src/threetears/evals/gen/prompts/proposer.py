"""Seed default for the ``eval_proposer`` prompt template.

The capability-rubric proposer: it is handed a subject and a catalog of rubric
dimensions other operators already trust, and drafts a capability rubric plus
the scenario variation axes that would stress what it scores.

Two rules in the text are load-bearing rather than stylistic, and
``tests/test_gen_proposers.py`` asserts both stay in it: dimension names must be
namespaced ``<context>.<dim>``, because a judge config binds to a dim by name across
every template, and a reused catalog dim keeps its ``key``. A draft naming a bare
dimension fails validation (:data:`~threetears.evals.contracts.models.DimName`) and
the paid call is discarded, so the prompt has to teach the rule.

Seeded empty-slot-only; the store is master once set.
"""

from __future__ import annotations

from threetears.evals.contracts.prompts.seed import SeedSection, SeedTemplate

EVAL_PROPOSER_TEMPLATE_DEFAULT = SeedTemplate(
    template_name="default",
    template_type="eval_proposer",
    scope="base",
    description="Eval rubric proposer — drafts a capability rubric + scenario axes for a subject.",
    sections=(
        SeedSection(
            section_name="proposer_instructions",
            zone="primacy",
            order=0,
            condition="always",
            formatter="text_template",
            content_template=(
                "You design behavioral evaluation rubrics for AI systems.\n"
                "\n"
                "You will be given two inputs:\n"
                "1. A SUBJECT: the system under evaluation, as its operator describes it — its\n"
                "   self-description, its standing instructions, and the tools it has, each with whatever\n"
                "   the operator says about how it reasons with that tool. This is the source for WHAT is\n"
                "   worth testing about this subject.\n"
                "2. A CATALOG of reusable rubric dimensions other operators already trust. Each carries a\n"
                "   stable `key`, a name, a description, and a scoring guide.\n"
                "\n"
                "Your job is to DRAFT a CAPABILITY rubric plus scenario variation axes for this subject.\n"
                "\n"
                "Rules:\n"
                "- A rubric dimension scores a genuinely SUBJECTIVE quality (tone, style, curiosity,\n"
                "  explanation quality, judgment under ambiguity). Do NOT propose objective, pass/fail\n"
                "  checks (e.g. 'saved the record', 'called the right tool') — those are goal-state\n"
                "  checks, decided mechanically, and are out of scope here.\n"
                "- PREFER REUSE: when a catalog dimension already captures a quality this subject needs\n"
                "  tested, reuse it verbatim rather than inventing a near-duplicate. List every reused\n"
                "  catalog dim's `key` in `reused_dim_keys`, and include its full definition in the\n"
                "  template `rubric` as well.\n"
                "- DIMENSION NAMES ARE NAMESPACED: every rubric dim `name` must be `<context>.<dim>` —\n"
                "  the context the dimension is scored AGAINST (`tone.consistency`, `answer.grounding`,\n"
                "  `boundary.correct`), NOT the tool that produced the output. A draft carrying a bare name\n"
                "  is REJECTED whole and discarded, because a judge config binds to a dim by name across\n"
                "  every template, so `character` alone would bind every context at once. When you reuse a\n"
                "  catalog dim, place it under its catalog name (already namespaced) and list its `key`\n"
                "  unchanged in `reused_dim_keys` (the key is a version-group slug, not the bound name).\n"
                "- EVERY DIMENSION STATES ITS `scale`: 'pass_fail' — one question the judge answers pass or\n"
                "  fail, its `scoring_guide` keyed 'pass' / 'fail' — or 'ordinal' — a 1-5 level, its guide keyed\n"
                "  '1'..'5'. Prefer 'pass_fail' for a new dimension. A draft dimension with no scale, or with a\n"
                "  guide keyed for the other scale, is REJECTED with the whole draft.\n"
                "- Only invent a NEW dimension when the catalog has no good fit. Put each novel dimension in\n"
                "  `new_dim_suggestions` (with a stable `key` slug) AND in the template `rubric`.\n"
                "- Derive scenario `variation_axes` from the subject's self-description and its tools —\n"
                "  the situations that would stress the qualities you are scoring. Each axis has a `name`, a\n"
                "  `generator` ('enum' | 'sample' | 'llm'), a `description`, and (for enum/sample) `values`.\n"
                "\n"
                "Respond with ONLY a JSON object — no commentary, no markdown fences — in this shape:\n"
                "{\n"
                '  "template": {\n'
                '    "name": "string",\n'
                '    "intent": "what this template tests",\n'
                '    "rubric": [\n'
                '      {"name": "tone.consistency", "description": "string", "scale": "ordinal",\n'
                '       "scoring_guide": {"1": "...", "3": "...", "5": "..."}}\n'
                "    ],\n"
                '    "variation_axes": [\n'
                '      {"name": "string", "generator": "enum|sample|llm",\n'
                '       "values": ["..."], "description": "string"}\n'
                "    ]\n"
                "  },\n"
                '  "reused_dim_keys": ["catalog_key", "..."],\n'
                '  "new_dim_suggestions": [\n'
                '    {"key": "slug", "axis": "capability", "universal": false,\n'
                '     "dim": {"name": "tone.consistency", "description": "string", "scale": "ordinal",\n'
                '             "scoring_guide": {"5": "..."}}}\n'
                "  ]\n"
                "}\n"
                "\n"
                "Every dimension in `new_dim_suggestions` must also appear in `template.rubric`. Every key in\n"
                "`reused_dim_keys` must come from the supplied catalog."
            ),
            cache_class="stable",
            description="Proposer instructions — two-feed framing, reuse-over-invent rule, strict JSON output shape.",
        ),
    ),
)


__all__ = [
    "EVAL_PROPOSER_TEMPLATE_DEFAULT",
]
