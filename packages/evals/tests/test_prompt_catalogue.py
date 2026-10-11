"""The package's prompt catalogue: every id declared once, and every entry a registry can seed from.

A host seeds its prompt registry from :data:`~threetears.evals.kernel.SEED_PROMPTS` by import, so an entry that
does not resolve, or resolves to a seed of another id, is a slot a fresh deployment never fills or fills wrongly
— and nothing fails until an operator opens it. These hold every entry to what it says, and hold the ids to the
catalogue: a module spelling one as a literal of its own is a second source the catalogue can drift from.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from threetears.evals.kernel import (
    ANALYSIS_GEN_PROMPT,
    BOUNDARY_GEN_PROMPT,
    KIND_TEMPLATE,
    KIND_TEXT,
    PROPOSER_PROMPT,
    SEED_PROMPTS,
    SeedPrompt,
    SeedTemplate,
)

#: The repository root, which every entry's ``definition_path`` is relative to.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: The package's source, scanned for a second spelling of an id.
SOURCE_ROOT = REPO_ROOT / "packages" / "evals" / "src" / "threetears" / "evals"

#: The one module allowed to spell an id.
CATALOGUE = SOURCE_ROOT / "kernel" / "prompts" / "catalogue.py"


def _entry(prompt_type: str) -> SeedPrompt:
    (entry,) = [entry for entry in SEED_PROMPTS if entry.prompt_type == prompt_type]
    return entry


def test_the_catalogue_declares_each_of_the_packages_prompts_once():
    assert [entry.prompt_type for entry in SEED_PROMPTS] == [ANALYSIS_GEN_PROMPT, PROPOSER_PROMPT, BOUNDARY_GEN_PROMPT]
    assert len({entry.prompt_type for entry in SEED_PROMPTS}) == len(SEED_PROMPTS)


@pytest.mark.parametrize("entry", SEED_PROMPTS, ids=lambda entry: entry.prompt_type)
def test_every_entry_resolves_by_import_to_a_seed_of_its_own_id(entry: SeedPrompt):
    module, attribute = entry.seed_ref
    seed = getattr(importlib.import_module(module), attribute)
    if entry.kind == KIND_TEMPLATE:
        assert isinstance(seed, SeedTemplate)
        assert seed.template_type == entry.prompt_type
        assert seed.template_name == "default"
        assert seed.sections, "a template seed with no section seeds an empty prompt"
    else:
        assert entry.kind == KIND_TEXT
        assert isinstance(seed, str) and seed.strip()


@pytest.mark.parametrize("entry", SEED_PROMPTS, ids=lambda entry: entry.prompt_type)
def test_every_entry_names_the_file_its_constant_is_defined_in(entry: SeedPrompt):
    """A promotion writes a tuned preset back into ``definition_path``, so the constant must be ASSIGNED there."""
    path, constant = entry.promotion_ref
    source = REPO_ROOT / path
    assert source.is_file(), f"{entry.prompt_type}: {path} does not exist"
    assigned = {
        target.id
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    assert constant in assigned, f"{entry.prompt_type}: {path} does not define {constant}"


def test_the_analysis_generators_seed_is_a_text_prompt_and_the_proposers_are_templates():
    assert _entry(ANALYSIS_GEN_PROMPT).kind == KIND_TEXT
    assert _entry(PROPOSER_PROMPT).kind == KIND_TEMPLATE
    assert _entry(BOUNDARY_GEN_PROMPT).kind == KIND_TEMPLATE


def test_no_module_but_the_catalogue_spells_an_id_as_a_literal():
    """Prose may NAME a slot inside a longer string; a string that IS an id is a second declaration of it."""
    ids = {entry.prompt_type for entry in SEED_PROMPTS}
    spelled: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if path == CATALOGUE:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and node.value in ids:
                spelled.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno} {node.value!r}")
    assert spelled == []


class TestASeedPromptIsRefusedWhereItIsDeclared:
    """A host declares its own prompts through the type, so a malformed one is refused at construction."""

    def test_a_kind_no_registry_seeds_is_refused(self):
        with pytest.raises(ValueError, match="declares kind 'prompt'"):
            SeedPrompt(kind="prompt", prompt_type="host_x", seed_module="host.x", definition_path="x.py", constant="X")

    def test_a_blank_field_is_refused_by_name(self):
        with pytest.raises(ValueError, match="leaves seed_module, constant blank"):
            SeedPrompt(kind=KIND_TEXT, prompt_type="host_x", seed_module=" ", definition_path="x.py", constant="")

    def test_a_hosts_own_declaration_constructs(self):
        own = SeedPrompt(
            kind=KIND_TEXT, prompt_type="host_x", seed_module="host.prompts", definition_path="host/p.py", constant="X"
        )
        assert own.seed_ref == ("host.prompts", "X")
        assert own.promotion_ref == ("host/p.py", "X")
