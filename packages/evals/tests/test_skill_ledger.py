"""The skill ledger: every analyst skill a build ships has an entry recording the basis it shipped on.

``skill/PROMOTIONS.toml`` sits beside the skill and is not part of it, so recording an entry never moves the digest it
records. The shipped skill must have an entry; it need not be measured, since an unmeasured entry with its reason is
the honest record while the writer cases are too few to measure an edit. An edited skill is a new digest, which has
no entry until one is written, and the refusal names the digests to copy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.analysis.analyst.skill_ledger import skill_record
from threetears.evals.kernel.errors import ValidationFailedError

#: The ledger as the source tree holds it, beside the skill's files.
LEDGER = Path(__file__).resolve().parents[1] / "src/threetears/evals/analysis/analyst/skill/PROMOTIONS.toml"


def test_the_shipped_skill_has_a_ledger_entry_with_its_own_digests() -> None:
    """Fails on a build whose skill files changed with no entry recorded: the message names the entry to add."""
    skill = load_skill()
    entry = skill_record(skill)
    assert (entry.skill, entry.analyst, entry.editor) == (skill.digest, skill.analyst().digest, skill.editor().digest)


def test_the_ledger_ships_beside_the_skill_and_is_not_part_of_it() -> None:
    assert LEDGER.is_file()
    assert not any(path.endswith(".toml") for path in load_skill().files)


def test_an_edited_skill_has_no_entry_and_the_refusal_names_its_digests() -> None:
    shipped = load_skill()
    edited = AnalystSkill(files={**shipped.files, "writing.md": shipped.files["writing.md"] + "\nOne more line.\n"})
    assert edited.digest != shipped.digest
    with pytest.raises(ValidationFailedError) as refused:
        skill_record(edited)
    assert edited.digest in refused.value.message
    assert edited.analyst().digest in refused.value.remedy
    assert "unmeasured" in refused.value.remedy


def test_the_baseline_entry_is_unmeasured_and_says_why() -> None:
    entry = skill_record(load_skill())
    assert entry.basis == "unmeasured"
    assert entry.reason == "baseline: shipped before the gate existed"
