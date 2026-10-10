"""The retired run-level control leaves nothing behind that still writes it.

A campaign's control used to be ``EvalCampaign.control_run_id`` — a run id, on the campaign,
beside its membership. It is now a variant key inside the declaration, and the difference is
not cosmetic: a run id is a POINTER, so curating or deleting the run it named destroyed the
campaign's design, and four write paths existed to manage that fragility. A variant key is an
IDENTITY, so nothing can dangle it and none of those paths has anything left to do.

**Why this file exists rather than the removal alone.** It was written when ``EvalCampaign`` was
a tolerant model: a surviving writer of ``control_run_id`` did not fail — it constructed
successfully and the value evaporated. **Construction and reads are both strict now**, so the
Python half fails loudly. Do not delete these scans on the strength of that: a name a writer
spells in a dict it validates later is still a writer. A browser kit restating the campaign shape is scanned on the
same terms where it ships, since no pydantic config reaches it. The
searches below state their own SHAPE, because an absence claim is only as strong as the shape
of the search that proved it: these are literal-substring scans over the package's Python source,
so they cannot see a name assembled at runtime (``getattr(campaign, "control_" + …)``) and they
cannot see a stored document.

**The bare name is forbidden, not only its campaign-shaped uses.** The derived design once
carried a ``control_run_id`` of its own — its answer to "which member run carried the declared
variant" — and while it did, the scan was narrowed to the attribute on a campaign and the key as
authored data. The design now keys on arms (``RealizedDesign.control_arm`` is an arm with every run
that measured it), so the reason for the narrowing is gone and the scan covers every mention.
"""

from __future__ import annotations

from pathlib import Path

import pytest


#: The package's source root.
_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"

#: The retired name.
_RETIRED = "control_run_id"

#: The uses of it that are retired, as literals a scan can find: the bare name, since nothing
#: live carries it any more.
_RETIRED_USES = (_RETIRED,)

#: The files allowed to name it, and why. The campaign model's docstring names it to say what the
#: control used to be. (The service once named it too, to refuse it with a pointer to its successor;
#: that compatibility message went with the other legacy-field readers, and the key now meets the
#: same unknown-field refusal as any other.)
_DELIBERATE_MENTIONS = {
    "threetears/evals/kernel/campaign.py": "EvalCampaign's docstring, recording what the control replaced",
}


def _sources(root: Path, *globs: str) -> list[Path]:
    """Every source file under ``root`` matching ``globs``, excluding build output.

    Args:
        root: The tree to walk.
        *globs: Patterns to collect.

    Returns:
        The matching paths, sorted.
    """
    found: list[Path] = []
    for pattern in globs:
        found.extend(
            path
            for path in root.rglob(pattern)
            if "node_modules" not in path.parts and "dist" not in path.parts and "__pycache__" not in path.parts
        )
    return sorted(found)


def test_no_production_source_still_writes_the_retired_control_field() -> None:
    """The whole point of the migration, asserted where the model cannot assert it.

    Both Python forms now raise — assignment always did, and ``EvalCampaign(control_run_id=…)``
    does too. What still needs finding is a writer that spells the key in a mapping it
    validates later, which no model config sees at the point of writing.
    """
    offenders: dict[str, str] = {}
    sources = _sources(_SOURCE_ROOT / "threetears", "*.py")
    assert any(path.name == "campaigns.py" for path in sources), "the scan read no package source"
    for path in sources:
        relative = path.relative_to(_SOURCE_ROOT).as_posix()
        if relative in _DELIBERATE_MENTIONS:
            continue
        source = path.read_text(encoding="utf-8")
        if hits := [use for use in _RETIRED_USES if use in source]:
            offenders[relative] = ", ".join(hits)

    assert offenders == {}, (
        f"`{_RETIRED}` on a CAMPAIGN was replaced by `declared_design.control`, a variant key. Still used as one: "
        f"{'; '.join(f'{f} ({why})' for f, why in sorted(offenders.items()))}. A writer that validates its "
        f"mapping later fails there, far from the line that wrote the key."
    )


def test_no_model_carries_the_name_so_the_scan_above_may_forbid_it_bare() -> None:
    """Pins the reason the scan forbids the bare name: neither the campaign nor the derived design carries it.

    If a model ever carries a field of this name again, the bare-name scan above would refuse
    it — this says why that refusal is right rather than a false red.
    """
    from threetears.evals.analysis.bundle.assemble import RealizedDesign
    from threetears.evals.kernel.campaign import EvalCampaign

    assert _RETIRED not in RealizedDesign.model_fields
    assert _RETIRED not in EvalCampaign.model_fields


def test_the_guard_that_names_the_retired_field_is_still_there() -> None:
    """The allowlist above must not become a way to let the name back in quietly.

    An entry naming a file that no longer mentions it is a stale exemption, and the next
    reintroduction of the field lands under it unnoticed.
    """
    stale = [
        relative
        for relative in _DELIBERATE_MENTIONS
        if _RETIRED not in (_SOURCE_ROOT / relative).read_text(encoding="utf-8")
    ]

    assert stale == [], f"allowlist entries for files that no longer mention `{_RETIRED}`: {', '.join(stale)}"


def test_the_campaign_no_longer_declares_the_field_and_the_declaration_does() -> None:
    """The acceptance criterion, in two lines: the field moved, it did not merely vanish.

    Asserting only the removal would pass on a change that deleted the control outright, which
    is a different product.
    """
    from threetears.evals.kernel.campaign import EvalCampaign
    from threetears.evals.kernel.declaration import CampaignDesign

    assert _RETIRED not in EvalCampaign.model_fields
    assert "control" in CampaignDesign.model_fields


def test_a_stored_campaign_carrying_the_retired_key_is_refused() -> None:
    """A campaign written before the control moved does not load — it is regenerated, never migrated.

    The refusal names the key, so the operator meeting it knows which document is from before.
    """
    from pydantic import ValidationError

    from threetears.evals.kernel.campaign import EvalCampaign

    legacy = EvalCampaign(
        scope_id="scope-1", name="c", subject_id="subject-1", behavior="triage", created_by="test:fixture"
    ).to_dict()
    legacy[_RETIRED] = "run-ctl"

    with pytest.raises(ValidationError, match=_RETIRED):
        EvalCampaign.from_dict(legacy)
