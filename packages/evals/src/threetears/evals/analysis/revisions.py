"""Two revisions of one template never pool: the refusal, and the name that keeps them apart where they are shown.

A definition reset (:func:`~threetears.evals.run.reset_definitions`) writes a template's new content as a new revision:
a new record under the same name, the older one kept for every run, result and case measured against it. Runs of two
revisions are measured against two definitions (a battery of four cases and one of seven, say), so their results are
two datasets. Nothing in the cell algebra separates them, since a template is no apparatus dimension: two revisions
run at one model are one variant under one rig. So a campaign holding runs of two revisions of one name is refused at
every door that would pool them — creating it, adding a run to it, and assembling its bundle, which every analysis,
contrast and report of a campaign reads — naming the template and its revisions. Where two revisions are shown side by
side rather than pooled (a two-run comparison's per-template rows), :func:`revision_label` keeps them apart by name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.schema.models import EvalTemplate

#: What a refusal tells its caller to do.
POOLED_REVISIONS_REMEDY = (
    "keep one revision's runs in this campaign (campaign_runs_remove) and put the other's in a campaign of its own"
)


def revision_label(template: EvalTemplate) -> str:
    """A template's name as a reader is shown it: with its revision once it has more than one."""
    return template.name if template.revision == 1 else f"{template.name} (revision {template.revision})"


def refuse_pooled_revisions(
    template_ids: Iterable[str | None], load_template: Callable[[str], EvalTemplate | None], *, what: str
) -> None:
    """Refuse a set of runs that holds more than one revision of one template.

    Two templates under one name at one revision number are not two revisions (a reset always writes the next
    number), so they are not refused.

    :param template_ids: The template each run ran; None for an ad-hoc run, which pools with none of them.
    :ptype template_ids: Iterable[str | None]
    :param load_template: Reads a template by id, archived or not; None when it no longer resolves, which can say
        nothing about its name and is passed over.
    :ptype load_template: Callable[[str], EvalTemplate | None]
    :param what: The campaign, as the refusal names it.
    :ptype what: str
    :raises ValidationFailedError: Naming each template whose runs span revisions, and every revision held.
    """
    # A revision is what a reset writes: a record one past the newest under its name, so two revisions of one template
    # always differ in number. Two records under one name at one number are not revisions of each other but two
    # templates that share a name (two authored or seeded separately, each revision 1), and runs of the two pool as a
    # campaign's runs of any two templates do. So a name spans revisions only when it holds more than one number.
    revisions: dict[str, dict[str, int]] = {}
    for template_id in sorted({template_id for template_id in template_ids if template_id}):
        template = load_template(template_id)
        if template is not None:
            revisions.setdefault(template.name, {})[template_id] = template.revision
    spanning = {name: held for name, held in sorted(revisions.items()) if len(set(held.values())) > 1}
    if spanning:
        named = "; ".join(
            f"{name!r} "
            + ", ".join(
                f"revision {revision} ({template_id})"
                for template_id, revision in sorted(held.items(), key=lambda item: (item[1], item[0]))
            )
            for name, held in spanning.items()
        )
        raise ValidationFailedError(
            f"{what} holds runs of more than one revision of a template: {named}. A reset wrote each revision's "
            "content as a new definition, so their results are two datasets and never pool as one",
            remedy=POOLED_REVISIONS_REMEDY,
        )


__all__ = ["POOLED_REVISIONS_REMEDY", "refuse_pooled_revisions", "revision_label"]
