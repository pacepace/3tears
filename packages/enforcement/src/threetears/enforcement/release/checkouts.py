"""the CI jobs that run the API-growth gate must see the release tags.

``actions/checkout`` fetches one commit and no tags by default. The gate refuses
-- rather than skips -- when it sees no tag past a repo's first release, so a job
that runs it from a default checkout goes red on every run, and a job that runs it
on the release commit itself sees one tag and compares against nothing. Both are
fixed by ``fetch-depth: 0`` on the checkout of the repository itself, and this
module is the binding that keeps that line from being dropped. It reads workflow
files; it cannot see a job that runs the gate through an indirection the text of
its ``run:`` steps does not name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from threetears.core.authored_yaml import safe_load_authored

__all__ = ["untagged_checkout_findings"]


def _own_checkouts(steps: list[Any]) -> list[dict[str, Any]]:
    """the job's ``actions/checkout`` steps that check out this repository.

    A checkout naming another ``repository:`` is a sibling, and needs no tags.

    :param steps: the job's steps
    :ptype steps: list[Any]
    :return: the checkout steps
    :rtype: list[dict[str, Any]]
    """
    found: list[dict[str, Any]] = []
    for step in steps:
        if isinstance(step, dict) and str(step.get("uses", "")).startswith("actions/checkout@"):
            options = step.get("with") or {}
            repository = str(options.get("repository", "${{ github.repository }}"))
            if repository == "${{ github.repository }}":
                found.append(step)
    return found


def untagged_checkout_findings(workflows: Path, runs: str) -> list[str]:
    """every workflow job that runs *runs* from a checkout of this repo without its tags.

    :param workflows: the ``.github/workflows`` directory
    :ptype workflows: Path
    :param runs: text a ``run:`` step contains when it runs the gate (``check-all.sh``, ``tests/``)
    :ptype runs: str
    :return: findings; one when no job runs *runs* at all, so a renamed command cannot pass vacuously
    :rtype: list[str]
    """
    findings: list[str] = []
    matched = 0
    for path in sorted([*workflows.glob("*.yml"), *workflows.glob("*.yaml")]):
        document = safe_load_authored(path.read_text(encoding="utf-8")) or {}
        for job_name, job in (document.get("jobs") or {}).items():
            steps = (job.get("steps") or []) if isinstance(job, dict) else []
            if not any(isinstance(step, dict) and runs in str(step.get("run", "")) for step in steps):
                continue
            matched += 1
            checkouts = _own_checkouts(steps)
            if not checkouts:
                findings.append(
                    f"{path.name} job {job_name!r} runs {runs!r} but checks out nothing of this repository."
                )
            for step in checkouts:
                depth = (step.get("with") or {}).get("fetch-depth")
                if depth != 0:
                    findings.append(
                        f"{path.name} job {job_name!r} runs {runs!r} from a checkout without the release tags "
                        f"(fetch-depth: {depth if depth is not None else 'unset, so 1'}). Add `fetch-depth: 0` to "
                        f"its actions/checkout step: the API-growth gate refuses when it cannot see a tag."
                    )
    if matched == 0:
        findings.append(f"no job in {workflows} runs {runs!r}; the binding matches nothing, so it proves nothing.")
    return findings
