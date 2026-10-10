"""``docs/reference.md`` is generated, committed, and held equal to what the generator makes of the package today.

The page is built from the package itself by ``scripts/generate_reference.py`` (dev tooling beside the package,
not in it): each public root's ``__all__``, docstrings, signatures, field descriptions, the goal-check language,
the action catalogue, the command line and the measure catalogue. A change to any of them moves the page, and
this test fails until the page is regenerated, so the reference a reader relies on never describes an older
package.
"""

from __future__ import annotations

from packages.evals.tests.script_loader import load_script


def test_the_committed_reference_is_what_the_package_generates() -> None:
    generator = load_script("generate_reference.py")
    page = generator.render()
    committed = generator.REFERENCE_PATH.read_text(encoding="utf-8") if generator.REFERENCE_PATH.exists() else ""
    assert committed == page, (
        f"packages/evals/docs/reference.md is stale: the package changed under it. Regenerate it with "
        f"`{generator.REGENERATE}` and commit the result."
    )
