"""The editor's rubric is read once per text, and a long-running host keeps a bounded number of texts read (#1117).

Contracts held here (``threetears.evals.ops.report_writer.rubric``):

- a text read twice is read once: the second read is the first one's rubric, the very object;
- however many distinct texts a host reads (a candidate-skill search writes a new one every generation), the readings
  it keeps are bounded, and each text still reads to the rubric it states.
"""

from __future__ import annotations

from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.ops.report_writer.rubric import editor_rubric


def test_a_text_read_twice_is_read_once() -> None:
    text = load_skill().editor().text
    assert editor_rubric(text) is editor_rubric(text)


def test_the_texts_a_host_keeps_read_are_bounded() -> None:
    held = editor_rubric.cache_info().maxsize
    assert held is not None, "an editor text is host- or search-supplied, so its readings are bounded"
    shipped = load_skill().editor().text
    texts = [f"{shipped}\n<!-- candidate {n} -->\n" for n in range(held + 8)]
    rubrics = [editor_rubric(text) for text in texts]
    assert editor_rubric.cache_info().currsize <= held
    assert {rubric.items for rubric in rubrics} == {editor_rubric(shipped).items}
