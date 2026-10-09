"""The decision surface serves its words with its numbers, over an analysis with no host vocabulary.

The sentence above the table and the word under each bar verdict are carried on the served
table (``provenance``, ``verdict_word``), so no render keeps its own copy. The analysis is
the invoice-extractor one ``test_hostneutral_surface_table.py`` builds by hand, and its helpers are
imported from there.
"""

from __future__ import annotations


from threetears.evals.analysis.surface_table import SURFACE_PROVENANCE, VERDICT_WORDS
from packages.evals.tests.test_hostneutral_surface_table import two_arm_table


class TestTheProvenanceSentenceIsServed:
    def test_a_table_over_a_surface_states_that_code_computed_its_numbers(self) -> None:
        table = two_arm_table()

        assert table.provenance == SURFACE_PROVENANCE
        assert table.model_dump(mode="json")["provenance"] == SURFACE_PROVENANCE


class TestEveryVerdictIsServedAsItsWord:
    def test_the_words_are_the_four_a_reader_acts_on(self) -> None:
        assert VERDICT_WORDS == {
            "clears": "clears",
            "misses": "misses",
            "no_interval": "no interval",
            "no_data": "no data",
        }

    def test_each_value_carries_the_word_for_its_verdict_and_a_merit_value_carries_none(self) -> None:
        values = [value for row in two_arm_table().rows for value in row.values if value is not None]
        judged = [value for value in values if value.verdict is not None]

        assert {value.verdict for value in judged} == {"clears", "no_data"}, "the fixture must reach two verdicts"
        for value in judged:
            assert value.verdict_word == VERDICT_WORDS[value.verdict]
            assert value.model_dump(mode="json")["verdict_word"] == VERDICT_WORDS[value.verdict]
        assert all(value.verdict_word is None for value in values if value.verdict is None)
