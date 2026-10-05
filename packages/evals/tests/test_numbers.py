"""The one number rule — what it writes at both ends of a value's range, and that it is the only one.

The shared cases live in `fixtures/number-format-cases.json` beside this suite and are asserted by
`test_viz_compiler.py` (a browser renderer that restates the rule pins the same table where it ships). What is here is the
rule's two ENDS stated as properties rather than cases, the signed form, and the canary that keeps a
second implementation from growing back: two formatters in one engine is how one analysis came to
print a latency as ``1.235e+04`` beside an interval that spelled the same magnitude ``12,346``.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

from threetears.evals.analysis import numbers, references, reporting, surface_table
from threetears.evals.analysis.numbers import ABSENT, WHOLE_FROM, format_number, format_signed
from threetears.evals.analysis.viz import quantities
from threetears.evals.contracts.models import LatencyMetrics


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


class TestTheLargeEnd:
    @pytest.mark.parametrize("value", [1000.5, 12345.6, 99999.99, 1234567.89, 9.87654321e12, -54321.9])
    def test_a_large_value_never_takes_an_exponent(self, value):
        assert "e" not in format_number(value)

    @pytest.mark.parametrize(("value", "text"), [(12345.6, "12346"), (1000.4, "1000"), (-54321.9, "-54322")])
    def test_a_large_value_is_rounded_to_the_integer(self, value, text):
        assert format_number(value) == text

    def test_the_line_is_where_four_figures_reach_the_units_digit(self):
        """Just under it, four significant figures; at it, the integer — the two agree on the digits."""
        assert format_number(WHOLE_FROM - 0.5) == "999.5"
        assert format_number(WHOLE_FROM + 0.25) == "1000"

    @pytest.mark.parametrize("value", [50300.0, 50300.5, 1234567.0, 1e21])
    def test_no_thousands_separator(self, value):
        """These numbers sit inside `CI [low, high]` and `a; b` lists, where a comma reads as an item."""
        assert "," not in format_number(value)


class TestTheSmallEnd:
    @pytest.mark.parametrize("value", [0.000042, 1e-9, -3.3e-7, 5e-324])
    def test_a_small_non_zero_value_never_reads_as_zero(self, value):
        text = format_number(value)
        assert float(text) != 0.0, text

    def test_it_is_four_significant_figures_in_the_middle(self):
        assert format_number(0.0123456) == "0.01235"
        assert format_number(12.3456) == "12.35"


class TestWholeAndAbsent:
    @pytest.mark.parametrize(
        ("value", "text"), [(0.0, "0"), (-0.0, "0"), (42.0, "42"), (1e21, "1000000000000000000000")]
    )
    def test_a_whole_number_is_written_whole_at_any_magnitude(self, value, text):
        assert format_number(value) == text

    @pytest.mark.parametrize("value", [None, math.nan, math.inf, -math.inf])
    def test_an_absent_or_non_finite_value_is_a_dash_never_a_zero(self, value):
        assert format_number(value) == ABSENT == "—"


class TestTheSignedForm:
    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (1.5, "+1.5"),
            (-1.5, "-1.5"),
            (0.0, "0"),
            (12345.6, "+12346"),
            (-0.000042, "-4.2e-05"),
            (None, "—"),
            (math.nan, "—"),
        ],
    )
    def test_a_change_carries_its_direction_and_the_one_spelling(self, value, text):
        assert format_signed(value) == text


class TestThereIsOneImplementation:
    def test_every_surface_that_spells_numbers_holds_the_same_function(self):
        """The chart quantities spell their values with it, as every chart arm does; nothing holds a copy.

        The arm table is not on the list: it prints no number since its per-measure renderer was
        retired, so it holds no formatter to compare.
        """
        for module in (quantities, references, surface_table):
            assert module.format_number is numbers.format_number, module.__name__

    def test_no_other_module_defines_a_number_formatter(self):
        """A second ``def format_number`` anywhere in the engine or its surfaces is the defect itself.

        Walks every module of the package rather than a list of the ones that once had a copy,
        because the copy that matters is the next one, in a module nobody thought to list.
        """
        defining = []
        for path in sorted((_SOURCE_ROOT / "threetears").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if any(isinstance(node, ast.FunctionDef) and node.name == "format_number" for node in ast.walk(tree)):
                defining.append(path.relative_to(_SOURCE_ROOT).as_posix())
        assert defining == ["threetears/evals/analysis/numbers.py"]


#: The trees and modules whose every number is shown to a reader: the analysis package -- what the
#: generator writes from, and the reporting module whose significance cells and partition refusals
#: are printed verbatim. A host's own renders (an operator console, a tool surface) are the host's
#: to walk on the same terms. A spelling of a number anywhere in them either goes through
#: :func:`~threetears.evals.analysis.numbers.format_number` / :func:`~threetears.evals.analysis.numbers.format_signed` or is
#: named below as a deliberate fixed spelling. A module sits here by itself, rather than its whole
#: directory, where the rest of that directory is not read as a report.
_SPELLING_TREES = ("threetears/evals/analysis",)

#: What sits inside :data:`_SPELLING_TREES` and is not read as a report, so is not walked:
#: ``numbers.py`` is the rule itself, and ``viz/`` spells numbers only in payload validation messages.
_NOT_READ_AS_REPORTS = (
    "threetears/evals/analysis/numbers.py",
    "threetears/evals/analysis/viz",
)

#: Presentation types that choose how many digits a reader sees. ``d`` and ``s`` do not.
_NUMERIC_TYPES = frozenset("eEfFgGn%")

#: A share of a whole spelled as a percentage (``.0%``) is a different sentence from a measured
#: value — "37% of it" — and carries its own fixed precision. Exempt by presentation type, since the
#: type itself says the value is a fraction.
_PERCENTAGE_TYPE = "%"

#: Sites that spell a number at a deliberate fixed precision, keyed by (module, expression, spec) so
#: a moved line keeps its exemption and a rewritten expression loses it. Each reason says why the
#: fixed spelling is the right one there; an entry no site matches any more fails the suite.
_FIXED_SPELLINGS: dict[tuple[str, str, str], str] = {
    # A figure inside a memo's sentence is read once, not compared down a column: two significant
    # figures below 10 are what a sentence can use, and the tables beside it keep the one rule.
    ("threetears/evals/analysis/prose_refs.py", "value", ".2g"): "a figure in a sentence",
    ("threetears/evals/analysis/prose_refs.py", "value", ".{}f"): "a figure in a sentence below 1e-4, written out",
}

_PRECISION = re.compile(r"\.(\d+|\{\})")
_PERCENT_DIRECTIVE = re.compile(r"%[-+ #0]*(\d+|\*)?(\.(\d+|\*))?[eEfFgG]")
_BRACE_FIELD_SPEC = re.compile(r"\{[^{}]*:([^{}]*)\}")


def _chooses_digits(spec: str) -> bool:
    """Whether a format spec decides how many digits a number is written with.

    A precision (fixed or nested), a numeric presentation type, a thousands separator, or a spec
    built at run time — which cannot be judged here and so is flagged.
    """
    if "{}" in spec or _PRECISION.search(spec) or "," in spec or "_" in spec:
        return True
    return bool(spec) and spec[-1] in _NUMERIC_TYPES


def _joined_spec(spec: ast.JoinedStr) -> str:
    return "".join(part.value if isinstance(part, ast.Constant) else "{}" for part in spec.values)


def _number_spellings(tree: ast.AST) -> Iterator[tuple[int, str, str]]:
    """Every ``(line, expression, spec)`` in ``tree`` that spells a value with a digit-choosing spec.

    Covers the four ways Python formats a number: an f-string field, the ``format`` builtin,
    ``str.format`` on a literal, and ``%`` on a literal.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FormattedValue) and node.format_spec is not None:
            spec = _joined_spec(node.format_spec)
            if _chooses_digits(spec):
                yield node.lineno, ast.unparse(node.value), spec
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "format"
            and len(node.args) == 2
        ):
            spec_node = node.args[1]
            spec = spec_node.value if isinstance(spec_node, ast.Constant) and isinstance(spec_node.value, str) else "{}"
            if _chooses_digits(spec):
                yield node.lineno, ast.unparse(node.args[0]), spec
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str)
        ):
            for spec in _BRACE_FIELD_SPEC.findall(node.func.value.value):
                if _chooses_digits(spec):
                    yield node.lineno, ast.unparse(node), spec
        elif (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
        ):
            for match in _PERCENT_DIRECTIVE.finditer(node.left.value):
                yield node.lineno, ast.unparse(node.right), match.group(0)


def _modules_under(trees: Iterable[str]) -> list[Path]:
    """Every module an entry of ``trees`` names — each ``.py`` under a directory, or the file itself."""
    modules = []
    for tree_root in trees:
        root = _SOURCE_ROOT / tree_root
        assert root.is_dir() or (root.is_file() and root.suffix == ".py"), tree_root
        modules.extend(sorted(root.rglob("*.py")) if root.is_dir() else [root])
    excluded = [_SOURCE_ROOT / entry for entry in _NOT_READ_AS_REPORTS]
    return [module for module in modules if not any(module == ex or ex in module.parents for ex in excluded)]


def _spellings_under(trees: Iterable[str]) -> list[tuple[str, int, str, str]]:
    found = []
    for path in _modules_under(trees):
        module = path.relative_to(_SOURCE_ROOT).as_posix()
        for line, expression, spec in _number_spellings(ast.parse(path.read_text(encoding="utf-8"))):
            found.append((module, line, expression, spec))
    return found


class TestTheWidenedModulesRenderByTheRule:
    """The reporting module's two reader-facing spellings, driven where their old specs disagree.

    ``.4g`` and the rule agree on every finite p-value and on any effect size below 1e4, so each
    case sits where they part: the large end, where ``.4g`` and a bare ``:g`` turn to an exponent,
    and a non-finite value, which ``.4g`` spells ``nan`` and the rule spells absent.
    """

    def test_a_large_effect_size_is_written_whole(self):
        assert f"{12345.6:.4g}" == "1.235e+04"
        cell = reporting.format_significance(significant=True, paired=True, p=3e-7, effect=12345.6, n=4)
        assert cell == f"{reporting.SIGNIFICANT_LABEL} (p={format_number(3e-7)}, d_z=12346, n=4)"

    def test_a_p_value_that_is_not_a_number_reads_as_absent(self):
        """The one input where ``.4g`` and the rule part on a p-value: ``.4g`` prints ``nan``.

        Every finite p lies in [0, 1], where the two spellings agree digit for digit, so this is the
        only render a p routed off the rule can be seen from.
        """
        assert f"{math.nan:.4g}" == "nan"
        cell = reporting.format_significance(significant=False, paired=True, p=math.nan, effect=0.5, n=4)
        assert f"p={ABSENT}, " in cell

    def test_a_large_partition_overrun_is_written_whole(self):
        overrun = 1234467.5
        assert f"{overrun:g}" == "1.23447e+06"
        partition = reporting.decompose_total_ms(LatencyMetrics(total_ms=100.0, llm_ms=overrun + 100.0, tool_ms=0.0))
        assert f"total_ms by {format_number(overrun)}ms, " in partition.withheld
        assert format_number(overrun) == "1234468"


class TestNoSurfaceSpellsANumberItsOwnWay:
    """The rule has one implementation only if nothing in the reader-facing trees writes digits itself.

    Derived from the source rather than from a function name: a second ``def format_number`` is one
    way to grow a second rule, and an inline ``f"{value:.4g}"`` is the commoner one — the shape
    that printed a latency ``1.235e+04`` in a results pivot beside an interval spelling the same
    magnitude ``12346``. Scoped to :data:`_SPELLING_TREES`; the engine's validation messages and the
    viz payload checks outside them are not read as reports and are not walked.
    """

    def test_a_module_named_by_itself_is_walked(self):
        """A single-module entry is parsed, not skipped as a tree with nothing under it.

        Asked of the walk's module list rather than of the spellings it found: once a module routes
        every number through the rule it yields no spelling, and "no spelling" would then read the
        same whether the module was clean or never opened.
        """
        single = "threetears/evals/analysis/reporting.py"
        assert [path.relative_to(_SOURCE_ROOT).as_posix() for path in _modules_under((single,))] == [single]
        walked = {path.relative_to(_SOURCE_ROOT).as_posix() for path in _modules_under(_SPELLING_TREES)}
        assert single in walked
        # And what _NOT_READ_AS_REPORTS names is inside a walked tree yet not walked.
        assert not {
            m
            for m in walked
            if m == "threetears/evals/analysis/numbers.py" or m.startswith("threetears/evals/analysis/viz/")
        }

    def test_every_spelling_goes_through_the_one_rule_or_is_named_as_fixed(self):
        offenders = [
            f"{module}:{line}: {{{expression}:{spec}}}"
            for module, line, expression, spec in _spellings_under(_SPELLING_TREES)
            if not spec.endswith(_PERCENTAGE_TYPE) and (module, expression, spec) not in _FIXED_SPELLINGS
        ]
        assert not offenders, (
            "these spell a number with their own precision; route them through format_number / format_signed, "
            "or name a deliberate fixed spelling in _FIXED_SPELLINGS with its reason:\n  " + "\n  ".join(offenders)
        )

    def test_every_named_fixed_spelling_still_names_a_site(self):
        """An exemption nothing matches is one a rewritten line walked out of without anyone deciding."""
        live = {(module, expression, spec) for module, _line, expression, spec in _spellings_under(_SPELLING_TREES)}
        assert sorted(set(_FIXED_SPELLINGS) - live) == []

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ('f"{value:.4g}"', ("value", ".4g")),
            ('f"{delta:+.3g}"', ("delta", "+.3g")),
            ('f"{latency:.0f} ms"', ("latency", ".0f")),
            ('f"{bound:g}"', ("bound", "g")),
            ('f"{count:,}"', ("count", ",")),
            ('f"{value:.{places}f}"', ("value", ".{}f")),
            ('format(value, ".3f")', ("value", ".3f")),
            ('"{:.2e}".format(value)', ("'{:.2e}'.format(value)", ".2e")),
            ('"%.3f" % value', ("value", "%.3f")),
        ],
    )
    def test_the_walk_sees_every_way_python_spells_a_number(self, source, expected):
        assert [found[1:] for found in _number_spellings(ast.parse(source))] == [expected]

    @pytest.mark.parametrize(
        "source", ['f"{name}"', 'f"{name!r}"', 'f"{count:d}"', 'f"{label:>10}"', 'f"{format_number(v)}"']
    )
    def test_the_walk_leaves_what_chooses_no_digits(self, source):
        assert list(_number_spellings(ast.parse(source))) == []
