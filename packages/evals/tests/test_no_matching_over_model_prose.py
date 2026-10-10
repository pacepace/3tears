"""No deterministic string matching over a field typed as model prose.

**The rule**: code checks the STRUCTURE of a
model's output, never its prose. A regex, a substring test, a ``startswith`` or an equality
against a literal over text a model wrote is an eval dimension wearing a validator's clothes, and
a gate built that way never converges — every real generation finds a phrasing it cannot parse.

**The population is derived, not listed.** A field is prose because its TYPE says so
(:data:`threetears.evals.schema.prose.ModelProse`); this test imports every module under ``threetears.evals``,
walks every Pydantic model reachable from ``BaseModel`` (the root, so a subsystem with its own
base is not lost), and collects the names of the fields carrying the marker. A new prose field is
covered the moment it is typed, and a hand-kept list — which fails by omission, silently — does
not exist to forget it.

**What is refused**, over an attribute whose name is a prose field's (optionally normalised
through ``.lower()``/``.strip()``/``.casefold()`` and friends or ``str(...)``):

* passing it to a regex method (``re.search(p, x.body)``, ``PATTERN.findall(x.body)``);
* calling a matching method on it (``x.body.startswith(...)``, ``.endswith``, ``.find``,
  ``.index``, ``.count``, ``.partition``, ``.removeprefix`` ...);
* ``... in x.body`` — a substring test;
* ``x.body == "literal"`` / ``!=`` against a non-empty string literal, and ``x.body in ("a", "b")``
  against a literal collection of strings.

**What is not refused**: emptiness (``x.body == ""``, ``not x.body``), length, copying the text,
rendering it, hashing it. Those read whether something was written, not what it says.

**Waivers, and what the scan cannot see, said beside the gate.** The scan is by ATTRIBUTE NAME,
statically, so it cannot tell ``finding.body`` from an unrelated ``response.body``; and a match
that is not a CHECK at all — an operator's own search term filtering a list — is still a match.
Either kind of site carries an inline waiver on the line (or the line above):
``# prose-canary: allow — <why this is not code judging what a model wrote>``. A waiver with no
reason is itself a violation, so every exception is greppable and argued where it sits. Nor does it see prose that is not a typed field — a transcript held as
``list[dict]`` has no field to mark, and a subscript ``turn["content"]`` is outside the scan.
Protocol parsers of a classifier's enum answer or a judge's JSON read raw response strings, not
fields typed as prose, so they are outside the population by construction rather than by waiver.
"""

from __future__ import annotations

import ast
import importlib
import itertools
import pkgutil
import re
from pathlib import Path

import pytest
from pydantic import BaseModel

import threetears.evals
from threetears.evals.schema.prose import field_is_prose


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
_EVAL_ROOT = _SOURCE_ROOT / "threetears" / "evals"

#: Methods of ``re`` and of a compiled pattern that read a string argument.
_REGEX_METHODS = frozenset({"search", "match", "fullmatch", "findall", "finditer", "sub", "subn", "split"})

#: ``str`` methods that match against the text they are called on.
_MATCHING_METHODS = frozenset(
    {
        "startswith",
        "endswith",
        "find",
        "rfind",
        "index",
        "rindex",
        "count",
        "partition",
        "rpartition",
        "removeprefix",
        "removesuffix",
    }
)

#: ``str`` methods that return the same text normalised — a match through one is still a match.
_NORMALISERS = frozenset({"lower", "upper", "casefold", "strip", "lstrip", "rstrip", "title"})

_WAIVER = re.compile(r"#\s*prose-canary:\s*allow\b(?P<reason>.*)$")


def _all_models() -> set[type[BaseModel]]:
    """Every Pydantic model class reachable from the root after importing all of ``threetears.evals``."""
    for module in pkgutil.walk_packages(threetears.evals.__path__, prefix="threetears.evals."):
        importlib.import_module(module.name)
    seen: set[type[BaseModel]] = set()
    stack: list[type] = [BaseModel]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return seen


def prose_field_names() -> frozenset[str]:
    """The names of every field typed as model prose, across every loaded model."""
    names: set[str] = set()
    for model in _all_models():
        fields = getattr(model, "__pydantic_fields__", None) or {}
        names.update(name for name, field in fields.items() if field_is_prose(field))
    return frozenset(names)


def _prose_subject(node: ast.AST, prose: frozenset[str]) -> str | None:
    """The prose field ``node`` reads, seeing through normalising calls, or None."""
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _NORMALISERS:
            node = node.func.value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "str"
            and len(node.args) == 1
        ):
            node = node.args[0]
        else:
            break
    if isinstance(node, ast.Attribute) and node.attr in prose:
        return node.attr
    return None


def _is_text_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value != ""


def _is_literal_text_collection(node: ast.AST) -> bool:
    return isinstance(node, (ast.Tuple, ast.List, ast.Set)) and any(_is_text_literal(element) for element in node.elts)


def matches_over_prose(source: str, prose: frozenset[str]) -> list[tuple[int, str, str]]:
    """Every deterministic string match over a prose field in ``source``.

    Args:
        source: Python source text.
        prose: The prose field names.

    Returns:
        ``(line, field, how)`` per match, in line order.
    """
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _REGEX_METHODS:
                for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                    if field := _prose_subject(argument, prose):
                        found.append((node.lineno, field, f"regex .{node.func.attr}()"))
            if node.func.attr in _MATCHING_METHODS and (field := _prose_subject(node.func.value, prose)):
                found.append((node.lineno, field, f".{node.func.attr}()"))
        elif isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            for op, (left, right) in zip(node.ops, itertools.pairwise(sides), strict=True):
                if isinstance(op, (ast.In, ast.NotIn)):
                    if field := _prose_subject(right, prose):
                        found.append((node.lineno, field, "substring `in`"))
                    if (field := _prose_subject(left, prose)) and _is_literal_text_collection(right):
                        found.append((node.lineno, field, "membership in literal strings"))
                elif isinstance(op, (ast.Eq, ast.NotEq)):
                    for subject, other in ((left, right), (right, left)):
                        if (field := _prose_subject(subject, prose)) and _is_text_literal(other):
                            found.append((node.lineno, field, "equality with a literal"))
    return sorted(found)


def _waiver(lines: list[str], lineno: int) -> tuple[bool, str]:
    """Whether ``lineno`` (1-based) carries a waiver on itself or the line above, and its reason."""
    for candidate in (lineno, lineno - 1):
        if 1 <= candidate <= len(lines) and (waived := _WAIVER.search(lines[candidate - 1])):
            return True, waived.group("reason").strip(" —-:")
    return False, ""


def _violations(prose: frozenset[str], root: Path = _EVAL_ROOT) -> list[str]:
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        rel = path.relative_to(_SOURCE_ROOT) if path.is_relative_to(_SOURCE_ROOT) else path.relative_to(root.parent)
        for lineno, field, how in matches_over_prose(source, prose):
            waived, reason = _waiver(lines, lineno)
            if waived and reason:
                continue
            suffix = " (waiver carries no reason)" if waived else ""
            violations.append(f"{rel}:{lineno}: {how} over prose field `{field}`{suffix}")
    return violations


class TestThePopulationIsDerivedFromTypes:
    def test_the_marked_fields_are_found(self):
        """Non-empty and containing a known marked field — an empty population passes every scan."""
        names = prose_field_names()
        assert {"caption", "mechanism", "statement"} <= names, sorted(names)

    def test_a_newly_typed_field_joins_the_population_without_a_list_edit(self):
        from threetears.evals.schema.prose import ModelProse

        class _Planted(BaseModel):
            planted_prose_field_for_the_canary: ModelProse = ""

        assert "planted_prose_field_for_the_canary" in prose_field_names()


class TestNoCodeMatchesOverModelProse:
    def test_the_scan_reads_the_package(self):
        """Non-vacuity: a scan over a root holding no module finds no violation by reading nothing."""
        assert (_EVAL_ROOT / "analysis" / "generator.py").is_file(), _EVAL_ROOT

    def test_no_deterministic_string_matching_over_prose_in_the_eval_package(self):
        violations = _violations(prose_field_names())
        assert not violations, (
            "code must not match over model prose — whether prose is right is an eval's question, not a check's:\n"
            + "\n".join(violations)
        )


class TestTheScannerFires:
    """The canary is evidence only if it goes red on the thing it forbids."""

    PROSE = frozenset({"body"})

    @pytest.mark.parametrize(
        "planted",
        [
            "re.search(r'significant', finding.body)",
            "_PATTERN.findall(finding.body)",
            "_PATTERN.search(finding.body.lower())",
            "finding.body.startswith('The')",
            "str(finding.body).endswith('.')",
            "'significant' in finding.body",
            "finding.body == 'n/a'",
            "'n/a' != finding.body.strip()",
            "finding.body in ('yes', 'no')",
        ],
    )
    def test_a_planted_match_over_a_prose_field_is_found(self, planted):
        assert matches_over_prose(planted, self.PROSE), planted

    @pytest.mark.parametrize(
        "allowed",
        [
            "finding.body == ''",
            "not finding.body",
            "len(finding.body) > 0",
            "render(finding.body)",
            "hashlib.sha256(finding.body.encode())",
            # A protocol parser of an enum answer reads a raw response string, not a prose field.
            "re.search(r'^(DIRECT|RELEVANT|NONE)$', response.content)",
            "answer.strip().upper() in ('ROUTINE', 'COMPLEX')",
            "json.loads(response.text)",
        ],
    )
    def test_structure_and_protocol_parsing_are_not_found(self, allowed):
        assert matches_over_prose(allowed, self.PROSE) == [], allowed

    def test_a_planted_regex_over_a_real_prose_field_fails_the_scan(self, tmp_path):
        """End to end over the REAL population: a planted module goes red, and a waived one does not."""
        tree = tmp_path / "eval"
        tree.mkdir()
        (tree / "planted.py").write_text("import re\nre.search(r'lab\\(', chart.caption)\n", encoding="utf-8")
        violations = _violations(prose_field_names(), root=tree)
        assert len(violations) == 1 and "`caption`" in violations[0], violations

        (tree / "planted.py").write_text(
            "import re\n# prose-canary: allow — a raw response, not the chart's caption\nre.search(r'x', r.caption)\n",
            encoding="utf-8",
        )
        assert _violations(prose_field_names(), root=tree) == []

    def test_a_waiver_needs_a_reason(self):
        lines = ["x = 1  # prose-canary: allow", "y = 2  # prose-canary: allow — a response, not a field"]
        assert _waiver(lines, 1) == (True, "")
        assert _waiver(lines, 2)[1] == "a response, not a field"
