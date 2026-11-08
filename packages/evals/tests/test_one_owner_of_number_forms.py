"""One owner of number forms: every place a figure's form is decided today is listed, and the list only shrinks.

A number's form is its digits, sign, unit and interval as printed. Spelling one number is centralised already
(``kernel/numbers.py``); deciding which figures form a set, and how many digits the set reads at, is not: it happens
at hundreds of sites, at three different times, by several owners, so one measure prints two ways on one page ("85%"
in a table, "85.3%" in the chart head). The number-forms plan moves every decision into ``analysis/report/forms.py``.
This test keeps that migration honest:

* **The walk** reads every module of ``threetears/evals`` and the kit's ``kit.js``, and finds each site that decides
  a form: a digit-choosing format spec, ``round(``, a call to a spelling or set primitive outside the two owners
  (``kernel/numbers.py`` and ``analysis/report/forms.py``), and in ``kit.js`` the digit primitives and the named
  formats used outside ``K_FMTS`` / ``fmt``. A package function that returns what a site produced is a primitive
  too, to the fixed point, so a call rerouted through a wrapper is still a site where it is called. So is a public
  free function of ``forms.py`` (a decision relocated there): only ``PageForms``, asked by measure and role, is the
  migrated route.
* **The ratchet** compares the walk with ``fixtures/number_form_sites.txt``, keyed by (module, enclosing qualname,
  form) with a count. A moved line keeps its entry; a new site in the same function, or in a new one, fails. An entry
  that overstates its count, or names nothing, also fails, so every migration cuts the file and the list only
  shrinks.
* **What the wrapper rule cannot see** is a stated limit, not a hidden one: a figure reached through another object's
  method of a common name, lines appended then joined, a ``yield``, a tuple unpacked, a lambda table, or digits
  derived as ``floor(v + 0.5)``. A function's nested returns count as its own, which over-counts, the safe way.
* **The permanent list** (:data:`_PERMANENT`) names the few sites whose fixed form is right for good (SVG
  coordinates and opacities, the kit's formats restated for the static page), each with its reason.

It also holds the report trees to the stricter rule ``test_numbers.py`` held them to before the ratchet existed:
under ``analysis/`` and ``ops/`` no format spec chooses digits except a share's ``%`` and a permanent site.
"""

from __future__ import annotations

import ast
import bisect
import functools
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
_PACKAGE = "threetears/evals"
_ALLOWLIST = Path(__file__).resolve().parent / "fixtures" / "number_form_sites.txt"

#: The rule's spelling primitives, and the module that decides a page's forms from C2 on. Nothing in either is a
#: site: they are where a form is meant to be decided. ``forms.py`` does not exist before C2, so it is optional.
_RULE = "threetears/evals/kernel/numbers.py"
_FORMS = "threetears/evals/analysis/report/forms.py"
_OWNERS = (_RULE, _FORMS)

#: The kit's script. Its copy under ``docs/eval-report-kit/kit/`` is pinned byte-identical to this one
#: (``test_report_kit_pages``), so walking one walks both.
_KIT_JS = "threetears/evals/analysis/report/kit/kit.js"

#: The kit's two named-format tables, its owners of a drawn figure's form: a site inside either is not a site.
_KIT_OWNERS = frozenset({"fmt", "K_FMTS"})

#: The set and precision helpers outside the rule that the plan's inventory names as deciding a figure's form. A
#: call to one, or one passed as a function value, is a site. The rule's own public callables join them by derivation
#: (:func:`_rule_primitives`), so a primitive added to ``numbers.py`` is covered without an edit here.
_SET_HELPERS = frozenset(
    {
        "bare_places",
        "dollars_text",
        "figure_places",
        "figure_text",
        "format_figure",
        "held_means",
        "interval_level",
        "interval_words",
    }
)

#: Families the plan names by prefix (``percent_*``, ``respell_*``, ``quantity*``), so a new member is a site too.
_SET_HELPER_PREFIXES = ("percent_", "respell_", "quantity")

#: Private helpers that decide a figure's places, keyed by the one module allowed to call them (a leading underscore
#: binds nothing outside its module). Matched by name alone they would also catch ``levers._places``, which lists
#: where an effect is read, and ``build._at``, which links a cell to a finding.
_PRIVATE_SET_HELPERS = frozenset(
    {
        ("threetears/evals/analysis/report/campaign/scorecard.py", "_one_precision"),
        ("threetears/evals/analysis/report/figure_keys.py", "_one_precision"),
        ("threetears/evals/harness/writer_eval.py", "_places"),
    }
)

#: The trees whose every number is shown to a reader, held to the stricter rule: the analysis package, what the
#: generator writes from, and the ops surfaces it renders through. A host's own renders (an operator console, a tool
#: surface) are the host's to walk on the same terms. A module sits here by itself, rather than its whole directory,
#: where the rest of that directory is not read as a report.
_SPELLING_TREES = ("threetears/evals/analysis", "threetears/evals/ops")

#: What sits inside :data:`_SPELLING_TREES` and is not held to the stricter rule (the ratchet still lists its sites):
#: ``report/svg.py`` spells only a glyph's drawing coordinates (to a tenth of a unit), every value it shows a reader
#: going through the rule. ``out_of_run_spend.py`` is the spend ledger, which spells only dollars, at the four
#: decimals a per-call price needs.
_NOT_READ_AS_REPORTS = (
    "threetears/evals/analysis/report/svg.py",
    "threetears/evals/analysis/out_of_run_spend.py",
)

#: Presentation types that choose how many digits a reader sees. ``d`` and ``s`` do not.
_NUMERIC_TYPES = frozenset("eEfFgGn%")

#: A share of a whole spelled as a percentage (``.0%``) is a different sentence from a measured value, "37% of it",
#: and carries its own fixed precision. The stricter rule passes it by presentation type; the ratchet still lists it,
#: because from C3 on a share prints through ``forms.share``.
_PERCENTAGE_TYPE = "%"

#: The static page's chart drawer, and why its kit formats spell as they do.
_CHART_SVG = "threetears/evals/analysis/report/chart_svg.py"
_KIT_FORMAT = (
    "the kit's named format, restated from kit.js for the static page and pinned to number-format-cases.json: "
)

#: Sites whose fixed form is right for good, keyed by (module, expression, spec or form) so a moved line keeps its
#: entry and a rewritten expression loses it. Each reason says why the fixed form is right there; an entry no site
#: matches any more fails the suite. Only C6 adds to it.
_PERMANENT: dict[tuple[str, str, str], str] = {
    # Money's spelling (two significant figures, cents from ten cents up) is numbers.spell_dollars, in the
    # rule's own module.
    # A static page's chart is the kit's drawing made on the server: its ticks and notes are the kit's named formats
    # (kit.js K_FMTS), restated so the static page reads the number the default page's script draws, and pinned to the
    # same cases table the kit and format_number are (test_report_chart_svg). The figure's table beside it keeps the one
    # rule. Its markup spells coordinates and opacities, which no reader reads as a number.
    (_CHART_SVG, "value", ",.{}f"): _KIT_FORMAT + "grouped dollars at the format's digits",
    (_CHART_SVG, "_rounded(size)", ","): _KIT_FORMAT + "whole dollars from $10",
    (_CHART_SVG, "_rounded(value)", ","): _KIT_FORMAT + "a whole amount, grouped",
    (_CHART_SVG, "n", ",.3f"): _KIT_FORMAT + "a tooltip's count, as toLocaleString writes it",
    (_CHART_SVG, "joined.width or 2", "g"): "an SVG stroke width",
    (_CHART_SVG, "1 if joined.opacity is None else joined.opacity", "g"): "an SVG opacity",
    (_CHART_SVG, "1 if item.opacity is None else item.opacity", "g"): "an SVG opacity",
    (_CHART_SVG, "opacity", "g"): "an SVG opacity",
    # The glyph's and the static charts' coordinates, every one written through svg.tenth_px.
    ("threetears/evals/analysis/report/svg.py", "round_figures(value)", ".1f"): (
        "an SVG coordinate, to a tenth of a px, which no reader reads as a number"
    ),
}

#: The one-line remedy every ratchet failure ends with.
_REMEDY = (
    "route the figure through analysis/report/forms.py (or, in C6 only, name the site in _PERMANENT with its reason)"
)


@dataclass(frozen=True)
class _Site:
    """One place a figure's form is decided.

    :ivar module: The file, relative to the package's ``src``.
    :ivar line: Its line, for the reader of a failure only: the ratchet's key leaves it out.
    :ivar qualname: The enclosing definition (``Class.method``, ``outer.inner``; ``<module>`` or ``<top>`` outside one).
    :ivar form: How the form is decided: ``f-string spec``, ``round()``, ``format_number()``, ``toFixed()`` ...
    :ivar expression: The value spelled, for a spec; the call or reference itself otherwise.
    :ivar spec: The format spec, for a spec site; ``None`` for any other form.
    """

    module: str
    line: int
    qualname: str
    form: str
    expression: str
    spec: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.module, self.qualname, self.form)

    @property
    def permanent_key(self) -> tuple[str, str, str]:
        return (self.module, self.expression, self.form if self.spec is None else self.spec)

    def shown(self) -> str:
        what = f"{{{self.expression}:{self.spec}}}" if self.spec is not None else self.expression
        return f"{self.module}:{self.line} ({self.qualname}, {self.form}): {what}"


# --- Python -------------------------------------------------------------------------------------------------------

_PRECISION = re.compile(r"\.(\d+|\{\})")
_PERCENT_DIRECTIVE = re.compile(r"%[-+ #0]*(\d+|\*)?(\.(\d+|\*))?[eEfFgG]")
_BRACE_FIELD_SPEC = re.compile(r"\{[^{}]*:([^{}]*)\}")


def _chooses_digits(spec: str) -> bool:
    """Whether a format spec decides how many digits a number is written with.

    A precision (fixed or nested), a numeric presentation type, a thousands separator, or a spec
    built at run time, which cannot be judged here and so is flagged.
    """
    if "{}" in spec or _PRECISION.search(spec) or "," in spec or "_" in spec:
        return True
    return bool(spec) and spec[-1] in _NUMERIC_TYPES


def _joined_spec(spec: ast.JoinedStr) -> str:
    return "".join(part.value if isinstance(part, ast.Constant) else "{}" for part in spec.values)


def _rule_primitives() -> tuple[frozenset[str], frozenset[str]]:
    """The rule's public callables: its top-level functions and classes, and apart, the public methods of those classes.

    A method's name (``spell``, ``places``) is also an ordinary local's, so it is a site only as a method call.
    """
    tree = ast.parse((_SOURCE_ROOT / _RULE).read_text(encoding="utf-8"))
    names: set[str] = set()
    methods: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            names.add(node.name)
            if isinstance(node, ast.ClassDef):
                methods |= {
                    member.name
                    for member in node.body
                    if isinstance(member, ast.FunctionDef) and not member.name.startswith("_")
                }
    return frozenset(names), frozenset(methods)


_RULE_CALLABLES, _RULE_METHODS = _rule_primitives()


def _forms_primitives(root: Path = _SOURCE_ROOT) -> frozenset[str]:
    """The public module-level functions of ``forms.py``: a decision relocated there and called as a free function
    (``at_decimals``, ``level_decimals``) is still decided where it is called, so its caller is a site. Only
    :class:`PageForms`, built once per page and asked by measure and role, is the migrated route: its classes (the
    page's forms and the value types it hands out) and their methods are not primitives. None before ``forms.py``
    exists.
    """
    path = root / _FORMS
    if not path.exists():
        return frozenset()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
    )


_FORMS_CALLABLES = _forms_primitives()

_BASE_PRIMITIVES = _RULE_CALLABLES | _SET_HELPERS | _FORMS_CALLABLES

#: Every method name of a builtin value. A wrapper method named like one (``format``, ``count``) is still a site
#: through ``self``, but an attribute call by that name alone is not taken to reach it.
_BUILTIN_METHODS = frozenset(name for kind in (str, bytes, list, dict, set, tuple, int, float) for name in dir(kind))


#: ``math.floor`` / ``math.ceil`` over a ``log10`` is digits derived from a magnitude (``2 - floor(log10(x))``
#: places); the ``log10`` alone is also a log axis's position, so it is a site only inside a floor or a ceiling.
_DIGITS_FROM_MAGNITUDE = frozenset({"floor", "ceil"})

#: Calls that hand a spelled figure on as text: a string method on it, or a builtin over it. What a function returns
#: through one of these is still the figure's form, so the function is a wrapper. A method is matched as an attribute
#: call and a builtin as a bare name, so ``dataclasses.replace(cell, text=...)`` is not read as ``str.replace``.
_TEXT_METHODS = frozenset(
    {
        "capitalize",
        "center",
        "join",
        "ljust",
        "lower",
        "lstrip",
        "removeprefix",
        "removesuffix",
        "replace",
        "rjust",
        "rstrip",
        "strip",
        "title",
        "upper",
    }
)
_TEXT_BUILTINS = frozenset({"abs", "int", "list", "max", "min", "sorted", "str", "tuple"})


def _dotted(module: str) -> str:
    """``threetears/evals/a/b.py`` as ``threetears.evals.a.b``; a package's ``__init__.py`` as the package."""
    parts = module.removesuffix(".py").split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


class _Index:
    """The walked modules' definitions and imports, to tell which function a call reaches.

    A call is resolved the way Python resolves it, as far as the source says: a name to its own module's definition
    or to the definition its import names (through any re-export), a module alias's attribute to that module's
    definition, ``self.name`` to the method of the module's own class. An attribute call on anything else is taken to
    reach a wrapper method only when every method of that name in the package is one.
    """

    def __init__(self, trees: dict[str, ast.Module]) -> None:
        self.trees = trees
        self.by_dotted = {_dotted(module): module for module in trees}
        self.defined: dict[str, set[str]] = {}
        self.imports: dict[str, dict[str, tuple[str, str | None]]] = {}
        #: Each method name, with ``(module, "Class.name")`` of every method that bears it.
        self.methods: dict[str, set[tuple[str, str]]] = {}
        for module, tree in trees.items():
            self.defined[module] = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            self.imports[module] = dict(self._imports(module, tree))
            for function, reached, cls in _definitions(tree):
                if reached == "method":
                    self.methods.setdefault(function.name, set()).add((module, f"{cls}.{function.name}"))

    def _imports(self, module: str, tree: ast.Module) -> Iterator[tuple[str, tuple[str, str | None]]]:
        package = _dotted(module).split(".") if module.endswith("__init__.py") else _dotted(module).split(".")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = ".".join([*package[: len(package) - node.level + 1], *([node.module] if node.module else [])])
                base = base if node.level else (node.module or "")
                for alias in node.names:
                    target = f"{base}.{alias.name}"
                    yield alias.asname or alias.name, (target, None) if target in self.by_dotted else (base, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        yield alias.asname, (alias.name, None)

    def resolve(self, module: str, name: str, depth: int = 0) -> tuple[str, str] | None:
        """The ``(module, name)`` of the definition ``name`` reaches in ``module``, or ``None`` outside the walk."""
        if name in self.defined.get(module, ()):
            return (module, name)
        target, attribute = self.imports.get(module, {}).get(name, ("", None))
        if attribute is None or target not in self.by_dotted or depth > 8:
            return None
        return self.resolve(self.by_dotted[target], attribute, depth + 1)

    def module_alias(self, module: str, name: str) -> str | None:
        target, attribute = self.imports.get(module, {}).get(name, ("", ""))
        return self.by_dotted.get(target) if attribute is None else None


@dataclass(frozen=True)
class _Lookup:
    """What a call names to be a site: the rule's and the plan's primitives, and the wrappers derived so far.

    :ivar index: The walked modules' definitions and imports.
    :ivar functions: ``(module, name)`` of every module-level or nested function found to be a wrapper.
    :ivar methods: ``(module, "Class.name")`` of every method found to be a wrapper.
    """

    index: _Index
    functions: frozenset[tuple[str, str]] = frozenset()
    methods: frozenset[tuple[str, str]] = frozenset()

    def named(self, node: ast.expr, where: tuple[str, str], *, as_value: bool = False) -> str | None:
        """The primitive ``node`` names, called (or, with ``as_value``, handed on as a function value), or ``None``.

        A family prefix or a method name is also an ordinary local's name (``quantity``, ``places``), so a value
        handed on is matched by exact name only, and a method only as an attribute call. ``where`` is the call's
        module and enclosing class (empty outside one), which is what ``self`` names.
        """
        module, cls = where
        if isinstance(node, ast.Name):
            name = node.id
            if (module, name) in _PRIVATE_SET_HELPERS or (module, name) in self.functions:
                return name
            if self.index.resolve(module, name) in self.functions:
                return name
        elif isinstance(node, ast.Attribute):
            name = node.attr
            if not as_value and name in _RULE_METHODS:
                return name
            owner = node.value
            if isinstance(owner, ast.Name):
                if owner.id in ("self", "cls") and (module, f"{cls}.{name}") in self.methods:
                    return name
                alias = self.index.module_alias(module, owner.id)
                if alias is not None and self.index.resolve(alias, name) in self.functions:
                    return name
            defined = self.index.methods.get(name, set())
            if not as_value and defined and name not in _BUILTIN_METHODS and defined <= self.methods:
                return name
        else:
            return None
        if name.startswith("_"):
            return None
        if name in _BASE_PRIMITIVES or (not as_value and name.startswith(_SET_HELPER_PREFIXES)):
            return name
        return None


def _call_form(node: ast.Call, where: tuple[str, str], lookup: _Lookup) -> str | None:
    """The form a call decides by itself (a format spec aside), or ``None``."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == "round":
        return "round()"
    if isinstance(func, ast.Attribute) and func.attr == "quantize":
        return "quantize()"
    if (
        isinstance(func, ast.Attribute)
        and func.attr in _DIGITS_FROM_MAGNITUDE
        and any(
            isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "log10"
            for argument in node.args
            for inner in ast.walk(argument)
        )
    ):
        return f"{func.attr}(log10)"
    name = lookup.named(func, where)
    return None if name is None else f"{name}()"


def _spec_sites(node: ast.AST) -> Iterator[tuple[str, str, str]]:
    """Every ``(form, expression, spec)`` the node itself spells with a digit-choosing spec.

    Covers the ways Python formats a number: an f-string or t-string field, the ``format`` builtin, ``str.format``
    on a literal, and ``%`` on a literal.
    """
    if isinstance(node, (ast.FormattedValue, ast.Interpolation)) and isinstance(node.format_spec, ast.JoinedStr):
        spec = _joined_spec(node.format_spec)
        if _chooses_digits(spec):
            form = "f-string spec" if isinstance(node, ast.FormattedValue) else "t-string spec"
            yield form, ast.unparse(node.value), spec
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            for match in _PERCENT_DIRECTIVE.finditer(node.left.value):
                yield "% spec", ast.unparse(node.right), match.group(0)
    elif isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "format" and len(node.args) == 2:
            spec_node = node.args[1]
            spec = spec_node.value if isinstance(spec_node, ast.Constant) and isinstance(spec_node.value, str) else "{}"
            if _chooses_digits(spec):
                yield "format() spec", ast.unparse(node.args[0]), spec
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "format"
            and isinstance(func.value, ast.Constant)
            and isinstance(func.value.value, str)
        ):
            for spec in _BRACE_FIELD_SPEC.findall(func.value.value):
                if _chooses_digits(spec):
                    yield "str.format() spec", ast.unparse(node), spec


class _PythonForms(ast.NodeVisitor):
    """Collects every site in one module, each under its enclosing definition."""

    def __init__(self, module: str, lookup: _Lookup) -> None:
        self.module = module
        self.lookup = lookup
        self.scope: list[str] = []
        self.classes: list[str] = []
        self.sites: list[_Site] = []

    @property
    def where(self) -> tuple[str, str]:
        return (self.module, self.classes[-1] if self.classes else "")

    def _add(self, node: ast.AST, form: str, expression: str, spec: str | None = None) -> None:
        qualname = ".".join(self.scope) or "<module>"
        self.sites.append(_Site(self.module, getattr(node, "lineno", 0), qualname, form, expression, spec))

    def _scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.scope.append(node.name)
        if isinstance(node, ast.ClassDef):
            self.classes.append(node.name)
        self.generic_visit(node)
        if isinstance(node, ast.ClassDef):
            self.classes.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node)

    def generic_visit(self, node: ast.AST) -> None:
        for form, expression, spec in _spec_sites(node):
            self._add(node, form, expression, spec)
        if isinstance(node, ast.Call):
            if (form := _call_form(node, self.where, self.lookup)) is not None:
                self._add(node, form, ast.unparse(node))
            # A primitive handed on as a function value (``map(format_number, ...)``, ``key=spell_at``) decides
            # where it is applied, which is here.
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
                if (handed := self.lookup.named(argument, self.where, as_value=True)) is not None:
                    self._add(argument, f"{handed}()", ast.unparse(argument))
        super().generic_visit(node)


def _spelled(node: ast.AST | None, where: tuple[str, str], lookup: _Lookup, spelled: set[str]) -> bool:
    """Whether ``node``'s value is a figure's form: a site's result, or text built from one.

    Only what carries the text on counts (a string built from it, a branch, a tuple, a string method or a builtin
    over it). An object built around it (a table cell, a page) does not, and neither do lines joined into a document:
    were they to, every builder that ever placed a figure would be a wrapper, and so would every caller of one, which
    over this package is most of it (3,357 of its 5,716 functions, measured 2026-11-08).
    """
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id in spelled
    module = where[0]
    if any((module, expression, spec) not in _PERMANENT for _form, expression, spec in _spec_sites(node)):
        return True
    if isinstance(node, ast.JoinedStr):
        return any(_spelled(part, where, lookup, spelled) for part in node.values)
    if isinstance(node, ast.FormattedValue):
        return _spelled(node.value, where, lookup, spelled)
    if isinstance(node, ast.BinOp):
        return _spelled(node.left, where, lookup, spelled) or _spelled(node.right, where, lookup, spelled)
    if isinstance(node, ast.UnaryOp):
        return _spelled(node.operand, where, lookup, spelled)
    if isinstance(node, ast.IfExp):
        return _spelled(node.body, where, lookup, spelled) or _spelled(node.orelse, where, lookup, spelled)
    if isinstance(node, ast.BoolOp):
        return any(_spelled(value, where, lookup, spelled) for value in node.values)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_spelled(element, where, lookup, spelled) for element in node.elts)
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return _spelled(node.elt, where, lookup, spelled)
    if isinstance(node, (ast.Starred, ast.Subscript, ast.Await)):
        return _spelled(node.value, where, lookup, spelled)
    if isinstance(node, ast.Call):
        form = _call_form(node, where, lookup)
        if form is not None and (module, ast.unparse(node), form) not in _PERMANENT:
            return True
        func = node.func
        if isinstance(func, ast.Name) and func.id in _TEXT_BUILTINS:
            carried = [*node.args, *(keyword.value for keyword in node.keywords)]
            return any(_spelled(part, where, lookup, spelled) for part in carried)
        if isinstance(func, ast.Attribute) and func.attr in _TEXT_METHODS:
            joins_lines = isinstance(func.value, ast.Constant) and "\n" in str(func.value.value)
            carried = [func.value, *node.args]
            return not joins_lines and any(_spelled(part, where, lookup, spelled) for part in carried)
    return False


def _definitions(tree: ast.Module) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str, str]]:
    """Every function in ``tree``, how a call reaches it (``module`` level, a ``method``, or ``nested``), and the
    class whose ``self`` it reads (empty outside one)."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, "module", ""
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield member, "method", node.name
    for top in tree.body:
        for node in ast.walk(top):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cls = top.name if isinstance(top, ast.ClassDef) else ""
                for inner in ast.walk(node):
                    if inner is not node and isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        yield inner, "nested", cls


def _returns_a_form(function: ast.FunctionDef | ast.AsyncFunctionDef, where: tuple[str, str], lookup: _Lookup) -> bool:
    """Whether what ``function`` returns is a figure's form, directly or through a local it spelled."""
    spelled: set[str] = set()
    body = list(ast.walk(function))
    for _ in range(2):  # twice, so a local spelled from one assigned below it in the source is still seen
        for node in body:
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if _spelled(node.value, where, lookup, spelled):
                    spelled |= {target.id for target in targets if isinstance(target, ast.Name)}
    return any(isinstance(node, ast.Return) and _spelled(node.value, where, lookup, spelled) for node in body)


def _reached_names(function: ast.FunctionDef | ast.AsyncFunctionDef, module: str, index: _Index) -> frozenset[str]:
    """The names a function's calls reach: each as written, and each import alias as the name it imports."""
    names = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            names.add(node.id)
            resolved = index.resolve(module, node.id)
            if resolved is not None:
                names.add(resolved[1])
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return frozenset(names)


def _derive_wrappers(index: _Index) -> _Lookup:
    """Every function over the walked modules that returns a form, to the fixed point.

    A helper that returns what a site produced is a primitive itself, and so is one that returns what such a helper
    produced, so a call rerouted through a wrapper is still a site where it is called. The owners are not walked, so
    routing through them stays the remedy, and a permanent site wraps nothing. After the first pass only a function
    that names a wrapper found in the pass before can become one, so only those are read again.
    """
    candidates = [
        (function, module, cls, reached, _reached_names(function, module, index))
        for module, tree in index.trees.items()
        for function, reached, cls in _definitions(tree)
        if not function.name.startswith("__")
    ]
    lookup = _Lookup(index)
    fresh: frozenset[str] | None = None
    while True:
        functions: set[tuple[str, str]] = set()
        methods: set[tuple[str, str]] = set()
        for function, module, cls, reached, names in candidates:
            key = (module, f"{cls}.{function.name}") if reached == "method" else (module, function.name)
            if key in lookup.functions or key in lookup.methods or (fresh is not None and not names & fresh):
                continue
            if _returns_a_form(function, (module, cls), lookup):
                (methods if reached == "method" else functions).add(key)
        if not functions and not methods:
            return lookup
        fresh = frozenset(name.rsplit(".", 1)[-1] for _module, name in functions | methods)
        lookup = _Lookup(index, lookup.functions | functions, lookup.methods | methods)


def _walk_python(sources: dict[str, str]) -> list[_Site]:
    """Every site in ``sources`` (module to text), with the wrappers derived over them as primitives."""
    index = _Index({module: ast.parse(text) for module, text in sources.items()})
    lookup = _derive_wrappers(index)
    sites = []
    for module, tree in index.trees.items():
        visitor = _PythonForms(module, lookup)
        visitor.visit(tree)
        sites.extend(visitor.sites)
    return sorted(sites, key=lambda site: (site.module, site.line, site.form, site.expression))


def _python_sites(source: str, module: str = "sample.py") -> list[_Site]:
    return _walk_python({module: source})


# --- kit.js -------------------------------------------------------------------------------------------------------

_JS_WORD = re.compile(r"[A-Za-z0-9_$]+")
#: After one of these a ``/`` opens a regular expression; after anything else it divides.
_REGEX_AFTER = frozenset("(,=:[!&|?{};+-*%<>~^")
_REGEX_AFTER_WORDS = frozenset(
    {"return", "typeof", "case", "in", "of", "new", "delete", "void", "throw", "else", "do", "yield", "await"}
)
_JS_TOKEN = re.compile(r"[A-Za-z_$][\w$]*|[()\[\]{};\n]")
_JS_DECLARES = frozenset({"function", "const", "let", "var", "class"})
_JS_DECLARED = re.compile(r"(?:\bfunction|\bconst|\blet|\bvar|\bclass)\s*$")


def _js_named(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w$.]){re.escape(name)}(?![\w$])")


#: Each kit.js form: the digit primitives (anything ``Intl`` offers among them), and the named formats picked or
#: applied by name. A call that applies a format the chart's data names (``f(v)``, ``opt.fmt(v)``) is not a site:
#: the data's ``fmt`` is decided in Python, where the walk sees it, and the pick is the ``namedFmt`` call.
_JS_FORMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("toFixed()", re.compile(r"\.\s*toFixed\s*\(")),
    ("toPrecision()", re.compile(r"\.\s*toPrecision\s*\(")),
    ("toLocaleString()", re.compile(r"\.\s*toLocaleString\s*\(")),
    ("Intl.", re.compile(r"(?<![\w$.])Intl\s*\.")),
    ("Math.round()", re.compile(r"(?<![\w$.])Math\s*\.\s*round\s*\(")),
    ("fmt.", re.compile(r"(?<![\w$.])fmt\s*\.")),
    ("K_FMTS", _js_named("K_FMTS")),
    ("namedFmt", _js_named("namedFmt")),
    ("kTickFmt", _js_named("kTickFmt")),
    ("kTickLabels", _js_named("kTickLabels")),
    ("kShort", _js_named("kShort")),
)


def _quoted_end(source: str, start: int) -> int:
    quote, at = source[start], start + 1
    while at < len(source):
        if source[at] == "\\":
            at += 2
        elif source[at] in (quote, "\n"):
            return at + 1 if source[at] == quote else at
        else:
            at += 1
    return len(source)


def _template_text_end(source: str, start: int) -> tuple[int, bool]:
    """Where a template literal's text run from ``start`` ends, and whether a ``${`` (rather than the close) ended it."""
    at = start
    while at < len(source):
        if source[at] == "\\":
            at += 2
        elif source[at] == "`":
            return at, False
        elif source.startswith("${", at):
            return at, True
        else:
            at += 1
    return len(source), False


def _regex_end(source: str, start: int) -> tuple[int, int] | None:
    """The end of a regex literal's body and of its flags, or ``None`` where the ``/`` at ``start`` opens none."""
    at, in_class = start + 1, False
    while at < len(source):
        char = source[at]
        if char == "\\":
            at += 2
            continue
        if char == "\n":
            return None
        if in_class:
            in_class = char != "]"
        elif char == "[":
            in_class = True
        elif char == "/":
            flags = at + 1
            while flags < len(source) and source[flags].isalpha():
                flags += 1
            return at, flags
        at += 1
    return None


def _code_only(source: str) -> str:
    """``source`` with every comment, string, template text and regex body blanked, offsets and newlines kept.

    A template's ``${...}`` expressions stay, since they are code; what is blanked can hold no site.
    """
    chars = list(source)

    def blank(start: int, end: int) -> None:
        for at in range(start, end):
            if chars[at] != "\n":
                chars[at] = " "

    at, last, templates = 0, "", []
    while at < len(source):
        char = source[at]
        if char.isspace():
            at += 1
            continue
        if source.startswith("//", at):
            end = source.find("\n", at)
            end = len(source) if end < 0 else end
            blank(at, end)
            at = end
            continue
        if source.startswith("/*", at):
            end = source.find("*/", at + 2)
            end = len(source) if end < 0 else end + 2
            blank(at, end)
            at = end
            continue
        if char in "'\"":
            end = _quoted_end(source, at)
            blank(at + 1, end - 1)
            at, last = end, "'"
            continue
        if char == "`" or (char == "}" and templates and templates[-1] == 0):
            if char == "}":
                templates.pop()
            end, opened = _template_text_end(source, at + 1)
            blank(at + 1, end)
            if opened:
                templates.append(0)
                at, last = end + 2, "("
            else:
                at, last = end + 1, "'"
            continue
        if char == "/" and (last == "" or last in _REGEX_AFTER or last in _REGEX_AFTER_WORDS):
            ends = _regex_end(source, at)
            if ends is not None:
                blank(at + 1, ends[0])
                at, last = ends[1], "'"
                continue
        word = _JS_WORD.match(source, at)
        if word:
            at, last = word.end(), word.group()
            continue
        if templates and char in "{}":
            templates[-1] += 1 if char == "{" else -1
        at, last = at + 1, char
    return "".join(chars)


def _js_statements(code: str) -> tuple[list[int], list[str]]:
    """The start offset and name of each top-level statement: its declared name, or ``<top>`` for any other."""
    offsets, names = [0], ["<top>"]
    depth, at_start, declaring = 0, True, -1
    for token in _JS_TOKEN.finditer(code):
        text = token.group()
        if text in ("(", "[", "{"):
            depth += 1
        elif text in (")", "]", "}"):
            depth -= 1
            at_start = at_start or (depth == 0 and text == "}")
        elif depth:
            continue
        elif text in (";", "\n"):
            at_start = True
        elif declaring >= 0:
            offsets.append(declaring)
            names.append(text)
            declaring = -1
        elif at_start and text != "async":
            at_start = False
            if text in _JS_DECLARES:
                declaring = token.start()
            else:
                offsets.append(token.start())
                names.append("<top>")
    return offsets, names


def _balanced_end(code: str, opening: int) -> int:
    depth = 0
    for at in range(opening, len(code)):
        if code[at] in "([{":
            depth += 1
        elif code[at] in ")]}":
            depth -= 1
            if depth == 0:
                return at + 1
    return len(code)


def _js_expression(source: str, code: str, start: int, end: int) -> str:
    """The site's text: the match, on through any member chain, and through the call or index that follows it."""
    if code[end - 1] == "(":
        end = _balanced_end(code, end - 1)
    else:
        while end < len(code) and (code[end].isalnum() or code[end] in "_$."):
            end += 1
        if end < len(code) and code[end] in "([":
            end = _balanced_end(code, end)
    return " ".join(source[start:end].split())


_JS_EXPRESSION_ARROW = re.compile(r"(?:const|let|var)\s+[\w$]+\s*=\s*(?:async\s*)?(?:\([^()]*\)|[\w$]+)\s*=>\s*(?!\{)")


def _js_returned(code: str, start: int, end: int) -> list[tuple[int, int]]:
    """The spans of the top-level statement ``code[start:end]`` whose value it returns.

    An expression-bodied arrow (``const f=v=>...``) returns the whole of itself; anything else returns what each
    ``return`` runs to, its ``;`` or the close of the block it sits in.
    """
    if _JS_EXPRESSION_ARROW.match(code, start):
        return [(start, end)]
    spans = []
    for found in re.finditer(r"(?<![\w$.])return(?![\w$])", code[start:end]):
        depth, at = 0, start + found.end()
        while at < end and not (depth == 0 and code[at] == ";") and depth >= 0:
            depth += 1 if code[at] in "([{" else -1 if code[at] in ")]}" else 0
            at += 1
        spans.append((start + found.end(), at))
    return spans


def _js_found(code: str, forms: Iterable[tuple[str, re.Pattern[str]]]) -> Iterator[tuple[str, re.Match[str]]]:
    for form, pattern in forms:
        for match in pattern.finditer(code):
            if not _JS_DECLARED.search(code, 0, match.start()):
                yield form, match


def _js_sites(source: str, module: str = "sample.js") -> list[_Site]:
    """Every site in a kit script, with each top-level function that returns one a form of its own.

    A function whose returned value holds a site (``const kAt=v=>v.toFixed(1)``) is a wrapper, and a call to it is a
    site, to the fixed point, so a figure rerouted through a new helper is still counted where it is called. A
    function that only holds a site in its body (``ticksRaw`` cleaning a float with ``toFixed(10)``) is not.
    """
    code = _code_only(source)
    offsets, names = _js_statements(code)
    bounds = [*offsets[1:], len(code)]
    spans = {}
    for start, end, name in zip(offsets, bounds, names, strict=True):
        if name != "<top>" and name not in _KIT_OWNERS:
            spans.setdefault(name, []).extend(_js_returned(code, start, end))
    named = {form for form, _pattern in _JS_FORMS}
    forms = list(_JS_FORMS)
    while True:
        returning = set()
        for _form, match in _js_found(code, forms):
            name = names[bisect.bisect_right(offsets, match.start()) - 1]
            if name not in named and any(low <= match.start() < high for low, high in spans.get(name, ())):
                returning.add(name)
        if not returning:
            break
        forms.extend((f"{name}()", re.compile(rf"(?<![\w$.]){re.escape(name)}\s*\(")) for name in sorted(returning))
        named |= returning
    sites = []
    for form, match in _js_found(code, forms):
        qualname = names[bisect.bisect_right(offsets, match.start()) - 1]
        if qualname in _KIT_OWNERS:
            continue
        line = code.count("\n", 0, match.start()) + 1
        sites.append(_Site(module, line, qualname, form, _js_expression(source, code, match.start(), match.end())))
    return sorted(sites, key=lambda site: (site.line, site.form, site.expression))


# --- The walk and the ratchet -------------------------------------------------------------------------------------


def _walked_modules(root: Path = _SOURCE_ROOT) -> list[Path]:
    """Every Python module of the package under ``root`` but the owners."""
    owners = {root / owner for owner in _OWNERS}
    return [path for path in sorted((root / _PACKAGE).rglob("*.py")) if path not in owners]


def _modules_under(trees: Iterable[str], root: Path = _SOURCE_ROOT) -> list[Path]:
    """Every module an entry of ``trees`` names under ``root`` (each ``.py`` under a directory, or the file itself),
    less the owners and the ones :data:`_NOT_READ_AS_REPORTS` sets aside."""
    modules = []
    for tree_root in trees:
        named = root / tree_root
        assert named.is_dir() or (named.is_file() and named.suffix == ".py"), tree_root
        modules.extend(sorted(named.rglob("*.py")) if named.is_dir() else [named])
    excluded = [root / entry for entry in (*_NOT_READ_AS_REPORTS, *_OWNERS)]
    return [module for module in modules if not any(module == ex or ex in module.parents for ex in excluded)]


@functools.cache
def _all_sites() -> tuple[_Site, ...]:
    """Every site of the package and the kit, walked once: deriving the wrappers reads the whole package."""
    sources = {
        path.relative_to(_SOURCE_ROOT).as_posix(): path.read_text(encoding="utf-8") for path in _walked_modules()
    }
    kit = _js_sites((_SOURCE_ROOT / _KIT_JS).read_text(encoding="utf-8"), _KIT_JS)
    return (*_walk_python(sources), *kit)


def _ratcheted(sites: Iterable[_Site]) -> list[_Site]:
    return [site for site in sites if site.permanent_key not in _PERMANENT]


def _tally(sites: Iterable[_Site]) -> Counter[tuple[str, str, str]]:
    return Counter(site.key for site in sites)


_ENTRY = re.compile(r"^(?P<module>[^|]+?) \| (?P<qualname>[^|]+?) \| (?P<form>[^|]+?) \| (?P<count>[1-9]\d*)$")


def _parse_allowlist(text: str) -> dict[tuple[str, str, str], int]:
    """The allowlist's entries; a malformed line, a repeated key or an unsorted file is refused."""
    entries: dict[tuple[str, str, str], int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        match = _ENTRY.match(line)
        assert match, f"{_ALLOWLIST.name}:{number}: not `module | qualname | form | count`: {line!r}"
        key = (match["module"], match["qualname"], match["form"])
        assert key not in entries, f"{_ALLOWLIST.name}:{number}: {key} is listed twice"
        entries[key] = int(match["count"])
    assert list(entries) == sorted(entries), f"{_ALLOWLIST.name} is not sorted by (module, qualname, form)"
    return entries


def _allowlist() -> dict[tuple[str, str, str], int]:
    return _parse_allowlist(_ALLOWLIST.read_text(encoding="utf-8"))


def _over(live: Counter[tuple[str, str, str]], listed: dict[tuple[str, str, str], int]) -> dict:
    return {key: (count, listed.get(key, 0)) for key, count in live.items() if count > listed.get(key, 0)}


def _under(live: Counter[tuple[str, str, str]], listed: dict[tuple[str, str, str], int]) -> dict:
    return {key: (live.get(key, 0), count) for key, count in listed.items() if count > live.get(key, 0)}


class TestTheRatchet:
    """Every site the walk finds is listed at its count, and every listed count is a site the tree still holds."""

    def test_no_site_is_new(self):
        sites = _ratcheted(_all_sites())
        over = _over(_tally(sites), _allowlist())
        shown = [
            f"{' | '.join(key)}: {live} in the tree, {listed} listed\n      "
            + "\n      ".join(site.shown() for site in sites if site.key == key)
            for key, (live, listed) in sorted(over.items())
        ]
        assert not over, f"new number-form sites: {_REMEDY}.\n  " + "\n  ".join(shown)

    def test_no_entry_outlives_its_sites(self):
        under = _under(_tally(_ratcheted(_all_sites())), _allowlist())
        shown = [
            f"{' | '.join(key)}: {listed} listed, {live} in the tree" for key, (live, listed) in sorted(under.items())
        ]
        assert not under, (
            f"the list only shrinks: cut each entry below to its count in the tree (drop it at 0) in "
            f"tests/fixtures/{_ALLOWLIST.name}.\n  " + "\n  ".join(shown)
        )

    def test_every_permanent_site_still_names_a_site(self):
        """An exemption nothing matches is one a rewritten line walked out of without anyone deciding."""
        live = {site.permanent_key for site in _all_sites()}
        assert sorted(set(_PERMANENT) - live) == []

    def test_the_walk_reads_every_module_but_the_owners_and_the_kit(self):
        """Derived from the tree, so a module added anywhere in the package is walked the day it lands."""
        walked = {path.relative_to(_SOURCE_ROOT).as_posix() for path in _walked_modules()}
        every = {path.relative_to(_SOURCE_ROOT).as_posix() for path in (_SOURCE_ROOT / _PACKAGE).rglob("*.py")}
        assert walked == every - set(_OWNERS)
        assert _RULE in every
        assert {site.module for site in _all_sites()} >= {_KIT_JS, _CHART_SVG}

    def test_the_key_keeps_a_moved_line_and_counts_a_new_one(self):
        before = _tally(_python_sites("def f(v):\n    return round(v, 1)\n"))
        moved = _tally(_python_sites("\n\n# a comment\ndef f(v):\n\n    return round(v, 1)\n"))
        added = _tally(_python_sites("def f(v):\n    return round(v, 1), round(v, 2)\n"))
        assert before == moved == Counter({("sample.py", "f", "round()"): 1})
        assert added == Counter({("sample.py", "f", "round()"): 2})

    def test_the_ratchet_fails_both_ways(self):
        live = _tally(_python_sites("def f(v):\n    return round(v, 1)\n\ndef g(v):\n    return round(v)\n"))
        listed = {("sample.py", "f", "round()"): 2}
        assert _over(live, listed) == {("sample.py", "g", "round()"): (1, 0)}
        assert _under(live, listed) == {("sample.py", "f", "round()"): (1, 2)}

    @pytest.mark.parametrize(
        "text",
        [
            "a.py | f | round() | 0\n",
            "a.py | f | round()\n",
            "a.py | f | round() | 1\na.py | f | round() | 1\n",
            "b.py | f | round() | 1\na.py | f | round() | 1\n",
        ],
    )
    def test_a_malformed_allowlist_is_refused(self, text):
        with pytest.raises(AssertionError):
            _parse_allowlist(text)


class TestTheReportTreesKeepTheStricterRule:
    """Under ``analysis/`` and ``ops/`` no format spec chooses digits but a share's ``%`` and a permanent site.

    Derived from the source rather than from a function name: a second ``def format_number`` is one way to grow a
    second rule, and an inline ``f"{value:.4g}"`` is the commoner one, the shape that printed a latency ``1.235e+04``
    in a results pivot beside an interval spelling the same magnitude ``12346``. The engine's validation messages
    outside these trees are not read as reports, so the ratchet lists them instead.
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
            m for m in walked if m in ("threetears/evals/kernel/numbers.py", "threetears/evals/analysis/report/svg.py")
        }

    def test_every_spelling_goes_through_the_one_rule_or_is_named_as_permanent(self):
        report_modules = {path.relative_to(_SOURCE_ROOT).as_posix() for path in _modules_under(_SPELLING_TREES)}
        offenders = [
            site.shown()
            for site in _all_sites()
            if site.module in report_modules
            if site.spec is not None
            and not site.spec.endswith(_PERCENTAGE_TYPE)
            and site.permanent_key not in _PERMANENT
        ]
        assert not offenders, (
            "these spell a number with their own precision; route them through format_number / format_signed, "
            "or name a deliberate fixed spelling in _PERMANENT with its reason:\n  " + "\n  ".join(offenders)
        )


def _specs(source: str) -> list[tuple[str, str | None]]:
    return [(site.expression, site.spec) for site in _python_sites(source) if site.spec is not None]


def _forms(sites: Iterable[_Site]) -> list[tuple[str, str, str]]:
    return [(site.qualname, site.form, site.expression) for site in sites]


class TestTheWalkSeesEveryPythonForm:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ('f"{value:.4g}"', ("value", ".4g")),
            ('f"{delta:+.3g}"', ("delta", "+.3g")),
            ('f"{latency:.0f} ms"', ("latency", ".0f")),
            ('f"{bound:g}"', ("bound", "g")),
            ('f"{count:,}"', ("count", ",")),
            ('f"{share:.0%}"', ("share", ".0%")),
            ('f"{value:.{places}f}"', ("value", ".{}f")),
            ('format(value, ".3f")', ("value", ".3f")),
            ("format(value, spec)", ("value", "{}")),
            ('"{:.2e}".format(value)', ("'{:.2e}'.format(value)", ".2e")),
            ('"%.3f" % value', ("value", "%.3f")),
            ('t"{value:.2f}"', ("value", ".2f")),
        ],
    )
    def test_the_walk_sees_every_way_python_spells_a_number(self, source, expected):
        assert _specs(source) == [expected]

    def test_a_t_string_field_is_its_own_form(self):
        assert [site.form for site in _python_sites('t"{value:.2f}"')] == ["t-string spec"]

    @pytest.mark.parametrize(
        "source", ['f"{name}"', 'f"{name!r}"', 'f"{count:d}"', 'f"{label:>10}"', 'f"{format_number(v)}"']
    )
    def test_the_walk_leaves_what_chooses_no_digits(self, source):
        assert _specs(source) == []

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("round(v, 2)", ("<module>", "round()", "round(v, 2)")),
            ("format_number(v)", ("<module>", "format_number()", "format_number(v)")),
            ("numbers.format_signed(v)", ("<module>", "format_signed()", "numbers.format_signed(v)")),
            ("spell_count(n)", ("<module>", "spell_count()", "spell_count(n)")),
            ("MoneyForm(decimals=2)", ("<module>", "MoneyForm()", "MoneyForm(decimals=2)")),
            ("form.spell(v)", ("<module>", "spell()", "form.spell(v)")),
            ("set_places(values)", ("<module>", "set_places()", "set_places(values)")),
            ("figure_places(key)", ("<module>", "figure_places()", "figure_places(key)")),
            ("charts.respell_bare(t, 1)", ("<module>", "respell_bare()", "charts.respell_bare(t, 1)")),
            ("percent_columns(rows)", ("<module>", "percent_columns()", "percent_columns(rows)")),
            ("look.quantity_of(u, v)", ("<module>", "quantity_of()", "look.quantity_of(u, v)")),
            ("list(map(format_number, xs))", ("<module>", "format_number()", "format_number")),
            ("sorted(xs, key=numbers.spell_at)", ("<module>", "spell_at()", "numbers.spell_at")),
            ("Decimal(v).quantize(step)", ("<module>", "quantize()", "Decimal(v).quantize(step)")),
            ("2 - math.floor(math.log10(v))", ("<module>", "floor(log10)", "math.floor(math.log10(v))")),
            ("math.ceil(-math.log10(top / 5))", ("<module>", "ceil(log10)", "math.ceil(-math.log10(top / 5))")),
        ],
    )
    def test_the_walk_sees_rounding_and_every_primitive(self, source, expected):
        assert _forms(_python_sites(source)) == [expected]

    def test_a_site_is_keyed_by_its_enclosing_definition(self):
        source = "class Look:\n    def text(self, v):\n        def inner():\n            return round(v)\n        return inner\n"
        assert _forms(_python_sites(source)) == [("Look.text.inner", "round()", "round(v)")]

    def test_a_private_helper_is_a_site_only_in_the_module_that_decides_with_it(self):
        deciding = "threetears/evals/harness/writer_eval.py"
        assert _forms(_python_sites("_places(vs)", deciding)) == [("<module>", "_places()", "_places(vs)")]
        assert _python_sites("_places(vs)", "threetears/evals/analysis/report/build.py") == []

    @pytest.mark.parametrize(
        "source",
        [
            "np.round(v)",
            "len(xs)",
            "places(v)",
            "round_to(v, places)",
            "f(quantity, spell)",
            "f(self.respelled)",
            "math.log10(v)",
            "math.floor(v / step)",
        ],
    )
    def test_the_walk_leaves_what_decides_no_form(self, source):
        assert _python_sites(source) == []


def _count(sites: Iterable[_Site]) -> int:
    return sum(_tally(sites).values())


class TestAWrapperIsAPrimitive:
    """A function that returns what a site produced is a primitive itself, so a call rerouted through it is a site."""

    def test_a_helper_that_wraps_a_site_makes_its_callers_sites(self):
        source = (
            "def bare_at(v, p):\n    return spell_at(v, p)\n\n"
            "def label(v):\n    return f'{bare_at(v, 1)} pts'\n\n"
            "def row(v):\n    return label(v)\n"
        )
        assert _forms(_python_sites(source)) == [
            ("bare_at", "spell_at()", "spell_at(v, p)"),
            ("label", "bare_at()", "bare_at(v, 1)"),
            ("row", "label()", "label(v)"),
        ]

    def test_rerouting_a_call_through_a_wrapper_does_not_shrink_the_count(self):
        """``spell_at`` moved into ``bare_at``: every caller keeps its site, and the wrapper holds one more."""
        direct = "def a(v):\n    return spell_at(v, 1)\n\ndef b(v):\n    return spell_at(v, 2)\n"
        rerouted = (
            "def bare_at(v, p):\n    return spell_at(v, p)\n\n"
            "def a(v):\n    return bare_at(v, 1)\n\ndef b(v):\n    return bare_at(v, 2)\n"
        )
        assert _count(_python_sites(direct)) == 2
        assert _count(_python_sites(rerouted)) == 3
        assert _tally(_python_sites(rerouted))[("sample.py", "a", "bare_at()")] == 1

    def test_a_wrapper_in_another_module_is_reached_through_its_import(self):
        sources = {
            "threetears/evals/a.py": "def bare_at(v, p):\n    text = spell_at(v, p)\n    return text.strip()\n",
            "threetears/evals/b.py": "from threetears.evals.a import bare_at as at\n\ndef f(v):\n    return at(v, 1)\n",
            "threetears/evals/c.py": "from threetears.evals import a\n\ndef g(v):\n    return a.bare_at(v, 1)\n",
            "threetears/evals/d.py": "from .a import bare_at\n\ndef h(v):\n    return bare_at(v, 1)\n",
            "threetears/evals/e.py": "def bare_at(v, p):\n    return v\n\ndef k(v):\n    return bare_at(v, 1)\n",
        }
        called = {(site.module, site.qualname, site.form) for site in _walk_python(sources)}
        assert called == {
            ("threetears/evals/a.py", "bare_at", "spell_at()"),
            ("threetears/evals/b.py", "f", "at()"),
            ("threetears/evals/c.py", "g", "bare_at()"),
            ("threetears/evals/d.py", "h", "bare_at()"),
        }

    def test_a_wrapper_method_is_reached_through_self_and_by_a_name_no_other_method_holds(self):
        source = (
            "class Ruler:\n"
            "    def figure(self, v):\n        return self.form.spell(v)\n"
            "    def head(self, v):\n        return self.figure(v)\n"
            "    def render(self, v):\n        return self.figure(v) + ' ms'\n\n"
            "class Page:\n"
            "    def render(self):\n        return Page()\n\n"
            "def show(ruler, page, v):\n    return ruler.figure(v), ruler.render(v), page.render()\n"
        )
        assert _forms(_python_sites(source)) == [
            ("Ruler.figure", "spell()", "self.form.spell(v)"),
            ("Ruler.head", "figure()", "self.figure(v)"),
            ("Ruler.render", "figure()", "self.figure(v)"),
            # ``render`` is also Page's, which returns no form, so ``ruler.render(v)`` is not taken to reach Ruler's.
            ("show", "figure()", "ruler.figure(v)"),
        ]

    def test_a_builder_around_a_figure_or_a_document_of_lines_is_not_a_wrapper(self):
        """Were it one, so would every page builder and every caller of one be (most of this package)."""
        source = (
            "def cell(v):\n    return TableCell(text=spell_at(v, 1))\n\n"
            "def page(v):\n    return '\\n'.join([spell_at(v, 1), 'end'])\n\n"
            "def table(v):\n    return [cell(v), page(v)]\n"
        )
        assert [(site.qualname, site.form) for site in _python_sites(source)] == [
            ("cell", "spell_at()"),
            ("page", "spell_at()"),
        ]

    def test_a_permanent_site_inside_a_helper_leaves_its_callers_unlisted(self):
        """An SVG opacity is permanent, so the helper that writes it wraps nothing and drawing through it is no site."""
        source = (
            "def _faded(opacity):\n    return f'{opacity:g}'\n\n"
            "def _swatch(opacity):\n    return f'<rect opacity=\"{_faded(opacity)}\"/>'\n"
        )
        sites = _python_sites(source, _CHART_SVG)
        assert [(site.qualname, site.form) for site in sites] == [("_faded", "f-string spec")]
        assert _ratcheted(sites) == []

    def test_a_relocated_decision_called_as_a_free_function_is_still_a_site(self):
        """A decision moved into ``forms.py`` and called from a surface is decided where it is called, so a module
        that reroutes through one never reaches zero entries by the move alone."""
        assert {"at_decimals", "level_decimals", "change_decimals", "measured_decimals"} <= _FORMS_CALLABLES
        assert _forms(_python_sites("at_decimals(v, 2, '%')")) == [
            ("<module>", "at_decimals()", "at_decimals(v, 2, '%')")
        ]
        assert _forms(_python_sites("forms.level_decimals(r, k, d)")) == [
            ("<module>", "level_decimals()", "forms.level_decimals(r, k, d)")
        ]

    def test_a_page_s_forms_asked_by_measure_and_role_are_the_migrated_route(self):
        """``PageForms``, its constructors, its methods and the values it hands out decide nothing at the call."""
        assert not _FORMS_CALLABLES & {"PageForms", "MeasureForm", "Printed", "PrintedInterval", "Issued", "of"}
        source = (
            "page = PageForms.of(report_input)\n"
            "codes = PageForms.of_campaign(readings, reports)\n"
            "a = page.level(key, v).text\n"
            "b = page.change(key, v, kind=kind).digits\n"
            "c = page.interval(key, lo, hi, change=True).text\n"
            "d = page.measure(key).level_decimals\n"
        )
        assert _python_sites(source) == []

    def test_forms_primitives_are_none_before_forms_exists(self, tmp_path):
        assert _forms_primitives(tmp_path) == frozenset()

    def test_an_owner_is_not_walked_so_routing_through_it_stays_the_remedy(self, tmp_path):
        """``forms.py`` is an owner before it exists: the walk and the report trees both pass over it."""
        for module in (_FORMS, _RULE, "threetears/evals/analysis/report/figures.py"):
            (tmp_path / module).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / module).write_text("x = round(1.5)\n", encoding="utf-8")
        walked = [path.relative_to(tmp_path).as_posix() for path in _walked_modules(tmp_path)]
        under = [path.relative_to(tmp_path).as_posix() for path in _modules_under(_SPELLING_TREES[:1], tmp_path)]
        assert walked == under == ["threetears/evals/analysis/report/figures.py"]
        assert _FORMS in _OWNERS


class TestTheWalkSeesEveryKitForm:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("function a(v){return v.toFixed(2);}", ("a", "toFixed()", ".toFixed(2)")),
            ("function a(v){return v.toPrecision(12);}", ("a", "toPrecision()", ".toPrecision(12)")),
            ('function a(v){return v.toLocaleString("en-US");}', ("a", "toLocaleString()", '.toLocaleString("en-US")')),
            (
                'function a(v){return new Intl.NumberFormat("en-US").format(v);}',
                ("a", "Intl.", 'Intl.NumberFormat("en-US")'),
            ),
            ("function a(v){return Math.round(v);}", ("a", "Math.round()", "Math.round(v)")),
            ("function a(v){return fmt.usd(v);}", ("a", "fmt.", "fmt.usd(v)")),
            ("function a(v){return K_FMTS.pct(v,1);}", ("a", "K_FMTS", "K_FMTS.pct(v,1)")),
            ('function a(d){return namedFmt(d.fmt||"pct:1");}', ("a", "namedFmt", 'namedFmt(d.fmt||"pct:1")')),
            ("function a(d){return kTickFmt(d.fmt);}", ("a", "kTickFmt", "kTickFmt(d.fmt)")),
            ("function a(d,t){return kTickLabels(d.fmt,t);}", ("a", "kTickLabels", "kTickLabels(d.fmt,t)")),
            ("function a(v){return kShort(v,1);}", ("a", "kShort", "kShort(v,1)")),
        ],
    )
    def test_the_walk_sees_every_way_the_kit_spells_a_number(self, source, expected):
        assert _forms(_js_sites(source)) == [expected]

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("const b=v=>v.toFixed(1);", "b"),
            ("let c;const b=v=>v.toFixed(1);", "b"),
            ("async function b(v){return v.toFixed(1);}", "b"),
            ("function b(v){const f=x=>x.toFixed(1);return f(v);}", "b"),
            ("document.title=String(v.toFixed(1));", "<top>"),
            ("function a(){}\nif(on)a(v.toFixed(1));", "<top>"),
            ("function b(v){return `${v.toFixed(1)} ms`;}", "b"),
            ("function b(v){return `${`${v.toFixed(1)}`}` + `${{a:1}.a}`;}", "b"),
        ],
    )
    def test_a_kit_site_is_keyed_by_its_top_level_statement(self, source, expected):
        assert [site.qualname for site in _js_sites(source)] == [expected]

    @pytest.mark.parametrize(
        "source",
        [
            'const s="v.toFixed(2)";',
            "const s='Math.round(v)';",
            "const s=`v.toFixed(2) ${x}`;",
            "// v.toFixed(2)\nconst a=1;",
            "/* Intl.NumberFormat */ const a=1;",
            "const r=/toFixed\\(\\d\\)/g;",
            "const r=/[/]toFixed(/;",
            "function a(d,v){return opt.fmt(v)+d.fmt+f(v);}",
            "const fmt={pct:(v,d=1)=>v.toFixed(d)+'%'};",
            "const K_FMTS={int:v=>Math.round(v).toLocaleString('en-US'),pct:(v,d)=>kShort(v,d)};",
            "function namedFmt(spec){return spec;}",
            "const kShort=(v,d)=>v;",
            "const K_STEP_FMTS=new Set(['num']);",
        ],
    )
    def test_the_walk_leaves_what_the_kit_does_not_decide(self, source):
        assert _js_sites(source) == []

    def test_a_function_returning_a_kit_site_makes_its_callers_sites(self):
        source = (
            "const kAt=v=>v.toFixed(1);\n"
            "function label(v){return kAt(v)+' ms';}\n"
            "function draw(v){el.textContent=label(v);return 1;}\n"
            "function clean(v){const t=+v.toFixed(10);return t;}\n"
            "function axis(v){return clean(v);}\n"
        )
        assert _forms(_js_sites(source)) == [
            ("kAt", "toFixed()", ".toFixed(1)"),
            ("label", "kAt()", "kAt(v)"),
            ("draw", "label()", "label(v)"),
            # A site held in a body and not returned (a float cleaned before placing a tick) wraps nothing.
            ("clean", "toFixed()", ".toFixed(10)"),
        ]

    def test_a_division_is_not_read_as_a_regex(self):
        source = "function a(v,w){return v/2+w/3+v.toFixed(1);}"
        assert _forms(_js_sites(source)) == [("a", "toFixed()", ".toFixed(1)")]
