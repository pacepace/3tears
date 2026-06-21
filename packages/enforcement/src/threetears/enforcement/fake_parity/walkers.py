"""AST walkers that detect protocol-fake parity violations.

every test fake (class named ``Fake<Name>`` or ``_Fake<Name>``) must
declare what production protocol it stands in for, via one of:

1. **subclass declaration** -- ``class _FakeKvBucket(NatsKvBucket):``
   the walker reads the AST class bases and skips parity checks for
   any fake that has at least one base other than ``object``. the
   subclass declaration is the parity declaration; mypy's structural
   typing already enforces method-surface compatibility on imported
   bases. the walker accepts ``object`` literally (no parity declared)
   as a violation, every other base name as "parity declared via
   subclass".

2. **marker comment** -- ``# parity-with: <fully.qualified.name>``
   on the line immediately preceding the class definition. the
   walker reads the marker, imports the named production class,
   reflects its public method names + signatures, and asserts every
   production public method exists on the fake with a parameter set
   that is a *superset* of production's required parameters (extras
   on the fake are fine; missing or renamed params are not).

3. **exempt marker comment** -- ``# parity-exempt: <rationale>`` on the
   line immediately preceding the class definition. a fake that
   genuinely cannot declare one production protocol (a hand-rolled
   subset stub) carries the rationale ON THE CLASS instead of in a
   central exemptions file. the marker travels with the class, so a
   reformat that shifts line numbers can never orphan it (the failure
   mode line-keyed file exemptions have). the rationale must be
   non-blank; a bare ``# parity-exempt:`` does not exempt.

fakes that have NEITHER a subclass, a ``# parity-with:`` marker, nor a
``# parity-exempt:`` marker raise ``fake_parity.no_declaration`` so the
discovery convention itself is load-bearing -- nobody can declare a fake
by name and then forget to say what it claims to mock.

return-type and parameter-type annotations are NOT compared. the
walker enforces *callability* (the fake must accept every required
production parameter); type compatibility is mypy's job for
subclassed fakes and is best-effort for marker fakes.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path

from threetears.enforcement.common.violations import Violation

__all__ = [
    "fake_parity_violations",
    "find_fakes_in_tree",
]


_FAKE_NAME_PREFIXES: tuple[str, ...] = ("Fake", "_Fake")
_PARITY_MARKER_PREFIX: str = "# parity-with:"
_PARITY_EXEMPT_PREFIX: str = "# parity-exempt:"


@dataclass(frozen=True)
class _FakeDecl:
    """one fake class declaration discovered in a test tree.

    :ivar file: absolute path to the source file
    :ivar name: class name (``_FakeKvBucket`` etc.)
    :ivar line: 1-based line of the ``class`` statement
    :ivar bases: list of base class names as written in the source
        (e.g. ``["NatsKvBucket"]`` for
        ``class _FakeKvBucket(NatsKvBucket):``). empty when the only
        base would have been ``object``
    :ivar marker_target: fully-qualified class name from the
        ``# parity-with:`` marker comment, or ``None`` when no marker
        is present
    :ivar exempt_rationale: non-blank rationale from the
        ``# parity-exempt:`` marker comment, or ``None`` when absent /
        blank; a present rationale exempts the fake from the check
    :ivar methods: mapping of method name to its AST function node;
        only top-level methods are recorded (no nested classes)
    """

    file: Path
    name: str
    line: int
    bases: list[str]
    marker_target: str | None
    exempt_rationale: str | None
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]


def find_fakes_in_tree(scan_root: Path) -> list[_FakeDecl]:
    """walk ``scan_root`` and return every ``Fake*`` / ``_Fake*`` class decl.

    pure-AST traversal; no imports of test code. the walker recurses
    into every ``.py`` file under the root, skipping ``__pycache__``
    directories. classes nested inside other classes or functions
    are collected -- the parity convention applies regardless of
    nesting (a fake-as-inner-class still claims to mock something).

    :param scan_root: directory to recurse into
    :ptype scan_root: Path
    :return: every discovered fake declaration
    :rtype: list[_FakeDecl]
    """
    fakes: list[_FakeDecl] = []
    if not scan_root.is_dir():
        return fakes
    for py_file in sorted(scan_root.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except OSError, SyntaxError:
            continue
        source_lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(node.name.startswith(p) for p in _FAKE_NAME_PREFIXES):
                continue
            rendered_bases = [_format_base(b) for b in node.bases]
            bases = [b for b in rendered_bases if b is not None and b != "object"]
            marker_target, exempt_rationale = _read_markers(source_lines, node.lineno)
            methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[child.name] = child
            fakes.append(
                _FakeDecl(
                    file=py_file,
                    name=node.name,
                    line=node.lineno,
                    bases=bases,
                    marker_target=marker_target,
                    exempt_rationale=exempt_rationale,
                    methods=methods,
                ),
            )
    return fakes


def _format_base(node: ast.expr) -> str | None:
    """render an AST base-class expression as its dotted name.

    handles bare names (``Foo``), attribute access (``mod.Foo``,
    ``a.b.Foo``), and subscript expressions used for generics
    (``BaseCollection[FakeRefEntity]``). returns ``None`` for shapes
    we don't try to interpret (call expressions, complex generics).

    :param node: AST node from a class def's ``bases`` list
    :ptype node: ast.expr
    :return: dotted base name, or ``None``
    :rtype: str | None
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: list[str] = []
        cur: ast.expr = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return None
    if isinstance(node, ast.Subscript):
        return _format_base(node.value)
    return None


def _read_markers(source_lines: list[str], class_lineno: int) -> tuple[str | None, str | None]:
    """search the lines preceding ``class_lineno`` for a parity marker.

    two markers are recognised on a line by itself (leading whitespace
    allowed): ``# parity-with: <fqname>`` and ``# parity-exempt:
    <rationale>``. the walker scans upward from the line immediately
    above the class, skipping decorator lines (lines that start with
    ``@``) and blank lines. the FIRST non-blank, non-decorator line
    determines the result -- if it is one of the two markers, its
    payload is returned; otherwise both results are ``None``. the two
    markers are mutually exclusive (a fake declares parity OR is exempt,
    never both via the closest comment).

    :param source_lines: source split on newline
    :ptype source_lines: list[str]
    :param class_lineno: 1-based line number of the ``class`` statement
    :ptype class_lineno: int
    :return: ``(parity_with_target, exempt_rationale)`` -- at most one is
        non-``None``; a blank payload collapses to ``None``
    :rtype: tuple[str | None, str | None]
    """
    for line_index in range(class_lineno - 2, -1, -1):
        line = source_lines[line_index].strip()
        if not line:
            continue
        if line.startswith("@"):
            continue
        if line.startswith(_PARITY_MARKER_PREFIX):
            return (line[len(_PARITY_MARKER_PREFIX) :].strip() or None, None)
        if line.startswith(_PARITY_EXEMPT_PREFIX):
            return (None, line[len(_PARITY_EXEMPT_PREFIX) :].strip() or None)
        return (None, None)
    return (None, None)


def fake_parity_violations(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
) -> list[Violation]:
    """run the parity check across every scan root and return violations.

    two violation categories:

    - ``fake_parity.no_declaration`` -- the fake has no parity
      declaration (no non-object base, no ``# parity-with:`` marker).
      surfaced as ``add a subclass declaration or
      ``# parity-with:`` marker, or exempt with rationale``.

    - ``fake_parity.method_missing`` -- a marker fake is missing a
      public method declared on its production target.

    - ``fake_parity.method_required_arg_missing`` -- a marker fake
      has the production method but is missing a required parameter.

    - ``fake_parity.import_failed`` -- the marker target could not
      be imported. usually a typo in the fully-qualified name; the
      walker treats it as a violation rather than silently skipping.

    :param scan_roots: roots whose ``rglob("*.py")`` yields the
        candidate test modules
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root, used only for diagnostic relative
        paths
    :ptype repo_root: Path
    :return: ordered list of violations
    :rtype: list[Violation]
    """
    del repo_root  # report formatter applies repo_root downstream
    violations: list[Violation] = []
    for scan_root in scan_roots:
        for fake in find_fakes_in_tree(scan_root):
            violations.extend(_check_fake(fake))
    return violations


def _check_fake(fake: _FakeDecl) -> list[Violation]:
    """produce zero or more violations for one fake declaration.

    routing:

    1. fake has at least one non-object base -> SKIP. mypy enforces
       structural parity on imported bases.
    2. fake has a ``# parity-exempt:`` rationale -> SKIP. the rationale
       on the class is the exemption (it travels with the code).
    3. fake has a ``# parity-with:`` marker -> import the target, compare
       method surfaces. emit method-missing / param-missing /
       import-failed violations as appropriate.
    4. fake has none -> emit ``no_declaration``.

    :param fake: declaration to check
    :ptype fake: _FakeDecl
    :return: violations contributed by this fake
    :rtype: list[Violation]
    """
    if fake.bases:
        return []
    if fake.exempt_rationale is not None:
        return []
    if fake.marker_target is None:
        return [
            Violation(
                category="fake_parity.no_declaration",
                file=fake.file,
                line=fake.line,
                symbol=fake.name,
                reason=(
                    "fake has no parity declaration. add a subclass "
                    "(`class _Fake(Production):`) or a `# parity-with: "
                    "module.Production` marker on the line above the "
                    "class. as a last resort, exempt it in place with a "
                    "`# parity-exempt: <rationale>` marker on the line above"
                ),
            ),
        ]
    return _check_marker_parity(fake)


def _check_marker_parity(fake: _FakeDecl) -> list[Violation]:
    """import ``fake.marker_target`` and compare method surfaces.

    failure modes:

    - target import fails -> ``fake_parity.import_failed``
    - production method missing on fake -> ``fake_parity.method_missing``
    - production method present but missing a required param ->
      ``fake_parity.method_required_arg_missing``

    :param fake: declaration with a non-None marker_target
    :ptype fake: _FakeDecl
    :return: violations for this fake's marker comparison
    :rtype: list[Violation]
    """
    assert fake.marker_target is not None
    target = _import_marker_target(fake.marker_target)
    if target is None:
        return [
            Violation(
                category="fake_parity.import_failed",
                file=fake.file,
                line=fake.line,
                symbol=fake.name,
                reason=(
                    f"could not import `# parity-with: {fake.marker_target}`. "
                    "fix the marker to point at an importable class, or "
                    "remove it and use a subclass declaration"
                ),
            ),
        ]
    violations: list[Violation] = []
    production_methods = _public_methods(target)
    for method_name, prod_method in production_methods.items():
        if method_name not in fake.methods:
            violations.append(
                Violation(
                    category="fake_parity.method_missing",
                    file=fake.file,
                    line=fake.line,
                    symbol=fake.name,
                    reason=(
                        f"production class {fake.marker_target} declares "
                        f"public method `{method_name}` but the fake does "
                        "not. add the method or exempt with rationale"
                    ),
                ),
            )
            continue
        missing_params = _missing_required_params(prod_method, fake.methods[method_name])
        for param_name in missing_params:
            violations.append(
                Violation(
                    category="fake_parity.method_required_arg_missing",
                    file=fake.file,
                    line=fake.methods[method_name].lineno,
                    symbol=fake.name,
                    reason=(
                        f"production `{fake.marker_target}.{method_name}` "
                        f"requires parameter `{param_name}` but the fake "
                        "method does not accept it"
                    ),
                ),
            )
    return violations


def _import_marker_target(fqname: str) -> type | None:
    """resolve ``fqname`` to a class object, or ``None`` on failure.

    ``fqname`` is split on the last dot: everything before is the
    module path, the last segment is the attribute name. import
    failures of any kind (ModuleNotFoundError, AttributeError, any
    runtime error during the module's top-level code) collapse to
    ``None`` -- the caller emits ``import_failed`` for the fake.

    :param fqname: ``module.Class`` style fully-qualified name
    :ptype fqname: str
    :return: the class object, or ``None``
    :rtype: type | None
    """
    if "." not in fqname:
        return None
    module_path, _, attr = fqname.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except Exception:
        return None
    candidate = getattr(module, attr, None)
    if not isinstance(candidate, type):
        return None
    return candidate


def _public_methods(
    cls: type,
) -> dict[str, inspect.Signature]:
    """return ``{name: signature}`` for every public method of ``cls``.

    public = does not start with ``_``. inherited methods count;
    static methods + class methods + async methods are all included.
    properties and class-level data attributes are excluded (they
    are not ``callable`` or are descriptors with no bound signature).

    a few categories are filtered to keep the comparison practical:

    - methods inherited from ``object`` (``__init_subclass__``,
      ``__subclasshook__``, etc.) — these are never named in
      ``Fake*`` classes and emit pure noise
    - dunder methods generally — fakes that need ``__init__`` etc.
      define them anyway; enforcing their presence is the language's
      job, not this walker's

    :param cls: production class to introspect
    :ptype cls: type
    :return: mapping of public method name to its signature
    :rtype: dict[str, inspect.Signature]
    """
    methods: dict[str, inspect.Signature] = {}
    for name, member in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        if not callable(member):
            continue
        if isinstance(member, type):
            # nested class on the production type -- skip; not a method
            continue
        try:
            methods[name] = inspect.signature(member)
        except TypeError, ValueError:
            continue
    return methods


def _missing_required_params(
    prod_sig: inspect.Signature,
    fake_method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    """return production-required parameter names absent from the fake.

    a parameter is "required on production" when it is positional-or-
    keyword or keyword-only and has no default (variadic ``*args`` /
    ``**kwargs`` are skipped — they cannot be required by name and
    the fake's variadic catch-all would satisfy any caller anyway).

    the fake's parameter set is read from its AST signature: every
    arg name that appears in ``args.posonlyargs``, ``args.args``, or
    ``args.kwonlyargs``. if the fake has ``**kwargs``, every
    production-required parameter is considered satisfied (kwargs
    catches anything by name).

    :param prod_sig: production method signature
    :ptype prod_sig: inspect.Signature
    :param fake_method: fake's method AST node
    :ptype fake_method: ast.FunctionDef | ast.AsyncFunctionDef
    :return: names of production-required params not accepted by fake
    :rtype: list[str]
    """
    required: list[str] = []
    for name, param in prod_sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        required.append(name)

    fake_args = fake_method.args
    fake_param_names: set[str] = set()
    for arg in fake_args.posonlyargs + fake_args.args + fake_args.kwonlyargs:
        fake_param_names.add(arg.arg)
    has_var_keyword = fake_args.kwarg is not None
    has_var_positional = fake_args.vararg is not None

    if has_var_keyword:
        # **kwargs catches everything by name; nothing is missing
        return []

    missing: list[str] = []
    for name in required:
        if name in fake_param_names:
            continue
        if has_var_positional:
            # production param could land in *args; cannot know
            # without runtime calls. treat as satisfied — best-effort
            continue
        missing.append(name)
    return missing
