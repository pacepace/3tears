"""
enforcement: every audit ``details`` key a 3tears package publishes is classified.

Erasure never deletes an audit record. It anonymizes the record's ``details``
through :func:`threetears.agent.audit.anonymize_details`, which keeps the value
under a key on the explicit safe list and masks every other leaf. That rule fails
safe on its own -- an unclassified key is masked, never leaked -- so this gate is
not what keeps personal data out of an erased record.

What it prevents is the classification going stale without anyone noticing. A
producer that adds a structural key (a count, an id, a status) nobody classified
gets it masked on erasure, and the loss is invisible until someone reads an
erased row and finds the one field they needed gone. A producer that adds a
personal key gets no review of that fact at all. So every key must be named in
``SAFE_DETAIL_KEYS``, in ``PERSONAL_DETAIL_KEYS``, or in a safe-key declaration
for the event's own family (``declare_safe_detail_keys``) -- a decision somebody
made, not a default somebody inherited.

**What it can see.** Every ``AuditEvent(...)`` construction (plain or
attribute-qualified name) under a workspace package's ``src/``, and the
``details`` argument when it is:

- a dict literal, including dict literals nested inside it (their keys are
  judged too, because the rule judges nested keys);
- a ``dict(k=v)`` call with keyword arguments only;
- a local name that the enclosing function builds from those, plus
  ``name["k"] = ...``, ``name.update({...})`` / ``name.update(k=...)``,
  ``name.setdefault("k", ...)`` and ``name |= {...}``.

**What it cannot see, and how it answers.** A key computed at runtime
(``details[k] = ...``), a ``**spread``, a details dict returned by a call, one
passed in as a parameter (the wrapper-helper shape), or one handed to another
function that adds keys to it. The first four are REFUSED rather than skipped:
a details argument this gate cannot read is reported as unreadable, because a
gate that passes what it cannot see is the one that silently falls behind. The
last (``helper(details)`` mutating its argument) is a genuine blind spot -- a
name passed out of the function is not followed -- and so is an ``AuditEvent``
built by ``model_validate`` or ``model_copy`` rather than by its constructor.

Static AST parsing only, no import of the scanned modules, consistent with the
rest of ``tests/enforcement``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from threetears.agent.audit import (
    PERSONAL_DETAIL_KEYS,
    SAFE_DETAIL_KEYS,
    declare_safe_detail_keys,
    is_classified_detail_key,
)
from threetears.enforcement.common import find_local_src_roots, iter_python_files, parse_python_file

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: the constructor this gate follows.
_ENVELOPE = "AuditEvent"

#: dict methods that add a key when called with a literal.
_ADDING_METHODS = frozenset({"update", "setdefault"})


@dataclass
class _Site:
    """
    one ``AuditEvent`` construction and what the gate could read of its details.

    :param lineno: line of the construction
    :ptype lineno: int
    :param event_type: the literal ``event_type`` argument, or ``None`` when absent or computed
    :ptype event_type: str | None
    :param keys: every details key read, nested literal keys included
    :ptype keys: set[str]
    :param unreadable: why part of the details argument could not be read, one reason per shape
    :ptype unreadable: list[str]
    """

    lineno: int
    event_type: str | None
    keys: set[str] = field(default_factory=set)
    unreadable: list[str] = field(default_factory=list)


def _callee(call: ast.Call) -> str | None:
    """
    the bare name a call invokes, for a plain or attribute-qualified callee.

    :param call: a call node
    :ptype call: ast.Call
    :return: the name, or ``None`` for a callee that is neither form
    :rtype: str | None
    """
    func = call.func
    name: str | None = None
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    return name


def _read_literal(node: ast.expr, site: _Site) -> None:
    """
    add a literal's keys to ``site``, recording any part it cannot read.

    :param node: a dict literal, ``dict(...)`` call, or list/tuple of them
    :ptype node: ast.expr
    :param site: the construction being read
    :ptype site: _Site
    :return: nothing
    :rtype: None
    """
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                site.unreadable.append(f"line {value.lineno}: a **spread inside the details literal")
            elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                site.keys.add(key.value)
            else:
                site.unreadable.append(f"line {key.lineno}: a computed key inside the details literal")
            _read_nested(value, site)
    elif isinstance(node, ast.Call) and _callee(node) == "dict" and not node.args:
        for keyword in node.keywords:
            if keyword.arg is None:
                site.unreadable.append(f"line {node.lineno}: a **spread inside dict(...)")
            else:
                site.keys.add(keyword.arg)
                _read_nested(keyword.value, site)
    else:
        site.unreadable.append(f"line {node.lineno}: details built from {type(node).__name__}, not a literal")


def _read_nested(value: ast.expr, site: _Site) -> None:
    """
    follow a details value into nested literals, whose keys the rule also judges.

    :param value: the value expression under a details key
    :ptype value: ast.expr
    :param site: the construction being read
    :ptype site: _Site
    :return: nothing
    :rtype: None
    """
    if isinstance(value, ast.Dict) or (isinstance(value, ast.Call) and _callee(value) == "dict"):
        _read_literal(value, site)
    elif isinstance(value, ast.List | ast.Tuple | ast.Set):
        for element in value.elts:
            _read_nested(element, site)


def _is_name(node: ast.expr, name: str) -> bool:
    """
    whether ``node`` is a load or store of the local ``name``.

    :param node: any expression
    :ptype node: ast.expr
    :param name: the local being traced
    :ptype name: str
    :return: ``True`` for a bare reference to ``name``
    :rtype: bool
    """
    return isinstance(node, ast.Name) and node.id == name


def _read_local(name: str, scope: ast.AST, site: _Site) -> None:
    """
    read the keys a function gives a local details dict before handing it over.

    :param name: the local passed as ``details=``
    :ptype name: str
    :param scope: the enclosing function (or module) body
    :ptype scope: ast.AST
    :param site: the construction being read
    :ptype site: _Site
    :return: nothing
    :rtype: None
    """
    bound = False
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if _is_name(target, name) and node.value is not None:
                    bound = True
                    _read_literal(node.value, site)
                elif isinstance(target, ast.Subscript) and _is_name(target.value, name):
                    key = target.slice
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        site.keys.add(key.value)
                        if node.value is not None:
                            _read_nested(node.value, site)
                    else:
                        site.unreadable.append(f"line {node.lineno}: {name}[<computed>] = ...")
        elif isinstance(node, ast.AugAssign) and _is_name(node.target, name):
            _read_literal(node.value, site)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _is_name(node.func.value, name)
            and node.func.attr in _ADDING_METHODS
        ):
            _read_method_call(node, name, site)
    if not bound:
        site.unreadable.append(f"line {site.lineno}: details={name} is not assigned in this function")


def _read_method_call(call: ast.Call, name: str, site: _Site) -> None:
    """
    read the keys one ``name.update(...)`` / ``name.setdefault(...)`` adds.

    :param call: the method call
    :ptype call: ast.Call
    :param name: the local being traced
    :ptype name: str
    :param site: the construction being read
    :ptype site: _Site
    :return: nothing
    :rtype: None
    """
    assert isinstance(call.func, ast.Attribute)
    if call.func.attr == "setdefault":
        first = call.args[0] if call.args else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            site.keys.add(first.value)
        else:
            site.unreadable.append(f"line {call.lineno}: {name}.setdefault(<computed>, ...)")
        return
    for argument in call.args:
        _read_literal(argument, site)
    for keyword in call.keywords:
        if keyword.arg is None:
            site.unreadable.append(f"line {call.lineno}: {name}.update(**...)")
        else:
            site.keys.add(keyword.arg)


def _scopes(tree: ast.Module) -> dict[int, ast.AST]:
    """
    map every node id to its nearest enclosing function, or the module.

    :param tree: a parsed module
    :ptype tree: ast.Module
    :return: ``id(node)`` -> the scope a local in it would be bound in
    :rtype: dict[int, ast.AST]
    """
    owner: dict[int, ast.AST] = {}

    def visit(node: ast.AST, scope: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            inner = child if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) else scope
            owner[id(child)] = inner
            visit(child, inner)

    visit(tree, tree)
    return owner


def _sites(tree: ast.Module) -> list[_Site]:
    """
    every ``AuditEvent`` construction in a module, with its details read.

    :param tree: a parsed module
    :ptype tree: ast.Module
    :return: one entry per construction, in source order
    :rtype: list[_Site]
    """
    scopes = _scopes(tree)
    found: list[_Site] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _callee(node) == _ENVELOPE):
            continue
        arguments = {keyword.arg: keyword.value for keyword in node.keywords}
        event_type = arguments.get("event_type")
        site = _Site(
            lineno=node.lineno,
            event_type=(
                event_type.value if isinstance(event_type, ast.Constant) and isinstance(event_type.value, str) else None
            ),
        )
        if None in arguments:
            site.unreadable.append(f"line {node.lineno}: AuditEvent(**...) may carry details")
        details = arguments.get("details")
        if isinstance(details, ast.Name):
            _read_local(details.id, scopes.get(id(node), tree), site)
        elif details is not None:
            _read_literal(details, site)
        found.append(site)
    return sorted(found, key=lambda site: site.lineno)


def _unclassified(site: _Site) -> list[str]:
    """
    the keys at one site that no classification names, for its event type.

    :param site: a construction whose details were read
    :ptype site: _Site
    :return: the unclassified keys, sorted
    :rtype: list[str]
    """
    event_type = site.event_type or ""
    return sorted(key for key in site.keys if not is_classified_detail_key(key, event_type=event_type))


def _workspace_sites() -> dict[str, list[_Site]]:
    """
    every construction under every workspace package's ``src/``.

    :return: repo-relative path -> the sites in that file, files with none omitted
    :rtype: dict[str, list[_Site]]
    """
    by_file: dict[str, list[_Site]] = {}
    for root in find_local_src_roots(_REPO_ROOT):
        for path in iter_python_files(root):
            tree = parse_python_file(path)
            if tree is None:
                continue
            sites = _sites(tree)
            if sites:
                by_file[path.relative_to(_REPO_ROOT).as_posix()] = sites
    return by_file


class TestAuditDetailsKeysAreClassified:
    """no 3tears package publishes a details key nobody decided about."""

    def test_every_published_details_key_is_classified(self) -> None:
        """an unclassified key fails here, naming the file, line, event type and key."""
        unclassified = [
            f"{path}:{site.lineno} [{site.event_type or '<computed event_type>'}] {key}"
            for path, sites in _workspace_sites().items()
            for site in sites
            for key in _unclassified(site)
        ]

        assert not unclassified, (
            "these audit details keys are neither safe nor personal:\n  "
            + "\n  ".join(unclassified)
            + "\n\nclassify each in packages/agent/audit/src/threetears/agent/audit/anonymize.py: "
            "SAFE_DETAIL_KEYS if its value is structural for every producer (an entity id, a count, "
            "a status, a duration, a version), PERSONAL_DETAIL_KEYS if it can carry a name, email, "
            "address, IP, free or user-typed text, a path, or a credential. when unsure it is personal. "
            "a key structural only within one event family belongs in that family's "
            "declare_safe_detail_keys(...) instead."
        )

    def test_every_details_argument_is_readable(self) -> None:
        """a details argument this gate cannot read is refused, not passed."""
        unreadable = [
            f"{path}:{site.lineno} {reason}"
            for path, sites in _workspace_sites().items()
            for site in sites
            for reason in site.unreadable
        ]

        assert not unreadable, (
            "these AuditEvent constructions pass details this gate cannot read statically:\n  "
            + "\n  ".join(unreadable)
            + "\n\nbuild details as a dict literal in the constructing function (keys as string "
            'literals; name["k"] = ... and name.update({...}) are followed). a key computed at '
            "runtime cannot be classified, so it would be masked on erasure without anyone having "
            "decided that."
        )

    def test_the_scan_reaches_every_known_producer(self) -> None:
        """a walker over no sites passes vacuously; pin that it found the producers it must.

        both inputs of the comparison are guarded: the sites (the tool server's baseline
        event and the workspace tools') and the classification it compares them against.
        """
        by_file = _workspace_sites()
        keys = {key for sites in by_file.values() for site in sites for key in site.keys}

        assert any(path.startswith("packages/agent/tools/") for path in by_file), sorted(by_file)
        assert any(path.startswith("packages/agent/workspace/") for path in by_file), sorted(by_file)
        assert {"tool_name", "failure_reason", "workspace_resource_id"} <= keys, sorted(keys)
        assert SAFE_DETAIL_KEYS
        assert PERSONAL_DETAIL_KEYS


_SYNTHETIC_PREFIX = "gatesynthetic"
declare_safe_detail_keys(_SYNTHETIC_PREFIX, ["gate_declared_count"])


def _read(source: str) -> _Site:
    """
    read the single construction in a synthetic module.

    :param source: module source holding exactly one ``AuditEvent(...)``
    :ptype source: str
    :return: its site
    :rtype: _Site
    """
    (site,) = _sites(ast.parse(source))
    return site


class TestTheGateItself:
    """synthetic modules, so a narrowed reader fails here instead of passing the tree."""

    def test_an_unclassified_literal_key_is_reported(self) -> None:
        """the base case: a new key in a literal."""
        site = _read("AuditEvent(event_type='x.y', details={'tool_name': t, 'mystery_key': m})")

        assert _unclassified(site) == ["mystery_key"]
        assert not site.unreadable

    def test_a_key_added_by_subscript_is_reported(self) -> None:
        """the tool server's shape: a literal, then one key added conditionally."""
        site = _read(
            "def f():\n"
            "    details = {'tool_name': t}\n"
            "    if r:\n"
            "        details['mystery_key'] = r\n"
            "    AuditEvent(event_type='x.y', details=details)\n"
        )

        assert _unclassified(site) == ["mystery_key"]

    def test_keys_added_by_update_and_setdefault_are_reported(self) -> None:
        """every literal-keyed way of growing the dict is followed."""
        site = _read(
            "def f():\n"
            "    details: dict[str, object] = {}\n"
            "    details.update({'mystery_a': 1}, mystery_b=2)\n"
            "    details.setdefault('mystery_c', 3)\n"
            "    details |= {'mystery_d': 4}\n"
            "    AuditEvent(event_type='x.y', details=details)\n"
        )

        assert _unclassified(site) == ["mystery_a", "mystery_b", "mystery_c", "mystery_d"]

    def test_a_nested_literal_key_is_reported(self) -> None:
        """the rule judges nested keys, so the gate must too."""
        site = _read("AuditEvent(event_type='x.y', details={'files_changed': [{'mystery_key': 1}]})")

        assert _unclassified(site) == ["mystery_key"]

    def test_an_attribute_qualified_constructor_is_found(self) -> None:
        """``audit.AuditEvent(...)`` is the same construction."""
        site = _read("audit.AuditEvent(event_type='x.y', details=dict(mystery_key=1))")

        assert _unclassified(site) == ["mystery_key"]

    def test_a_family_declaration_is_credited_to_its_family_only(self) -> None:
        """a declared key passes for its own event family and fails for any other."""
        own = _read(f"AuditEvent(event_type='{_SYNTHETIC_PREFIX}.x', details={{'gate_declared_count': 1}})")
        other = _read("AuditEvent(event_type='other.x', details={'gate_declared_count': 1})")

        assert _unclassified(own) == []
        assert _unclassified(other) == ["gate_declared_count"]

    def test_a_computed_event_type_gets_no_family_credit(self) -> None:
        """when the gate cannot read the event type it cannot credit a declaration."""
        site = _read("AuditEvent(event_type=kind, details={'gate_declared_count': 1})")

        assert _unclassified(site) == ["gate_declared_count"]

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param("AuditEvent(event_type='x.y', details=build())", id="details-from-a-call"),
            pytest.param("AuditEvent(event_type='x.y', details={**extra})", id="spread-in-literal"),
            pytest.param("AuditEvent(event_type='x.y', details={key: 1})", id="computed-literal-key"),
            pytest.param("AuditEvent(event_type='x.y', **fields)", id="spread-into-constructor"),
            pytest.param(
                "def f(details):\n    AuditEvent(event_type='x.y', details=details)\n",
                id="details-passed-in-as-a-parameter",
            ),
            pytest.param(
                "def f():\n    details = {}\n    details[k] = 1\n    AuditEvent(event_type='x.y', details=details)\n",
                id="computed-subscript-key",
            ),
        ],
    )
    def test_a_shape_the_gate_cannot_read_is_refused(self, source: str) -> None:
        """what the gate cannot see is reported as unreadable, never passed as clean.

        :param source: a synthetic module with one unreadable details argument
        :ptype source: str
        """
        assert _read(source).unreadable

    def test_a_construction_without_details_carries_no_keys(self) -> None:
        """an event with no details is readable and empty."""
        site = _read("AuditEvent(event_type='x.y', action='a')")

        assert site.keys == set()
        assert not site.unreadable
