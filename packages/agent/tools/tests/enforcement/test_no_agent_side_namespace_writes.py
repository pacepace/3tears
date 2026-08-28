"""enforcement: the pod does not write ``namespaces`` rows for its own tools.

the agent-side emitter is GONE and must not come back. it was two methods on
:class:`~threetears.agent.tools.server.ToolServer` --
``_emit_tool_namespace`` and ``_delete_tool_namespace`` -- fed by a
``namespace_collection`` constructor parameter, and the pair failed in two
different ways at once:

1. **the delete raised on every call.** it passed a BARE ``UUID`` to
   ``NamespaceCollection.delete``, whose ``primary_key_column`` is the
   composite ``("row_scope", "namespace_id")``. ``normalize_pk`` refuses that
   arity, so a pod dropping a tool -- stale-tool pruning reaches
   ``deregister_tool`` from three live call sites -- raised
   ``ValueError: primary key arity mismatch``. its only test replaced the
   collection with an ``AsyncMock``, which accepts any arity and hid it.
2. **the write had already moved.** the hub reconciles tool namespaces off the
   registration manifest, under a verified signature, in
   ``aibots.hub.tools.namespace_emitter``. a pod that also wrote its own rows
   would write them from the one side that cannot prove who it is.

so the pod publishes a manifest and nothing else. anything that reintroduces a
pod-side write to ``namespaces`` -- a parameter to thread a collection in, an
attribute to hold one, or a method to emit or delete a row -- is the same
defect returning, and fails here.

pure AST over the package source, no import of the code under test.

FALLIBILITY PROOF: every check below is exercised against a synthetic violating
source in ``TestTheseChecksCanFail``. a guard nobody has ever seen fail is a
guard nobody knows is wired up.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "agent" / "tools"

#: constructor parameter that threaded a ``NamespaceCollection`` into the pod.
_BANNED_PARAMETER = "namespace_collection"

#: instance attribute that held it.
_BANNED_ATTRIBUTE = "_namespace_collection"

#: the two methods that wrote and deleted ``namespaces`` rows pod-side.
_BANNED_METHODS = frozenset({"_emit_tool_namespace", "_delete_tool_namespace"})

#: classes whose constructors must not accept the parameter.
_POD_CLASSES = frozenset({"ToolServer", "DynamicToolPod"})


def _collect_src_files() -> list[Path]:
    """collect production python source files to scan.

    :return: sorted list of source file paths under agent/tools src
    :rtype: list[Path]
    """
    return sorted(_SRC_ROOT.rglob("*.py"))


def _parse(path: Path) -> ast.Module:
    """parse one source file into a module tree.

    :param path: file to parse
    :ptype path: Path
    :return: parsed module
    :rtype: ast.Module
    """
    return ast.parse(path.read_text(encoding="utf-8"))


def _parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """collect every declared parameter name of a function.

    covers positional, positional-only, keyword-only, ``*args`` and
    ``**kwargs``, so a rename of the calling convention cannot smuggle the
    parameter back.

    :param node: function definition node
    :ptype node: ast.FunctionDef | ast.AsyncFunctionDef
    :return: set of parameter names
    :rtype: set[str]
    """
    args = node.args
    names = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _constructor_parameter_violations(tree: ast.Module) -> list[str]:
    """find pod constructors declaring the banned parameter.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: list of ``Class.__init__`` names that declare it
    :rtype: list[str]
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in _POD_CLASSES:
            continue
        for member in node.body:
            if not isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if member.name != "__init__":
                continue
            if _BANNED_PARAMETER in _parameter_names(member):
                found.append(f"{node.name}.__init__")
    return found


def _method_violations(tree: ast.Module) -> list[str]:
    """find definitions of the banned emitter methods.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: list of method names defined anywhere in the module
    :rtype: list[str]
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in _BANNED_METHODS:
            found.append(node.name)
    return found


def _attribute_violations(tree: ast.Module) -> list[int]:
    """find reads or writes of the banned instance attribute.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: line numbers carrying the attribute
    :rtype: list[int]
    """
    return [
        node.lineno for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == _BANNED_ATTRIBUTE
    ]


def _called_name(node: ast.Call) -> str | None:
    """the simple name of a call's callee, for ``Name`` and ``a.b`` forms.

    :param node: call node
    :ptype node: ast.Call
    :return: callee name, or ``None`` for a shape with no simple name
    :rtype: str | None
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _keyword_violations(tree: ast.Module) -> list[int]:
    """find pod constructions passing the banned parameter as a keyword.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: line numbers of offending calls
    :rtype: list[int]
    """
    found: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _called_name(node) not in _POD_CLASSES:
            continue
        for keyword in node.keywords:
            if keyword.arg == _BANNED_PARAMETER:
                found.append(node.lineno)
    return found


class TestPodWritesNoNamespaceRows:
    """the pod-side ``namespaces`` emitter stays deleted."""

    def test_no_pod_constructor_accepts_a_namespace_collection(self) -> None:
        """``ToolServer`` / ``DynamicToolPod`` declare no such parameter.

        :return: nothing
        :rtype: None
        """
        offenders = {
            str(path.relative_to(_SRC_ROOT)): _constructor_parameter_violations(_parse(path))
            for path in _collect_src_files()
        }
        offenders = {path: names for path, names in offenders.items() if names}
        assert not offenders, (
            f"pod constructors must not accept {_BANNED_PARAMETER!r}: {offenders}. "
            "the hub reconciles tool namespaces off the registration manifest; "
            "a pod-side write cannot prove who it is and its delete raised on every call."
        )

    def test_no_module_defines_the_emitter_methods(self) -> None:
        """``_emit_tool_namespace`` / ``_delete_tool_namespace`` are gone.

        :return: nothing
        :rtype: None
        """
        offenders = {
            str(path.relative_to(_SRC_ROOT)): _method_violations(_parse(path)) for path in _collect_src_files()
        }
        offenders = {path: names for path, names in offenders.items() if names}
        assert not offenders, f"pod-side namespace emitter methods must stay deleted: {offenders}"

    def test_no_module_holds_a_namespace_collection_attribute(self) -> None:
        """nothing stores a ``NamespaceCollection`` on the pod.

        :return: nothing
        :rtype: None
        """
        offenders = {
            str(path.relative_to(_SRC_ROOT)): _attribute_violations(_parse(path)) for path in _collect_src_files()
        }
        offenders = {path: lines for path, lines in offenders.items() if lines}
        assert not offenders, f"the pod holds no namespace collection: {_BANNED_ATTRIBUTE} found at {offenders}"

    def test_no_call_threads_a_namespace_collection_through(self) -> None:
        """no call site inside the package passes the keyword on.

        :return: nothing
        :rtype: None
        """
        offenders = {
            str(path.relative_to(_SRC_ROOT)): _keyword_violations(_parse(path)) for path in _collect_src_files()
        }
        offenders = {path: lines for path, lines in offenders.items() if lines}
        assert not offenders, f"no call may pass {_BANNED_PARAMETER!r}: {offenders}"


class TestTheseChecksCanFail:
    """each check above is proved fallible against a synthetic violation."""

    def test_constructor_check_catches_a_declared_parameter(self) -> None:
        """a pod constructor declaring the parameter is reported.

        :return: nothing
        :rtype: None
        """
        source = "class ToolServer:\n    def __init__(self, *, namespace_collection):\n        pass\n"
        assert _constructor_parameter_violations(ast.parse(source)) == ["ToolServer.__init__"]

    def test_constructor_check_ignores_other_classes(self) -> None:
        """the same parameter on an unrelated class is not a violation.

        :return: nothing
        :rtype: None
        """
        source = "class SomethingElse:\n    def __init__(self, *, namespace_collection):\n        pass\n"
        assert _constructor_parameter_violations(ast.parse(source)) == []

    def test_method_check_catches_a_reintroduced_emitter(self) -> None:
        """a redefined ``_emit_tool_namespace`` is reported.

        :return: nothing
        :rtype: None
        """
        source = "class ToolServer:\n    async def _emit_tool_namespace(self, tool):\n        pass\n"
        assert _method_violations(ast.parse(source)) == ["_emit_tool_namespace"]

    def test_attribute_check_catches_a_stored_collection(self) -> None:
        """an assignment to the banned attribute is reported.

        :return: nothing
        :rtype: None
        """
        source = "class ToolServer:\n    def __init__(self, coll):\n        self._namespace_collection = coll\n"
        assert _attribute_violations(ast.parse(source)) == [3]

    def test_keyword_check_catches_a_threaded_keyword(self) -> None:
        """a pod construction passing the keyword on is reported.

        :return: nothing
        :rtype: None
        """
        source = "server = ToolServer(nats_url='n', namespace_collection=None)\n"
        assert _keyword_violations(ast.parse(source)) == [1]

    def test_keyword_check_ignores_non_pod_calls(self) -> None:
        """the same keyword on an unrelated call is not a violation.

        the workspace tool set still needs a ``NamespaceCollection`` --
        :class:`WorkspaceCreateTool` writes the paired workspace row --
        so this check must not sweep that plumbing away with the pod's.

        :return: nothing
        :rtype: None
        """
        source = "register_workspace_tools(namespace_collection=coll)\n"
        assert _keyword_violations(ast.parse(source)) == []
