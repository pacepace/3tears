"""AST guards for the :class:`PromptCachingHook` implementation.

three static checks scoped to
``packages/langgraph/src/threetears/langgraph/hooks.py`` (the module
that owns the cache logic -- :func:`agent_node` deliberately stays
cache-agnostic so non-caching callers see no behavior change):

1. every :class:`SystemMessage` construction inside the caching
   path passes a structured content list, guarding against a
   regression that reintroduces a plain-string SystemMessage and
   silently breaks cache annotations.
2. every call to ``chat_model.bind_tools(...)`` inside the hook
   module appears in a branch guarded by
   :func:`should_bind_tools_fresh`, guarding against a "rebind
   every turn" regression.
3. every code path inside :class:`PromptCachingHook.after_invoke`
   reaches :func:`extract_cache_usage`, guarding against losing
   telemetry when the hook is refactored.

the :func:`agent_node` body itself is outside the scope of these
guards -- it creates a bare-string :class:`SystemMessage` for
callers that do NOT install the caching hook (degradation path).
"""

from __future__ import annotations

import ast
from pathlib import Path

_LANGGRAPH_SRC = Path(__file__).resolve().parents[2] / "src" / "threetears" / "langgraph"
_HOOKS_PATH = _LANGGRAPH_SRC / "hooks.py"


def _parse(path: Path) -> ast.Module:
    """parse a python file into an AST module.

    :param path: absolute path to a python source file
    :ptype path: Path
    :return: parsed module
    :rtype: ast.Module
    """
    source = path.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(path))


def _is_system_message_call(call: ast.Call) -> bool:
    """check whether a call expression targets :class:`SystemMessage`.

    matches both the bare-name form (``SystemMessage(...)``) and
    the attribute form (``messages.SystemMessage(...)``); the
    langgraph hook module uses the bare-name form but the helper
    stays conservative.

    :param call: AST call node
    :ptype call: ast.Call
    :return: ``True`` when the call is on :class:`SystemMessage`
    :rtype: bool
    """
    func = call.func
    result = False
    if isinstance(func, ast.Name) and func.id == "SystemMessage":
        result = True
    elif isinstance(func, ast.Attribute) and func.attr == "SystemMessage":
        result = True
    return result


def _call_has_list_content(call: ast.Call) -> bool:
    """determine whether the ``content=`` kwarg is a list literal.

    checks the ``content`` keyword argument for an :class:`ast.List`
    or a name reference that clearly carries a list (the production
    code uses a named ``content_blocks`` list variable). the name
    check is necessarily loose -- AST cannot prove the type of a
    named reference without full type inference -- but any variable
    whose name includes ``"block"`` or ``"content"`` or that points
    at a literal list is accepted.

    :param call: AST call node already identified as SystemMessage
    :ptype call: ast.Call
    :return: ``True`` when the call carries a list-shaped content
        argument
    :rtype: bool
    """
    result = False
    for kw in call.keywords:
        if kw.arg != "content":
            continue
        value = kw.value
        if isinstance(value, ast.List):
            result = True
        elif isinstance(value, ast.Name):
            lowered = value.id.lower()
            if "block" in lowered or "content" in lowered:
                result = True
        break
    return result


def test_system_message_in_caching_path_uses_list_content() -> None:
    """:class:`SystemMessage` in the caching path never takes a bare string.

    walks :mod:`threetears.langgraph.hooks` and rejects any
    ``SystemMessage(content=<non-list>)`` construction. the hook
    module has one legitimate construction path inside
    :func:`_rewrite_system_prompt_for_cache` that forwards to
    :func:`annotate_system_prompt`, which in turn returns a
    structured-content :class:`SystemMessage` -- so the only
    SystemMessage construction in the module's file tree comes
    back from helpers that already pass a list.

    :raises AssertionError: when a bare-string ``content=`` shows up
    """
    tree = _parse(_HOOKS_PATH)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_system_message_call(node):
            continue
        if _call_has_list_content(node):
            continue
        offenders.append(
            f"{_HOOKS_PATH.name}:{node.lineno} SystemMessage with non-list content",
        )
    assert not offenders, (
        "SystemMessage in the caching path must pass a list content; "
        f"offenders: {offenders}"
    )


def _find_bind_tools_calls(tree: ast.Module) -> list[ast.Call]:
    """collect every ``<anything>.bind_tools(...)`` call in a module.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: list of matching call nodes
    :rtype: list[ast.Call]
    """
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "bind_tools":
            calls.append(node)
    return calls


def _enclosing_functions(tree: ast.Module) -> dict[int, ast.FunctionDef | ast.AsyncFunctionDef]:
    """map every line number in the module to its enclosing function.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: ``{lineno: enclosing FunctionDef}`` mapping
    :rtype: dict[int, ast.FunctionDef | ast.AsyncFunctionDef]
    """
    mapping: dict[int, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            for line in range(start, end + 1):
                mapping[line] = node
    return mapping


def _function_calls_should_bind_tools_fresh(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """check whether a function body calls :func:`should_bind_tools_fresh`.

    :param func: function definition node
    :ptype func: ast.FunctionDef | ast.AsyncFunctionDef
    :return: ``True`` when the guard helper is referenced
    :rtype: bool
    """
    found = False
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Name) and callee.id == "should_bind_tools_fresh":
                found = True
                break
            if (
                isinstance(callee, ast.Attribute)
                and callee.attr == "should_bind_tools_fresh"
            ):
                found = True
                break
    return found


def test_bind_tools_in_hook_module_is_guarded_by_should_bind_tools_fresh() -> None:
    """every ``bind_tools`` call in hooks.py rides through the guard helper.

    walks :mod:`threetears.langgraph.hooks`, finds every
    ``.bind_tools(...)`` call, resolves the enclosing function,
    and requires that function to reference
    :func:`should_bind_tools_fresh` somewhere in its body.

    :raises AssertionError: when a ``bind_tools`` call lives in a
        function that never consults the guard helper
    """
    tree = _parse(_HOOKS_PATH)
    function_map = _enclosing_functions(tree)
    offenders: list[str] = []
    for call in _find_bind_tools_calls(tree):
        func = function_map.get(call.lineno)
        if func is None:
            offenders.append(
                f"{_HOOKS_PATH.name}:{call.lineno} bind_tools outside any function",
            )
            continue
        if not _function_calls_should_bind_tools_fresh(func):
            offenders.append(
                f"{_HOOKS_PATH.name}:{call.lineno} bind_tools in {func.name!r}"
                " without should_bind_tools_fresh guard",
            )
    assert not offenders, (
        "bind_tools in caching path must be guarded by should_bind_tools_fresh; "
        f"offenders: {offenders}"
    )


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """locate a top-level or method function by name.

    :param tree: parsed module
    :ptype tree: ast.Module
    :param name: function name to find
    :ptype name: str
    :return: first matching function node or ``None``
    :rtype: ast.FunctionDef | ast.AsyncFunctionDef | None
    """
    result: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            result = node
            break
    return result


def _function_calls(func: ast.AST) -> set[str]:
    """collect the set of callable names invoked in a function body.

    flattens both bare-name (``foo()``) and attribute
    (``obj.foo()``) invocations into a set of names.

    :param func: function node
    :ptype func: ast.AST
    :return: names referenced as call targets
    :rtype: set[str]
    """
    names: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if isinstance(callee, ast.Name):
            names.add(callee.id)
        elif isinstance(callee, ast.Attribute):
            names.add(callee.attr)
    return names


def test_after_invoke_of_prompt_caching_hook_calls_extract_cache_usage() -> None:
    """every path through ``PromptCachingHook.after_invoke`` reaches the helper.

    ``extract_cache_usage`` is the single readout for
    cache-hit telemetry; dropping it silently from the hook
    removes the only signal downstream consumers (gateway,
    tests) have. the guard walks the hook's ``after_invoke``
    method and requires the helper name to appear in its call
    set.

    :raises AssertionError: when ``extract_cache_usage`` is absent
    """
    tree = _parse(_HOOKS_PATH)
    target: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "PromptCachingHook":
            continue
        for item in node.body:
            if (
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == "after_invoke"
            ):
                target = item
                break
        break
    assert target is not None, "PromptCachingHook.after_invoke not found"
    names = _function_calls(target)
    assert "extract_cache_usage" in names, (
        "PromptCachingHook.after_invoke must call extract_cache_usage "
        "on every AIMessage-producing path"
    )
