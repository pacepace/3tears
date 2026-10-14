"""Generate ``packages/evals/docs/reference.md`` from the package itself.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed.

    # Rewrite the page:
    uv run python packages/evals/scripts/generate_reference.py

    # Exit 1 when the committed page is stale, without writing it:
    uv run python packages/evals/scripts/generate_reference.py --check

    # List the public items that carry no description:
    uv run python packages/evals/scripts/generate_reference.py --gaps

Everything on the page is read from code: the package's own ``__all__`` and each public root's (the roots are
``threetears.evals.PUBLIC_ROOTS``), first docstring paragraphs and ``#:`` comments, signatures, Pydantic field
descriptions, the goal-check language's module docstring and builtins, the action catalogue, the command line's
argparse definition and ``METRIC_DESCRIPTORS``. Nothing on it is written here except headings and the intro, so a
change to the package moves the page and ``tests/test_reference_doc.py`` fails until it is regenerated. The one
thing not read from the package is each name's stability tier: a name ``HOST_CONTRACT`` in ``public_surface.py``
lists is marked *host contract*, so a reader sees what is promised without opening ``api/surface.json``.

The walk over the public roots is ``public_surface.py``'s, shared with the API snapshot
(``generate_api_snapshot.py``), so the page and the snapshot never disagree about a name's kind or signature. The
output is deterministic: stable ordering, no timestamps, no object addresses, sets sorted.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import inspect
import io
import os
import re
import sys
import types
import typing
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from public_surface import (
    PACKAGE_ROOT,
    Item,
    assigned_names,
    describe,
    factory_name,
    field_rows,
    first_sentence,
    one_line,
    own_doc,
    paragraphs,
    public_names,
    stable_repr,
    tier_of,
    walked_modules,
)
from pydantic import BaseModel

#: The page this script writes.
REFERENCE_PATH = PACKAGE_ROOT / "docs" / "reference.md"

#: The command that rewrites the page, as the page and the staleness test tell a reader.
REGENERATE = "uv run python packages/evals/scripts/generate_reference.py"

#: What a public item with no description reads as, so a gap is visible rather than papered over.
NO_DESCRIPTION = "*(no description)*"

#: The width ``--help`` output is wrapped at on the page.
HELP_WIDTH = 100

#: Gaps found while rendering: (where, name). Filled by :func:`render`.
GAPS: list[tuple[str, str]] = []


# --- text --------------------------------------------------------------------------------------------------------

_ROLE = re.compile(r":[a-z]+(?::[a-z]+)?:`(?:[^`<]*<)?~?([^`>]+)>?`")
_DOUBLE_TICK = re.compile(r"``(.+?)``")


def _role_target(match: re.Match[str]) -> str:
    """A Sphinx cross-reference as a code span: the last dotted part when it was ``~``-shortened."""
    whole = match.group(0)
    target = match.group(1)
    if "`~" in whole:
        target = target.rsplit(".", 1)[-1]
    return f"`{target}`"


def md_text(text: str) -> str:
    """Docstring reStructuredText as Markdown: roles and double backticks become code spans, ``<`` is escaped."""
    text = _ROLE.sub(_role_target, text)
    text = _DOUBLE_TICK.sub(lambda m: f"`{m.group(1)}`", text)
    parts = re.split(r"(`[^`]*`)", text)
    return "".join(p if p.startswith("`") else p.replace("<", "&lt;") for p in parts)


def cell(text: str) -> str:
    """Text safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


# --- sections ----------------------------------------------------------------------------------------------------


def anchor(slug: str) -> str:
    return f'<a id="{slug}"></a>'


def root_slug(root: str) -> str:
    return "api-" + root.removeprefix("threetears.evals.").replace(".", "-")


def api_section() -> Iterator[str]:
    yield anchor("public-api")
    yield "## Public API"
    yield ""
    yield (
        "Import only from these roots, and only the names below: a module under a root is internal and may move. "
        "The package itself, `threetears.evals`, exports only the list of roots and the installed version. "
        "Within a root, names are grouped as functions, classes, types and constants, each sorted by name. A "
        "class's kind says what it is: a `model` is a Pydantic model, a `protocol` is something a host implements. "
        "A name marked *host contract* is one a host writes against: it moves only with a deprecation cycle. Every "
        "other name may move in any release, recorded in `MIGRATIONS.toml` "
        "([stability tiers](adopting-a-host.md#stability-tiers))."
    )
    yield ""
    from threetears.evals import ROOT_EXTRAS

    for root in walked_modules():
        extra = ROOT_EXTRAS.get(root)
        needs = f" (fails to import without the `[{extra}]` extra)" if extra else ""
        yield f"- [`{root}`](#{root_slug(root)}){needs}"
    yield ""
    first_root: dict[int, str] = {}
    names = list(public_names())
    for root in walked_modules():
        module = importlib.import_module(root)
        yield anchor(root_slug(root))
        yield f"### `{root}`"
        yield ""
        if root in ROOT_EXTRAS:
            yield (
                f"*Fails to import without the `[{ROOT_EXTRAS[root]}]` extra "
                f'(`pip install "3tears-evals[{ROOT_EXTRAS[root]}]"`); see `threetears.evals.ROOT_EXTRAS`. Every root '
                "not so marked imports on the base install.*"
            )
            yield ""
        intro = paragraphs(module.__doc__)
        if intro:
            yield md_text(intro[0])
            yield ""
        items: dict[str, list[Item]] = {}
        promised: dict[str, bool] = {}
        again: list[tuple[str, str]] = []
        for name, obj in ((n, o) for r, n, o in names if r == root):
            # A class or function exported under its own name is one object however many roots export it, and is
            # described once. One bound to another name (``UsageRole = str``) is an alias of it, which says what
            # the name means, so it is described where it is exported rather than collapsed into the first alias
            # of the same type.
            shared = isinstance(obj, (type, types.FunctionType)) and getattr(obj, "__name__", None) == name
            if shared and id(obj) in first_root:
                again.append((name, first_root[id(obj)]))
                continue
            if shared:
                first_root[id(obj)] = root
            item = describe(root, name, obj)
            if not item.description:
                GAPS.append((root, name))
            items.setdefault(item.group, []).append(item)
            promised[item.name] = tier_of(f"{root}.{name}") == "host-contract"
        for group in ("Functions", "Classes", "Types", "Constants"):
            if group not in items:
                continue
            yield f"**{group}**"
            yield ""
            for item in items[group]:
                description = md_text(item.description) if item.description else NO_DESCRIPTION
                tier = " · *host contract*" if promised[item.name] else ""
                yield f"- **`{item.name}`**{tier} · {item.kind} · {description}"
                if item.detail:
                    detail = item.detail if item.detail.startswith(("`", "values")) else f"`{item.detail}`"
                    yield f"  <br>{detail}"
            yield ""
        if again:
            yield "**Also exported here**"
            yield ""
            yield ", ".join(f"`{n}` ([`{r}`](#{root_slug(r)}))" for n, r in again)
            yield ""


def _field_rows(model: type, *, full: bool) -> Iterator[tuple[str, str, str, str]]:
    """(name, type, default, description) for each field of a Pydantic model or a dataclass, in declaration order.

    A Pydantic field's default factory is called and its value shown, which is what a reader of a settings table
    wants to see; a dataclass's is named.
    """
    describe_text = (lambda d: paragraphs(d)[0] if paragraphs(d) else "") if full else one_line
    for row in field_rows(model):
        default = row.default
        if row.factory is not None:
            default = stable_repr(row.factory()) if issubclass(model, BaseModel) else factory_name(row.factory)
        yield row.name, row.annotation, default, describe_text(row.description)


def field_table(model: type, *, full: bool) -> Iterator[str]:
    yield "| Field | Type | Default | Description |"
    yield "|---|---|---|---|"
    for name, annotation, default, text in _field_rows(model, full=full):
        if not text:
            GAPS.append((f"{model.__name__} field", name))
        description = cell(md_text(text)) if text else NO_DESCRIPTION
        shown = default if default in ("required", "derived") else f"`{cell(default)}`"
        yield f"| `{name}` | `{cell(annotation)}` | {shown} | {description} |"
    yield ""


def configuration_section() -> Iterator[str]:
    from threetears.evals.kernel.host import HostProfile
    from threetears.evals.run import LaunchSettings

    yield anchor("configuration")
    yield "## Configuration"
    yield ""
    for slug, model in (("launch-settings", LaunchSettings), ("host-profile", HostProfile)):
        yield anchor(slug)
        yield f"### `{model.__name__}`"
        yield ""
        yield md_text(paragraphs(model.__doc__)[0])
        yield ""
        yield from field_table(model, full=True)


def documents_section() -> Iterator[str]:
    from threetears.evals.analysis import (
        AnalysisContextBundle,
        ChartBlock,
        DisclosureBlock,
        Report,
        ReportSource,
        TableBlock,
        TextBlock,
    )

    yield anchor("documents")
    yield "## The report and the analysis bundle"
    yield ""
    yield anchor("report")
    yield "### `Report`"
    yield ""
    yield md_text(paragraphs(Report.__doc__)[0])
    yield ""
    yield from field_table(Report, full=False)
    yield "Its `source`, and each kind of block in `blocks`:"
    yield ""
    for part in (ReportSource, TextBlock, TableBlock, ChartBlock, DisclosureBlock):
        yield f"#### `{part.__name__}`"
        yield ""
        summary = one_line(own_doc(part))
        if summary:
            yield md_text(summary)
            yield ""
        yield from field_table(part, full=False)
    yield anchor("analysis-bundle")
    yield "### `AnalysisContextBundle`"
    yield ""
    yield md_text(paragraphs(AnalysisContextBundle.__doc__)[0])
    yield ""
    yield "Its top-level fields, in declaration order; each one's type is described in the API section."
    yield ""
    yield from field_table(AnalysisContextBundle, full=False)


_SECTION_UNDERLINE = re.compile(r"^-{3,}$")


def rst_to_markdown(doc: str, heading: str) -> Iterator[str]:
    """A module docstring's body (after its summary line) as Markdown: sections, ``::`` blocks and roles."""
    lines = inspect.cleandoc(doc).splitlines()[1:]
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and _SECTION_UNDERLINE.match(lines[i + 1]) and line.strip():
            yield f"{heading} {line.strip()}"
            i += 2
            continue
        if line.rstrip().endswith("::"):
            yield md_text(line.rstrip()[:-1])
            yield ""
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            block: list[str] = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith("    ")):
                block.append(lines[i][4:])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            yield "```python"
            yield from block
            yield "```"
            yield ""
            continue
        yield md_text(line)
        i += 1


def goal_language_section() -> Iterator[str]:
    from threetears.evals.kernel import dsl
    from threetears.evals.schema import goal_grammar

    yield anchor("goal-checks")
    yield "## The goal-check language"
    yield ""
    yield md_text(paragraphs(goal_grammar.__doc__)[0])
    yield ""
    yield from rst_to_markdown(typing.cast(str, goal_grammar.__doc__), "####")
    yield ""
    yield anchor("goal-check-functions")
    yield "### Functions"
    yield ""
    yield "| Function | Arguments | Description |"
    yield "|---|---|---|"
    for name, arguments, description in goal_functions(inspect.getsource(goal_grammar), inspect.getsource(dsl)):
        yield f"| `{name}` | `{arguments}` | {cell(md_text(description))} |"
    yield ""


#: The handler a goal-check function's description is read from, where it is not ``_builtin_<name>``.
_SHARED_HANDLERS = {
    "any": "_eval_any_all",
    "all": "_eval_any_all",
    "fired": "_builtin_fired",
    "fired_armed": "_builtin_fired",
}

#: Handler parameters the evaluator supplies, which an expression never writes.
_SUPPLIED = {"ledger", "ctx", "predicate", "call", "func_name", "node"}


def goal_functions(grammar_source: str, evaluator_source: str) -> list[tuple[str, str, str]]:
    """(name, arguments, description) for each function the goal-check language admits, read from its source.

    Read from the module's text, as a documentation generator reads it, rather than by importing the module's
    private names: the function set is the frozenset the parser admits calls against, and each description is
    the first line of the handler that evaluates it. A function with no handler to read stops the generator.
    """
    names: list[str] = []
    handlers: dict[str, ast.FunctionDef] = {}
    for node in ast.parse(evaluator_source).body:
        if isinstance(node, ast.FunctionDef):
            handlers[node.name] = node
    for node in ast.parse(grammar_source).body:
        if "_BUILTINS" in assigned_names(node) and isinstance(node, ast.Assign):
            names = sorted(ast.literal_eval(node.value.args[0]))  # type: ignore[attr-defined]
    if not names:
        raise SystemExit("the goal-check language's function set (_BUILTINS in schema/goal_grammar.py) was not found")
    rows = []
    for name in names:
        handler = handlers.get(_SHARED_HANDLERS.get(name, f"_builtin_{name}"))
        if handler is None:
            raise SystemExit(f"goal-check function {name!r} has no handler in kernel/dsl.py to document it from")
        if name in ("any", "all"):
            arguments = "<body> for it in <path>"
        else:
            arguments = ", ".join(a.arg for a in handler.args.args if a.arg not in _SUPPLIED)
        rows.append((name, arguments, one_line(ast.get_docstring(handler))))
    return rows


def _json_type(schema: dict[str, Any]) -> str:
    if "enum" in schema:
        return " | ".join(f"`{v!r}`" for v in schema["enum"])
    if "anyOf" in schema:
        return " or ".join(_json_type(s) for s in schema["anyOf"])
    kind = schema.get("type", "any")
    if kind == "array":
        return f"array of {_json_type(schema.get('items', {}))}"
    return f"`{kind}`"


def actions_section() -> Iterator[str]:
    from threetears.evals.actions import PERMISSION_CLASSES, engine_actions, standard_tools

    actions = engine_actions()
    yield anchor("actions")
    yield "## The action catalogue (MCP)"
    yield ""
    yield (
        "Every engine action, as every transport mounts it (the FastMCP tools, a host's own CLI or REST mapping). "
        "A mounted tool takes `action` plus the union of its actions' parameters; `help` is generated. A "
        "parameter marked `?` is optional. `job` marks long work: it returns jobs to poll with `job_poll`. A "
        "host may contribute actions of its own, which are not listed here."
    )
    yield ""
    yield "| Tool (default prefix) | Classes it mounts | Description |"
    yield "|---|---|---|"
    for tool in standard_tools():
        classes = ", ".join(f"`{c}`" for c in PERMISSION_CLASSES if c in tool.permissions)
        yield f"| `{tool.name}` | {classes} | {cell(md_text(tool.description))} |"
    yield ""
    params: dict[tuple[str, str, str], list[str]] = {}
    workflows: dict[str, list[Any]] = {}
    for action in actions:
        workflows.setdefault(action.workflow, []).append(action)
    for workflow, members in workflows.items():
        yield f"### {workflow}"
        yield ""
        yield "| Action | Class | Parameters | Description |"
        yield "|---|---|---|---|"
        for action in members:
            schema = action.params.model_json_schema()
            names = []
            for pname, info in action.params.model_fields.items():
                names.append(f"`{pname}`" if info.is_required() else f"`{pname}?`")
                prop = schema["properties"][pname]
                params.setdefault((pname, _json_type(prop), info.description or ""), []).append(action.name)
            permission = f"`{action.permission}`" + (", job" if action.long_running else "")
            yield (f"| `{action.name}` | {permission} | {', '.join(names) or '—'} | {cell(md_text(action.summary))} |")
        yield ""
    yield anchor("action-parameters")
    yield "### Parameters"
    yield ""
    yield "| Parameter | Type | Description |"
    yield "|---|---|---|"
    for (pname, ptype, description), used_by in sorted(params.items()):
        shared = (pname for (other, _, _) in params if other == pname)
        note = f" (on {', '.join(f'`{a}`' for a in used_by)})" if sum(1 for _ in shared) > 1 else ""
        yield f"| `{pname}`{note} | {cell(ptype)} | {cell(md_text(description))} |"
    yield ""


def cli_section() -> Iterator[str]:
    from threetears.evals.quick import cli

    yield anchor("cli")
    yield "## The command line"
    yield ""
    yield f"`{cli.DEFAULT_PROG} <command> --host MODULE:FACTORY --scope SCOPE [options]`"
    yield ""
    yield (
        "A product mounting the commands under its own CLI (`run_cli(host_factory=...)`) drops `--host`. "
        "Every command takes `--scope`."
    )
    yield ""
    exit_codes = [p for p in paragraphs(cli.__doc__) if p.startswith("Exit codes")]
    if not exit_codes:
        raise SystemExit(
            "the CLI module docstring no longer states its exit codes in a paragraph starting 'Exit codes'"
        )
    yield md_text(exit_codes[0])
    yield ""
    yield "Each command's options, as its `--help` prints them:"
    yield ""
    for name in cli.ENGINE_COMMANDS:
        yield f"### `{name}`"
        yield ""
        yield "```text"
        yield command_help(cli.build_parser(cli.DEFAULT_PROG), name)
        yield "```"
        yield ""


def command_help(parser: argparse.ArgumentParser, command: str) -> str:
    """What ``<command> --help`` prints, at a fixed width and without colour, so it is the same on every machine."""
    saved = {key: os.environ.get(key) for key in ("COLUMNS", "NO_COLOR", "FORCE_COLOR", "PYTHON_COLORS")}
    os.environ.update({"COLUMNS": str(HELP_WIDTH), "NO_COLOR": "1"})
    for key in ("FORCE_COLOR", "PYTHON_COLORS"):
        os.environ.pop(key, None)
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed), contextlib.suppress(SystemExit):
            parser.parse_args([command, "--help"])
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return printed.getvalue().rstrip()


def measures_section() -> Iterator[str]:
    from threetears.evals.kernel import METRIC_DESCRIPTORS

    yield anchor("measures")
    yield "## Measures"
    yield ""
    yield (
        "The engine's own measures (`METRIC_DESCRIPTORS`), grouped by family in the order the engine declares "
        "them. Direction is which end is better (`—` for a coordinate, condition or raw count); scale says "
        "whether a relative change means anything (`ratio`) or only a difference does (`interval`), `—` when "
        "undeclared. A host "
        "declares its own measures on its `MeasureRegistry`, and a template mints one per rubric dimension and "
        "goal check; neither is listed here."
    )
    yield ""
    families: dict[str, list[Any]] = {}
    for descriptor in METRIC_DESCRIPTORS.values():
        families.setdefault(descriptor.family or "unclassified", []).append(descriptor)
    for family, members in families.items():
        yield f"### {family}"
        yield ""
        yield "| Measure | Direction | Scale | Unit | Description |"
        yield "|---|---|---|---|---|"
        for d in members:
            direction = {True: "higher", False: "lower", None: "—"}[d.higher_is_better]
            flags = [label for label, on in (("diagnostic", d.diagnostic), ("guardrail", d.guardrail)) if on]
            if flags:
                direction += f" ({', '.join(flags)})"
            yield (
                f"| `{d.name}` | {direction} | {d.scale or '—'} | {d.unit or '—'} | "
                f"{cell(md_text(first_sentence(d.description)))} |"
            )
        yield ""


SECTIONS: tuple[tuple[str, str, Callable[[], Iterable[str]]], ...] = (
    ("public-api", "Public API", api_section),
    ("configuration", "Configuration", configuration_section),
    ("documents", "The report and the analysis bundle", documents_section),
    ("goal-checks", "The goal-check language", goal_language_section),
    ("actions", "The action catalogue (MCP)", actions_section),
    ("cli", "The command line", cli_section),
    ("measures", "Measures", measures_section),
)


def render() -> str:
    """The whole page."""
    GAPS.clear()
    out = [
        "# Reference",
        "",
        "**For:** people who already know the engine and agents that need exact names. It answers: what does "
        "this root export, what does this setting or field mean, what can a goal check say, which actions and "
        "commands exist, and what does each measure measure. To learn the engine, start with the "
        "[tutorial](tutorial.md) instead.",
        "",
        f"**Generated** from the package by `packages/evals/scripts/generate_reference.py`. Do not edit it by "
        f"hand: change the docstring, field description or definition it came from, then run `{REGENERATE}`. "
        "`tests/test_reference_doc.py` fails while this page is stale.",
        "",
        "## Contents",
        "",
    ]
    out += [f"- [{title}](#{slug})" for slug, title, _ in SECTIONS]
    out.append("")
    for _, _, section in SECTIONS:
        out += list(section())
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).rstrip() + "\n"
    return text


def main(argv: list[str] | None = None) -> int:
    """Write the page, check it, or list its gaps."""
    parser = argparse.ArgumentParser(description=typing.cast(str, __doc__).splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="exit 1 when the committed page is stale; write nothing")
    mode.add_argument("--gaps", action="store_true", help="list the public items that carry no description")
    args = parser.parse_args(argv)
    page = render()
    if args.gaps:
        for where, name in GAPS:
            print(f"{where}: {name}")
        return 0
    if args.check:
        current = REFERENCE_PATH.read_text(encoding="utf-8") if REFERENCE_PATH.exists() else ""
        if current != page:
            print(f"{REFERENCE_PATH} is stale; run: {REGENERATE}", file=sys.stderr)
            return 1
        return 0
    REFERENCE_PATH.write_text(page, encoding="utf-8")
    print(f"wrote {REFERENCE_PATH} ({len(page.encode())} bytes, {page.count(chr(10))} lines, {len(GAPS)} gaps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
