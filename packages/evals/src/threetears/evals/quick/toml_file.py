"""Rung 0: an eval in one TOML file, no Python — ``python -m threetears.evals triage.toml``.

The file holds the same three slots an :class:`~threetears.evals.quick.contract.Eval` takes, and builds one::

    cases = "emails.jsonl"
    recording = "recordings/triage.jsonl"
    grader = ["category", "urgent", "needs_human"]

    [runner]
    model = "deepseek/deepseek-v4-flash"
    prompt = \"\"\"
    Triage this support email:

    {email}
    \"\"\"

    [runner.output]
    category = ["billing", "bug", "account", "sales", "spam"]
    urgent = "bool"
    needs_human = "bool"

How the file is read:

- **``cases``** is a case file beside it (``.jsonl``, ``.json``, ``.csv``, ``.toml``), or the cases inline, as an
  array of tables (``[[cases]]``).
- **``runner``** is a prompt: ``model`` answers ``prompt`` for each case. ``{field}`` in the prompt is filled from
  the case; a field a case lacks is refused by name before anything runs, and ``{{`` ``}}`` give a literal brace. The
  prompt's first line is what each case asks, which a judge reads.
- **``output``** has two rules: a type name (``str``, ``int``, ``float``, ``bool``) gives that type, and a list of
  strings gives the values allowed. The answer is that model, sent as a strict schema and validated on return. With
  no ``output`` the answer is the reply's text.
- **``grader``** is a list of field names and expressions, as an eval's, and a judge as an inline table:
  ``{judge = {model = "...", rubric = {helpful = "says what to do next"}}}``.
- **``recording``** covers every model call the file makes, the runner's and the judge's.
- **``name``** and **``k``** are the eval's, when given.

Every path is read relative to the file.
"""

from __future__ import annotations

import string
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, create_model

from threetears.evals.quick.client import Model
from threetears.evals.quick.contract import Eval, load_cases
from threetears.evals.quick.judged import Judge

#: The type names ``output`` takes, and the types they give.
OUTPUT_TYPES: Mapping[str, type] = {"str": str, "int": int, "float": float, "bool": bool}

#: The keys a file may hold.
_KEYS = frozenset({"cases", "runner", "grader", "recording", "name", "k"})


def _refuse(path: Path, problem: str) -> ValueError:
    return ValueError(f"{path}: {problem}")


def _placeholders(path: Path, prompt: str) -> list[str]:
    """The fields ``{field}`` a prompt fills from each case, refusing a placeholder that names no field.

    Raises:
        ValueError: A placeholder that is not a field's name (``{0}``, ``{a.b}``, ``{x!r}``).
    """
    fields: list[str] = []
    try:
        parsed = list(string.Formatter().parse(prompt))
    except ValueError as malformed:
        raise _refuse(
            path, f"the prompt's braces do not parse ({malformed}); write a literal brace as {{{{ or }}}}"
        ) from malformed
    for _literal, field, spec, conversion in parsed:
        if field is None:
            continue
        if not field.isidentifier() or spec or conversion:
            raise _refuse(path, f"the prompt's {{{field}}} names no case field; a placeholder is {{field}}")
        fields.append(field)
    return fields


def _output_model(path: Path, output: Any) -> type[BaseModel]:
    """The answer's model, from ``[runner.output]``.

    Raises:
        ValueError: An output that is not a table, a field of no known type name, or a list of values that are not
            non-blank strings.
    """
    if not isinstance(output, Mapping) or not output:
        raise _refuse(path, "runner.output is a table of field = type name, or field = [allowed values]")
    fields: dict[str, Any] = {}
    for name, declared in output.items():
        if not name.isidentifier():
            raise _refuse(path, f"runner.output field {name!r} is not an identifier")
        if isinstance(declared, list):
            if not declared or not all(isinstance(value, str) and value.strip() for value in declared):
                raise _refuse(path, f"runner.output.{name} lists the values allowed, each a non-blank string")
            # A Literal of the values allowed: a strict schema's string enum, validated on return.
            fields[name] = (cast(Any, Literal)[tuple(declared)], ...)
        elif isinstance(declared, str) and declared in OUTPUT_TYPES:
            fields[name] = (OUTPUT_TYPES[declared], ...)
        else:
            raise _refuse(
                path,
                f"runner.output.{name} is {declared!r}: a type name ({', '.join(OUTPUT_TYPES)}) or a list of values",
            )
    model: type[BaseModel] = create_model("Output", **fields)
    return model


def _prompt_runner(path: Path, runner: Any, recording: Path | None) -> Any:
    """The runner ``[runner]`` declares: the model answering the prompt, filled from each case.

    Raises:
        ValueError: A runner that is not a table of ``model``, ``prompt`` and, optionally, ``output``.
    """
    if not isinstance(runner, Mapping) or set(runner) - {"model", "prompt", "output"}:
        raise _refuse(path, "runner is a table of model, prompt and, optionally, output")
    model_id, prompt = runner.get("model"), runner.get("prompt")
    if not isinstance(model_id, str) or not model_id.strip():
        raise _refuse(path, "runner.model is the model's id, a non-blank string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise _refuse(path, "runner.prompt is the prompt, a non-blank string")
    prompt = prompt.strip("\n")
    schema = _output_model(path, runner["output"]) if "output" in runner else None
    model = Model(model_id, recording=recording)

    async def answer(case: Mapping[str, Any]) -> Any:
        asked = prompt.format_map(dict(case))
        return await model.text(asked) if schema is None else await model.json(asked, schema)

    answer.__name__ = path.stem
    answer.__doc__ = prompt.strip().splitlines()[0]
    return answer


def _graders(path: Path, graders: Any, recording: Path | None) -> list[Any]:
    """The graders ``grader`` lists: names and expressions as given, and a ``{judge = {...}}`` as a Judge.

    Raises:
        ValueError: A grader that is neither a string nor a ``judge`` table, or a judge with no model or rubric.
    """
    listed = graders if isinstance(graders, list) else [graders]
    built: list[Any] = []
    for given in listed:
        if isinstance(given, str):
            built.append(given)
            continue
        judge = given.get("judge") if isinstance(given, Mapping) and set(given) == {"judge"} else None
        if not isinstance(judge, Mapping) or set(judge) - {"model", "rubric"}:
            raise _refuse(
                path, f"a grader is a field name, an expression or {{judge = {{model, rubric}}}}, not {given!r}"
            )
        model_id, rubric = judge.get("model"), judge.get("rubric")
        if not isinstance(model_id, str) or not isinstance(rubric, Mapping):
            raise _refuse(path, "a judge is {model = \"<id>\", rubric = {dimension = \"what it measures\"}}")
        built.append(Judge(Model(model_id, recording=recording), dict(rubric)))
    return built


def load_toml(path: str | Path) -> Eval:
    """The eval a TOML file declares.

    Args:
        path: The file.

    Returns:
        The eval, not yet run.

    Raises:
        ValueError: A file that is not TOML, a key it does not take, a slot missing or malformed, a case lacking a
            field the prompt fills, or anything the eval refuses — each naming the file.
    """
    path = Path(path)
    try:
        table = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as unreadable:
        raise _refuse(path, f"cannot be read as TOML: {unreadable}") from unreadable
    if unknown := sorted(set(table) - _KEYS):
        raise _refuse(path, f"takes no key {', '.join(unknown)}; its keys are {', '.join(sorted(_KEYS))}")
    if missing := [key for key in ("cases", "runner", "grader") if key not in table]:
        raise _refuse(path, f"lacks {', '.join(missing)}: an eval is its cases, its runner and its graders")
    here = path.parent
    given_cases = table["cases"]
    cases: Sequence[Mapping[str, Any]] | Path = here / given_cases if isinstance(given_cases, str) else given_cases
    loaded = load_cases(cases)
    recording = here / table["recording"] if isinstance(table.get("recording"), str) else None
    runner = _prompt_runner(path, table["runner"], recording)
    prompt = table["runner"]["prompt"]
    for field in _placeholders(path, prompt):
        if lacking := [str(case.get("id", index)) for index, case in enumerate(loaded) if field not in case]:
            raise _refuse(path, f"the prompt fills {{{field}}}, which case(s) {', '.join(lacking)} lack")
    k = table.get("k")
    return Eval(
        loaded,
        runner,
        _graders(path, table["grader"], recording),
        name=table.get("name") or path.stem,
        **({} if k is None else {"k": k}),
    )


__all__ = ["OUTPUT_TYPES", "load_toml"]
