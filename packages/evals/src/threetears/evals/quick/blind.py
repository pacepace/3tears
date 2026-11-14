"""The blind read as files: the page and the sheet a judged result writes, the labels file read back, and the lines
the quick print says about them.

The mechanism is :mod:`threetears.evals.analysis.blind_read`; nothing here draws, pairs or tests. This module is
the quick path's door to it:

- :func:`blind_read_html` is one self-contained static page over a :class:`~threetears.evals.schema.BlindSheet`:
  one item at a time, in the drawn order, a control per criterion per output, a preference control for a pair, and
  a button that saves ``labels.csv``. It loads nothing from anywhere, works opened from disk, and keeps its progress
  in the browser's storage when the browser allows it.
- :func:`blind_read_csv` is the same order as rows, for a spreadsheet (:data:`SHEET_COLUMNS`).
- :func:`read_labels` reads a labels file back: the page's, the sheet's, or an older ``to_label.csv``. Rows on this
  draw become answers; rows that are not (an older sheet, another run's draw) are paired by the output they were
  given on and count as ``selected``, never as a random sample.
- :func:`answer_sheet` has a model answer the sheet through the completion port, one call per item: a reference
  reviewer. Its answers are a reference's on every surface, never a person's.
- :func:`judge_lines` is what the quick print says.

**Blindness here is discipline, not enforcement.** The labeller holds the store and the report, where every judge
score is. The page shows none of it, and the order is random; nothing stops a person from looking.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, NamedTuple

from threetears.evals.analysis.blind_read import PREFERENCE_MIN_ITEMS, blind_sheet, exact_sign_test_p
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA
from threetears.evals.kernel import CALIBRATION_MIN_AGREEMENT, CALIBRATION_MIN_RESULTS
from threetears.evals.kernel.text_block import under
from threetears.evals.schema import (
    SCALES,
    BlindAgreement,
    BlindAnswer,
    BlindPreference,
    BlindPreferenceAnswer,
    BlindRead,
    BlindSheet,
    BlindStanding,
    BlindVerdict,
    CalibrationRating,
    CompletionClient,
    EvalResult,
    RubricScale,
)

#: The page a judged result writes for a person to read blind.
BLIND_READ_HTML: Final = "blind_read.html"

#: The same order as rows, for a spreadsheet.
BLIND_READ_CSV: Final = "blind_read.csv"

#: The columns of :data:`BLIND_READ_CSV`. ``label``, ``reason``, ``rater``, ``rater_kind`` and ``prefer`` are the
#: labeller's to fill: ``prefer`` once per item, on its first row. ``rater_kind`` blank is a person.
SHEET_COLUMNS: tuple[str, ...] = (
    "position",
    "item",
    "slot",
    "dimension",
    "case_material",
    "answer",
    "label",
    "reason",
    "rater",
    "rater_kind",
    "prefer",
    "output_fingerprint",
    "criterion_fingerprint",
)

#: The columns a labels file must carry: which answer a row is of, and the label.
_REQUIRED: tuple[str, ...] = ("dimension", "label")

#: What a label is read as, on a pass/fail dimension.
_PASS_FAIL = {"pass": 1, "fail": 0, "1": 1, "0": 0, "true": 1, "false": 0}

#: What a row's label says when the rater passed on its item.
_SKIP = "skip"


class LabelsRead(NamedTuple):
    """A labels file, read against one draw.

    :ivar answers: The rows on this draw, as answers to its items.
    :ivar loose: People's labels that are not on this draw (an older sheet, another run's draw), as ratings bound to
        the output each was given on: they count toward agreement as ``selected``.
    :ivar unread: Rows that could not be read against this run at all, each with why.
    """

    answers: list[BlindAnswer]
    loose: list[CalibrationRating]
    unread: list[str]


def blind_read_csv(read: BlindRead) -> str:
    """``read`` as a sheet of rows in the drawn order: one per item, output and criterion, the labels blank.

    :param read: The read.
    :ptype read: BlindRead
    :return: The CSV text, in :data:`SHEET_COLUMNS`. It names no arm, run or result, and carries no judge score.
    :rtype: str
    """
    written = io.StringIO()
    writer = csv.DictWriter(written, fieldnames=SHEET_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for position, item in enumerate(read.items, start=1):
        for output in item.outputs:
            for dim in read.criteria:
                writer.writerow(
                    {
                        "position": position,
                        "item": item.item,
                        "slot": output.slot,
                        "dimension": dim.name,
                        "case_material": output.evidence.case_material,
                        "answer": output.evidence.artifact,
                        "label": "",
                        "reason": "",
                        "rater": "",
                        "rater_kind": "",
                        "prefer": "",
                        "output_fingerprint": output.output_fingerprint,
                        "criterion_fingerprint": output.criterion_fingerprints[dim.name],
                    }
                )
    return written.getvalue()


def _score(label: str, scale: RubricScale) -> int | None:
    if scale == "pass_fail":
        return _PASS_FAIL.get(label)
    return int(label) if label.isdigit() and 1 <= int(label) <= 5 else None


def read_labels(path: Path, read: BlindRead, results: Sequence[EvalResult]) -> LabelsRead:
    """Read a labels file against ``read``: the page's download, the sheet filled in, or an older ``to_label.csv``.

    A row whose ``item`` and ``slot`` are on this draw is an answer to it. Any other labelled row is paired by its
    fingerprints with a judged answer of ``results``, and is a rating outside the read; one that reaches none is
    listed unread. A blank ``rater`` is the file's name; a blank ``rater_kind`` is a person, and ``reference`` names
    a model in ``rater``.

    :param path: The labels file.
    :ptype path: Path
    :param read: This run's draw.
    :ptype read: BlindRead
    :param results: The results the draw was over.
    :ptype results: Sequence[EvalResult]
    :return: The answers, the ratings outside the read, and the rows not read.
    :rtype: LabelsRead
    :raises ValueError: A missing column, a label not on its dimension's scale, a dimension the judge does not score,
        a preference that is not A, B or tie, or a rater kind that is not person or reference.
    """
    with path.open(encoding="utf-8", newline="") as opened:
        rows = list(csv.DictReader(opened))
    if rows and (missing := [column for column in _REQUIRED if column not in rows[0]]):
        raise ValueError(
            f"{path} lacks the column(s) {', '.join(missing)}; save the labels from the {BLIND_READ_HTML} a result "
            f"wrote, or fill in its {BLIND_READ_CSV}"
        )
    scales = {dim.name: dim.scale for dim in read.criteria}
    items = {item.item: item for item in read.items}
    by_key: dict[tuple[str, str, str], EvalResult] = {}
    for result in results:
        for score in result.judge_scores():
            if score.output_fingerprint is not None and score.criterion_fingerprint is not None:
                by_key.setdefault((score.dim, score.output_fingerprint, score.criterion_fingerprint), result)
    answers: dict[tuple[str, str], dict[str, Any]] = {}
    loose: list[CalibrationRating] = []
    unread: list[str] = []
    for number, row in enumerate(rows, start=2):
        label = (row.get("label") or "").strip().lower()
        prefer = (row.get("prefer") or "").strip()
        if not label and not prefer:
            continue
        kind = (row.get("rater_kind") or "").strip().lower() or "person"
        if kind not in ("person", "reference"):
            raise ValueError(f"{path} line {number} has rater_kind {kind!r}; it is person (or blank) or reference")
        rater = (row.get("rater") or "").strip()
        if kind == "reference" and not rater:
            raise ValueError(f"{path} line {number} is a reference reviewer's and names no model in rater")
        rater = rater or path.name
        item = items.get((row.get("item") or "").strip())
        slot = (row.get("slot") or "").strip().upper() or "A"
        output = None if item is None else next((shown for shown in item.outputs if shown.slot == slot), None)
        dimension = (row.get("dimension") or "").strip()
        given = (row.get("output_fingerprint") or "").strip()
        on_draw = item is not None and output is not None and given in ("", output.output_fingerprint)
        if on_draw and item is not None:
            answer = answers.setdefault(
                (item.item, rater),
                {
                    "item": item.item,
                    "rater": rater,
                    "rater_kind": kind,
                    "verdicts": [],
                    "prefer": None,
                    "skipped": False,
                },
            )
            if answer["rater_kind"] != kind:
                raise ValueError(
                    f"{path} line {number} names {rater!r} as a {kind}, and an earlier row as another kind"
                )
            if prefer and answer["prefer"] is None:
                choices: dict[str, BlindPreferenceAnswer] = {"a": "A", "b": "B", "tie": "tie"}
                choice = choices.get(prefer.lower())
                if choice is None or (choice != "tie" and all(shown.slot != choice for shown in item.outputs)):
                    raise ValueError(f"{path} line {number} prefers {prefer!r}; a preference is A, B or tie")
                answer["prefer"] = choice
            if label == _SKIP:
                answer["skipped"] = True
                continue
        if not label or label == _SKIP:
            continue
        scale = scales.get(dimension)
        if scale is None:
            raise ValueError(f"{path} line {number} labels {dimension!r}, which the judge does not score")
        score = _score(label, scale)
        if score is None:
            scale_words = "pass or fail" if scale == "pass_fail" else "1 to 5"
            raise ValueError(f"{path} line {number} labels {dimension!r} {label!r}; its scale is {scale_words}")
        reason = (row.get("reason") or "").strip()
        if on_draw and item is not None:
            answers[(item.item, rater)]["verdicts"].append(
                BlindVerdict(slot=slot, dim=dimension, score=score, reason=reason)
            )
            continue
        criterion = (row.get("criterion_fingerprint") or "").strip()
        result = by_key.get((dimension, given, criterion))
        run_id = result.eval_run_id if result is not None else (row.get("run_id") or "").strip()
        result_id = result.id if result is not None else (row.get("result_id") or "").strip()
        if kind != "person":
            unread.append(f"line {number}: a reference reviewer's label that is not on this run's draw")
        elif not given or not criterion or not run_id or not result_id:
            unread.append(f"line {number}: it is not on this run's draw and carries no fingerprint to pair it by")
        else:
            loose.append(
                CalibrationRating(
                    scope_id=read.scope_id,
                    run_id=run_id,
                    result_id=result_id,
                    rubric_dim=dimension,
                    rater=rater,
                    rater_kind="person",
                    scale=scale,
                    score=score,
                    reason=reason or f"labelled in {path.name}",
                    output_fingerprint=given,
                    criterion_fingerprint=criterion,
                    rated_withheld=f"read from {path.name}, which carries the answer and its case as the person read them",
                )
            )
    built: list[BlindAnswer] = []
    for answer in answers.values():
        if answer["skipped"]:
            answer = {**answer, "verdicts": [], "prefer": None}
        elif not answer["verdicts"] and answer["prefer"] is None:
            continue
        built.append(BlindAnswer(collected_by=path.name if answer["rater_kind"] == "person" else None, **answer))
    return LabelsRead(built, loose, unread)


# --- the page ---------------------------------------------------------------------------------------------------

_PAGE_STYLE: Final = """
body{font:16px/1.5 system-ui,sans-serif;margin:0;background:#fff;color:#111}
main{max-width:60rem;margin:0 auto;padding:1rem}
pre{white-space:pre-wrap;overflow-wrap:anywhere;border:1px solid #999;padding:.75rem;margin:.5rem 0;font:inherit}
details{margin:.5rem 0}fieldset{border:1px solid #999;margin:.75rem 0}
label{margin-right:1rem;white-space:nowrap}button{font:inherit;padding:.4rem .8rem;margin:.25rem .5rem .25rem 0}
input[type=text]{font:inherit;padding:.25rem;width:min(100%,30rem)}
.note{border-left:4px solid #b60;padding-left:.75rem}textarea{width:100%;height:10rem}
@media (prefers-color-scheme:dark){body{background:#111;color:#eee}pre,fieldset{border-color:#777}}
"""

_PAGE_SCRIPT: Final = r"""
(function () {
  "use strict";
  var sheet = JSON.parse(document.getElementById("sheet").textContent);
  var key = "blind-read:" + sheet.read_id;
  var state = { rater: "", at: 0, answers: {} };
  function load() {
    try {
      var kept = JSON.parse(window.localStorage.getItem(key) || "null");
      if (kept ? kept.answers : null) { state = kept; }
    } catch (unavailable) { /* storage is blocked or absent: the page works without it */ }
  }
  function save() {
    try { window.localStorage.setItem(key, JSON.stringify(state)); } catch (unavailable) { /* as above */ }
  }
  function answerOf(item) {
    if (!state.answers[item.item]) { state.answers[item.item] = { verdicts: {}, reasons: {}, prefer: "", skipped: false }; }
    return state.answers[item.item];
  }
  function complete(item) {
    var answer = state.answers[item.item];
    if (!answer) { return false; }
    if (answer.skipped) { return true; }
    var all = item.outputs.every(function (output) {
      return item.ask.verdicts.every(function (dim) { return answer.verdicts[output.slot + "|" + dim]; });
    });
    if (!all) { return false; }
    return !item.ask.preference || Boolean(answer.prefer);
  }
  function fromTheTop() {
    var count = 0;
    while (count < sheet.items.length) {
      if (!complete(sheet.items[count])) { break; }
      count += 1;
    }
    return count;
  }
  function answered() { return sheet.items.filter(complete).length; }
  function el(tag, text, attrs) {
    var node = document.createElement(tag);
    if (text) { node.textContent = text; }
    Object.keys(attrs || {}).forEach(function (name) { node.setAttribute(name, attrs[name]); });
    return node;
  }
  function radios(legend, name, values, chosen, onPick) {
    var set = el("fieldset");
    set.appendChild(el("legend", legend));
    values.forEach(function (value) {
      var label = el("label");
      var input = el("input", "", { type: "radio", name: name, value: value[0] });
      input.checked = chosen === value[0];
      input.addEventListener("change", function () { onPick(value[0]); });
      label.appendChild(input);
      label.appendChild(document.createTextNode(" " + value[1]));
      set.appendChild(label);
    });
    return set;
  }
  function render() {
    var root = document.getElementById("item");
    var item = sheet.items[state.at];
    root.textContent = "";
    if (!item) { root.appendChild(el("p", "This read has no items.")); return; }
    var answer = answerOf(item);
    var top = fromTheTop();
    var status = "Item " + (state.at + 1) + " of " + sheet.items.length + ". Answered " + answered() +
      ", " + top + " of them from the top.";
    document.getElementById("status").textContent = status;
    var ahead = document.getElementById("ahead");
    ahead.hidden = !(answered() > top || state.at > top);
    root.appendChild(el("h2", "Item " + (state.at + 1), { tabindex: "-1", id: "heading" }));
    var material = el("details");
    material.open = true;
    material.appendChild(el("summary", "What the output answers"));
    material.appendChild(el("pre", item.case_material));
    root.appendChild(material);
    item.outputs.forEach(function (output) {
      var title = item.outputs.length > 1 ? "Output " + output.slot : "The output";
      root.appendChild(el("h3", title));
      root.appendChild(el("pre", output.text));
      item.ask.verdicts.forEach(function (dim) {
        var criterion = sheet.criteria.filter(function (entry) { return entry.dim === dim; })[0];
        var slot = output.slot + "|" + dim;
        var values = criterion.levels.map(function (level) {
          return [level.value, level.meaning ? level.value + " (" + level.meaning + ")" : level.value];
        });
        root.appendChild(radios(title + ", " + dim + ": " + criterion.question, item.item + "|" + slot, values,
          answer.verdicts[slot] || "", function (value) {
            answer.verdicts[slot] = value; answer.skipped = false; save(); refresh();
          }));
      });
    });
    if (item.ask.preference) {
      root.appendChild(radios("Which output is better?", item.item + "|prefer",
        [["A", "A"], ["B", "B"], ["tie", "Neither"]], answer.prefer, function (value) {
          answer.prefer = value; answer.skipped = false; save(); refresh();
        }));
    }
    var why = el("label", "A note on this item (optional) ");
    var note = el("input", "", { type: "text" });
    note.value = answer.reasons.note || "";
    note.addEventListener("input", function () { answer.reasons.note = note.value; save(); });
    why.appendChild(note);
    root.appendChild(why);
    root.appendChild(el("p", complete(item) ? (answer.skipped ? "Skipped." : "Answered.") :
      "Answer every question for this item to count it.", { id: "done" }));
  }
  function refresh() {
    var item = sheet.items[state.at];
    var done = document.getElementById("done");
    if (item ? done : null) {
      var answer = answerOf(item);
      done.textContent = complete(item) ? (answer.skipped ? "Skipped." : "Answered.") :
        "Answer every question for this item to count it.";
    }
    var top = fromTheTop();
    document.getElementById("status").textContent = "Item " + (state.at + 1) + " of " + sheet.items.length +
      ". Answered " + answered() + ", " + top + " of them from the top.";
    document.getElementById("ahead").hidden = !(answered() > top || state.at > top);
  }
  function go(to) {
    state.at = Math.max(0, Math.min(sheet.items.length - 1, to));
    save(); render();
    var heading = document.getElementById("heading");
    if (heading) { heading.focus(); }
  }
  function cell(value) {
    var text = String(value === undefined || value === null ? "" : value);
    return /[",\n\r]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }
  function labels() {
    var rows = [["item", "slot", "dimension", "label", "reason", "rater", "rater_kind", "prefer"]];
    sheet.items.forEach(function (item) {
      var answer = state.answers[item.item];
      if (!answer || !complete(item)) { return; }
      var note = answer.reasons.note || "";
      if (answer.skipped) { rows.push([item.item, "A", "", "skip", note, state.rater, "person", ""]); return; }
      var first = true;
      item.outputs.forEach(function (output) {
        item.ask.verdicts.forEach(function (dim) {
          rows.push([item.item, output.slot, dim, answer.verdicts[output.slot + "|" + dim], note, state.rater,
            "person", first ? answer.prefer : ""]);
          first = false;
        });
      });
      if (first ? answer.prefer : "") { rows.push([item.item, "A", "", "", note, state.rater, "person", answer.prefer]); }
    });
    return rows.map(function (row) { return row.map(cell).join(","); }).join("\n") + "\n";
  }
  function download() {
    var text = labels();
    var shown = document.getElementById("csv");
    shown.value = text;
    try {
      var link = el("a", "", { href: URL.createObjectURL(new Blob([text], { type: "text/csv" })), download: "labels.csv" });
      document.body.appendChild(link); link.click(); document.body.removeChild(link);
    } catch (unavailable) {
      document.getElementById("fallback").open = true;
    }
  }
  load();
  if (state.at >= sheet.items.length) { state.at = 0; }
  document.getElementById("name").textContent = sheet.name;
  document.getElementById("instructions").textContent = sheet.instructions;
  var rater = document.getElementById("rater");
  rater.value = state.rater || "";
  rater.addEventListener("input", function () { state.rater = rater.value; save(); });
  document.getElementById("previous").addEventListener("click", function () { go(state.at - 1); });
  document.getElementById("next").addEventListener("click", function () { go(state.at + 1); });
  document.getElementById("first").addEventListener("click", function () { go(fromTheTop()); });
  document.getElementById("skip").addEventListener("click", function () {
    var item = sheet.items[state.at];
    if (!item) { return; }
    state.answers[item.item] = { verdicts: {}, reasons: answerOf(item).reasons, prefer: "", skipped: true };
    go(state.at + 1);
  });
  document.getElementById("download").addEventListener("click", download);
  render();
})();
"""


def blind_read_html(sheet: BlindSheet) -> str:
    """One self-contained static page over ``sheet``: read blind, in order, and save the labels.

    The page carries the sheet and nothing else: no arm, run or result, no judge score. It loads no script, style or
    font from anywhere, so it works opened from disk with no network. Every control is a native form control, so it
    is usable from the keyboard. Progress is kept in the browser's storage where the browser allows it, and the page
    works where it does not.

    :param sheet: The blinded view of a read.
    :ptype sheet: BlindSheet
    :return: The page.
    :rtype: str
    """
    # Inlined as JSON in a script element: ``<`` is escaped so no output's text can close the element.
    data = json.dumps(sheet.model_dump(mode="json"), ensure_ascii=False).replace("<", "\\u003c")
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Blind read</title>\n<style>{_PAGE_STYLE}</style>\n</head>\n<body>\n<main>\n"
        '<h1 id="name"></h1>\n<p id="instructions"></p>\n'
        '<p><label>Your name <input type="text" id="rater" autocomplete="name"></label></p>\n'
        '<p id="status" role="status" aria-live="polite"></p>\n'
        '<p id="ahead" class="note" hidden>You have answered out of order. Answers given out of order are not a '
        "random sample; go back to the first unanswered item to keep one.</p>\n"
        '<div id="item"></div>\n'
        '<p><button type="button" id="previous">Previous</button>'
        '<button type="button" id="next">Next</button>'
        '<button type="button" id="skip">Skip this item</button>'
        '<button type="button" id="first">Go to the first unanswered item</button></p>\n'
        '<p><button type="button" id="download">Download labels.csv</button></p>\n'
        '<details id="fallback"><summary>The labels as text</summary>'
        "<p>If the download did not start, copy this into a file named labels.csv.</p>"
        '<textarea id="csv" readonly aria-label="labels.csv"></textarea></details>\n'
        "</main>\n"
        f'<script type="application/json" id="sheet">{data}</script>\n'
        f"<script>{_PAGE_SCRIPT}</script>\n</body>\n</html>\n"
    )


def write_blind_read(out_dir: Path, read: BlindRead) -> dict[str, Path]:
    """Write ``read``'s page and sheet into ``out_dir``.

    :param out_dir: The directory; created when missing.
    :ptype out_dir: Path
    :param read: The read.
    :ptype read: BlindRead
    :return: The page under ``blind_read`` and the sheet under ``blind_read_csv``.
    :rtype: dict[str, Path]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    page, sheet = out_dir / BLIND_READ_HTML, out_dir / BLIND_READ_CSV
    page.write_text(blind_read_html(blind_sheet(read)), encoding="utf-8")
    sheet.write_text(blind_read_csv(read), encoding="utf-8", newline="")
    return {"blind_read": page, "blind_read_csv": sheet}


# --- a reference reviewer ---------------------------------------------------------------------------------------

_REVIEW_SYSTEM: Final = (
    "You are reviewing outputs against stated criteria. You are shown what each output answers, then one output or "
    "two outputs labelled A and B in a random order. Judge each output on each criterion from what is shown and "
    "nothing else. Reply with one JSON object and no other text: "
    '{"verdicts": {"<slot>": {"<criterion>": <answer>}}, "prefer": "A" | "B" | "tie" | null, "reason": "<one '
    'sentence>"}. Give every slot and criterion shown. Set "prefer" only where two outputs are shown and you are '
    "asked which is better; otherwise null."
)


def _review_prompt(sheet: BlindSheet, index: int) -> str:
    item = sheet.items[index]
    asked = [criterion for criterion in sheet.criteria if criterion.dim in item.ask.verdicts]
    lines = ["# Criteria"]
    for criterion in asked:
        answers = " or ".join(
            f'"{level.value}"' + (f" ({level.meaning})" if level.meaning else "") for level in criterion.levels
        )
        lines += under(f"{criterion.dim}: {criterion.question} Answer {answers}.", "  ", first="- ")
    lines += ["", "# What the output answers", item.case_material]
    for output in item.outputs:
        lines += ["", f"# Output {output.slot}", output.text]
    if item.ask.preference:
        lines += ["", "Then say which output is better: A, B, or tie."]
    return "\n".join(lines)


def _json_object(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


async def answer_sheet(sheet: BlindSheet, client: CompletionClient, *, model: str) -> list[BlindAnswer]:
    """Have a model answer ``sheet``, one call per item, each with nothing of any other item in it.

    The model is shown exactly what a person is: the sheet's criteria, the case material and the outputs in their
    drawn slots. Each answer is a reference reviewer's, named by ``model``, and carries the model that served it.

    :param sheet: The blinded view of a read.
    :ptype sheet: BlindSheet
    :param client: The completion client the reviewer is called through.
    :ptype client: CompletionClient
    :param model: The reviewer's model id, which names it on every surface.
    :ptype model: str
    :return: One answer per item, in order.
    :rtype: list[BlindAnswer]
    :raises ValueError: A reply that is not the JSON object asked for, or that misses a slot or criterion or gives an
        answer off its scale, naming the item's position.
    """
    scales = {criterion.dim: criterion.scale for criterion in sheet.criteria}
    answers: list[BlindAnswer] = []
    for index, item in enumerate(sheet.items):
        completion = await client.generate(system=_REVIEW_SYSTEM, user=_review_prompt(sheet, index))
        where = f"the reference reviewer's reply to item {index + 1}"
        try:
            reply = _json_object(completion.content)
        except json.JSONDecodeError as unparsed:
            raise ValueError(f"{where} is not JSON: {unparsed}") from unparsed
        given = reply.get("verdicts") if isinstance(reply, dict) else None
        if not isinstance(given, dict):
            raise ValueError(f'{where} carries no "verdicts" object')
        reason = str(reply.get("reason") or "")
        verdicts: list[BlindVerdict] = []
        for output in item.outputs:
            of_slot = given.get(output.slot)
            for dim in item.ask.verdicts:
                raw = of_slot.get(dim) if isinstance(of_slot, dict) else None
                score = None if raw is None else _score(str(raw).strip().lower(), scales[dim])
                if score is None:
                    levels = " or ".join(SCALES[scales[dim]].levels)
                    raise ValueError(f"{where} gives {raw!r} for output {output.slot} on {dim}; its answer is {levels}")
                verdicts.append(BlindVerdict(slot=output.slot, dim=dim, score=score, reason=reason))
        prefer = reply.get("prefer")
        if item.ask.preference and prefer not in ("A", "B", "tie"):
            raise ValueError(f"{where} prefers {prefer!r}; a preference is A, B or tie")
        served = getattr(completion, "served_model", None) or getattr(completion, "model", None)
        answers.append(
            BlindAnswer(
                item=item.item,
                rater=model,
                rater_kind="reference",
                reviewer_served_model=served if isinstance(served, str) and served.strip() else None,
                verdicts=verdicts,
                prefer=prefer if item.ask.preference else None,
            )
        )
    return answers


# --- what the quick print says ----------------------------------------------------------------------------------


def _figure(value: float) -> str:
    return PageForms.off_page().reading(value).text


def _standing_words(standing: BlindStanding, *, several: bool) -> str:
    who = ""
    if standing.rater_kind == "reference":
        who = f" by {standing.rater}, a reference reviewer"
    elif several:
        who = f" by {standing.rater}"
    read = f"{standing.answered} of {standing.items} items read blind{who}"
    skipped = f"; {standing.skipped} skipped" if standing.skipped else ""
    if standing.standing == "random":
        sample = "a random sample of the cases answered" if standing.skipped else "a random sample"
        return f"{read}, in the drawn order ({sample}{skipped})"
    return f"{read}, out of the drawn order (selected, not a random sample{skipped})"


def _agreement_clause(reading: BlindAgreement) -> str:
    who = "you" if reading.anchor == "person" else "the reviewer"
    counted = f"{reading.pairs} results from {reading.cases} cases"
    table = reading.table
    caught = ""
    if table is not None and (failed := table.both_fail + table.rater_fail_judge_pass):
        caught = f"; the judge failed {table.both_fail} of the {failed} answers {who} failed"
    if reading.kappa is None:
        return (
            f"{counted}  kappa undefined: {who} and the judge gave every answer the same verdict; read further{caught}"
        )
    figure = f"kappa {_figure(reading.kappa)}"
    if reading.lower is not None and reading.upper is not None:
        figure += f" [{_figure(reading.lower)}, {_figure(reading.upper)}]"
    tier = reading.tier
    if reading.anchor == "reference" and tier == "calibrated":
        tier = f"calibrated against a reference model ({reading.anchor_names[0]})"
    if reading.calibration == "not_met":
        tier += f": agreement is shown below the bar of {_figure(CALIBRATION_MIN_AGREEMENT)}"
    elif reading.calibration == "insufficient" and reading.results_needed is not None:
        tier += f": needs {reading.results_needed} more results"
    elif reading.calibration == "undecided" and reading.results_needed is not None:
        tier += f": about {reading.results_needed} more results would decide it"
    return f"{counted}  {figure}  {tier}{caught}"


def _preference_line(preference: BlindPreference, shown: Mapping[str, str]) -> str:
    head = "preference"
    if preference.anchor == "reference":
        head = f"preference of {preference.anchor_names[0]} (a reference reviewer, not a person)"
    if preference.verdict == "withheld":
        return (
            f"{head}: shown when the first {preference.planned} items are answered "
            f"({preference.answered} of {preference.planned})"
        )
    candidate, control = (
        shown.get(preference.candidate, preference.candidate),
        shown.get(preference.control, preference.control),
    )
    decided = preference.prefer_candidate + preference.prefer_control
    ties = f" ({preference.ties} tie{'' if preference.ties == 1 else 's'})" if preference.ties else ""
    counted = f"{candidate} over {control} on {preference.prefer_candidate} of {decided}{ties}"
    notes = "" if preference.random else ", from a selected sample"
    notes += f", {preference.skipped} skipped" if preference.skipped else ""
    # A rater who leans to a slot whatever is in it costs the test power, never validity: said where it shows.
    lean = exact_sign_test_p(preference.slot_a, preference.slot_choices)
    if lean is not None and lean < SIGNIFICANCE_ALPHA:
        notes += f", slot A chosen on {preference.slot_a} of {preference.slot_choices}"
    if preference.verdict == "too_few" or preference.p is None:
        return f"{head}: {counted}{notes}: too few to test ({PREFERENCE_MIN_ITEMS} needed)"
    test = "sign test" if preference.test == "exact_sign" else "bounded test on shares"
    said = {
        "improved": f"{candidate} preferred",
        "regressed": f"{control} preferred",
        "not_separated": "cannot tell apart",
    }[preference.verdict]
    p = PageForms.off_page().p(preference.p).text
    return f"{head}: {counted}{notes}, {test} p {p}: {said}"


def _first_step(read: BlindRead, labels_name: str | None) -> str:
    save = f"save its labels as {labels_name}" if labels_name else "save its labels where Judge(labels=...) names"
    step = f"; next step: open the {BLIND_READ_HTML} the report writes, answer its items from the top, and {save}"
    held = len(read.items) * len(read.arms)
    if held < CALIBRATION_MIN_RESULTS:
        step += (
            f"; the calibrated tier's floor is {CALIBRATION_MIN_RESULTS} judged answers per dimension and the read "
            f"holds {held}, so it needs {CALIBRATION_MIN_RESULTS - held} more"
        )
    return step


def judge_lines(
    read: BlindRead | None,
    *,
    labels_name: str | None,
    labels_there: bool,
    loose: int = 0,
    unread: Sequence[str] = (),
    shown: Mapping[str, str] | None = None,
) -> list[str]:
    """How the judge stands against whoever read it blind, as the quiet print says it.

    :param read: The run's blind read, closed over its labels; None where the run has no judged answer to show.
    :ptype read: BlindRead | None
    :param labels_name: The labels file's name, or None where the judge names none.
    :ptype labels_name: str | None
    :param labels_there: Whether that file exists.
    :ptype labels_there: bool
    :param loose: How many labels in the file are not on this draw and were counted as selected.
    :ptype loose: int
    :param unread: The rows that could not be read against this run, each with why.
    :ptype unread: Sequence[str]
    :param shown: The name each arm is printed by, by its key.
    :ptype shown: Mapping[str, str] | None
    :return: The lines.
    :rtype: list[str]
    """
    if read is None:
        return ["judge: no judged answer to read blind: judge uncalibrated"]
    report = read.report
    standings = [standing for standing in (report.standings if report else []) if standing.standing != "none"]
    lines: list[str] = []
    if not standings and not loose:
        why = (
            "Judge(labels=...) names the labels file"
            if labels_name is None
            else f"{labels_name} is not there yet"
            if not labels_there
            else f"{labels_name} holds no label that pairs with this run's answers"
        )
        lines.append(
            f"judge: 0 of {len(read.items)} items read blind: judge uncalibrated ({why}){_first_step(read, labels_name)}"
        )
        return lines + [f"  not read: {entry}" for entry in unread]
    assert report is not None  # standings or loose labels come from a closed read
    people = [standing for standing in standings if standing.rater_kind == "person"]
    lines += [f"judge: {_standing_words(standing, several=len(people) > 1)}" for standing in standings]
    if loose:
        lines.append(
            f"judge: {loose} label(s) in {labels_name} are not on this run's draw (an older sheet, or another run's): "
            "counted toward agreement as selected, not a random sample"
        )
    lines += [f"  not read: {entry}" for entry in unread]
    width = max((len(reading.dim) for reading in report.agreement), default=0)
    by_person = [reading for reading in report.agreement if reading.anchor == "person"]
    lines += [f"  {reading.dim.ljust(width)}  {_agreement_clause(reading)}" for reading in by_person]
    reviewers = dict.fromkeys(reading.anchor_names[0] for reading in report.agreement if reading.anchor == "reference")
    for reviewer in reviewers:
        own = [r for r in report.agreement if r.anchor == "reference" and r.anchor_names[0] == reviewer]
        checked = [check for check in report.anchor_checks if check.reviewer == reviewer]
        against = "" if checked else ", not yet against people"
        lines.append(f"  {'beside it, ' if by_person else ''}checked against {reviewer}{against}:")
        lines += [f"    {reading.dim.ljust(width)}  {_agreement_clause(reading)}" for reading in own]
        if any(reading.shares_family for reading in own):
            lines.append(
                f"    {reviewer} and the judge are one provider's models: a lean they share would not show here"
            )
        for check in checked:
            kappa = "kappa undefined" if check.kappa is None else f"kappa {_figure(check.kappa)}"
            lines += under(f"{reviewer} against people on {check.dim}: {check.outputs} outputs, {kappa}", "    ")
    for reviewer in dict.fromkeys(refused.reviewer for refused in report.refused_anchors):
        lines.append(
            f"  {reviewer} is the judge's own model: its answers repeat the judge and are not read as calibration"
        )
    lines += [_preference_line(preference, shown or {}) for preference in report.preferences]
    if people:
        furthest = max(people, key=lambda standing: standing.answered)
        if all(standing.standing == "random" for standing in people):
            if furthest.answered < furthest.items:
                lines.append(f"next: keep going from item {furthest.answered + 1} in {BLIND_READ_HTML}")
        else:
            lines.append(f"next: for a random sample, answer {BLIND_READ_HTML} from item 1, in order")
    return lines


#: What the quick result calls. The page, the sheet and their names are this module's own, reached through
#: :func:`write_blind_read`.
__all__ = ["answer_sheet", "judge_lines", "read_labels", "write_blind_read"]
