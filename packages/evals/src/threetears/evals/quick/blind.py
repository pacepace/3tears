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
  given on and are kept as not drawn at random.
- :func:`answer_sheet` has a model answer the sheet through the completion port, one call per item, each admitted
  against the eval's cost cap and written to the out-of-run ledger. Its answers are a model's on every surface, never
  a person's. :func:`keep_reviewed` keeps them as a labels file, so the same draw is not asked twice.
- :func:`judge_lines` is what the quick print says.

**Blindness here is discipline, not enforcement.** The labeller holds the store and the report, where every judge
score is. The page shows none of it, and the order is random; nothing stops a person from looking.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, NamedTuple

from threetears.evals.analysis.blind_read import PREFERENCE_MIN_ITEMS, blind_sheet, exact_sign_test_p
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.words import judged_heading
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA
from threetears.evals.kernel import CALIBRATION_MIN_AGREEMENT, CALIBRATION_MIN_RESULTS
from threetears.evals.kernel.authored import strict_response_format
from threetears.evals.kernel.out_of_run import AdmittedCall, OutOfRunBudget, PlannedCall
from threetears.evals.schema.completion import CompletionProtocolDefect, PricedCompletion
from threetears.evals.schema.logs import contain, get_logger
from threetears.evals.kernel.text_block import under
from threetears.evals.schema import (
    SCALES,
    CalibrationRating,
    EvalResult,
    RubricScale,
)
from threetears.evals.schema.blind_read import (
    BlindAgreement,
    BlindAnswer,
    BlindPreference,
    BlindPreferenceAnswer,
    BlindRead,
    BlindReview,
    BlindSheet,
    BlindSample,
    BlindVerdict,
)

log = get_logger(__name__)

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
        the output each was given on: they count toward agreement, as not drawn at random.
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

    A row whose ``item`` and ``slot`` are on this draw is an answer to it. Any other labelled row (the population
    changed since the labels were saved, or the file is an older sheet) is paired by its fingerprints with a judged
    answer of ``results`` that is byte for byte the output it was given on: a rating outside the read, kept, and not
    drawn at random. One whose output is no longer among the answers is listed unread, with why; so is a preference
    given on another draw, since the pair it chose between is gone. A blank ``rater`` is the file's name; a blank
    ``rater_kind`` is a person, and ``reference`` names a model in ``rater`` (with the model that served it in
    ``served_model``, where the file has that column). A file's word for who answered is taken as written.

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
    for judged in results:
        for stored in judged.judge_scores():
            if stored.output_fingerprint is not None and stored.criterion_fingerprint is not None:
                by_key.setdefault((stored.dim, stored.output_fingerprint, stored.criterion_fingerprint), judged)
    answers: dict[tuple[str, str, str], dict[str, Any]] = {}
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
            raise ValueError(f"{path} line {number} is a model's (rater_kind reference) and names no model in rater")
        rater = rater or path.name
        item = items.get((row.get("item") or "").strip())
        slot = (row.get("slot") or "").strip().upper() or "A"
        output = None if item is None else next((shown for shown in item.outputs if shown.slot == slot), None)
        dimension = (row.get("dimension") or "").strip()
        given = (row.get("output_fingerprint") or "").strip()
        on_draw = item is not None and output is not None and given in ("", output.output_fingerprint)
        if on_draw and item is not None:
            served = (row.get("served_model") or "").strip() or None
            answer = answers.setdefault(
                (item.item, rater, kind),
                {
                    "item": item.item,
                    "rater": rater,
                    "rater_kind": kind,
                    "reviewer_served_model": served if kind == "reference" else None,
                    "verdicts": [],
                    "prefer": None,
                    "skipped": False,
                },
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
        elif prefer:
            unread.append(f"line {number}: a preference given on another draw, whose pair is no longer shown together")
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
        if on_draw and item is not None and output is not None:
            answers[(item.item, rater, kind)]["verdicts"].append(
                BlindVerdict(slot=output.slot, dim=dimension, score=score, reason=reason)
            )
            continue
        criterion = (row.get("criterion_fingerprint") or "").strip()
        result = by_key.get((dimension, given, criterion))
        if kind != "person":
            unread.append(f"line {number}: a model's label that is not on this run's draw")
        elif not given or not criterion:
            unread.append(f"line {number}: it is not on this run's draw and carries no fingerprint to pair it by")
        elif result is None:
            unread.append(f"line {number}: the output it was given on is no longer among this run's answers")
        else:
            loose.append(
                CalibrationRating(
                    scope_id=read.scope_id,
                    run_id=result.eval_run_id,
                    result_id=result.id,
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
      input.onchange = (function () { onPick(value[0]); });
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
    if (!item) { root.appendChild(el("p", "This read has no cases.")); return; }
    var answer = answerOf(item);
    var top = fromTheTop();
    var status = "Case " + (state.at + 1) + " of " + sheet.items.length + ". Answered " + answered() +
      ", " + top + " of them from the top.";
    document.getElementById("status").textContent = status;
    var ahead = document.getElementById("ahead");
    ahead.hidden = !(answered() > top || state.at > top);
    root.appendChild(el("h2", "Case " + (state.at + 1), { tabindex: "-1", id: "heading" }));
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
    var why = el("label", "A note on this case (optional) ");
    var note = el("input", "", { type: "text" });
    note.value = answer.reasons.note || "";
    note.oninput = (function () { answer.reasons.note = note.value; save(); });
    why.appendChild(note);
    root.appendChild(why);
    root.appendChild(el("p", complete(item) ? (answer.skipped ? "Skipped." : "Answered.") :
      "Answer every question for this case to count it.", { id: "done" }));
  }
  function refresh() {
    var item = sheet.items[state.at];
    var done = document.getElementById("done");
    if (item ? done : null) {
      var answer = answerOf(item);
      done.textContent = complete(item) ? (answer.skipped ? "Skipped." : "Answered.") :
        "Answer every question for this case to count it.";
    }
    var top = fromTheTop();
    document.getElementById("status").textContent = "Case " + (state.at + 1) + " of " + sheet.items.length +
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
    var rows = [["item", "slot", "dimension", "label", "reason", "rater", "rater_kind", "prefer",
      "output_fingerprint", "criterion_fingerprint"]];
    sheet.items.forEach(function (item) {
      var answer = state.answers[item.item];
      if (!answer || !complete(item)) { return; }
      var note = answer.reasons.note || "";
      if (answer.skipped) {
        rows.push([item.item, "A", "", "skip", note, state.rater, "person", "", item.outputs[0].output_fingerprint, ""]);
        return;
      }
      var first = true;
      item.outputs.forEach(function (output) {
        item.ask.verdicts.forEach(function (dim) {
          rows.push([item.item, output.slot, dim, answer.verdicts[output.slot + "|" + dim], note, state.rater,
            "person", first ? answer.prefer : "", output.output_fingerprint, output.criterion_fingerprints[dim]]);
          first = false;
        });
      });
      if (first ? answer.prefer : "") {
        rows.push([item.item, "A", "", "", note, state.rater, "person", answer.prefer,
          item.outputs[0].output_fingerprint, ""]);
      }
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
  rater.oninput = (function () { state.rater = rater.value; save(); });
  document.getElementById("previous").onclick = (function () { go(state.at - 1); });
  document.getElementById("next").onclick = (function () { go(state.at + 1); });
  document.getElementById("first").onclick = (function () { go(fromTheTop()); });
  document.getElementById("skip").onclick = (function () {
    var item = sheet.items[state.at];
    if (!item) { return; }
    state.answers[item.item] = { verdicts: {}, reasons: answerOf(item).reasons, prefer: "", skipped: true };
    go(state.at + 1);
  });
  document.getElementById("download").onclick = (download);
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
        "random sample; go back to the first unanswered case to keep one.</p>\n"
        '<div id="item"></div>\n'
        '<p><button type="button" id="previous">Previous</button>'
        '<button type="button" id="next">Next</button>'
        '<button type="button" id="skip">Skip this case</button>'
        '<button type="button" id="first">Go to the first unanswered case</button></p>\n'
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


# --- a model as the reader ---------------------------------------------------------------------------------------

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


def _reply_format(sheet: BlindSheet, index: int) -> dict[str, Any]:
    """The strict reply shape one item asks for: a verdict per criterion per output, the preference, and a reason.
    Every answer is a string enum, the one form every provider's strict mode enforces."""
    item = sheet.items[index]
    levels = {criterion.dim: [level.value for level in criterion.levels] for criterion in sheet.criteria}
    per_output = {
        "type": "object",
        "properties": {dim: {"type": "string", "enum": levels[dim]} for dim in item.ask.verdicts},
        "required": list(item.ask.verdicts),
        "additionalProperties": False,
    }
    slots = [output.slot for output in item.outputs]
    schema = {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "object",
                "properties": dict.fromkeys(slots, per_output),
                "required": slots,
                "additionalProperties": False,
            },
            "prefer": {"type": "string", "enum": ["A", "B", "tie"] if item.ask.preference else ["none"]},
            "reason": {"type": "string"},
        },
        "required": ["verdicts", "prefer", "reason"],
        "additionalProperties": False,
    }
    return strict_response_format("blind_review", schema)


class _ReplyUnusable(ValueError):
    """A reviewer's reply that is not the answer its item asked for."""


def _answer_of(sheet: BlindSheet, index: int, content: str) -> tuple[list[BlindVerdict], BlindPreferenceAnswer | None]:
    """The verdicts and preference one reply gives for the item at ``index``.

    :raises _ReplyUnusable: The reply is not the JSON object asked for, misses a slot or criterion, or gives an
        answer off its scale.
    """
    item = sheet.items[index]
    scales = {criterion.dim: criterion.scale for criterion in sheet.criteria}
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        reply = json.loads(text)
    except json.JSONDecodeError as unparsed:
        raise _ReplyUnusable(f"its reply is not JSON ({unparsed.msg})") from unparsed
    given = reply.get("verdicts") if isinstance(reply, dict) else None
    if not isinstance(given, dict):
        raise _ReplyUnusable('its reply carries no "verdicts" object')
    reason = str(reply.get("reason") or "")
    verdicts: list[BlindVerdict] = []
    for output in item.outputs:
        of_slot = given.get(output.slot)
        for dim in item.ask.verdicts:
            raw = of_slot.get(dim) if isinstance(of_slot, dict) else None
            score = None if raw is None else _score(str(raw).strip().lower(), scales[dim])
            if score is None:
                levels = " or ".join(SCALES[scales[dim]].levels)
                raise _ReplyUnusable(f"it gave {raw!r} for output {output.slot} on {dim}, whose answer is {levels}")
            verdicts.append(BlindVerdict(slot=output.slot, dim=dim, score=score, reason=reason))
    if not item.ask.preference:
        return verdicts, None
    choices: dict[str, BlindPreferenceAnswer] = {"A": "A", "B": "B", "tie": "tie"}
    prefer = choices.get(str(reply.get("prefer")))
    if prefer is None:
        raise _ReplyUnusable(f"it preferred {reply.get('prefer')!r}, and a preference is A, B or tie")
    return verdicts, prefer


#: The ledger purpose a model reviewer's calls are written under: measurement spend, on its own line.
REVIEWER_PURPOSE: Final = "reviewer"


def plan_review(sheet: BlindSheet, items: Collection[str] | None = None) -> dict[str, PlannedCall]:
    """The calls a model's reading of ``sheet`` makes: one per item, by item, in the drawn order.

    The planner half: its caller admits these under its budget before any is made, and :func:`answer_sheet` makes
    only what was admitted.

    :param sheet: The blinded view of a read.
    :ptype sheet: BlindSheet
    :param items: The items to plan; None for every item.
    :ptype items: Collection[str] | None
    :return: Each item's call: the prompt pair and the strict reply shape it asks for.
    :rtype: dict[str, PlannedCall]
    """
    return {
        item.item: PlannedCall(
            system=_REVIEW_SYSTEM, user=_review_prompt(sheet, index), response_format=_reply_format(sheet, index)
        )
        for index, item in enumerate(sheet.items)
        if items is None or item.item in items
    }


async def answer_sheet(
    sheet: BlindSheet,
    client: PricedCompletion,
    budget: OutOfRunBudget,
    admitted: Mapping[str, AdmittedCall],
    *,
    reused: int = 0,
    max_cost_usd: float | None = None,
    landed_usd: float | None = 0.0,
) -> tuple[list[BlindAnswer], BlindReview]:
    """Have a model answer ``sheet``, one call per item, each with nothing of any other item in it.

    The model is shown exactly what a person is: the sheet's criteria, the case material and the outputs in their
    drawn slots. Each answer is a model's, named by the client's model, and carries the model that served it.

    **Spend.** Every call was admitted through ``budget`` by this function's caller (:func:`plan_review` plans them)
    and is written to the out-of-run ledger as it is made, under the purpose ``reviewer``: what it cost to check the judge, never a candidate's cost. Under a cost
    cap the calls are held to what the eval has left, by the rule the judge's examples are: before each call what has
    landed is read against the cap, and once it is over, or any of it went unpriced, no further call is made and the
    reading says why.

    **A reply that cannot be used** (not the JSON asked for, a verdict missing or off its scale) leaves its item not
    answered, as a skip, with the reason on the reading; the rest are asked. A completion that breaks the protocol
    stops the reading, said the same way. Either way what was answered is returned.

    :param sheet: The blinded view of a read.
    :ptype sheet: BlindSheet
    :param client: The reviewer's client, which names its model and prices its calls where it can.
    :ptype client: PricedCompletion
    :param budget: The ledger the calls are admitted against and written to.
    :ptype budget: OutOfRunBudget
    :param admitted: Each item's admitted call, by item: an item with none is not asked. An item answered before is
        left out by the caller.
    :ptype admitted: Mapping[str, AdmittedCall]
    :param reused: How many items the caller read from answers recorded earlier, for the reading to state.
    :ptype reused: int
    :param max_cost_usd: The eval's cost cap, over everything it spends; None for none.
    :ptype max_cost_usd: float | None
    :param landed_usd: What the eval has landed before these calls; None where any of it went unpriced.
    :ptype landed_usd: float | None
    :return: An answer per item asked, in order (a skip where the reply could not be used), and the reading.
    :rtype: tuple[list[BlindAnswer], BlindReview]
    """
    model = client.model_name
    forms = PageForms.off_page()
    answers: list[BlindAnswer] = []
    unanswered: list[str] = []
    served: set[str] = set()
    priced: list[float] = []
    calls = unpriced = 0
    stopped: str | None = None
    for index, item in enumerate(sheet.items):
        call = admitted.get(item.item)
        if call is None:
            continue
        if max_cost_usd is not None:
            if landed_usd is None:
                stopped = "the eval spent what nobody priced, which its cost cap cannot bound"
            elif unpriced:
                stopped = "the reviewer's calls spent what nobody priced, which the eval's cost cap cannot bound"
            elif (landed := landed_usd + math.fsum(priced)) > max_cost_usd:
                stopped = (
                    f"the eval's cost cap stopped it, the eval having landed {forms.money_total(landed).text}, over "
                    f"its {forms.money_total(max_cost_usd).text} cap"
                )
            if stopped is not None:
                break
        try:
            recorded = await budget.generate(client, call)
        except CompletionProtocolDefect as broken:
            # The call was made and its row written; the reading says what stopped it, and what was answered stands.
            contain(
                broken, log=log, what=f"a model reviewer's reading of a blind read stopped: {broken}", level="error"
            )
            stopped = f"the reviewer's completion broke the protocol ({broken})"
            calls += 1
            unpriced += 1
            break
        calls += 1
        if recorded.spend.cost_usd is None:
            unpriced += 1
        else:
            priced.append(recorded.spend.cost_usd)
        which = getattr(recorded.result, "served_model", None)
        which = which if isinstance(which, str) and which.strip() else None
        if which is not None:
            served.add(which)
        try:
            verdicts, prefer = _answer_of(sheet, index, str(getattr(recorded.result, "content", "")))
        except _ReplyUnusable as unusable:
            unanswered.append(f"case {index + 1}: {unusable}")
            answers.append(BlindAnswer(item=item.item, rater=model, rater_kind="reference", skipped=True))
            continue
        answers.append(
            BlindAnswer(
                item=item.item,
                rater=model,
                rater_kind="reference",
                reviewer_served_model=which,
                verdicts=verdicts,
                prefer=prefer,
            )
        )
    review = BlindReview(
        reviewer=model,
        served_models=sorted(served),
        asked=calls,
        reused=reused,
        calls=calls,
        cost_usd=math.fsum(priced) if priced else None,
        unpriced_calls=unpriced,
        unanswered=unanswered,
        stopped=stopped,
    )
    return answers, review


#: The columns a reviewer's kept answers are written in: a labels file, which :func:`read_labels` reads back.
_KEPT_COLUMNS: tuple[str, ...] = (
    "item",
    "slot",
    "dimension",
    "label",
    "reason",
    "rater",
    "rater_kind",
    "served_model",
    "prefer",
    "output_fingerprint",
    "criterion_fingerprint",
)


def keep_reviewed(path: Path, read: BlindRead, reviewer: str) -> None:
    """Write ``reviewer``'s answers on ``read`` to ``path`` as a labels file, so the same draw is not asked again.

    :param path: Where they are kept; its directory is created when missing.
    :ptype path: Path
    :param read: The read, carrying the answers.
    :ptype read: BlindRead
    :param reviewer: The reviewer's model id.
    :ptype reviewer: str
    """
    items = {item.item: item for item in read.items}
    scales = {dim.name: SCALES[dim.scale] for dim in read.criteria}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as opened:
        writer = csv.DictWriter(opened, fieldnames=_KEPT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for answer in read.answers:
            if (answer.rater, answer.rater_kind) != (reviewer, "reference"):
                continue
            outputs = {output.slot: output for output in items[answer.item].outputs}
            row = {"item": answer.item, "rater": reviewer, "rater_kind": "reference"}
            row["served_model"] = answer.reviewer_served_model or ""
            if answer.skipped:
                writer.writerow({**row, "slot": "A", "label": _SKIP})
                continue
            for number, verdict in enumerate(answer.verdicts):
                spec = scales[verdict.dim]
                writer.writerow(
                    {
                        **row,
                        "slot": verdict.slot,
                        "dimension": verdict.dim,
                        "label": spec.labels.get(verdict.score, str(verdict.score)),
                        "reason": verdict.reason,
                        "prefer": (answer.prefer or "") if number == 0 else "",
                        "output_fingerprint": outputs[verdict.slot].output_fingerprint,
                        "criterion_fingerprint": outputs[verdict.slot].criterion_fingerprints[verdict.dim],
                    }
                )


# --- what the quick print says ----------------------------------------------------------------------------------

#: How many rows a line names of those it counts.
_NAMED = 3


def _figure(value: float) -> str:
    return PageForms.off_page().reading(value).text


def _who(kind: str, name: str) -> str:
    return f"a model ({name})" if kind == "reference" else name


def _whose(kind: str, names: Sequence[str], labels_name: str | None) -> str:
    """Whose read a line states: a model by its id, a person by the name their labels give, else ``a person``."""
    if kind == "reference":
        return _who(kind, names[0])
    named = [name for name in names if name != labels_name]
    return ", ".join(named) if named else "a person"


def _sample_words(sample: BlindSample, *, labels_name: str | None) -> str:
    who = f" by {_whose(sample.rater_kind, [sample.rater], labels_name)}"
    read = f"{sample.answered} of {sample.items} cases read blind{who}"
    left = [f"{sample.skipped} skipped"] if sample.skipped else []
    left += [f"{sample.partial} answered in part"] if sample.partial else []
    aside = f"; {', '.join(left)}" if left else ""
    if sample.sample == "random":
        words = "a random sample of the cases answered" if left else "a random sample"
        return f"{read}, in the drawn order ({words}{aside})"
    return f"{read}, out of the drawn order (not drawn at random{aside})"


def _agreement_clause(reading: BlindAgreement, *, labels_name: str | None, left: bool) -> str:
    """One dimension's agreement; ``left`` says whether its reader has cases on the sheet still to read."""
    person = reading.anchor == "person"
    who = _whose(reading.anchor, reading.anchor_names, labels_name) if person else "the model"
    counted = f"{reading.pairs} results from {reading.cases} cases"
    table = reading.table
    caught = ""
    if table is not None and (failed := table.both_fail + table.rater_fail_judge_pass):
        caught = f"; the judge failed {table.both_fail} of the {failed} answers {who} failed"
    # More results come from the cases left on the sheet, or, once it is read through, from more cases.
    more = "" if left else ", from more cases than the run has"
    if reading.kappa is None:
        further = "; read further" if left else ""
        return f"{counted}  kappa undefined: {who} and the judge gave every answer the same verdict{further}{caught}"
    figure = f"kappa {_figure(reading.kappa)}"
    if reading.lower is not None and reading.upper is not None:
        figure += f" [{_figure(reading.lower)}, {_figure(reading.upper)}]"
    bar = _figure(CALIBRATION_MIN_AGREEMENT)
    # A person's line states the tier, as the page beside it does. A model's never does: it says what the judge was
    # checked against (the line above it) and how the check came out, in words.
    said = reading.tier if person else ""
    if reading.calibration == "met" and not person:
        said = f"agrees with the model, above the bar of {bar}"
    elif reading.calibration == "not_met":
        said += f"{': ' if person else ''}agreement is shown below the bar of {bar}"
    elif reading.calibration == "insufficient" and reading.results_needed is not None:
        said += f"{': ' if person else ''}needs {reading.results_needed} more results{more}"
    elif reading.calibration == "undecided" and reading.results_needed is not None:
        said += f"{': ' if person else ''}about {reading.results_needed} more results would decide it{more}"
    return f"{counted}  {figure}  {said or 'not decided'}{caught}"


def _preference_line(preference: BlindPreference, shown: Mapping[str, str]) -> str:
    head = "preference"
    if preference.anchor == "reference":
        head = f"preference of {_who('reference', preference.anchor_names[0])}, not a person"
    if preference.withheld:
        return (
            f"{head}: shown when the first {preference.planned} cases are answered "
            f"({preference.answered} of {preference.planned})"
        )
    candidate, control = (
        shown.get(preference.candidate, preference.candidate),
        shown.get(preference.control, preference.control),
    )
    decided = preference.prefer_candidate + preference.prefer_control
    ties = f" ({preference.ties} tie{'' if preference.ties == 1 else 's'})" if preference.ties else ""
    counted = f"{candidate} over {control} on {preference.prefer_candidate} of {decided}{ties}"
    notes = "" if preference.random else ", not drawn at random"
    notes += f", {preference.missing} with no preference given" if preference.missing else ""
    # A rater who leans to one side whatever is on it costs the test power, never validity: said where it shows.
    lean = exact_sign_test_p(preference.slot_a, preference.slot_choices)
    if lean is not None and lean < SIGNIFICANCE_ALPHA:
        notes += f", output A chosen on {preference.slot_a} of {preference.slot_choices}"
    if preference.verdict is None or preference.p is None:
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
    step = f"; next step: open the {BLIND_READ_HTML} the report writes, answer its cases from the top, and {save}"
    held = len(read.items) * len(read.arms)
    if held < CALIBRATION_MIN_RESULTS:
        step += (
            f"; the calibrated tier's floor is {CALIBRATION_MIN_RESULTS} judged answers per dimension and the read "
            f"holds {held}, so it needs {CALIBRATION_MIN_RESULTS - held} more"
        )
    return step


def _counted(what: str, entries: Sequence[str]) -> list[str]:
    """One line for ``entries``: how many, and the first few."""
    if not entries:
        return []
    named = "; ".join(entries[:_NAMED]) + (f"; and {len(entries) - _NAMED} more" if len(entries) > _NAMED else "")
    return under(f"{len(entries)} {what}: {named}", "  ")


def _review_lines(review: BlindReview, kept: str | None) -> list[str]:
    """What one model's reading cost and how far it got: measurement spend, on its own line."""
    forms = PageForms.off_page()
    asked = f"{review.calls} call(s)"
    if review.calls:
        money = "unpriced" if review.cost_usd is None else forms.money_total(review.cost_usd).text
        asked = f"{money} over {review.calls} call(s)"
        if review.unpriced_calls and review.cost_usd is not None:
            asked += f", {review.unpriced_calls} of them unpriced"
    reused = (
        f", {review.reused} case(s) read from {kept or 'its recorded answers'} and not asked again"
        if review.reused
        else ""
    )
    line = (
        f"reviewer: {_who('reference', review.reviewer)} read the blind read for {asked}{reused}; the cost of checking "
        "the judge, in no candidate's cost"
    )
    lines = [line]
    if review.stopped is not None:
        lines += under(f"stopped before every case was asked, since {review.stopped}", "  ")
    return lines + _counted("case(s) not answered", review.unanswered)


def judge_lines(
    read: BlindRead | None,
    *,
    labels_name: str | None,
    labels_there: bool,
    loose: int = 0,
    unread: Sequence[str] = (),
    shown: Mapping[str, str] | None = None,
    kept: str | None = None,
) -> list[str]:
    """How the judge reads against whoever read it blind, as the quiet print says it.

    :param read: The run's blind read, closed over its labels; None where the run has no judged answer to show.
    :ptype read: BlindRead | None
    :param labels_name: The labels file's name, or None where the judge names none.
    :ptype labels_name: str | None
    :param labels_there: Whether that file exists.
    :ptype labels_there: bool
    :param loose: How many labels in the file are not on this draw and were kept, as not drawn at random.
    :ptype loose: int
    :param unread: The rows that could not be read against this run, each with why.
    :ptype unread: Sequence[str]
    :param shown: The name each arm is printed by, by its key.
    :ptype shown: Mapping[str, str] | None
    :param kept: The file a model reviewer's answers are kept in, by name; None where they are not kept.
    :ptype kept: str | None
    :return: The lines.
    :rtype: list[str]
    """
    if read is None:
        return ["judge: no judged answer to read blind: judge uncalibrated"]
    report = read.report
    samples = [sample for sample in (report.samples if report else []) if sample.sample != "none"]
    reviews = [line for review in read.reviews for line in _review_lines(review, kept)]
    not_read = _counted(f"label(s) in {labels_name} not read", unread)
    why = (
        "Judge(labels=...) names the labels file"
        if labels_name is None
        else f"{labels_name} is not there yet"
        if not labels_there
        else f"{labels_name} holds no label that pairs with this run's answers"
    )
    if not samples and not loose:
        return [
            f"judge: 0 of {len(read.items)} cases read blind: judge uncalibrated ({why}){_first_step(read, labels_name)}",
            *not_read,
            *reviews,
        ]
    assert report is not None  # samples or loose labels come from a closed read
    people = [sample for sample in samples if sample.rater_kind == "person"]
    # Each model's read heads its own lines, below the people's, so no reading sits under another reader's count.
    models = [sample for sample in samples if sample.rater_kind != "person"]
    lines = [f"judge: {_sample_words(sample, labels_name=labels_name)}" for sample in people]
    if not people and not loose:
        # A model's read earns no tier, so the person's first step is never dropped beside it.
        lines.append(
            f"judge: 0 of {len(read.items)} cases read blind by a person: judge uncalibrated ({why})"
            f"{_first_step(read, labels_name)}"
        )

    def clause(reading: BlindAgreement) -> str:
        # Whether the reader has cases left on the sheet: what "read further" and "more results" are decided by.
        own = [sample for sample in samples if sample.rater in reading.anchor_names]
        left = not own or any(sample.answered < sample.items for sample in own)
        return _agreement_clause(reading, labels_name=labels_name, left=left)

    if loose:
        lines.append(
            f"judge: {loose} label(s) in {labels_name} are not on this run's draw (the answers changed since, or it is "
            "an older sheet): kept, each paired with the same output, and not drawn at random"
        )
    lines += not_read
    # A dimension is named as every surface names it (``answer grounded (judged)``), never by its stored id.
    width = max((len(judged_heading(reading.dim)) for reading in report.agreement), default=0)
    by_person = [reading for reading in report.agreement if reading.anchor == "person"]
    lines += [f"  {judged_heading(reading.dim).ljust(width)}  {clause(reading)}" for reading in by_person]
    reviewers = dict.fromkeys(reading.anchor_names[0] for reading in report.agreement if reading.anchor == "reference")
    for reviewer in reviewers:
        lines += [f"judge: {_sample_words(s, labels_name=labels_name)}" for s in models if s.rater == reviewer]
        own = [r for r in report.agreement if r.anchor == "reference" and r.anchor_names[0] == reviewer]
        checked = [check for check in report.reviewer_checks if check.reviewer == reviewer]
        against = "" if checked else ", not yet against people"
        lines += under(
            f"{'beside it, ' if by_person else ''}checked against {_who('reference', reviewer)}{against}:", "  "
        )
        lines += [f"    {judged_heading(reading.dim).ljust(width)}  {clause(reading)}" for reading in own]
        if any(reading.shares_family for reading in own):
            lines += under(
                f"{reviewer} and the judge are one provider's models: a lean they share would not show here", "    "
            )
        for check in checked:
            kappa = "kappa undefined" if check.kappa is None else f"kappa {_figure(check.kappa)}"
            on = judged_heading(check.dim)
            lines += under(f"{reviewer} against people on {on}: {check.outputs} outputs, {kappa}", "    ")
    lines += [f"judge: {_sample_words(s, labels_name=labels_name)}" for s in models if s.rater not in reviewers]
    for reviewer, reason in ((reviewer, reason) for reviewer, why in report.not_checked.items() for reason in why):
        lines += under(f"not checked against {_who('reference', reviewer)}: {reason}", "  ")
    lines += reviews
    lines += [_preference_line(preference, shown or {}) for preference in report.preferences]
    if people:
        furthest = max(people, key=lambda sample: sample.answered)
        if all(sample.sample == "random" for sample in people):
            if furthest.answered < furthest.items:
                lines.append(f"next: keep going from case {furthest.answered + 1} in {BLIND_READ_HTML}")
        else:
            lines.append(f"next: for a random sample, answer {BLIND_READ_HTML} from case 1, in order")
    return lines


#: What the quick result calls.
__all__ = [
    "REVIEWER_PURPOSE",
    "answer_sheet",
    "judge_lines",
    "keep_reviewed",
    "plan_review",
    "read_labels",
]
