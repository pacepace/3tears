"""How much of each held-out app's shipped data its planted defect reaches: ``python held_out_shares.py``.

Each held-out app is loaded twice from a temporary copy, as a trial lays it out: as shipped (``apps/<app>/``), and with
its fixed twin laid over it (``twins/<app>/``). Every case in the app's data is run through both, and a case counts as
reached when the shipped app's answer differs from the twin's on what that data's labels judge (for the graded apps,
when the twin's graded reading is better). ``held-out-defects.md`` records the shares this prints.

Standard library only, no network, and nothing here is copied into a trial.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent


def _rows(path: Path) -> list[dict[str, Any]]:  # Any: the apps' own JSON lines
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load(scratch: Path, app: str, module: str, *, twin: bool) -> tuple[ModuleType, Path]:
    root = scratch / f"{app}-{'twin' if twin else 'app'}"
    shutil.copytree(HERE / "apps" / app, root, ignore=shutil.ignore_patterns("__pycache__"))
    if twin:
        shutil.copytree(HERE / "twins" / app, root, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    name = f"held_out_{app}_{'twin' if twin else 'app'}"
    spec = importlib.util.spec_from_file_location(name, root / f"{module}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {root / module}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded, root


def parcelrate(app: ModuleType, twin: ModuleType, root: Path) -> tuple[int, int, str]:
    rows = _rows(root / "data" / "courier_invoice.jsonl")
    wrong = sum(app.quote(r["weight_kg"], r["zone"], signed=r["signed"]).pence != r["charged_pence"] for r in rows)
    assert all(twin.quote(r["weight_kg"], r["zone"], signed=r["signed"]).pence == r["charged_pence"] for r in rows)
    return wrong, len(rows), "invoice lines whose quote is not the courier's charge"


def weatherboard(app: ModuleType, twin: ModuleType, root: Path) -> tuple[int, int, str]:
    rows = _rows(root / "data" / "dual_logged.jsonl")
    off = sum(abs(app.display_c(r["temp_f"]) - r["temp_c"]) > 0.5 + 1e-9 for r in rows)
    assert all(abs(twin.display_c(r["temp_f"]) - r["temp_c"]) <= 0.5 + 1e-9 for r in rows)
    return off, len(rows), "dual-logged readings shown more than half a degree from the logger's Celsius"


def docnames(app: ModuleType, twin: ModuleType, root: Path) -> tuple[int, int, str]:
    rows = _rows(root / "data" / "archive_index.jsonl")
    wrong = sum(app.stored_name(r["title"], r["uploaded"], r["extension"]) != r["archive_name"] for r in rows)
    assert all(twin.stored_name(r["title"], r["uploaded"], r["extension"]) == r["archive_name"] for r in rows)
    return wrong, len(rows), "archive documents whose stored name is not the archive's"


def previews(app: ModuleType, twin: ModuleType, root: Path) -> tuple[int, int, str]:
    rows = _rows(root / "data" / "signed_off.jsonl")
    wrong = sum(app.preview(r["body"], r["channel"]) != r["approved"] for r in rows)
    assert all(twin.preview(r["body"], r["channel"]) == r["approved"] for r in rows)
    return wrong, len(rows), "signed-off messages whose preview is not the approved one"


def tickets(app: ModuleType, twin: ModuleType, root: Path) -> tuple[int, int, str]:
    rows = _rows(root / "data" / "export.jsonl")

    def seen(module: ModuleType, row: dict[str, Any]) -> tuple[str, int]:  # Any: an export row
        ticket = module.replay(row["ticket"], row["events"])
        return ticket.status, ticket.reopens

    wrong = sum(seen(app, r) != (r["old_desk"]["status"], r["old_desk"]["reopens"]) for r in rows)
    status = sum(seen(app, r)[0] != r["old_desk"]["status"] for r in rows)
    assert all(seen(twin, r) == (r["old_desk"]["status"], r["old_desk"]["reopens"]) for r in rows)
    return wrong, len(rows), f"tickets off the old desk's status or reopens ({status} off on status alone)"


def _ndcg(ranked: list[str], grades: dict[str, int], k: int = 5) -> float:
    gained = sum(grades.get(aid, 0) / math.log2(i + 2) for i, aid in enumerate(ranked[:k]))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(grades.values(), reverse=True)[:k]))
    return gained / ideal


def faqrank(app: ModuleType, twin: ModuleType, root: Path) -> tuple[int, int, str]:
    queries = {r["id"]: r["query"] for r in _rows(root / "data" / "queries.jsonl")}
    grades = {r["query_id"]: r["grades"] for r in _rows(root / "data" / "judgments.jsonl")}
    worse = top = 0
    means = [0.0, 0.0]
    for qid, query in queries.items():
        shipped = [h.id for h in app.search(query)]
        fixed = [h.id for h in twin.search(query)]
        a, t = _ndcg(shipped, grades[qid]), _ndcg(fixed, grades[qid])
        means[0] += a / len(queries)
        means[1] += t / len(queries)
        worse += t > a + 1e-9
        best = max(grades[qid], key=grades[qid].__getitem__)
        top += fixed[:1] == [best] and shipped[:1] != [best]
    return (
        worse,
        len(queries),
        (
            f"queries with a lower NDCG@5 than the twin's (mean {means[0]:.3f} against {means[1]:.3f}; "
            f"{top} lose the top answer the twin finds)"
        ),
    )


def digest(app: ModuleType, twin: ModuleType, root: Path) -> tuple[int, int, str]:
    rows = _rows(root / "data" / "key_facts.jsonl")

    def carried(module: ModuleType, row: dict[str, Any]) -> float:  # Any: a key-facts row
        text = module.digest_file(root / "data" / row["report"]).lower()
        return sum(fact.lower() in text for fact in row["facts"]) / len(row["facts"])

    worse = sum(carried(twin, r) > carried(app, r) for r in rows)
    means = [sum(carried(m, r) for r in rows) / len(rows) for m in (app, twin)]
    better = sum(carried(app, r) > carried(twin, r) for r in rows)
    return (
        worse,
        len(rows),
        (
            f"reports whose digest carries fewer key facts than the twin's (mean share {means[0]:.3f} against "
            f"{means[1]:.3f}; {better} where the app carries more)"
        ),
    )


def statements(app: ModuleType, twin: ModuleType, root: Path) -> tuple[int, int, str]:
    ledger = _rows(root / "data" / "ledger.jsonl")
    files = sorted({r["statement"] for r in ledger})

    def read(module: ModuleType, name: str) -> list[tuple[str, str, int]]:
        return [(t.date.isoformat(), t.description, t.pence) for t in module.read_statement_file(root / "data" / name)]

    expected = {
        name: [(r["date"], r["description"], r["pence"]) for r in ledger if r["statement"] == name] for name in files
    }
    assert all(read(twin, name) == expected[name] for name in files)
    rows_off = sum(got != want for name in files for got, want in zip(read(app, name), expected[name], strict=True))
    files_off = sum(read(app, name) != expected[name] for name in files)
    return rows_off, len(ledger), f"ledger transactions the import gets wrong ({files_off} of {len(files)} statements)"


SHARES: dict[str, tuple[str, Callable[[ModuleType, ModuleType, Path], tuple[int, int, str]]]] = {
    "parcelrate": ("rates", parcelrate),
    "weatherboard": ("board", weatherboard),
    "docnames": ("naming", docnames),
    "previews": ("previews", previews),
    "tickets": ("desk", tickets),
    "faqrank": ("search", faqrank),
    "digest": ("digest", digest),
    "statements": ("importer", statements),
}


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        for app_name, (module, share) in SHARES.items():
            shipped, root = _load(Path(scratch), app_name, module, twin=False)
            fixed, _ = _load(Path(scratch), app_name, module, twin=True)
            reached, total, what = share(shipped, fixed, root)
            print(f"{app_name}: {reached} of {total} ({reached / total:.0%}) {what}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
