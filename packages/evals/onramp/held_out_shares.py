"""How much of each held-out app's shipped data its planted defect reaches: ``python held_out_shares.py``.

Each held-out app is loaded twice from a temporary copy, as a trial lays it out: as shipped (``apps/<app>/``), and with
its fixed twin laid over it (``twins/<app>/``). Every case in the app's data is run through both, and :func:`measure`
returns a :class:`Reach`:

- ``reached``: the cases where the shipped app's answer is wrong by the data's labels (for the graded apps, where the
  twin's graded reading is better than the app's);
- ``twin_off``: the cases where the twin's answer is wrong by the same labels (for faqrank and digest, whose labels
  grade rather than decide, where the app's reading is better than the twin's). A sound held-out app has none.

``packages/evals/tests/test_onramp_held_out.py`` holds every app in ``held-out.jsonl`` to these, and
``held-out-defects.md`` records the shares this prints. Standard library only, no network, and nothing here is copied
into a trial.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
#: The held-out apps, in ``apps.jsonl``'s format.
HELD_OUT = HERE / "held-out.jsonl"
#: How close weatherboard's board must be to the logger's Celsius, as its README states.
TOLERANCE_C = 0.5


@dataclass(frozen=True)
class Reach:
    reached: int
    total: int
    twin_off: int
    what: str

    @property
    def share(self) -> float:
        return self.reached / self.total


def held_out_apps() -> list[str]:
    """The apps ``held-out.jsonl`` lists."""
    return [json.loads(line)["app"] for line in HELD_OUT.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy_app(app: str, dest: Path, *, twin: bool) -> Path:
    """A copy of ``app`` at ``dest`` as a trial has it, with its twin laid over it when ``twin``; no bytecode copied."""
    no_cache = shutil.ignore_patterns("__pycache__")
    shutil.copytree(HERE / "apps" / app, dest, ignore=no_cache)
    if twin:
        shutil.copytree(HERE / "twins" / app, dest, dirs_exist_ok=True, ignore=no_cache)
    return dest


def _rows(path: Path) -> list[dict[str, Any]]:  # Any: the apps' own JSON lines
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load(root: Path, module: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, root / f"{module}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {root / module}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def parcelrate(app: ModuleType, twin: ModuleType, root: Path) -> Reach:
    rows = _rows(root / "data" / "courier_invoice.jsonl")

    def off(module: ModuleType) -> int:
        return sum(
            module.quote(r["weight_kg"], r["zone"], signed=r["signed"]).pence != r["charged_pence"] for r in rows
        )

    return Reach(off(app), len(rows), off(twin), "invoice lines whose quote is not the courier's charge")


def weatherboard(app: ModuleType, twin: ModuleType, root: Path) -> Reach:
    rows = _rows(root / "data" / "dual_logged.jsonl")

    def off(module: ModuleType) -> int:
        return sum(abs(module.display_c(r["temp_f"]) - r["temp_c"]) > TOLERANCE_C + 1e-9 for r in rows)

    return Reach(
        off(app), len(rows), off(twin), "dual-logged readings shown more than half a degree from the logger's Celsius"
    )


def docnames(app: ModuleType, twin: ModuleType, root: Path) -> Reach:
    rows = _rows(root / "data" / "archive_index.jsonl")

    def off(module: ModuleType) -> int:
        return sum(module.stored_name(r["title"], r["uploaded"], r["extension"]) != r["archive_name"] for r in rows)

    return Reach(off(app), len(rows), off(twin), "archive documents whose stored name is not the archive's")


def previews(app: ModuleType, twin: ModuleType, root: Path) -> Reach:
    rows = _rows(root / "data" / "signed_off.jsonl")

    def off(module: ModuleType) -> int:
        return sum(module.preview(r["body"], r["channel"]) != r["approved"] for r in rows)

    return Reach(off(app), len(rows), off(twin), "signed-off messages whose preview is not the approved one")


def tickets(app: ModuleType, twin: ModuleType, root: Path) -> Reach:
    rows = _rows(root / "data" / "export.jsonl")

    def seen(module: ModuleType, row: dict[str, Any]) -> tuple[str, int]:  # Any: an export row
        ticket = module.replay(row["ticket"], row["events"])
        return ticket.status, ticket.reopens

    def off(module: ModuleType) -> int:
        return sum(seen(module, r) != (r["old_desk"]["status"], r["old_desk"]["reopens"]) for r in rows)

    status = sum(seen(app, r)[0] != r["old_desk"]["status"] for r in rows)
    return Reach(
        off(app), len(rows), off(twin), f"tickets off the old desk's status or reopens ({status} off on status alone)"
    )


def _ndcg(ranked: list[str], grades: dict[str, int], k: int = 5) -> float:
    gained = sum(grades.get(aid, 0) / math.log2(i + 2) for i, aid in enumerate(ranked[:k]))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(grades.values(), reverse=True)[:k]))
    return gained / ideal


def faqrank(app: ModuleType, twin: ModuleType, root: Path) -> Reach:
    queries = {r["id"]: r["query"] for r in _rows(root / "data" / "queries.jsonl")}
    grades = {r["query_id"]: r["grades"] for r in _rows(root / "data" / "judgments.jsonl")}
    worse = better = top = 0
    means = [0.0, 0.0]
    for qid, query in queries.items():
        shipped = [h.id for h in app.search(query)]
        fixed = [h.id for h in twin.search(query)]
        a, t = _ndcg(shipped, grades[qid]), _ndcg(fixed, grades[qid])
        means[0] += a / len(queries)
        means[1] += t / len(queries)
        worse += t > a + 1e-9
        better += a > t + 1e-9
        best = max(grades[qid], key=grades[qid].__getitem__)
        top += fixed[:1] == [best] and shipped[:1] != [best]
    return Reach(
        worse,
        len(queries),
        better,
        f"queries with a lower NDCG@5 than the twin's (mean {means[0]:.3f} against {means[1]:.3f}; "
        f"{top} lose the top answer the twin finds)",
    )


def digest(app: ModuleType, twin: ModuleType, root: Path) -> Reach:
    rows = _rows(root / "data" / "key_facts.jsonl")

    def carried(module: ModuleType, row: dict[str, Any]) -> float:  # Any: a key-facts row
        text = module.digest_file(root / "data" / row["report"]).lower()
        return sum(fact.lower() in text for fact in row["facts"]) / len(row["facts"])

    worse = sum(carried(twin, r) > carried(app, r) for r in rows)
    better = sum(carried(app, r) > carried(twin, r) for r in rows)
    means = [sum(carried(m, r) for r in rows) / len(rows) for m in (app, twin)]
    return Reach(
        worse,
        len(rows),
        better,
        f"reports whose digest carries fewer key facts than the twin's (mean share {means[0]:.3f} against "
        f"{means[1]:.3f})",
    )


def statements(app: ModuleType, twin: ModuleType, root: Path) -> Reach:
    ledger = _rows(root / "data" / "ledger.jsonl")
    files = sorted({r["statement"] for r in ledger})
    expected = {
        name: [(r["date"], r["description"], r["pence"]) for r in ledger if r["statement"] == name] for name in files
    }

    def read(module: ModuleType, name: str) -> list[tuple[str, str, int]]:
        return [(t.date.isoformat(), t.description, t.pence) for t in module.read_statement_file(root / "data" / name)]

    def off(module: ModuleType) -> int:
        found = {name: read(module, name) for name in files}
        if any(len(found[name]) != len(expected[name]) for name in files):
            return len(ledger)
        return sum(got != want for name in files for got, want in zip(found[name], expected[name], strict=True))

    files_off = sum(read(app, name) != expected[name] for name in files)
    return Reach(
        off(app),
        len(ledger),
        off(twin),
        f"ledger transactions the import gets wrong ({files_off} of {len(files)} statements)",
    )


#: Each held-out app's module, and how its reach is measured.
SHARES: dict[str, tuple[str, Callable[[ModuleType, ModuleType, Path], Reach]]] = {
    "parcelrate": ("rates", parcelrate),
    "weatherboard": ("board", weatherboard),
    "docnames": ("naming", docnames),
    "previews": ("previews", previews),
    "tickets": ("desk", tickets),
    "faqrank": ("search", faqrank),
    "digest": ("digest", digest),
    "statements": ("importer", statements),
}


def measure(app: str, scratch: Path) -> Reach:
    """``app``'s reach over its shipped data, from copies made under ``scratch``."""
    module, share = SHARES[app]
    shipped_root = copy_app(app, scratch / f"{app}-app", twin=False)
    fixed_root = copy_app(app, scratch / f"{app}-twin", twin=True)
    shipped = _load(shipped_root, module, f"held_out_{app}_app")
    fixed = _load(fixed_root, module, f"held_out_{app}_twin")
    return share(shipped, fixed, shipped_root)


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        for app in held_out_apps():
            reach = measure(app, Path(scratch))
            print(
                f"{app}: {reach.reached} of {reach.total} ({reach.share:.0%}) {reach.what}; twin off on {reach.twin_off}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
