"""collection-census enforcement domain: one table, one line of classes, one write-generation declaration.

A table's write generation lets a pod trust "the generation did not move" as "nothing changed" only
while every class that writes the table declares the same thing. Two unrelated classes for one table
are two places that declaration can disagree, and when they live in different repositories no check
inside one of them sees the other. So:

- :func:`run_census` imports every module under the source trees it is given, in a process of its
  own, and describes every ``BaseCollection`` subclass they define. With ``framework=True`` the
  installed 3tears framework's classes join the census, so a product repository's census sees a
  class of its own that names a table the framework already has a class for.
- :func:`find_census_problems` turns a census into the rule's violations: the classes that name one
  table must descend from one class that names it (a subclass adding queries is the same class, the
  way ``HubGroupMemberCollection`` extends ``GroupMemberCollection``), and they must all declare the
  same write generation. A concrete class of the trees' own whose table cannot be read off the class
  is listed with why, in ``named_per_instance``.

3tears runs it over its own packages; a product runs it over its own ``src`` with ``framework=True``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = ["find_census_problems", "run_census"]


def run_census(roots: list[Path], *, framework: bool = False) -> dict[str, Any]:
    """enumerate every collection class under ``roots``, in a process of its own.

    :param roots: source roots to import
    :ptype roots: list[Path]
    :param framework: whether the installed 3tears framework's classes join the census
    :ptype framework: bool
    :return: the census: ``classes``, ``import_failures`` and ``framework_import_failures``
    :rtype: dict[str, Any]
    :raises AssertionError: when the census process fails
    """
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    arguments = [*(["--framework"] if framework else []), *(str(root) for root in roots)]
    completed = subprocess.run(  # noqa: S603 - this interpreter, this package's own module
        [sys.executable, "-m", "threetears.enforcement.collection_census", *arguments],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr
    result: dict[str, Any] = json.loads(completed.stdout)
    return result


def find_census_problems(census: dict[str, Any], named_per_instance: dict[str, str]) -> list[str]:
    """every way a census breaks the one-class-per-table rule, or its enumeration is incomplete.

    The framework's own classes (``framework`` in a record) are judged by the framework's census;
    here they only take part in the comparison by table.

    :param census: what :func:`run_census` found
    :ptype census: dict[str, Any]
    :param named_per_instance: concrete classes of the trees' own whose table is named per
        instance, by qualified name, with why
    :ptype named_per_instance: dict[str, str]
    :return: one line per problem
    :rtype: list[str]
    """
    problems = [f"module did not import, so its classes were not enumerated: {f}" for f in census["import_failures"]]
    classes: list[dict[str, Any]] = census["classes"]
    names = {record["name"] for record in classes}
    has_subclass = {ancestor for record in classes for ancestor in record["ancestors"]}
    by_table: dict[str, list[dict[str, Any]]] = {}
    for record in classes:
        if record["table"] is not None:
            by_table.setdefault(record["table"], []).append(record)
        if record.get("framework", False):
            continue
        if record["declaration"] == "invalid":
            problems.append(f"{record['name']}: write_generation is not one of the three declarations")
        if record["declaration"] == "opted_out" and not str(record["reason"] or "").strip():
            problems.append(f"{record['name']}: NoWriteGeneration without a reason")
        if (
            record["table"] is None
            and not record["abstract"]
            and record["name"] not in has_subclass
            and record["name"] not in named_per_instance
        ):
            problems.append(
                f"{record['name']}: its table cannot be read off the class; name it in a schema or a "
                f"table_name property, or list it as named per instance with why"
            )
    for stale in sorted(set(named_per_instance) - names):
        problems.append(f"named per instance lists {stale}, which the census no longer finds")
    for table, records in sorted(by_table.items()):
        if all(record.get("framework", False) for record in records):
            continue
        members = {record["name"] for record in records}
        roots = sorted(record["name"] for record in records if not set(record["ancestors"]) & members)
        if len(roots) > 1:
            problems.append(f"table {table!r} is named by unrelated classes: {', '.join(roots)}")
        kinds = sorted({record["declaration"] for record in records})
        if len(kinds) > 1:
            problems.append(
                f"table {table!r} has classes declaring different write generations ({', '.join(kinds)}): "
                f"{', '.join(sorted(members))}"
            )
    return problems
