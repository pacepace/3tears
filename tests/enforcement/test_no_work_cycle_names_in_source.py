"""Enforcement: a shipped comment carries its why and its numbers, never a work cycle's name.

A comment or docstring that anchors its reason to a lane of a release night (``SG1``, ``TF1``, ``IT3``) names
something no reader of the package can resolve once that night's records are gone: the measurement the name stood for
(its date, its host, its numbers) is what stays. The review rule ("review and finding ids, chunk numbers and review
history never belong in a shipped comment") had no check; this is it, for the lane names the release nights used.
A dated measurement is fine and stays.

A commit sha in a comment or docstring is the same kind of pointer: it sends the reader to history for the reason
instead of stating it, and it names nothing once a branch is rewritten or the history is read elsewhere. The comment
keeps its why inline; the version or date it shipped in may stay.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Final

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]

#: a release night's lane name: its letters, then its number (``TF1``, ``SG1``, ``IT3``, ``FR4``)
_LANE: Final = re.compile(r"\b(?:SG|TF|TL|TT|FR|WS|IT|RP|LM|LG|NS|PL|HB|SDKR|OPS|RF|IG|LT|TC|RC|SYN|GATE)\d+[a-z]?\b")

#: a run of 7 to 40 lowercase hex digits holding a letter and a digit, standing alone: a commit's sha as git prints it
_SHA: Final = re.compile(r"(?<![\w-])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}(?![\w-])")

#: hex runs in shipped prose that name no commit, each with what it is
_NOT_A_COMMIT: Final[dict[str, str]] = {
    "ed25519": "the signature scheme NATS keys use",
    "3f9a1c2b7e04": "an example edge-token qualifier in http_cache's docstrings",
    "0123456789ab": "an example digest in the evals report's naming rules",
    "01a097bc": "a production conversation id, the dated finding text_window and web_fetch answer",
}


def lane_names(text: str) -> list[str]:
    """the lane names ``text`` holds.

    :param text: a module's source
    :ptype text: str
    :return: each one, in order
    :rtype: list[str]
    """
    return [found.group() for found in _LANE.finditer(text)]


def prose(source: str) -> list[str]:
    """the comments and docstrings of a module's source.

    :param source: a module's source
    :ptype source: str
    :return: each comment, then each module, class and function docstring
    :rtype: list[str]
    """
    found = [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    ]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node, clean=False)
            if docstring is not None:
                found.append(docstring)
    return found


def commit_shas(source: str) -> list[str]:
    """the commit shas the comments and docstrings of ``source`` name.

    :param source: a module's source
    :ptype source: str
    :return: each one, in order of its comment or docstring
    :rtype: list[str]
    """
    return [
        found.group() for text in prose(source) for found in _SHA.finditer(text) if found.group() not in _NOT_A_COMMIT
    ]


def test_the_rule_sees_a_lane_name_and_not_a_dated_measurement() -> None:
    assert lane_names("# 1.70 ms on base (TF1, 2026-11-18)") == ["TF1"]
    assert lane_names('"""moved from reports (SG1 2026-11-18)"""') == ["SG1"]
    assert lane_names("# 2026-11-16, the plan read with and without the literals; UTF8, utf-16") == []


def test_no_package_source_names_a_work_cycle() -> None:
    found = {
        str(path.relative_to(_REPO_ROOT)): names
        for root in ("packages/*/src", "packages/agent/*/src")
        for path in sorted(_REPO_ROOT.glob(f"{root}/**/*.py"))
        if (names := lane_names(path.read_text(encoding="utf-8")))
    }
    assert not found, (
        f"shipped source names a work cycle: {found}. Keep the measurement's date, host and numbers inline and drop "
        "the lane name"
    )


def test_the_rule_sees_a_commit_sha_in_a_comment_or_docstring_and_nothing_else() -> None:
    assert commit_shas("x = 1  # a page in two round trips (f93ef9522)\n") == ["f93ef9522"]
    assert commit_shas('def f() -> None:\n    """renamed in 8060d75d7c0ffee1234567890abcdef12345678."""\n') == [
        "8060d75d7c0ffee1234567890abcdef12345678"
    ]
    assert commit_shas('x = "f93ef9522"\n') == []
    assert commit_shas("# 2026-11-15, 1234567 rows, deadbeef-cafe, sha256, abcdefabc, ed25519 keys, #1172\n") == []


def test_no_package_source_names_a_commit() -> None:
    found = {
        str(path.relative_to(_REPO_ROOT)): shas
        for root in ("packages/*/src", "packages/agent/*/src")
        for path in sorted(_REPO_ROOT.glob(f"{root}/**/*.py"))
        if (shas := commit_shas(path.read_text(encoding="utf-8")))
    }
    assert not found, (
        f"shipped comments name commits: {found}. State the reason inline (the version or date it shipped in may "
        "stay) and drop the sha"
    )
