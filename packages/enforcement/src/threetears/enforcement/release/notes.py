"""release notes: moving unreleased notes under the version that releases them.

Two shapes of notes, and each is pure text in, text out:

- a **changelog** with an unreleased heading on top and one dated heading per
  release. Two heading styles are known: Keep a Changelog
  (``## [Unreleased]`` / ``## [X.Y.Z] - YYYY-MM-DD``) and 3tears'
  (``## Unreleased`` / ``## vX.Y.Z -- YYYY-MM-DD``).
- a **prawduct change-log**, whose entries carry a ``<!-- prawduct: ... -->``
  tag line. An entry with a ``scope=`` and no ``release=`` is release-pending;
  releasing it appends ``release=vX.Y.Z``.

Every function refuses (:class:`NotesRefusal`) rather than produce a release with
no notes or a second section for the same version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "CHANGELOG_STYLES",
    "KEEPACHANGELOG",
    "THREETEARS",
    "ChangelogStyle",
    "NotesRefusal",
    "release_changelog",
    "release_prawduct_log",
]

_PRAWDUCT_TAG = re.compile(r"^<!-- prawduct: (.*?) -->\s*$")


class NotesRefusal(Exception):
    """the notes cannot be released as asked; the message says why."""


@dataclass(frozen=True)
class ChangelogStyle:
    """how one changelog spells its headings.

    :param name: style name, as configured
    :ptype name: str
    :param unreleased: the unreleased heading, exactly
    :ptype unreleased: str
    :param version_prefix: what a release heading carries before the version
    :ptype version_prefix: str
    :param version_close: what closes the version itself
    :ptype version_close: str
    :param date_separator: what separates the closed version from the date
    :ptype date_separator: str
    """

    name: str
    unreleased: str
    version_prefix: str
    version_close: str
    date_separator: str

    def heading(self, version: str, today: str) -> str:
        """the heading that releases *version* on *today*.

        :param version: ``X.Y.Z``
        :ptype version: str
        :param today: ``YYYY-MM-DD``
        :ptype today: str
        :return: the heading line
        :rtype: str
        """
        return f"## {self.version_prefix}{version}{self.version_close}{self.date_separator}{today}"

    def matches(self, line: str, version: str) -> bool:
        """whether *line* is a heading for *version*, dated or not.

        :param line: one changelog line
        :ptype line: str
        :param version: ``X.Y.Z``
        :ptype version: str
        :return: ``True`` for that version's heading
        :rtype: bool
        """
        opener = f"## {self.version_prefix}{version}{self.version_close}"
        return line.rstrip() == opener or line.startswith(opener + " ")


#: ``## [Unreleased]`` / ``## [X.Y.Z] - YYYY-MM-DD``
KEEPACHANGELOG = ChangelogStyle("keepachangelog", "## [Unreleased]", "[", "]", " - ")
#: ``## Unreleased`` / ``## vX.Y.Z -- YYYY-MM-DD``
THREETEARS = ChangelogStyle("3tears", "## Unreleased", "v", "", " -- ")

#: every known style by its configured name.
CHANGELOG_STYLES = {style.name: style for style in (KEEPACHANGELOG, THREETEARS)}


def _trimmed(lines: list[str]) -> list[str]:
    """*lines* without leading or trailing blank lines.

    :param lines: lines
    :ptype lines: list[str]
    :return: the trimmed slice
    :rtype: list[str]
    """
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _subsections(lines: list[str]) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """the prose before the first ``###`` heading, then each ``###`` heading and its body.

    :param lines: a section's body
    :ptype lines: list[str]
    :return: preamble, and ``(heading, body)`` pairs in order
    :rtype: tuple[list[str], list[tuple[str, list[str]]]]
    """
    preamble: list[str] = []
    parts: list[tuple[str, list[str]]] = []
    for line in lines:
        if line.startswith("### "):
            parts.append((line.strip(), []))
        elif parts:
            parts[-1][1].append(line)
        else:
            preamble.append(line)
    return _trimmed(preamble), [(name, _trimmed(body)) for name, body in parts]


def _merged(section: list[str], unreleased: list[str]) -> list[str]:
    """folds unreleased notes into an existing section, subsection by subsection.

    ``### Added`` items join the section's ``### Added``; a new subsection goes
    last; loose prose follows the section's own.

    :param section: the existing section's body
    :ptype section: list[str]
    :param unreleased: the unreleased body
    :ptype unreleased: list[str]
    :return: the merged body
    :rtype: list[str]
    """
    pre_a, parts_a = _subsections(section)
    pre_b, parts_b = _subsections(unreleased)
    preamble = pre_a + ([""] if pre_a and pre_b else []) + pre_b
    merged = [(name, list(body)) for name, body in parts_a]
    for name, body in parts_b:
        for index, (existing, existing_body) in enumerate(merged):
            if existing == name:
                merged[index] = (existing, existing_body + body)
                break
        else:
            merged.append((name, body))
    out: list[str] = []
    if preamble:
        out += preamble + [""]
    for name, body in merged:
        out += [name, ""] + body + [""]
    return out


def _next_heading(lines: list[str], start: int) -> int:
    """index of the first ``## `` heading at or after *start*, or ``len(lines)``.

    :param lines: changelog lines
    :ptype lines: list[str]
    :param start: first index to look at
    :ptype start: int
    :return: heading index
    :rtype: int
    """
    return next((index for index in range(start, len(lines)) if lines[index].startswith("## ")), len(lines))


def release_changelog(text: str, style: ChangelogStyle, version: str, today: str, absorb: str | None = None) -> str:
    """releases a changelog's unreleased notes as *version*, dated *today*.

    With *absorb*, the existing section for that version -- one declared but never
    released -- is taken over: retitled *version*, dated, and the unreleased notes
    folded into it. ``release`` absorbs the version it releases; a minor bump from
    a declared-but-untagged patch absorbs that patch.

    :param text: the changelog
    :ptype text: str
    :param style: its heading style
    :ptype style: ChangelogStyle
    :param version: the version being released
    :ptype version: str
    :param today: ``YYYY-MM-DD``
    :ptype today: str
    :param absorb: an unreleased version whose section becomes *version*'s, if any
    :ptype absorb: str | None
    :return: the released changelog
    :rtype: str
    :raises NotesRefusal: if the unreleased heading is missing or repeated,
        *version* already has a section, or there are no notes to release
    """
    lines = text.split("\n")
    starts = [index for index, line in enumerate(lines) if line.rstrip() == style.unreleased]
    if len(starts) != 1:
        raise NotesRefusal(f"the changelog needs exactly one `{style.unreleased}` heading; found {len(starts)}")
    top = starts[0]
    after = _next_heading(lines, top + 1)
    body = _trimmed(lines[top + 1 : after])
    head = lines[: top + 1] + [""]
    if version != absorb and any(style.matches(line, version) for line in lines):
        raise NotesRefusal(f"the changelog already has a section for {version}")
    found = [index for index, line in enumerate(lines) if absorb is not None and style.matches(line, absorb)]
    if found:
        start = found[0]
        if start < top:
            raise NotesRefusal(f"the {absorb} section sits above `{style.unreleased}`; fix the order by hand")
        end = _next_heading(lines, start + 1)
        section = _merged(_trimmed(lines[start + 1 : end]), body)
        if not _trimmed(section):
            raise NotesRefusal(f"the {absorb} section and `{style.unreleased}` are both empty; there are no notes")
        lines = head + lines[after:start] + [style.heading(version, today), ""] + section + lines[end:]
    else:
        if not body:
            raise NotesRefusal(f"nothing under `{style.unreleased}` -- write the release notes first")
        lines = head + [style.heading(version, today), ""] + body + [""] + lines[after:]
    return "\n".join(lines).rstrip("\n") + "\n"


def release_prawduct_log(text: str, version: str, absorb: str | None = None) -> tuple[str, int]:
    """marks every release-pending prawduct change-log entry ``release=v<version>``.

    With *absorb*, entries already marked for that unreleased version move to
    *version* too.

    :param text: the change-log
    :ptype text: str
    :param version: the version being released
    :ptype version: str
    :param absorb: an unreleased version whose entries become *version*'s, if any
    :ptype absorb: str | None
    :return: the released change-log, and how many entries it marked
    :rtype: tuple[str, int]
    :raises NotesRefusal: if no entry is release-pending
    """
    absorbed = None
    if absorb is not None and absorb != version:
        absorbed = re.compile(r"(\brelease=)v" + re.escape(absorb) + r"(?=\s|\||$)")
    out: list[str] = []
    marked = 0
    for line in text.split("\n"):
        matched = _PRAWDUCT_TAG.match(line)
        keys = matched.group(1) if matched else ""
        if matched and "scope=" in keys and "release=" not in keys:
            line = f"<!-- prawduct: {keys} | release=v{version} -->"
            marked += 1
        elif matched and absorbed is not None and absorbed.search(keys):
            line = f"<!-- prawduct: {absorbed.sub(r'\g<1>v' + version, keys)} -->"
            marked += 1
        out.append(line)
    if marked == 0:
        raise NotesRefusal("the change-log has no release-pending entry (a `scope=` tag with no `release=`)")
    return "\n".join(out), marked
