"""bump a repository's release version, and move its release notes under it, in one step.

**The rule** (3tears BLD-7QM3; every aibots repo by owner ruling, 2026-09-30):
``minor`` for any new public API or new feature -- the API-growth gate refuses a
patch that grew the API -- ``patch`` for everything else, ``major`` only on the
owner's word.

**Actions.**

- ``patch`` / ``minor`` / ``major``: move the version, release the notes under
  it, relock.
- ``release``: release the version ALREADY declared -- one declared before its
  release was cut. Dates its notes, folds in anything written under the
  unreleased heading since. No version change, no relock.
- ``sync``: bring every lockstep location back to the declared version (drift
  repair; no notes, no tag checks).
- ``verify X.Y.Z``: confirm every lockstep location says ``X.Y.Z``; edits nothing.

**What a bump touches** -- the whole list: every version file (the canonical one
and any lockstep members), smoke-test assertions, a bake file, intra-family
bounds, the release notes, and ``uv.lock``, relocked with ``uvx uv@latest lock``
(not the local uv: an older uv writes an absolute exclude-newer into the lock).

**Local 3tears mode.** When the root ``pyproject.toml`` or ``uv.lock`` differs
from ``HEAD`` -- a sources block and a local resolution that must never be
committed -- the committable bump is built from ``HEAD`` in a scratch worktree
whose parent mirrors this checkout's siblings, relocked there in released mode,
and STAGED. The working tree gets the same edits and a local relock. Commit with a
plain ``git commit`` and no pathspec: a pathspec commits the working-tree files.

**It changes nothing unless it can change everything.** Every check runs before
the first edit; a failure after it restores every file it touched and unstages
them, so a refused or failed run can simply be re-run. It refuses:

- a version line edited by hand (the working tree's version differs from ``HEAD``'s);
- staged changes to any file it touches, or uncommitted edits to any of them other
  than the root ``pyproject.toml`` / ``uv.lock`` (local mode);
- a ``patch`` while the declared version is untagged -- release it, or tag it if it
  shipped. A ``minor``/``major`` from an untagged version proceeds when that
  version's minor line has a tag: the API gate's remedy for growth after a
  declared patch. The untagged version's notes move to the new one;
- a version whose tag or notes section already exists, and a release with no notes.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from threetears.enforcement.release.config import ReleaseConfig
from threetears.enforcement.release.lockstep import lockstep_mismatches, rewrite_lockstep_file
from threetears.enforcement.release.notes import (
    CHANGELOG_STYLES,
    NotesRefusal,
    release_changelog,
    release_prawduct_log,
)
from threetears.enforcement.release.versions import (
    Version,
    format_version,
    parse_version,
    project_version,
    release_tags,
    run_git,
)

__all__ = ["BUMP_KINDS", "Refusal", "next_version", "run_release"]

#: the actions that move the version.
BUMP_KINDS = ("patch", "minor", "major")

#: files that may differ from ``HEAD`` without being a refusal: local 3tears mode.
_LOCAL_MODE_FILES = ("pyproject.toml", "uv.lock")


class Refusal(Exception):
    """the release tool will not proceed; the message says why and what to do."""


def next_version(current: Version, kind: str) -> Version:
    """the version a *kind* bump moves *current* to.

    :param current: the declared version
    :ptype current: Version
    :param kind: ``patch``, ``minor`` or ``major``
    :ptype kind: str
    :return: the bumped version
    :rtype: Version
    :raises ValueError: for any other *kind*
    """
    major, minor, patch = current
    bumped = {"major": (major + 1, 0, 0), "minor": (major, minor + 1, 0), "patch": (major, minor, patch + 1)}
    if kind not in bumped:
        raise ValueError(f"unknown bump kind {kind!r}")
    return bumped[kind]


@dataclass
class _Plan:
    """everything a run will do, decided before anything is written.

    :param action: the requested action
    :ptype action: str
    :param current: the declared version
    :ptype current: str
    :param new: the version after the run
    :ptype new: str
    :param absorb: an unreleased version whose notes become *new*'s, if any
    :ptype absorb: str | None
    :param touched: repo-relative files the run may write
    :ptype touched: list[str]
    :param notes_text: the released notes, or ``None`` when notes are not touched
    :ptype notes_text: str | None
    :param local_mode: whether the committable bump must be built from ``HEAD``
    :ptype local_mode: bool
    """

    action: str
    current: str
    new: str
    absorb: str | None
    touched: list[str] = field(default_factory=list)
    notes_text: str | None = None
    local_mode: bool = False


class _Run:
    """one invocation: its checks, its edits, and its guarantee to undo them."""

    def __init__(self, config: ReleaseConfig, today: str, out: TextIO, err: TextIO) -> None:
        """binds the run to one repository.

        :param config: the repository's release configuration
        :ptype config: ReleaseConfig
        :param today: ``YYYY-MM-DD`` written into the notes
        :ptype today: str
        :param out: progress and instructions
        :ptype out: TextIO
        :param err: warnings
        :ptype err: TextIO
        :return: nothing
        :rtype: None
        """
        self.config = config
        self.root = config.root
        self.today = today
        self.out = out
        self.err = err

    def say(self, line: str = "") -> None:
        """writes one line of progress.

        :param line: text
        :ptype line: str
        :return: nothing
        :rtype: None
        """
        self.out.write(line + "\n")
        self.out.flush()

    # ------------------------------------------------------------------ checks

    def _git_ok(self, *args: str) -> bool:
        """whether a git command exits zero in the repository.

        :param args: git arguments
        :ptype args: str
        :return: ``True`` on exit status 0
        :rtype: bool
        """
        return run_git(self.root, *args).ok

    def _tag_exists(self, version: str) -> bool:
        """whether ``v<version>`` is a tag here.

        :param version: ``X.Y.Z``
        :ptype version: str
        :return: ``True`` when the tag exists
        :rtype: bool
        """
        return self._git_ok("rev-parse", "-q", "--verify", f"refs/tags/v{version}")

    def _head_version(self, path: str) -> str:
        """the version *path* declares at ``HEAD``.

        :param path: repo-relative ``pyproject.toml``
        :ptype path: str
        :return: ``X.Y.Z``
        :rtype: str
        :raises Refusal: if ``HEAD`` has no such file or it declares no version
        """
        shown = run_git(self.root, "show", f"HEAD:{path}")
        if not shown.ok:
            raise Refusal(f"{path} is not committed at HEAD. Commit it first; the bump commit carries only the bump.")
        try:
            return format_version(project_version(shown.text))
        except ValueError as exc:
            raise Refusal(f"{path} at HEAD: {exc}") from None

    def _check_repository(self) -> None:
        """refuses a root that is not the top of a git checkout.

        :return: nothing
        :rtype: None
        :raises Refusal: if *root* is not a checkout's top level
        """
        top = run_git(self.root, "rev-parse", "--show-toplevel")
        if not top.ok or Path(top.text).resolve() != self.root.resolve():
            raise Refusal(f"{self.root} is not the top of a git checkout")

    def _check_versions(self) -> str:
        """the declared version, refusing a hand edit or lockstep drift.

        :return: ``X.Y.Z``
        :rtype: str
        :raises Refusal: if a version line differs from ``HEAD``'s
        """
        current = ""
        for path in self.config.version_files:
            try:
                working = format_version(project_version((self.root / path).read_text(encoding="utf-8")))
            except ValueError as exc:
                raise Refusal(f"{path}: {exc}") from None
            head = self._head_version(path)
            if working != head:
                raise Refusal(
                    f"the version line is dirty: {path} says {working} and HEAD says {head}. This tool owns that "
                    f"line -- revert the hand edit and run it again."
                )
            current = current or working
        return current

    def _check_touched(self, touched: list[str]) -> None:
        """refuses staged or uncommitted changes to anything the run writes.

        :param touched: repo-relative files the run may write
        :ptype touched: list[str]
        :return: nothing
        :rtype: None
        :raises Refusal: if any is staged, untracked, or edited outside local-mode files
        """
        if not self._git_ok("diff", "--cached", "--quiet", "--", *touched):
            raise Refusal(f"{', '.join(touched)}: something here has STAGED changes. Commit or unstage them first.")
        for path in touched:
            if not self._git_ok("ls-files", "--error-unmatch", "--", path):
                raise Refusal(f"{path} is not tracked by git. Commit it first.")
            if path not in _LOCAL_MODE_FILES and not self._git_ok("diff", "--quiet", "--", path):
                raise Refusal(
                    f"{path} has uncommitted edits. Commit them first, so the bump commit carries only the bump."
                )

    def _target(self, action: str, current: str) -> tuple[str, str | None]:
        """the version the run produces, and the unreleased version it absorbs.

        :param action: requested action
        :ptype action: str
        :param current: the declared version
        :ptype current: str
        :return: ``(new, absorb)``
        :rtype: tuple[str, str | None]
        :raises Refusal: if the tag state forbids the action
        """
        command = self.config.command
        if action == "sync":
            return current, None
        if action == "release":
            if self._tag_exists(current):
                raise Refusal(
                    f"v{current} is already tagged, so it is released. Bump instead: patch, or minor for new API."
                )
            return current, current
        parsed = parse_version(current)
        absorb: str | None = None
        if not self._tag_exists(current):
            line_tagged = any(version[:2] == parsed[:2] for version in release_tags(self.root).values())
            if action == "patch" or not line_tagged:
                raise Refusal(
                    f"{current} has no v{current} tag, so it is declared but not released.\n"
                    f"  - not shipped yet:  {command} release\n"
                    f"  - already shipped:  tag it on its main merge commit, then bump\n"
                    f"  - tag exists on the remote but not here:  git fetch --tags"
                    + (
                        ""
                        if action == "patch"
                        else f"\n  ({action} from an untagged version needs a tag on its own minor line, and "
                        f"{parsed[0]}.{parsed[1]}.x has none)"
                    )
                )
            absorb = current
        new = format_version(next_version(parsed, action))
        if self._tag_exists(new):
            raise Refusal(f"v{new} is already tagged; versions only move forward")
        return new, absorb

    def _released_notes(self, new: str, absorb: str | None) -> str:
        """the release notes as they will be written.

        :param new: version being released
        :ptype new: str
        :param absorb: an unreleased version whose notes become *new*'s, if any
        :ptype absorb: str | None
        :return: the new notes text
        :rtype: str
        :raises Refusal: if the notes cannot be released
        """
        text = (self.root / self.config.notes_file).read_text(encoding="utf-8")
        try:
            if self.config.notes_style == "prawduct":
                released, marked = release_prawduct_log(text, new, absorb)
                self.say(f"  {marked} change-log entries marked release=v{new}")
            else:
                released = release_changelog(text, CHANGELOG_STYLES[self.config.notes_style], new, self.today, absorb)
        except NotesRefusal as exc:
            raise Refusal(f"{self.config.notes_file}: {exc}") from None
        return released

    def plan(self, action: str) -> _Plan:
        """runs every check and decides every edit, writing nothing.

        :param action: requested action
        :ptype action: str
        :return: the plan
        :rtype: _Plan
        :raises Refusal: on the first check that fails
        """
        self._check_repository()
        if action in (*BUMP_KINDS, "sync") and shutil.which("uvx") is None:
            raise Refusal("uvx is not on PATH (the relock is `uvx uv@latest lock`)")
        current = self._check_versions()
        new, absorb = self._target(action, current)
        result = _Plan(action=action, current=current, new=new, absorb=absorb)
        if action != "release":
            result.touched += list(self.config.edited_files)
            if (self.root / "uv.lock").is_file():
                result.touched.append("uv.lock")
        if action != "sync":
            result.touched.append(self.config.notes_file)
        self._check_touched(result.touched)
        if absorb is not None and absorb != new:
            self.say(
                f"  {absorb} was declared but never tagged, so it never shipped: its notes move to {new}. If {absorb} "
                f"DID ship, stop here -- tag it on its main merge commit and bump from there."
            )
        if action != "sync":
            result.notes_text = self._released_notes(new, absorb)
        result.local_mode = action != "release" and not self._git_ok(
            "diff", "--quiet", "HEAD", "--", *_LOCAL_MODE_FILES
        )
        return result

    # ------------------------------------------------------------------- edits

    def _apply(self, tree: Path, plan: _Plan) -> None:
        """writes the plan's version and notes edits into *tree*.

        :param tree: the working tree, or a scratch worktree of ``HEAD``
        :ptype tree: Path
        :param plan: the plan
        :ptype plan: _Plan
        :return: nothing
        :rtype: None
        """
        if plan.action != "release":
            for path in self.config.edited_files:
                target = tree / path
                text = target.read_text(encoding="utf-8")
                rewritten = rewrite_lockstep_file(self.config, path, text, plan.new)
                if rewritten != text:
                    target.write_text(rewritten, encoding="utf-8")
        if plan.notes_text is not None:
            (tree / self.config.notes_file).write_text(plan.notes_text, encoding="utf-8")

    def _relock(self, tree: Path) -> bool:
        """relocks *tree* with ``uvx uv@latest lock``, when it has a lock.

        :param tree: directory holding ``uv.lock``
        :ptype tree: Path
        :return: ``True`` when relocked or there is nothing to relock
        :rtype: bool
        """
        if not (tree / "uv.lock").is_file():
            return True
        self.say(f"  relocking {tree} with uvx uv@latest lock ...")
        done = subprocess.run(["uvx", "uv@latest", "lock", "--directory", str(tree), "--quiet"], check=False)
        return done.returncode == 0

    def _scratch_worktree(self, scratch: Path) -> Path:
        """a detached worktree of ``HEAD`` in *scratch*, beside symlinks to every sibling.

        The worktree carries THIS repository's directory name and every sibling is
        a symlink beside it. A sibling path source (``../14-eng-ai-bot-agents``) must
        resolve as it does here, and a sibling pointing BACK at this repository must
        resolve to the worktree itself -- under any other name uv records this
        project a second time as ``editable = "../<repo>"``.

        :param scratch: an empty scratch directory
        :ptype scratch: Path
        :return: the worktree
        :rtype: Path
        :raises Refusal: if the worktree cannot be created
        """
        for sibling in sorted(self.root.parent.iterdir()):
            if sibling.is_dir() and sibling.name != self.root.name:
                (scratch / sibling.name).symlink_to(sibling)
        worktree = scratch / self.root.name
        added = run_git(self.root, "worktree", "add", "--quiet", "--detach", str(worktree), "HEAD")
        if not added.ok:
            raise Refusal(f"could not create the scratch worktree: {added.stderr}")
        return worktree

    def _stage_from(self, worktree: Path, paths: list[str]) -> None:
        """stages *worktree*'s copy of each path, leaving the working tree as it is.

        :param worktree: the scratch worktree
        :ptype worktree: Path
        :param paths: repo-relative files to stage
        :ptype paths: list[str]
        :return: nothing
        :rtype: None
        :raises Refusal: if git refuses a blob or an index entry
        """
        for path in paths:
            listed = run_git(self.root, "ls-files", "-s", "--", path).text.split()
            mode = listed[0] if listed else "100644"
            blob = run_git(self.root, "hash-object", "-w", "--", str(worktree / path))
            staged = (
                run_git(self.root, "update-index", "--cacheinfo", f"{mode},{blob.text},{path}") if blob.ok else blob
            )
            if not staged.ok:
                raise Refusal(f"could not stage {path}: {staged.stderr}")

    def execute(self, plan: _Plan) -> str:
        """performs the plan, restoring every touched file if any step fails.

        :param plan: the plan
        :ptype plan: _Plan
        :return: the commit command to print
        :rtype: str
        :raises Refusal: if a step fails; every touched file is restored first
        """
        backups = {path: (self.root / path).read_bytes() for path in plan.touched if (self.root / path).is_file()}
        scratch: Path | None = None
        worktree: Path | None = None
        hint = ""
        try:
            if plan.action == "release":
                self._apply(self.root, plan)
                hint = f'git commit -m "release: v{plan.new}" -- {self.config.notes_file}'
            elif not plan.local_mode:
                self._apply(self.root, plan)
                if not self._relock(self.root):
                    raise Refusal("uvx uv@latest lock failed; see above")
                changed = [path for path in plan.touched if (self.root / path).read_bytes() != backups.get(path)]
                hint = f'git commit -m "release: v{plan.new}" -- {" ".join(changed)}'
            else:
                scratch = Path(tempfile.mkdtemp(prefix="threetears-release-"))
                self.say("  local mode: building the committable bump from HEAD in a scratch worktree")
                worktree = self._scratch_worktree(scratch)
                self._apply(worktree, plan)
                if not self._relock(worktree):
                    raise Refusal("uvx uv@latest lock failed in the scratch worktree; see above")
                self._apply(self.root, plan)
                if not self._relock(self.root):
                    self.err.write(
                        "warning: the working tree's LOCAL lock did not relock; run `uvx uv@latest lock` here once "
                        "the bump is committed. The staged bump is unaffected.\n"
                    )
                self._stage_from(worktree, plan.touched)
                self.say(f"  staged in RELEASED mode (no sources block): {', '.join(plan.touched)}")
                hint = f'git commit -m "release: v{plan.new}"    # NO pathspec: the index holds the released-mode files'
        except BaseException:
            for path, content in backups.items():
                (self.root / path).write_bytes(content)
            if not self._git_ok("reset", "-q", "--", *plan.touched):
                self.err.write(f"warning: could not unstage {', '.join(plan.touched)}; check `git status`\n")
            self.err.write("error: nothing was changed; every file this run touched is restored\n")
            raise
        finally:
            if worktree is not None and worktree.exists():
                if not self._git_ok("worktree", "remove", "--force", str(worktree)):
                    self.err.write(f"warning: could not remove {worktree}; remove it and run `git worktree prune`\n")
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)
                run_git(self.root, "worktree", "prune")
        return hint

    def report(self, plan: _Plan, hint: str) -> None:
        """prints what was done and what to run next.

        :param plan: the executed plan
        :ptype plan: _Plan
        :param hint: the commit command
        :ptype hint: str
        :return: nothing
        :rtype: None
        """
        self.say("")
        self.say(f"Done. Version: {plan.new}")
        self.say("")
        self.say("Commit:")
        self.say(f"  {hint}")
        if plan.new != plan.current:
            needle = f'source = {{ editable = "../{self.root.name}" }}'
            dependents = sorted(
                lock.parent.name
                for lock in self.root.parent.glob("*/uv.lock")
                if lock.parent != self.root and needle in lock.read_text(encoding="utf-8", errors="replace")
            )
            if dependents:
                self.say("")
                self.say("These sibling checkouts lock this project by path and record its version; relock them")
                self.say("(uvx uv@latest lock) in the same release, or their locked syncs go stale:")
                self.say("  " + " ".join(dependents))
        if plan.action != "sync":
            self.say("")
            self.say("After the release merges to main, tag the merge commit:")
            self.say(f'  git tag -a v{plan.new} <merge sha> -m "{self.root.name} {plan.new}"')
            self.say(f"  git push origin v{plan.new}")


def _verify(config: ReleaseConfig, version: str, out: TextIO, err: TextIO) -> int:
    """checks every lockstep location says *version*.

    :param config: the repository's release configuration
    :ptype config: ReleaseConfig
    :param version: ``X.Y.Z``
    :ptype version: str
    :param out: where success is reported
    :ptype out: TextIO
    :param err: where mismatches are reported
    :ptype err: TextIO
    :return: exit status
    :rtype: int
    """
    mismatches = lockstep_mismatches(config, version)
    for mismatch in mismatches:
        err.write(f"  MISMATCH {mismatch}\n")
    if mismatches:
        err.write(f"\nerror: lockstep verification failed for version {version}.\n")
    else:
        out.write(f"All version locations at {version}.\n")
    return 1 if mismatches else 0


def run_release(
    config: ReleaseConfig,
    action: str,
    today: str,
    out: TextIO,
    err: TextIO,
    version: str | None = None,
) -> int:
    """runs one release action against one repository.

    :param config: the repository's release configuration
    :ptype config: ReleaseConfig
    :param action: ``patch``, ``minor``, ``major``, ``release``, ``sync`` or ``verify``
    :ptype action: str
    :param today: ``YYYY-MM-DD`` written into the notes
    :ptype today: str
    :param out: progress and instructions
    :ptype out: TextIO
    :param err: refusals and warnings
    :ptype err: TextIO
    :param version: the version ``verify`` checks against
    :ptype version: str | None
    :return: exit status: 0 done, 1 refused or failed
    :rtype: int
    """
    status = 0
    if action == "verify":
        status = _verify(config, version or "", out, err)
    else:
        run = _Run(config, today, out, err)
        try:
            plan = run.plan(action)
            if plan.new == plan.current:
                run.say(f"{'Releasing the declared version' if action == 'release' else 'Syncing'} {plan.current}")
            else:
                run.say(f"Bumping {plan.current} -> {plan.new} ({action})")
            hint = run.execute(plan)
            run.report(plan, hint)
        except Refusal as exc:
            err.write(f"error: {exc}\n")
            status = 1
    return status
