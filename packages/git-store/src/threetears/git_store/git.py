"""the ``git`` program, run as a subprocess, for one working tree.

Plain ``git`` over :mod:`subprocess` (no pygit2), lifted from scriob's ``git.py`` and bound to one
working tree. Every call is BLOCKING: async callers run it with :func:`asyncio.to_thread`.

Three rules hold for every call:

- **never a force push.** :meth:`Git.push` is a plain push, which git refuses when it is not a
  fast-forward, so a stale writer is rejected instead of overwriting origin. Nothing in this package
  passes ``--force`` or ``--force-with-lease``; a test reads the source to keep it so.
- **identity is explicit.** A commit names its author and committer on the call, so a fresh clone
  with no ``user.name`` commits, and nobody's global config decides who wrote the data.
- **paths are literal.** ``--literal-pathspecs`` is on for every call, so a file name can never be
  read as a pathspec pattern (``:(top)``, ``*``).

A deploy key (a private SSH key for one repo) is written to a private temporary file for the one
network call that needs it and removed after; it is never logged.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal, overload

from pydantic import SecretStr

__all__ = [
    "Author",
    "CommitEntry",
    "FileChange",
    "Git",
    "GitError",
    "deploy_key_ssh_command",
]

#: the record and field separators a log is read with, so a subject holding any text parses.
_FIELD: Final[str] = "\x1f"
_RECORD: Final[str] = "\x1e"


class GitError(Exception):
    """a ``git`` call failed.

    :ivar git_args: the arguments after ``git``
    :ivar returncode: its exit code
    :ivar stderr: what it wrote to stderr
    """

    def __init__(self, args: Sequence[str], returncode: int, stderr: str) -> None:
        """record the failed call.

        :param args: the arguments after ``git``
        :ptype args: Sequence[str]
        :param returncode: the exit code
        :ptype returncode: int
        :param stderr: what git wrote to stderr
        :ptype stderr: str
        :return: None
        :rtype: None
        """
        self.git_args = tuple(args)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"git {' '.join(args)} failed (exit {returncode}): {stderr.strip()}")

    @property
    def index_locked(self) -> bool:
        """whether git refused because another git process holds the index (``index.lock``).

        :return: ``True`` for the busy case worth a retry
        :rtype: bool
        """
        return "index.lock" in self.stderr


@dataclass(frozen=True, slots=True)
class Author:
    """who a commit names as its author or committer.

    :ivar name: the name
    :ivar email: the email address
    """

    name: str
    email: str


@dataclass(frozen=True, slots=True)
class CommitEntry:
    """one commit, as a history lists it.

    :ivar sha: the full commit id
    :ivar author: who wrote it
    :ivar date: the author date, timezone-aware
    :ivar subject: the message's first line
    :ivar body: the rest of the message, trailers included
    """

    sha: str
    author: Author
    date: datetime
    subject: str
    body: str


@dataclass(frozen=True, slots=True)
class FileChange:
    """one path that differs between two commits.

    :ivar path: the path, relative to the working tree's root
    :ivar status: ``"A"`` added, ``"M"`` modified, ``"D"`` deleted
    """

    path: str
    status: Literal["A", "M", "D"]


@contextmanager
def deploy_key_ssh_command(key: SecretStr) -> Iterator[str]:
    """a ``GIT_SSH_COMMAND`` that uses one deploy key, for the length of the block.

    The key is written to a ``0600`` temporary file, used with ``IdentitiesOnly=yes`` (no other
    identity is offered) and ``StrictHostKeyChecking=accept-new`` (a first-contact host key is
    trusted, a changed one refused), and overwritten and removed when the block ends.

    :param key: the private key's text
    :ptype key: SecretStr
    :return: the command, for the block
    :rtype: Iterator[str]
    """
    fd, name = tempfile.mkstemp(prefix="git-store-key-", suffix=".key")
    path = Path(name)
    try:
        os.fchmod(fd, 0o600)
        material = key.get_secret_value()
        if not material.endswith("\n"):
            material += "\n"  # OpenSSH refuses a key file without a final newline
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(material)
        yield f"ssh -i {path} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
    finally:
        with _ignoring_os_errors():
            path.write_bytes(b"\0" * 256)
        path.unlink(missing_ok=True)


@contextmanager
def _ignoring_os_errors() -> Iterator[None]:
    """run a best-effort scrub whose failure the following unlink makes harmless.

    :return: the block
    :rtype: Iterator[None]
    """
    try:
        yield
    except OSError:  # NOSILENT: a best-effort scrub; the unlink after it is the guarantee
        return


def _environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """the environment every git call runs in.

    ``LC_ALL=C`` keeps git's messages in English, which :attr:`GitError.index_locked` reads;
    ``GIT_TERMINAL_PROMPT=0`` makes a call that would ask for a password fail instead of hang.

    :param extra: variables to add
    :ptype extra: Mapping[str, str] | None
    :return: the environment
    :rtype: dict[str, str]
    """
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    env["GIT_TERMINAL_PROMPT"] = "0"
    if extra:
        env.update(extra)
    return env


class Git:
    """``git`` for one working tree.

    :ivar work_tree: the working tree's root
    """

    def __init__(self, work_tree: Path, *, deploy_key: SecretStr | None = None) -> None:
        """bind to a working tree.

        :param work_tree: the working tree's root (it need not exist yet; see :meth:`init` and
            :meth:`clone`)
        :ptype work_tree: Path
        :param deploy_key: the private key network calls use, or ``None`` for the ambient SSH
            configuration
        :ptype deploy_key: SecretStr | None
        :return: None
        :rtype: None
        """
        self.work_tree = work_tree
        self._deploy_key = deploy_key

    @overload
    def run(
        self,
        *args: str,
        stdin: bytes | None = ...,
        env: Mapping[str, str] | None = ...,
        raw: Literal[False] = ...,
        ok_codes: tuple[int, ...] = ...,
    ) -> str: ...

    @overload
    def run(
        self,
        *args: str,
        stdin: bytes | None = ...,
        env: Mapping[str, str] | None = ...,
        raw: Literal[True],
        ok_codes: tuple[int, ...] = ...,
    ) -> bytes: ...

    def run(
        self,
        *args: str,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
        raw: bool = False,
        ok_codes: tuple[int, ...] = (0,),
    ) -> str | bytes:
        """run ``git <args>`` in the working tree.

        :param args: the arguments after ``git``
        :ptype args: str
        :param stdin: bytes for git's standard input
        :ptype stdin: bytes | None
        :param env: variables added to the environment
        :ptype env: Mapping[str, str] | None
        :param raw: return stdout as bytes rather than text
        :ptype raw: bool
        :param ok_codes: the exit codes that are not a failure
        :ptype ok_codes: tuple[int, ...]
        :return: what git wrote to stdout
        :rtype: str | bytes
        :raises GitError: on any other exit code
        """
        result = subprocess.run(
            ["git", "--literal-pathspecs", *args],
            cwd=self.work_tree,
            input=stdin,
            capture_output=True,
            env=_environment(env),
            check=False,
        )
        if result.returncode not in ok_codes:
            raise GitError(args, result.returncode, result.stderr.decode("utf-8", "replace"))
        return result.stdout if raw else result.stdout.decode("utf-8")

    def network(self, *args: str) -> str:
        """run a call that reaches a remote, with the deploy key when there is one.

        :param args: the arguments after ``git``
        :ptype args: str
        :return: stdout
        :rtype: str
        :raises GitError: when the call fails
        """
        if self._deploy_key is None:
            return self.run(*args)
        with deploy_key_ssh_command(self._deploy_key) as command:
            return self.run(*args, env={"GIT_SSH_COMMAND": command})

    # ------------------------------------------------------------------ the repository itself

    def toplevel(self) -> Path | None:
        """the root of the git repository the working tree is in: itself, or one it sits inside.

        :return: that root, resolved; ``None`` when the folder is in no repository (or isn't there)
        :rtype: Path | None
        """
        if not self.work_tree.is_dir():
            return None
        try:
            out = self.run("rev-parse", "--show-toplevel").strip()
        except GitError:
            return None
        return Path(out).resolve() if out else None

    def exists(self) -> bool:
        """whether the working tree is a git checkout of its own.

        A folder inside another repository is not: ``--is-inside-work-tree`` says "true" for any folder of a
        checkout, and a store there committed into the outer repository (found adopting it in samhla, whose
        characters folder sits inside its code checkout).

        :return: ``True`` when the working tree is the root of its repository
        :rtype: bool
        """
        top = self.toplevel()
        return top is not None and top == self.work_tree.resolve()

    def enclosing(self) -> Path | None:
        """the root of another repository the working tree sits inside (or would, once made), if any.

        :return: that repository's root, or ``None`` when the working tree is a repo of its own or in none
        :rtype: Path | None
        """
        probe = self.work_tree
        while not probe.is_dir() and probe != probe.parent:
            probe = probe.parent
        top = Git(probe).toplevel()
        return top if top is not None and top != self.work_tree.resolve() else None

    def init(self, branch: str) -> None:
        """make a new, empty repository in the working tree, on ``branch``.

        :param branch: the branch HEAD names
        :ptype branch: str
        :return: None
        :rtype: None
        :raises GitError: when git refuses
        """
        self.work_tree.mkdir(parents=True, exist_ok=True)
        self.run("init", "-q")
        self.run("symbolic-ref", "HEAD", f"refs/heads/{branch}")

    def clone(self, url: str, branch: str) -> None:
        """clone ``url`` into the working tree, atomically.

        The clone is made beside the working tree and renamed onto it only when complete, so an
        interrupted clone leaves nothing behind. An origin with no commits yet clones as an empty
        repository on ``branch``.

        :param url: the origin
        :ptype url: str
        :param branch: the branch to check out
        :ptype branch: str
        :return: None
        :rtype: None
        :raises GitError: when the clone fails
        """
        parent = self.work_tree.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = parent / f".{self.work_tree.name}.cloning"
        if staging.exists():
            shutil.rmtree(staging)
        cloner = Git(parent, deploy_key=self._deploy_key)
        try:
            cloner.network("clone", "-q", "--", url, str(staging))
            staged = Git(staging)
            if staged.head() is None:
                staged.run("symbolic-ref", "HEAD", f"refs/heads/{branch}")
            elif staged.run("rev-parse", "--abbrev-ref", "HEAD").strip() != branch:
                staged.run("checkout", "-q", branch)
        except GitError:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        os.replace(staging, self.work_tree)

    def git_path(self, name: str) -> Path:
        """where a file inside the repository's own directory lives (``git rev-parse --git-path``).

        :param name: the file's name under the git directory
        :ptype name: str
        :return: its absolute path
        :rtype: Path
        """
        found = Path(self.run("rev-parse", "--git-path", name).strip())
        return found if found.is_absolute() else self.work_tree / found

    def has_remote(self, remote: str = "origin") -> bool:
        """whether the repository names ``remote``.

        :param remote: the remote's name
        :ptype remote: str
        :return: ``True`` when it is configured
        :rtype: bool
        """
        return remote in self.run("remote").split()

    def set_remote(self, url: str, remote: str = "origin") -> None:
        """point ``remote`` at ``url``, adding it when absent.

        :param url: the remote's URL
        :ptype url: str
        :param remote: the remote's name
        :ptype remote: str
        :return: None
        :rtype: None
        """
        if self.has_remote(remote):
            self.run("remote", "set-url", remote, url)
        else:
            self.run("remote", "add", remote, url)

    # ------------------------------------------------------------------ reading

    def rev_parse(self, rev: str) -> str | None:
        """the commit ``rev`` names, or ``None`` when it names none.

        :param rev: a revision (``HEAD``, ``origin/main``, a sha)
        :ptype rev: str
        :return: the full sha, or ``None``
        :rtype: str | None
        """
        out = self.run("rev-parse", "-q", "--verify", f"{rev}^{{commit}}", ok_codes=(0, 1)).strip()
        return out or None

    def head(self) -> str | None:
        """the commit HEAD names, or ``None`` on a branch with no commit yet.

        :return: the full sha, or ``None``
        :rtype: str | None
        """
        return self.rev_parse("HEAD")

    def show(self, rev: str, path: str) -> bytes | None:
        """a file's bytes as of ``rev``, or ``None`` when it did not exist there.

        :param rev: the commit
        :ptype rev: str
        :param path: the path, relative to the root
        :ptype path: str
        :return: the bytes, or ``None``
        :rtype: bytes | None
        """
        try:
            return self.run("cat-file", "blob", f"{rev}:{path}", raw=True)
        except GitError:  # NOSILENT: the file did not exist at that commit, which None says
            return None

    def in_head(self, paths: Sequence[str]) -> set[str]:
        """which of ``paths`` HEAD holds.

        :param paths: paths, relative to the root
        :ptype paths: Sequence[str]
        :return: the ones HEAD holds
        :rtype: set[str]
        """
        if not paths or self.head() is None:
            return set()
        out = self.run("ls-tree", "-r", "-z", "--name-only", "HEAD", "--", *paths, raw=True)
        return {name.decode("utf-8") for name in out.split(b"\0") if name}

    def dirty(self, paths: Sequence[str] | None = None) -> list[tuple[str, str]]:
        """files that differ from HEAD in the working tree or the index, untracked ones included.

        :param paths: limit to these paths; ``None`` for the whole tree
        :ptype paths: Sequence[str] | None
        :return: ``(status, path)`` pairs, the status as ``git status --porcelain`` gives it
        :rtype: list[tuple[str, str]]
        """
        if paths is not None and not paths:
            return []
        args = ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames"]
        if paths is not None:
            args += ["--", *paths]
        out = self.run(*args, raw=True)
        found: list[tuple[str, str]] = []
        for entry in out.split(b"\0"):
            if len(entry) > 3:
                text = entry.decode("utf-8")
                found.append((text[:2], text[3:]))
        return found

    def changed(self, old: str | None, new: str) -> list[FileChange]:
        """the paths that differ between two commits.

        :param old: the earlier commit; ``None`` for the empty tree (everything in ``new`` is added)
        :ptype old: str | None
        :param new: the later commit
        :ptype new: str
        :return: each changed path with how it changed
        :rtype: list[FileChange]
        """
        if old is None:
            out = self.run("ls-tree", "-r", "-z", "--name-only", new, raw=True)
            return [FileChange(name.decode("utf-8"), "A") for name in out.split(b"\0") if name]
        out = self.run("diff", "--name-status", "-z", "--no-renames", old, new, raw=True)
        fields = [field.decode("utf-8") for field in out.split(b"\0") if field]
        changes: list[FileChange] = []
        for status, path in zip(fields[::2], fields[1::2], strict=True):
            kind: Literal["A", "M", "D"] = "A" if status == "A" else "D" if status == "D" else "M"
            changes.append(FileChange(path, kind))
        return changes

    def log(self, *, path: str | None = None, rev: str = "HEAD", limit: int = 50) -> list[CommitEntry]:
        """the newest commits first, optionally only those that touched ``path``.

        :param path: only commits that changed this path (followed by name, not across renames)
        :ptype path: str | None
        :param rev: where to start
        :ptype rev: str
        :param limit: the most commits to return
        :ptype limit: int
        :return: the commits
        :rtype: list[CommitEntry]
        """
        if self.rev_parse(rev) is None:
            return []
        fmt = _FIELD.join(("%H", "%an", "%ae", "%aI", "%s", "%b")) + _RECORD
        args = ["log", f"--max-count={limit}", f"--format={fmt}", rev]
        if path is not None:
            args += ["--", path]
        entries: list[CommitEntry] = []
        for record in self.run(*args).split(_RECORD):
            record = record.strip("\n")
            if not record:
                continue
            sha, name, email, date, subject, body = record.split(_FIELD)
            entries.append(CommitEntry(sha, Author(name, email), datetime.fromisoformat(date), subject, body.strip()))
        return entries

    def commit_time(self, rev: str = "HEAD") -> datetime | None:
        """when ``rev`` was committed, or ``None`` when it names no commit.

        :param rev: the commit
        :ptype rev: str
        :return: the committer date, timezone-aware
        :rtype: datetime | None
        """
        if self.rev_parse(rev) is None:
            return None
        return datetime.fromisoformat(self.run("log", "-1", "--format=%cI", rev).strip())

    def is_ancestor(self, older: str, newer: str) -> bool:
        """whether ``older`` is ``newer`` or one of its ancestors.

        :param older: the candidate ancestor
        :ptype older: str
        :param newer: the descendant
        :ptype newer: str
        :return: ``True`` when ``newer`` contains ``older``
        :rtype: bool
        """
        # exit 0 is yes and 1 is no; --is-ancestor writes nothing to stdout either way, so the answer
        # is asked for again as the merge base, which equals ``older`` exactly when it is an ancestor
        base = self.run("merge-base", older, newer, ok_codes=(0, 1)).strip()
        return base != "" and base == self.rev_parse(older)

    def ahead_behind(self, branch: str, remote: str = "origin") -> tuple[int, int]:
        """how far ``branch`` is ahead of and behind ``<remote>/<branch>``, as last fetched.

        :param branch: the local branch
        :ptype branch: str
        :param remote: the remote
        :ptype remote: str
        :return: ``(ahead, behind)``; ``(0, 0)`` with no remote-tracking ref, and ``(commits, 0)``
            when the remote has none yet
        :rtype: tuple[int, int]
        """
        local = self.rev_parse(branch)
        upstream = self.rev_parse(f"{remote}/{branch}")
        if local is None:
            return (0, 0)
        if upstream is None:
            count = int(self.run("rev-list", "--count", local).strip() or 0)
            return (count, 0)
        out = self.run("rev-list", "--left-right", "--count", f"{upstream}...{local}").strip()
        behind, _, ahead = out.partition("\t")
        return (int(ahead or 0), int(behind or 0))

    # ------------------------------------------------------------------ writing

    def stage(self, written: Sequence[str], removed: Sequence[str]) -> None:
        """stage written files and removals.

        :param written: paths whose working-tree bytes are staged
        :ptype written: Sequence[str]
        :param removed: paths whose removal is staged (a path git never knew is skipped)
        :ptype removed: Sequence[str]
        :return: None
        :rtype: None
        :raises GitError: when git refuses
        """
        if written:
            self.run("add", "--", *written)
        if removed:
            self.run("rm", "-q", "--cached", "--ignore-unmatch", "--", *removed)

    def commit(
        self,
        paths: Sequence[str],
        message: str,
        *,
        author: Author,
        committer: Author,
        date: datetime | None = None,
    ) -> str:
        """commit exactly ``paths``, whatever else is staged, and return the new commit.

        ``--only``: other staged or edited files are left as they are, out of the commit. Hooks do
        not run (``--no-verify``) and nothing is signed: a machine commit of data is not gated by the
        hooks a person installed for their own commits.

        :param paths: the paths the commit takes, as staged by :meth:`stage`
        :ptype paths: Sequence[str]
        :param message: the whole message, trailers included
        :ptype message: str
        :param author: who wrote the change
        :ptype author: Author
        :param committer: who committed it
        :ptype committer: Author
        :param date: the author date; ``None`` for now
        :ptype date: datetime | None
        :return: the new commit's sha
        :rtype: str
        :raises GitError: when git refuses
        """
        env = {
            "GIT_AUTHOR_NAME": author.name,
            "GIT_AUTHOR_EMAIL": author.email,
            "GIT_COMMITTER_NAME": committer.name,
            "GIT_COMMITTER_EMAIL": committer.email,
        }
        if date is not None:
            env["GIT_AUTHOR_DATE"] = date.isoformat()
        self.run(
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "--no-verify",
            "--only",
            "-F",
            "-",
            "--",
            *paths,
            stdin=message.encode("utf-8"),
            env=env,
        )
        sha = self.head()
        assert sha is not None  # a commit just landed
        return sha

    def restore(self, paths: Sequence[str]) -> None:
        """put ``paths`` back as HEAD has them, in the index and the working tree.

        A path HEAD holds is checked out from it; one HEAD does not hold is unstaged and removed.

        :param paths: the paths
        :ptype paths: Sequence[str]
        :return: None
        :rtype: None
        :raises GitError: when git refuses
        """
        if not paths:
            return
        held = self.in_head(paths)
        kept = [path for path in paths if path in held]
        gone = [path for path in paths if path not in held]
        if kept:
            self.run("checkout", "-q", "HEAD", "--", *kept)
        if gone:
            self.run("rm", "-q", "--cached", "--ignore-unmatch", "--", *gone)
            for path in gone:
                (self.work_tree / path).unlink(missing_ok=True)

    # ------------------------------------------------------------------ origin

    def fetch(self, remote: str = "origin") -> None:
        """fetch ``remote`` (remote-tracking refs only; the working tree is untouched).

        :param remote: the remote
        :ptype remote: str
        :return: None
        :rtype: None
        :raises GitError: when the fetch fails
        """
        self.network("fetch", "-q", remote)

    def merge_ff_only(self, ref: str) -> None:
        """move the checked-out branch forward to ``ref``; refuse when it is not a fast-forward.

        :param ref: the commit to move to (``origin/main``)
        :ptype ref: str
        :return: None
        :rtype: None
        :raises GitError: when the branch has commits ``ref`` lacks, or a file in the way is edited
        """
        self.run("merge", "-q", "--ff-only", ref)  # on a branch with no commit yet, it takes ref's

    def push(self, branch: str, remote: str = "origin") -> None:
        """push ``branch`` to ``remote``: a plain push, which git refuses when it is not a fast-forward.

        :param branch: the branch
        :ptype branch: str
        :param remote: the remote
        :ptype remote: str
        :return: None
        :rtype: None
        :raises GitError: when the push is rejected or fails
        """
        self.network("push", "-q", remote, f"refs/heads/{branch}:refs/heads/{branch}")
