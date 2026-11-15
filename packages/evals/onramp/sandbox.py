"""The docs build an agent works against, and the sandbox it works in: one app's directory and three tools.

A :class:`DocsBuild` is one arm: ``3tears-evals`` installed, where the package lives, and the ``llms.txt`` the build
ships. For a measured run the build is installed in a trial image (:meth:`DocsBuild.in_image`) and every command runs
in a container of it (:class:`~.isolation.Container`): no network, not root, nothing of this machine but the trial's
own directory. A :class:`Sandbox` is a fresh copy of one app. The agent reads files in it or in the installed package,
writes files in it, and runs ``python`` (so ``python -m threetears.evals``), ``ls`` and ``cat``, with no shell.
"""

from __future__ import annotations

import hashlib
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .isolation import Container, HostRunner, Runner, copy_out_of_image, package_dir, trial_image

HERE = Path(__file__).resolve().parent
APPS = HERE / "apps"
TWINS = HERE / "twins"

#: Lines one read returns, from the line it starts at.
READ_LINES = 200
#: The largest file a read takes, in bytes: no more of a file than this is ever held in memory.
READ_BYTES = 2 * 1024 * 1024
#: Characters of a command's output the agent is shown: the head and the tail, the middle cut.
SHOWN_CHARS = 6000
#: How long one command may run, in seconds.
COMMAND_TIMEOUT_S = 180.0
#: What ``run`` admits, by program.
PROGRAMS = ("python", "python3", "ls", "cat")
#: Exit codes a run of the eval door ends well on: 0, or 4 (a comparison's gate failed, which is a finding).
DOOR_EXITS_OK = (0, 4)


@dataclass(frozen=True)
class DocsBuild:
    """One docs build: the interpreter it is installed in, where the package is, and its ``llms.txt``.

    :ivar name: The arm it is measured as (``docs@control``).
    :ivar python: The interpreter.
    :ivar package_dir: The installed ``threetears/evals`` directory.
    :ivar llms_txt: The build's ``llms.txt``: inside ``package_dir`` for a wheel.
    :ivar source: What it was built from, recorded on every run: a wheel and its digest, or this interpreter.
    :ivar image: The trial image the build is installed in, whose containers run every command; ``None`` for a build
        on this machine, which only the scripted agent may use. ``package_dir`` is then a copy of the image's.
    :ivar image_package_dir: Where the package is inside the image: the path the agent is shown and its commands see.
    """

    name: str
    python: Path
    package_dir: Path
    llms_txt: Path
    source: str
    image: str | None = None
    image_package_dir: Path | None = None

    @classmethod
    def here(cls, name: str) -> DocsBuild:
        """The package this interpreter imports, with the repository's ``llms.txt``: offline runs and tests."""
        import threetears.evals

        package_dir = Path(threetears.evals.__file__).resolve().parent
        return cls(name, Path(sys.executable), package_dir, HERE.parent / "llms.txt", "this interpreter")

    @classmethod
    def from_wheel(cls, name: str, wheel: Path, root: Path) -> DocsBuild:
        """Install ``wheel`` into a virtual environment of its own under ``root``, reused while the wheel is unchanged.

        :raises subprocess.CalledProcessError: ``uv`` could not make the environment or install the wheel.
        """
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        venv = root / f"venv-{digest[:12]}"
        python = venv / "bin" / "python"
        if not python.exists():
            subprocess.run(["uv", "venv", "--quiet", "--python", sys.executable, str(venv)], check=True)
            subprocess.run(["uv", "pip", "install", "--quiet", "--python", str(python), str(wheel)], check=True)
        where = subprocess.run(
            [str(python), "-c", "import pathlib, threetears.evals as e; print(pathlib.Path(e.__file__).parent)"],
            check=True,
            capture_output=True,
            text=True,
        )
        package_dir = Path(where.stdout.strip())
        return cls(name, python, package_dir, package_dir / "llms.txt", f"{wheel.name} sha256:{digest}")

    @classmethod
    def in_image(cls, name: str, wheel: Path, root: Path) -> DocsBuild:
        """``wheel`` installed in a trial image (built at set-up when it is not there yet), its docs copied under
        ``root`` for the agent's reads, which then read exactly what the image holds.

        :raises ~.isolation.IsolationUnavailable: Docker is unavailable, or the image did not build.
        """
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        image = trial_image(wheel)
        inside = package_dir(image)
        docs = root / f"docs-{image.rsplit(':', 1)[1]}"
        if not docs.is_dir():
            copy_out_of_image(image, inside, docs)
        return cls(
            name, Path(Container.python), docs, docs / "llms.txt", f"{wheel.name} sha256:{digest}", image, Path(inside)
        )

    @property
    def shown_package_dir(self) -> Path:
        """Where the agent is told the package is: where its commands find it."""
        return self.image_package_dir or self.package_dir

    @property
    def shown_llms_txt(self) -> Path:
        """Where the agent is told the build's ``llms.txt`` is."""
        if self.image_package_dir is None:
            return self.llms_txt
        return self.image_package_dir / self.llms_txt.relative_to(self.package_dir)

    def runner(self) -> Runner:
        """Where this build's commands run: a container of its image, or this machine for a build with none."""
        return HostRunner(self.python) if self.image is None else Container(self.image)

    def as_record(self) -> dict[str, str | None]:
        """The build as a run records it."""
        return {
            "name": self.name,
            "source": self.source,
            "package_dir": str(self.shown_package_dir),
            "image": self.image,
        }


@dataclass(frozen=True)
class Observation:
    """What one tool call did.

    :ivar ok: Whether it did what was asked; a run is ok when it exits 0, or 4 for the eval door.
    :ivar text: What the agent is shown.
    :ivar exit_code: A run's exit code; ``None`` for a read or a write, and for a run that was refused or timed out.
    :ivar first_error: A run's first line of stderr, or why the call was refused; empty when there was none.
    """

    ok: bool
    text: str
    exit_code: int | None = None
    first_error: str = ""


def refused(why: str) -> Observation:
    """A call the sandbox did not make."""
    return Observation(ok=False, text=f"refused: {why}", first_error=why)


def _shown(text: str) -> str:
    """At most :data:`SHOWN_CHARS` of ``text``: its head and tail, saying how much was cut between."""
    if len(text) <= SHOWN_CHARS:
        return text
    half = SHOWN_CHARS // 2
    return f"{text[:half]}\n[... {len(text) - SHOWN_CHARS} characters cut ...]\n{text[-half:]}"


def is_door(argv: list[str]) -> bool:
    """Whether a command runs the eval door (``python -m threetears.evals ...``)."""
    return len(argv) >= 3 and argv[0] in ("python", "python3") and argv[1:3] == ["-m", "threetears.evals"]


class Sandbox:
    """One app's directory, copied fresh, and the tools an agent works in it with.

    ``read``, ``write``, ``ls`` and ``cat`` are the harness's own file operations, held to the app's directory and the
    installed package; every command (``python``) runs where :attr:`runner` puts it.
    """

    def __init__(self, root: Path, app: str, build: DocsBuild, runner: Runner | None = None) -> None:
        self.root = root.resolve()
        self.app = app
        self.build = build
        #: Where every command runs: the build's (:meth:`DocsBuild.runner`) unless one is given.
        self.runner = build.runner() if runner is None else runner
        #: Every file the agent wrote, relative to the root, in the order first written.
        self.written: list[str] = []
        #: Every read of the package's own files (``read`` or ``cat``): the turn, the page and the lines read.
        self.pages: list[dict[str, int | str]] = []
        #: The turn the agent is on, which a read is recorded under.
        self.turn = 0

    @classmethod
    def create(cls, root: Path, app: str, build: DocsBuild, runner: Runner | None = None) -> Sandbox:
        """A fresh copy of ``apps/<app>`` at ``root``, which must not exist yet."""
        shutil.copytree(APPS / app, root, ignore=shutil.ignore_patterns("__pycache__"))
        return cls(root, app, build, runner)

    def readable(self, path: str) -> Path | None:
        """``path`` resolved, when it is in the sandbox, the installed package or the build's ``llms.txt``.

        A path in the image's package (the one the agent is shown) is read from the copy of it at ``package_dir``.
        """
        shown = self.build.image_package_dir
        if shown is not None and Path(path).is_absolute() and Path(path).is_relative_to(shown):
            path = str(self.build.package_dir / Path(path).relative_to(shown))
        target = (self.root / path).resolve()
        allowed = (self.root, self.build.package_dir)
        if target == self.build.llms_txt.resolve() or any(target.is_relative_to(base) for base in allowed):
            return target
        return None

    def page(self, target: Path) -> str | None:
        """How a read of ``target`` is recorded when it is the package's: its path in the package, or ``llms.txt``."""
        if target == self.build.llms_txt.resolve():
            return "llms.txt"
        if target.is_relative_to(self.build.package_dir):
            return target.relative_to(self.build.package_dir).as_posix()
        return None

    def read(self, path: str, start: int = 1) -> Observation:
        """Up to :data:`READ_LINES` lines of a file from line ``start``; a read of the package's own is recorded."""
        target = self.readable(path)
        if target is None:
            return refused(f"{path} is outside the app's directory and the installed package")
        if not target.is_file():
            return Observation(ok=False, text=f"{path}: no such file", first_error=f"{path}: no such file")
        with target.open("rb") as handle:
            data = handle.read(READ_BYTES + 1)
        if len(data) > READ_BYTES:
            why = f"{path} is larger than the {READ_BYTES // 2**20} MB a read takes"
            return Observation(ok=False, text=f"refused: {why}", first_error=why)
        lines = data.decode("utf-8", errors="replace").splitlines()
        first = max(start, 1)
        if lines and first > len(lines):
            why = f"{path} has {len(lines)} lines; start at a line from 1 to {len(lines)}"
            return Observation(ok=False, text=why, first_error=why)
        chunk = lines[first - 1 : first - 1 + READ_LINES]
        more = f"\n[lines {first}-{first + len(chunk) - 1} of {len(lines)}]" if len(lines) > len(chunk) else ""
        numbered = "\n".join(f"{first + offset:>5}  {line}" for offset, line in enumerate(chunk))
        if (page := self.page(target)) is not None:
            self.pages.append({"turn": self.turn, "page": page, "lines": len(chunk)})
        return Observation(ok=True, text=numbered + more)

    def write(self, path: str, text: str) -> Observation:
        """Write ``text`` to a file in the sandbox, ``path`` relative to it."""
        target = (self.root / path).resolve()
        if Path(path).is_absolute() or not target.is_relative_to(self.root) or target == self.root:
            return refused(f"{path} is not a file in the app's directory; write a path relative to it")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        relative = target.relative_to(self.root).as_posix()
        if relative not in self.written:
            self.written.append(relative)
        return Observation(ok=True, text=f"wrote {relative} ({len(text.splitlines())} lines)")

    async def run(self, command: str) -> Observation:
        """Run an admitted command in the sandbox: ``python``, ``ls`` or ``cat``, with no shell."""
        try:
            argv = shlex.split(command)
        except ValueError as unparsed:
            return refused(f"the command does not parse: {unparsed}")
        if not argv or argv[0] not in PROGRAMS:
            return refused(f"run takes {', '.join(PROGRAMS)} (python -m threetears.evals is python); not {command!r}")
        if argv[0] == "ls":
            return self._ls(argv[1:])
        if argv[0] == "cat":
            return self._cat(argv[1:])
        return await self.execute([self.runner.python, *argv[1:]], cwd=self.root, door=is_door(argv))

    async def execute(
        self,
        argv: list[str],
        *,
        cwd: Path,
        door: bool = False,
        timeout_s: float = COMMAND_TIMEOUT_S,
        grading: bool = False,
    ) -> Observation:
        """Run ``argv`` in ``cwd`` where :attr:`runner` puts it; its output shown with its directory as ``.``.

        ``argv[0]`` is the runner's interpreter (:attr:`~.isolation.Runner.python`). ``grading`` is the regrade's own
        command, which alone may run ``rerun.py``.
        """
        ran = await self.runner.run(argv, cwd=cwd, timeout_s=min(timeout_s, COMMAND_TIMEOUT_S), grading=grading)
        stdout = ran.stdout.replace(ran.cwd, ".")
        stderr = ran.stderr.replace(ran.cwd, ".")
        said = (f"\n{stdout}" if stdout else "") + (f"\n[stderr]\n{stderr}" if stderr.strip() else "")
        if ran.refused:
            # It ran, or could not, and its directory is as it was before it: a failed command, whatever it exited.
            return Observation(ok=False, text=_shown(f"refused: {ran.refused}{said}"), first_error=ran.refused)
        if ran.exit_code is None:
            why = f"timed out after {min(timeout_s, COMMAND_TIMEOUT_S):.0f}s"
            return Observation(ok=False, text=why, first_error=why)
        code = ran.exit_code
        first_error = next((line for line in stderr.splitlines() if line.strip()), "")
        text = _shown(f"exit {code}\n{stdout}" + (f"\n[stderr]\n{stderr}" if stderr.strip() else ""))
        ok = code in DOOR_EXITS_OK if door else code == 0
        return Observation(ok=ok, text=text, exit_code=code, first_error=first_error)

    def _ls(self, args: list[str]) -> Observation:
        target = self.readable(args[0] if args else ".")
        if target is None or not target.is_dir():
            why = f"ls: {args[0] if args else '.'} is no directory in the app or the installed package"
            return Observation(ok=False, text=why, exit_code=2, first_error=why)
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir() if p.name != "__pycache__")
        return Observation(ok=True, text="\n".join(names), exit_code=0)

    def _cat(self, args: list[str]) -> Observation:
        if len(args) != 1:
            return Observation(ok=False, text="cat takes one file", exit_code=2, first_error="cat takes one file")
        observation = self.read(args[0])
        return Observation(observation.ok, observation.text, 0 if observation.ok else 1, observation.first_error)

    def grading_copy(self, dest: Path, *, twin: bool, overlay: list[str]) -> Path:
        """A copy of the app as shipped (or its fixed twin), with ``overlay``'s files from the sandbox laid over it.

        Files of the app itself are never overlaid, so an agent that edited the app is graded on the app.
        """
        shutil.copytree(APPS / self.app, dest, ignore=shutil.ignore_patterns("__pycache__"))
        if twin:
            shutil.copytree(TWINS / self.app, dest, ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
        shipped = {p.relative_to(APPS / self.app).as_posix() for p in (APPS / self.app).rglob("*") if p.is_file()}
        for relative in overlay:
            source = self.root / relative
            if relative in shipped or not source.is_file():
                continue
            (dest / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest / relative)
        return dest

    def manifest(self) -> dict[str, str]:
        """Every file in the sandbox, its store and caches aside, by its path relative to the root, with the sha256 of
        its bytes: the state of the directory, as a run logs it after each turn that may change it."""
        return {
            path.relative_to(self.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(self.root.rglob("*"))
            if path.is_file()
            and not path.relative_to(self.root).as_posix().startswith(".evals/")
            and "__pycache__" not in path.parts
        }

    def made(self) -> list[str]:
        """Every file in the sandbox the app did not ship, its store and caches aside: what the agent left."""
        shipped = {p.relative_to(APPS / self.app).as_posix() for p in (APPS / self.app).rglob("*") if p.is_file()}
        return sorted(
            relative
            for path in self.root.rglob("*")
            if path.is_file()
            for relative in [path.relative_to(self.root).as_posix()]
            if relative not in shipped and not relative.startswith(".evals/") and "__pycache__" not in relative
        )
