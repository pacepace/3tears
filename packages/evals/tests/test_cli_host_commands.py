"""A host's own subcommands beside the engine's: mounted, parsed, handed the factory's host, and refused on a clash.

A product mounting :func:`~threetears.evals.quick.run_cli` under its own CLI passes
:class:`~threetears.evals.quick.HostCommand` values; each is parsed like the engine's commands and its
handler is handed the very host object the factory built. A name the engine already uses — or one two
host commands share — is refused before anything is parsed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.contracts.errors import NotFoundError
from threetears.evals.quick import ENGINE_COMMANDS, HostCommand, callable_host, run_cli
from threetears.evals.quick.cli import build_parser


def _graded(case: Mapping[str, Any], answer: Any) -> float:
    return 1.0


class _Capture:
    """A host command that records what it was handed, and exits with a chosen code."""

    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self.seen: list[tuple[Any, argparse.Namespace]] = []

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("session", help="the session to capture")
        parser.add_argument("--note", default=None)

    def handle(self, host: Any, args: argparse.Namespace) -> int:
        self.seen.append((host, args))
        return self.exit_code

    def command(self, name: str = "capture") -> HostCommand:
        return HostCommand(name=name, help="Record a witnessed session.", configure=self.configure, handler=self.handle)


def test_a_host_command_runs_with_the_host_the_factory_built() -> None:
    built: list[Any] = []

    def factory() -> Any:
        # A fresh host per call, so "the host the factory built" is one object and not merely an equal one.
        built.append(callable_host([_graded]))
        return built[-1]

    capture = _Capture(exit_code=7)
    code = run_cli(
        ["capture", "s-1", "--scope", "dev", "--note", "tuesday"], host_factory=factory, commands=[capture.command()]
    )
    assert code == 7
    ((handed, args),) = capture.seen
    assert len(built) == 1 and handed is built[0]
    assert (args.command, args.scope, args.session, args.note) == ("capture", "dev", "s-1", "tuesday")


def test_a_coroutine_handler_is_run_to_completion() -> None:
    seen: list[str] = []

    async def handle(host: Any, args: argparse.Namespace) -> int:
        seen.append(args.scope)
        return 0

    command = HostCommand(name="capture", help="capture", configure=lambda _parser: None, handler=handle)
    assert (
        run_cli(["capture", "--scope", "dev"], host_factory=lambda: callable_host([_graded]), commands=[command]) == 0
    )
    assert seen == ["dev"]


def test_a_host_command_named_on_the_command_line_takes_host_too() -> None:
    capture = _Capture()
    parser = build_parser(commands=[capture.command()])
    args = parser.parse_args(["capture", "s-1", "--scope", "dev", "--host", "pkg.mod:factory"])
    assert (args.host, args.session) == ("pkg.mod:factory", "s-1")


def test_a_refusal_a_host_command_raises_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    def handle(host: Any, args: argparse.Namespace) -> int:
        raise NotFoundError("session", "s-404")

    command = HostCommand(name="capture", help="capture", configure=lambda _parser: None, handler=handle)
    assert (
        run_cli(["capture", "--scope", "dev"], host_factory=lambda: callable_host([_graded]), commands=[command]) == 2
    )
    assert "capture: " in capsys.readouterr().err


@pytest.mark.parametrize("name", ENGINE_COMMANDS)
def test_a_host_command_taking_an_engine_commands_name_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match=f"host command '{name}' collides with the engine's own"):
        run_cli(
            [name, "--scope", "dev"], host_factory=lambda: callable_host([_graded]), commands=[_Capture().command(name)]
        )


def test_two_host_commands_sharing_a_name_are_refused() -> None:
    with pytest.raises(ValueError, match="two host commands are both named 'capture'"):
        build_parser(commands=[_Capture().command(), _Capture().command()])


def test_the_engine_commands_still_work_beside_a_host_command(capsys: pytest.CaptureFixture[str]) -> None:
    capture = _Capture()
    assert (
        run_cli(["ls", "--scope", "empty"], host_factory=lambda: callable_host([_graded]), commands=[capture.command()])
        == 0
    )
    assert capsys.readouterr().out.splitlines() == ["templates (0)", "runs (0)", "campaigns (0)"]
    assert capture.seen == []


def test_the_usage_line_names_the_host_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser(commands=[_Capture().command()]).parse_args([])
    assert "{run,ls,report,bundle,spend,frontier,capture}" in capsys.readouterr().err
