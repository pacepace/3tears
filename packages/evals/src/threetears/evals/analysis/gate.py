"""The CI gate: fail a build on the verdicts a caller names, read from a report's typed verdicts.

A gate reads :attr:`~threetears.evals.analysis.report.model.Report.verdicts` — the typed verdicts every printed
verdict is rendered from — never the words. The caller names the outcomes that fail it (``fail_on``), as
tokens:

- ``regressed``: a contrast shown worse than the control.
- ``not-separated`` and ``untested``: a contrast the evidence could not decide.
- ``breached``: a guardrail shown worse than the control by more than its margin.
- ``undecided-guardrail``: a guardrail neither shown held nor breached.
- ``missed``: a bar a cell is shown to fall short of.
- ``undecided-bar``: a bar a cell is not shown to clear or miss (``undecided``, ``no_interval``, ``no_data``).

**The default fails on ``regressed``, ``breached`` and ``undecided-guardrail``** (:data:`DEFAULT_FAIL_ON`). A
regression and a breach are what a change must not ship with. A guardrail is what the candidate must never get
worse on, and an undecided one is not known to be safe, so it fails too: a gate that passed it would read
"not shown unsafe" as "safe". A contrast that is not separated is not a regression the evidence shows, so it
fails only when named; add ``not-separated`` and ``untested`` to require every arm be shown improved or
equivalent.

**Undecided is never a pass.** A verdict outside ``fail_on`` that the evidence did not decide — a contrast not
separated or untested, a guardrail or bar undecided — leaves the gate ``undecided``, not ``passed``; it does not
fail the build, because the caller did not name it, and the gate says how many it is. ``passed`` means every
verdict gated was decided and none failed. A gate over no verdict at all is ``undecided``: nothing was shown.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from threetears.evals.analysis.report.model import Verdict

#: An outcome a gate can fail on; see the module docstring.
GateToken = Literal[
    "regressed", "not-separated", "untested", "breached", "undecided-guardrail", "missed", "undecided-bar"
]

#: Every token, in the order the module docstring lists them.
GATE_TOKENS: tuple[GateToken, ...] = get_args(GateToken)

#: What a gate fails on when the caller names nothing: a regression, a breached guardrail, and a guardrail not
#: shown held.
DEFAULT_FAIL_ON: tuple[GateToken, ...] = ("regressed", "breached", "undecided-guardrail")

#: What a gate came to: ``failed`` (a verdict named in ``fail_on``), ``undecided`` (none failed, and some verdict
#: was not decided, or there was none), ``passed`` (every verdict decided, none failing).
GateOutcome = Literal["passed", "failed", "undecided"]

#: Each verdict's token, by kind and outcome; a decided outcome no gate fails on (improved, equivalent, held,
#: cleared) has none.
_TOKEN_OF: dict[tuple[str, str], GateToken] = {
    ("contrast", "regressed"): "regressed",
    ("contrast", "not_separated"): "not-separated",
    ("contrast", "untested"): "untested",
    ("guardrail", "breached"): "breached",
    ("guardrail", "undecided"): "undecided-guardrail",
    ("bar", "missed"): "missed",
    ("bar", "undecided"): "undecided-bar",
    ("bar", "no_interval"): "undecided-bar",
    ("bar", "no_data"): "undecided-bar",
}

#: The tokens that are an absence of a decision rather than a decided failure.
_UNDECIDED_TOKENS: frozenset[GateToken] = frozenset(
    {"not-separated", "untested", "undecided-guardrail", "undecided-bar"}
)


def verdict_token(verdict: Verdict) -> GateToken | None:
    """The gate token a verdict counts under, or None for a decided outcome no gate fails on.

    Args:
        verdict: The verdict.

    Returns:
        Its token.
    """
    return _TOKEN_OF.get((verdict.kind, verdict.outcome))


def parse_fail_on(text: str) -> tuple[GateToken, ...]:
    """A comma-separated ``--fail-on`` list as tokens, refusing one that is no token.

    Args:
        text: ``"regressed,breached"``.

    Returns:
        The tokens, in the order given, each once.

    Raises:
        ValueError: A name that is no token, or an empty list.
    """
    names = [name.strip() for name in text.split(",") if name.strip()]
    if not names:
        raise ValueError(f"name at least one outcome to fail on: {', '.join(GATE_TOKENS)}")
    return _refuse_unknown_tokens(names)


def _refuse_unknown_tokens(names: Iterable[str]) -> tuple[GateToken, ...]:
    """The names as gate tokens, each once, in order.

    Raises:
        ValueError: A name that is no token.
    """
    given = list(dict.fromkeys(names))
    if unknown := [name for name in given if name not in GATE_TOKENS]:
        raise ValueError(
            f"{', '.join(map(repr, unknown))} is no outcome a gate fails on; the outcomes are {', '.join(GATE_TOKENS)}"
        )
    return tuple(name for name in GATE_TOKENS if name in given)


@dataclass(frozen=True)
class GateResult:
    """What a gate read and what it came to.

    Attributes:
        fail_on: The tokens it failed on.
        gated: Every verdict it read, after any filter by reading.
        failures: The verdicts whose token is in ``fail_on``.
        undecided: The verdicts outside ``fail_on`` that the evidence did not decide.
    """

    fail_on: tuple[GateToken, ...]
    gated: tuple[Verdict, ...]
    failures: tuple[Verdict, ...]
    undecided: tuple[Verdict, ...]

    @property
    def outcome(self) -> GateOutcome:
        """``failed``, ``undecided`` or ``passed``; see :data:`GateOutcome`."""
        if self.failures:
            return "failed"
        if self.undecided or not self.gated:
            return "undecided"
        return "passed"

    @property
    def failed(self) -> bool:
        """Whether a verdict named in ``fail_on`` occurred — what a build fails on."""
        return bool(self.failures)

    @property
    def passed(self) -> bool:
        """Whether every verdict gated was decided and none failed. Never true with an undecided verdict."""
        return self.outcome == "passed"

    def render(self) -> str:
        """The gate as a few lines for a terminal: the outcome, then each failure and each undecided verdict.

        Returns:
            The text, without a trailing newline.
        """
        named = ", ".join(self.fail_on)
        if not self.gated:
            return f"gate undecided: no verdict to read (fail on {named}), so nothing was shown either way"
        head = {
            "failed": f"gate FAILED: {len(self.failures)} of {len(self.gated)} verdict(s) are ones it fails on ({named})",
            "undecided": (
                f"gate undecided: none of {len(self.gated)} verdict(s) is one it fails on ({named}), and "
                f"{len(self.undecided)} were not decided, which is not a pass"
            ),
            "passed": f"gate passed: all {len(self.gated)} verdict(s) decided, none it fails on ({named})",
        }[self.outcome]
        lines = [head]
        lines.extend(f"  failed: {_line(verdict)}" for verdict in self.failures)
        lines.extend(f"  undecided: {_line(verdict)}" for verdict in self.undecided)
        return "\n".join(lines)


def _line(verdict: Verdict) -> str:
    """One verdict as a gate prints it: what, whose, against what, and the report's words."""
    against = f" vs {verdict.control}" if verdict.control is not None else ""
    what = {"contrast": "", "bar": f"bar {verdict.threshold} on ", "guardrail": "guardrail "}[verdict.kind]
    return f"{what}{verdict.heading}, {verdict.arm}{against}: {verdict.words}"


def gate_verdicts(
    verdicts: Sequence[Verdict],
    *,
    fail_on: Iterable[str] = DEFAULT_FAIL_ON,
    readings: Iterable[str] | None = None,
) -> GateResult:
    """Gate a report's typed verdicts: fail on the outcomes named, and never pass one the evidence left undecided.

    Args:
        verdicts: The verdicts, as a report carries them (:attr:`Report.verdicts`).
        fail_on: The tokens to fail on (:data:`GATE_TOKENS`); the default is :data:`DEFAULT_FAIL_ON`.
        readings: Only the verdicts on these readings, by key (``"accuracy"``) or as the report heads them
            (``"Accuracy"``); ``None`` gates every verdict.

    Returns:
        The gate's result.

    Raises:
        ValueError: A ``fail_on`` name that is no token, or a ``readings`` name no verdict carries.
    """
    tokens = _refuse_unknown_tokens(fail_on)
    gated = list(verdicts)
    if readings is not None:
        wanted = set(readings)
        known = {verdict.name for verdict in gated} | {verdict.heading for verdict in gated}
        if unknown := sorted(wanted - known):
            raise ValueError(
                f"no verdict is on {', '.join(map(repr, unknown))}; the readings are "
                f"{', '.join(sorted({verdict.name for verdict in gated})) or 'none'}"
            )
        gated = [verdict for verdict in gated if verdict.name in wanted or verdict.heading in wanted]
    failures = [verdict for verdict in gated if verdict_token(verdict) in tokens]
    undecided = [
        verdict for verdict in gated if (token := verdict_token(verdict)) in _UNDECIDED_TOKENS and token not in tokens
    ]
    return GateResult(fail_on=tokens, gated=tuple(gated), failures=tuple(failures), undecided=tuple(undecided))


__all__ = [
    "DEFAULT_FAIL_ON",
    "GATE_TOKENS",
    "GateOutcome",
    "GateResult",
    "GateToken",
    "gate_verdicts",
    "parse_fail_on",
    "verdict_token",
]
