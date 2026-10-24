"""One aspect's depths as every surface states them: each arm's, whether they differ, and the depth pass^k compares at.

The page, the writer's view and the memo's style pass read an aspect's depth from here, so the three say one thing:
each arm's :class:`~threetears.evals.analysis.depth.Depth` off the scoreboard, the plain sentence when the arms ran to
different depths, and the frontier's planned ``k``, its common depth and the arms pass^k at ``k`` is unmeasured for
(:mod:`threetears.evals.analysis.depth`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.depth import (
    Depth,
    common_depth_sentence,
    depth_sentence,
    depths_differ,
    unmeasured_sentence,
)

__all__ = ["AspectDepth", "PAIRING_METHOD", "aspect_depth"]

#: How a pass-rate comparison weighs its cases, said once in the Methods wherever one is read.
PAIRING_METHOD = (
    "A pass-rate interval weighs each shared case's share passed equally, however many times the case ran; the "
    "passed-of-measured counts beside it pool every trial."
)


@dataclass(frozen=True)
class AspectDepth:
    """One aspect's depths, each arm named by its label."""

    aspect: str
    #: Each arm's depth over every rig, by label, in scoreboard order; an arm with no usable trial is absent.
    arms: dict[str, Depth]
    #: The plain sentence saying the arms ran to different depths; None where they ran alike.
    said: str | None
    #: The frontier's planned depth, which pass^k is ranked at; None with no frontier subject.
    k: int | None = None
    #: The fewest runs any arm gave a case every arm shares; None with no frontier subject or no scored attempt.
    common_depth: int | None = None
    #: The arms, by label, that scored no case ``k`` times.
    unmeasured_for: list[str] = field(default_factory=list)
    #: The most cases any one arm loses at the common depth.
    left_out: int = 0
    #: Whether the frontier compares two or more arms, so a common depth is a comparison's.
    compared: bool = False

    @property
    def differ(self) -> bool:
        """Whether the arms ran to different depths."""
        return depths_differ(self.arms.values())

    def lacks(self, k: int) -> bool:
        """Whether pass^k at ``k`` is at a depth some arm never reached: above the common depth."""
        return self.common_depth is not None and k > self.common_depth

    def pass_k_lines(self) -> list[str]:
        """The sentences on pass^k's depth: the common depth, and each arm it is unmeasured for at ``k``."""
        lines = [common_depth_sentence(self.common_depth, left_out=self.left_out)] if self.compared else []
        if self.k is not None:
            lines.append(unmeasured_sentence(self.k, self.unmeasured_for))
        return [line for line in lines if line]


def aspect_depth(bundle: AnalysisContextBundle, label_of: Callable[[str], str]) -> AspectDepth:
    """One aspect's depths, read off its bundle.

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param label_of: What a reader calls the arm a variant key names.
    :ptype label_of: Callable[[str], str]
    :return: The depths.
    :rtype: AspectDepth
    """
    arms = {label_of(board.variant_key): board.depth for board in bundle.scoreboard if board.depth is not None}
    subject = next((subject for subject in bundle.frontier.subjects if subject.points), None)
    if subject is None:
        return AspectDepth(aspect=bundle.aspect, arms=arms, said=depth_sentence(bundle.aspect, dict(arms)))
    return AspectDepth(
        aspect=bundle.aspect,
        arms=arms,
        said=depth_sentence(bundle.aspect, dict(arms)),
        k=subject.k,
        common_depth=subject.common_depth,
        unmeasured_for=[label_of(key) for key in subject.unmeasured_for],
        left_out=max((point.n_cases_left_out for point in subject.points), default=0),
        compared=len(subject.points) > 1,
    )
