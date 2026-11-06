"""One aspect's depths as every surface states them: each arm's, whether they differ, and the depth pass^k compares at.

The page, the writer's view and the frontier's text read an aspect's depth from here, so they say one thing:
each arm's :class:`~threetears.evals.analysis.depth.Depth` off the scoreboard, the plain sentence when the arms ran to
different depths, and the frontier's planned ``k``, its common depth and the arms pass^k at ``k`` is unmeasured for
(:mod:`threetears.evals.analysis.depth`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.lenses.frontier import SubjectFrontier
from threetears.evals.analysis.depth import (
    Depth,
    common_depth_sentence,
    depth_sentence,
    depths_differ,
    unmeasured_sentence,
)

__all__ = ["AspectDepth", "PAIRING_METHOD", "SubjectDepth", "aspect_depth", "subject_depth"]

#: How a pass-rate comparison weighs its cases, said once in the Methods wherever one is read.
PAIRING_METHOD = (
    "A pass-rate interval weighs each shared case's share passed equally, however many times the case ran; the "
    "passed-of-measured counts beside it pool every trial."
)


#: Where a subject's depth sentences say they are, where an aspect's frontier holds several subjects.
WHERE_SUBJECT = "{aspect} ({subject})"


@dataclass(frozen=True)
class SubjectDepth:
    """One frontier subject's pass^k depths: its planned ``k``, its common depth, and the arms it leaves short.

    Every surface that says a subject's pass^k depth (the page, the writer's view, the frontier's text) reads it from
    here, so they say one thing.
    """

    #: Where its sentences say it is: the aspect, with the subject where an aspect holds several.
    where: str
    #: The planned depth, which pass^k is ranked at.
    k: int
    #: The fewest runs any arm gave a case every arm shares; None where the arms share no case.
    common_depth: int | None
    #: The arms, by label, that ran no case ``k`` times.
    unmeasured_for: list[str]
    #: The most cases any one arm loses for the others never running them.
    left_out: int
    #: Each compared arm's deepest case, by label: an arm with no scored attempt is not compared.
    reach: dict[str, int]

    @property
    def compared(self) -> bool:
        """Whether two or more arms scored an attempt, so a common depth is a comparison's."""
        return len(self.reach) > 1

    @property
    def shares_no_case(self) -> bool:
        """Whether two or more arms are compared and they share no case, so pass^k compares nothing between them."""
        return self.compared and self.common_depth is None

    def lines(self) -> list[str]:
        """The sentences on pass^k's depth: the common depth or that the arms share none, and each arm pass^k at the
        planned ``k`` is unmeasured for."""
        lines = [common_depth_sentence(self.where, self.common_depth, left_out=self.left_out)] if self.compared else []
        unmeasured = unmeasured_sentence(self.where, self.k, self.unmeasured_for)
        return lines + ([unmeasured] if unmeasured else [])


def subject_depth(subject: SubjectFrontier, label_of: Callable[[str], str], *, where: str) -> SubjectDepth:
    """One frontier subject's pass^k depths, each arm named by ``label_of``.

    :param subject: The subject's frontier.
    :ptype subject: SubjectFrontier
    :param label_of: What a reader calls the arm a variant key names.
    :ptype label_of: Callable[[str], str]
    :param where: Where its sentences say it is.
    :ptype where: str
    :return: The depths.
    :rtype: SubjectDepth
    """
    return SubjectDepth(
        where=where,
        k=subject.k,
        common_depth=subject.common_depth,
        unmeasured_for=[label_of(key) for key in subject.unmeasured_for],
        left_out=max((point.n_cases_left_out for point in subject.points), default=0),
        reach={label_of(point.variant_key): point.depth.repeats_max for point in subject.points if point.depth},
    )


@dataclass(frozen=True)
class AspectDepth:
    """One aspect's depths, each arm named by its label, and each of its frontier subjects' pass^k depths."""

    aspect: str
    #: Each arm's depth over every rig, by label, in scoreboard order; an arm with no usable trial is absent.
    arms: dict[str, Depth]
    #: The plain sentence saying the arms ran to different depths; None where they ran alike.
    said: str | None
    #: Each frontier subject's pass^k depths, in the frontier's order.
    subjects: list[SubjectDepth] = field(default_factory=list)

    @property
    def differ(self) -> bool:
        """Whether the arms ran to different depths."""
        return depths_differ(self.arms.values())

    @property
    def compared(self) -> bool:
        """Whether any subject compares two or more arms."""
        return any(subject.compared for subject in self.subjects)

    def pass_k_lines(self) -> list[str]:
        """Every subject's sentences on pass^k's depth, in order."""
        return [line for subject in self.subjects for line in subject.lines()]


def aspect_depth(
    bundle: AnalysisContextBundle, label_of: Callable[[str], str], *, name: str | None = None
) -> AspectDepth:
    """One aspect's depths, read off its bundle: every arm's, and every frontier subject's.

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param label_of: What a reader calls the arm a variant key names.
    :ptype label_of: Callable[[str], str]
    :param name: What its sentences call the aspect; None calls it by its label.
    :ptype name: str | None
    :return: The depths.
    :rtype: AspectDepth
    """
    arms = {label_of(board.variant_key): board.depth for board in bundle.scoreboard if board.depth is not None}
    subjects = [subject for subject in bundle.frontier.subjects if subject.points]
    called = bundle.aspect if name is None else name
    return AspectDepth(
        aspect=bundle.aspect,
        arms=arms,
        said=depth_sentence(called, dict(arms)),
        subjects=[
            subject_depth(
                subject,
                label_of,
                where=called
                if len(subjects) == 1
                else WHERE_SUBJECT.format(aspect=called, subject=subject.subject_label or subject.subject_id),
            )
            for subject in subjects
        ],
    )
