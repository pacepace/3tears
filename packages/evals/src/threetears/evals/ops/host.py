"""What the operations run over: a launching host, and how it generates an analysis in the background.

Every operation that reads takes the :class:`~threetears.evals.contracts.host.EvalHost`; every one that
starts long work — a launch, an analysis generation — needs the process's job manager too, which the
:class:`~threetears.evals.run.LaunchHost` builds. :class:`OpsHost` is the two together with the one
thing neither carries: what a background generation is told (its prompt, its output cap and its
wall-clock budget), which is the host's configuration and never the engine's guess.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from threetears.evals.contracts.host import EvalHost
from threetears.evals.run.launch import LaunchHost


@dataclass(frozen=True, kw_only=True)
class AnalysisGeneration:
    """How this host generates an analysis as a background job.

    Attributes:
        prompt_id: The registry key the generator prompt resolves under, recorded on every analysis as
            its provenance.
        resolve_prompt: Resolves the generator prompt; awaited once a generation's bundle has evidence.
        max_output_tokens: The output cap the host's generator clients are built with, for the
            pre-spend disclosure.
        budget_s: The job's wall-clock ceiling. Derive it ABOVE the generation's own ceiling
            (:func:`~threetears.evals.analysis.generation_ceiling_s`): a budget below it cancels a
            repair round-trip still writing inside its cap, after both calls were billed.
    """

    prompt_id: str
    resolve_prompt: Callable[[], Awaitable[str]]
    max_output_tokens: int
    budget_s: float

    def __post_init__(self) -> None:
        """Refuse an empty prompt id, and a cap or budget that is not positive.

        Raises:
            ValueError: ``prompt_id`` is blank, or ``max_output_tokens`` or ``budget_s`` is not positive.
        """
        if not self.prompt_id.strip():
            raise ValueError("AnalysisGeneration.prompt_id is blank; name the registry key the prompt resolves under")
        if self.max_output_tokens <= 0:
            raise ValueError(f"AnalysisGeneration.max_output_tokens must be positive; got {self.max_output_tokens}")
        if self.budget_s <= 0:
            raise ValueError(f"AnalysisGeneration.budget_s must be positive; got {self.budget_s}")


@dataclass(frozen=True, kw_only=True)
class OpsHost:
    """The host the operations, and the actions over them, work in.

    Attributes:
        launch: The launching host: its :class:`~threetears.evals.contracts.host.EvalHost`, its kinds and
            the job manager every long-running operation is tracked by.
        generation: How an analysis is generated in the background; ``None`` for a host that does not
            generate analyses here, whose generation start is refused saying so.
    """

    launch: LaunchHost
    generation: AnalysisGeneration | None = None

    @property
    def eval_host(self) -> EvalHost:
        """The host every read goes through — the launching host's own."""
        return self.launch.eval_host


__all__ = ["AnalysisGeneration", "OpsHost"]
