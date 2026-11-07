"""What one generation of an analysis sent, spent and was refused: the record an attempt's ending is stored from.

``analysis_generate``'s job reads the report writer's turns into a :class:`GenerationTally` once the writer stops,
whatever it stopped on, and the service stores the attempt's record from it
(:func:`~threetears.evals.analysis.service.record_written_generation`). A generation either stores an analysis or
raises, and a cancellation unwinds it without either, so the tally is what describes an attempt that ended any other
way.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["GenerationTally"]


class GenerationTally(BaseModel):
    """What one generation has sent, spent and been refused so far, kept by the caller.

    Plain ``BaseModel``: the refusals are the validators' messages verbatim, and stripping them would record a refusal
    that is not the one the validator raised.

    **No field constraints, deliberately.** It is read from a tally kept while a paid generation was in flight, so a
    constraint here would turn a bookkeeping anomaly into an exception raised between a billed call and the attempt
    record it paid for. The bounds belong where the tally is read into a record that crosses a boundary:
    :class:`~threetears.evals.kernel.campaign.EvalAnalysisAttempt` refuses a negative count or cost at construction.
    """

    calls: int = Field(default=0, description="Provider calls sent, counted before each is awaited.")
    returned: int = Field(default=0, description="Calls that came back with a result.")
    unreported_cost: int = Field(default=0, description="Returned calls whose provider reported no cost.")
    token_cost: float = Field(default=0.0, description="Dollars reported across every returned call.")
    reported_models: list[str] = Field(default_factory=list, description="The model each returned call ran on.")
    refusals: list[str] = Field(
        default_factory=list,
        description="Each refusal of a finished call's output, in order, naming cells by their full cell_ref.",
    )
    prompt_version: str | None = Field(default=None, description="The assembled prompt's version, once it is known.")

    @property
    def unpriced_calls(self) -> int:
        """Calls that returned with no reported cost. A call cut off before a result is not one: no reply reported
        its usage, so it adds nothing to ``token_cost`` and never makes it a floor (owner, 2026-10-27, applied to the
        attempt record 2026-11-04); ``calls - returned`` counts it."""
        return self.unreported_cost
