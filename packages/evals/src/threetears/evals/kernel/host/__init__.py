"""The host contract — what a consuming product declares, and what the engine never interprets.

Everything under this package holds **no host vocabulary**. The engine keeps the arithmetic; the
host keeps the taxonomy — the same move the spend port makes: the thing that made a call reports
what it consumed, in units the engine does not interpret.

A product adopts the engine by building one :class:`EvalHost` (:mod:`~threetears.evals.kernel.host.eval_host`)
and handing it to every entrypoint: its :class:`HostProfile` (the registries below), its storage
and its services. There is no installed host, so two of them in one process are two values.

Not every member is a registry, and the two that are not point in opposite directions.
:mod:`~threetears.evals.kernel.host.apparatus` is a leaf holding one exception — the fault in the
measuring rig that a host raises and the engine acts on — and it imports nothing, which is what
lets a host's tool layer import it without either side acquiring a dependency on the other's
internals. :mod:`~threetears.evals.schema.traces` is a port rather than a vocabulary: the engine
states what it needs from a host's tracing and takes an implementation of it, because a cell's
wall-clock and its spans are produced by a runtime the engine did not instrument.
:mod:`~threetears.evals.kernel.host.timeouts` is the same kind of port for a cell's wall-clock budget.

**Why a package rather than a module.** Each registry answers a different question and a host
adopts them at different rates — a host with dozens of levers and no simulated world registers
sweepables and measures and supplies no ``world`` at all. One module per registry keeps a partial
adoption expressible.

Two gates hold this boundary, and they fail for different reasons:

* ``tests/test_extraction_import_boundary.py`` — no module here imports a host package.
* ``tests/test_no_host_names_in_shared_contract.py`` — no module here *names* a host concept,
  which the import gate cannot see: a host's inputs read fields on the engine's own ``EvalRun``,
  so there is no import to catch.

**This module is the package's public root.** A host imports from here and from no module below
it, and only the names in ``__all__``; ``tests/test_package_matrix.py`` holds that. Code inside the
package imports its own modules directly.
"""

from __future__ import annotations

from threetears.evals.kernel.host.apparatus import ApparatusError
from threetears.evals.kernel.host.bars import (
    DEFAULT_PASS_THRESHOLD,
    Bar,
    BarRegistry,
    PassThreshold,
    pass_threshold_label,
    BarProposal,
    BarRegistrationError,
)
from threetears.evals.kernel.host.eval_host import CompletionClients, CompletionRole, EvalHost
from threetears.evals.kernel.host.kinds import (
    ActsOn,
    Interval,
    KindContract,
    KindContractError,
    MemberActsOn,
    Ordinal,
    ResolvesInto,
    freeze,
)
from threetears.evals.kernel.host.measures import MeasureRegistrationError, MeasureRegistry
from threetears.evals.kernel.host.profile import (
    Coverage,
    HostProfile,
    UNSEATED_LEVEL,
    ActionParameterReader,
    CoverageState,
    ProfileRegistrationError,
    ToolActionReader,
    VariantLeverReader,
)
from threetears.evals.kernel.host.style import (
    CHART_FONT_CHARACTERS,
    SERIES_SLOTS,
    VALIDATED_SLOTS,
    ChartFont,
    ChartPalette,
    StyleError,
    StyleProfile,
    require_resolved_colour,
    ToneRegister,
)
from threetears.evals.kernel.host.sweepables import (
    CANDIDATE_KIND_LEVER,
    CANDIDATE_MODEL_LEVER,
    SHARED_CORE,
    RolePins,
    Sweepable,
    SweepableRegistry,
    served_models_by_score,
    Comparability,
    FamilyMemberTest,
    ProductionDepartureReader,
    RegistrationError,
    ResidualReader,
    ResolvedLevers,
    SweepableReader,
    SweepableRole,
)
from threetears.evals.kernel.host.timeouts import CellTimeoutFactory, EvalCellTimeout, default_cell_timeout
from threetears.evals.kernel.host.world import (
    Triggered,
    WorldDimension,
    WorldRegistry,
    resolve_preconditions,
    Evidence,
    TriggerKind,
    When,
    WorldCapability,
    WorldRegistrationError,
)
from threetears.evals.kernel.host.world_conformance import (
    CheckName,
    ConformanceResult,
    ObligationRow,
    Outcome,
    Qualification,
    WorldConformanceError,
    WorldConformanceReport,
    check_world_conformance,
    obligation_rows,
    obligations,
)
from threetears.evals.kernel.host.world_schema import UnsupportedSchemaError, schema_violations
from threetears.evals.kernel.host.world_seed import SeedRefused, SeedWrite, check_seed, SeedRefusalKind

__all__ = [
    "CANDIDATE_KIND_LEVER",
    "CANDIDATE_MODEL_LEVER",
    "CHART_FONT_CHARACTERS",
    "SERIES_SLOTS",
    "SHARED_CORE",
    "UNSEATED_LEVEL",
    "VALIDATED_SLOTS",
    "ActionParameterReader",
    "ActsOn",
    "ApparatusError",
    "Bar",
    "DEFAULT_PASS_THRESHOLD",
    "PassThreshold",
    "pass_threshold_label",
    "BarProposal",
    "BarRegistrationError",
    "BarRegistry",
    "CellTimeoutFactory",
    "ChartFont",
    "ChartPalette",
    "CheckName",
    "Comparability",
    "CompletionClients",
    "CompletionRole",
    "ConformanceResult",
    "Coverage",
    "CoverageState",
    "EvalCellTimeout",
    "EvalHost",
    "Evidence",
    "FamilyMemberTest",
    "HostProfile",
    "Interval",
    "KindContract",
    "KindContractError",
    "MeasureRegistrationError",
    "MeasureRegistry",
    "MemberActsOn",
    "ObligationRow",
    "Ordinal",
    "Outcome",
    "ProductionDepartureReader",
    "ProfileRegistrationError",
    "Qualification",
    "RegistrationError",
    "ResidualReader",
    "ResolvedLevers",
    "ResolvesInto",
    "RolePins",
    "SeedRefusalKind",
    "SeedRefused",
    "SeedWrite",
    "StyleError",
    "StyleProfile",
    "Sweepable",
    "SweepableReader",
    "SweepableRegistry",
    "SweepableRole",
    "ToneRegister",
    "ToolActionReader",
    "TriggerKind",
    "Triggered",
    "UnsupportedSchemaError",
    "VariantLeverReader",
    "When",
    "WorldCapability",
    "WorldConformanceError",
    "WorldConformanceReport",
    "WorldDimension",
    "WorldRegistrationError",
    "WorldRegistry",
    "resolve_preconditions",
    "check_seed",
    "check_world_conformance",
    "default_cell_timeout",
    "freeze",
    "obligation_rows",
    "obligations",
    "require_resolved_colour",
    "schema_violations",
    "served_models_by_score",
]
