"""corpus-mapping coverage for ``dsm-task-01a``'s slice of the definition model.

This is the shard's acceptance bar, not a fixture set. D3 is explicit that
``dsm-task-01a``-``d`` are accepted against ``corpus-task-01``'s mapping
rather than against fixtures drawn from the design, because a fixture set
drawn from the same document lets the model pass by omission.

Each row below names a semantic the corpus carries, an ``anchor`` that
must appear verbatim in ``docs/corpus-mapping.md``, and exactly one
resolution: a dotted path that must resolve against
:mod:`threetears.datasources.definition`, or a recorded decision.

Scope is this shard's fields only -- ``GrainSpec``, ``ParameterSpec``,
``Unit``, ``Resolution``, ``Qualification``, the seven-name namespace, and
``Comparison``. ``dsm-task-01b``, ``-01c``, and ``-01d`` extend this
module as each lands; ``dsm-task-01d`` (DSM-01D-20) is where it asserts
coverage of the WHOLE mapping.

The mapping document lives in the sibling ripple repo, which 3tears must
not take a build dependency on. The row table is therefore checked in
here, and :class:`TestMappingDocumentCrossCheck` re-verifies every anchor
against the live document when that repo is reachable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

import threetears.datasources.definition as definition


@dataclass(frozen=True)
class MappingRow:
    """one corpus-mapping row inside this shard's scope.

    :ivar row_id: mapping part / section / row identifier
    :ivar anchor: literal substring that must appear in ``corpus-mapping.md``
    :ivar semantic: one-line restatement of what the corpus carries
    :ivar field_path: dotted path resolved against the definition package,
        or ``None`` when the row resolves to a recorded decision
    :ivar decision: recorded decision text, or ``None`` when the row
        resolves to a field
    """

    row_id: str
    anchor: str
    semantic: str
    field_path: str | None = None
    decision: str | None = None


IN_SCOPE_ROWS: tuple[MappingRow, ...] = (
    # ---- Part 1, the cross-slice reconciliation -------------------------
    MappingRow(
        "P1/1.2",
        "Duplicate unit names: three layers that disagree",
        "one name means the union of two in SQL and the second one in every reporting path",
        field_path="validate_unique_unit_names",
    ),
    MappingRow(
        "P1/1.4",
        "`Qualification.applies_to` is an extension, not a port",
        "every unit-set-scoped qualification in the corpus was hand-written outside the tool",
        field_path="Qualification.applies_to",
    ),
    MappingRow(
        "P1/1.5",
        "domains must cover both `-1` and `NULL`",
        "the two record_year sentinels are opposite failures and both are silent",
        field_path="SentinelKind",
    ),
    MappingRow(
        "P1/1.6",
        "An audience unit that is silently annihilated",
        "a unit named in no qualification arm has every row dropped, not filtered, not errored",
        field_path="validate_qualification_coverage",
    ),
    MappingRow(
        "P1/1.10",
        "A parameter sweep the model cannot express",
        "ten hand-unrolled arms over thresholds 1 through 10 for one knob",
        field_path="ParameterSweep",
    ),
    # ---- Part 2, section B1: the renderer -------------------------------
    MappingRow(
        "P2/B1/R04",
        "duplicate `Unit.name` must be a **parse error**",
        "the metadata dict is last-wins while the SQL unions; no migration should choose silently",
        field_path="DuplicateUnitName",
    ),
    MappingRow(
        "P2/B1/R05",
        "Duplicate unit name in SQL: no dedup",
        "every duplicate emits its own INSERT and the long table receives the union",
        field_path="Unit.resolutions",
    ),
    MappingRow(
        "P2/B1/R06",
        "Custom unit name derived from filename via",
        "the filename derivation is not carried forward; the name is authored",
        field_path="Unit.name",
    ),
    MappingRow(
        "P2/B1/R09",
        "Schema derivation chain:",
        "extraction_schema, match_schema, analytics_schema all derived from release_schema",
        field_path="TemplateDerivation.template",
    ),
    MappingRow(
        "P2/B1/R10",
        "enforced only when truthy",
        "vf_suffix carries an enumeration that the prototype bypasses when the value is None",
        field_path="ParameterSpec.enum",
    ),
    MappingRow(
        "P2/B1/R11",
        "stated in a help string, never checked",
        "tsmart_comm is valid only with the TargetSmart voter file",
        field_path="ParameterConstraint.requires_parameter",
    ),
    MappingRow(
        "P2/B1/R12",
        "help documents a derivation the code does not implement",
        "the argparse help and the code disagree about the analytics_schema derivation",
        decision=(
            "the CODE is authoritative and the help is drift; the derivation is authored once on "
            "ParameterSpec.derivation, so there is no second place for a description to disagree"
        ),
    ),
    MappingRow(
        "P2/B1/R13",
        "must be `YYYYmmdd`, unvalidated",
        "the date parameter is typed, and the run year is derived from it by a string slice",
        field_path="SubstringDerivation.length",
    ),
    # ---- Part 2, section B2: the hand-match materializer -----------------
    MappingRow(
        "P2/B2/H08",
        "Hardcoded operational defaults",
        "target_schema, table, and date carry defaults rather than being required",
        field_path="ParameterSpec.default",
    ),
    # ---- Part 2, section B4: the shared templates ------------------------
    MappingRow(
        "P2/B4/T05",
        "`facts_table` optional at stage 1",
        "a resolution with no fact source is bridge-only",
        field_path="Resolution.source",
    ),
    MappingRow(
        "P2/B4/T08",
        "a **group key** `empl.record_year` for linkedin units",
        "record_year is a MAX or a group key depending on the resolution path",
        field_path="Namespace.RESOLVED",
    ),
    MappingRow(
        "P2/B4/T17",
        "Filter block emitted only if at least one of three fields is present",
        "a resolution may carry no predicate at all",
        field_path="Resolution.predicate",
    ),
    MappingRow(
        "P2/B4/T31",
        "Unconditional entity join defining the qualified set",
        "the qualification stage has joins, not just predicates",
        field_path="Qualification.relations",
    ),
    MappingRow(
        "P2/B4/T32",
        "Qualification scoped **per unit** by an OR-chain",
        "the committed template expresses only per-unit scoping",
        field_path="Qualification.applies_to",
    ),
    MappingRow(
        "P2/B4/T33",
        "Qualification scoped to a named unit **set**",
        "a genuine extension over the prototype; every real use was hand-written",
        field_path="Qualification.name",
    ),
    MappingRow(
        "P2/B4/T34",
        "Working-age window `age > 20 AND age < (70 + year - record_year)`",
        "a comparison with an expression on the right-hand side",
        field_path="ArithmeticExpression.arith",
    ),
    MappingRow(
        "P2/B4/T35",
        "bind to the **materialized long row**",
        "unqualified record_year and candidate_count bind resolved.*, vf.* binds entity.*",
        field_path="Namespace.ENTITY",
    ),
    MappingRow(
        "P2/B4/T84",
        "**28 occurrences** in `universal_2026_core/standard_audience_units.yaml`",
        "ILIKE must be in the operator vocabulary",
        field_path="ComparisonOperator.ILIKE",
    ),
    MappingRow(
        "P2/B4/T85",
        "POSIX regex with backslash classes",
        "POSIX ~ and ~* round-tripping doubled escapes byte-exactly",
        field_path="ComparisonOperator.REGEX_CASE_INSENSITIVE",
    ),
    MappingRow(
        "P2/B4/T86",
        "Unit names containing `>` — 4 distinct names, 12 declaration sites",
        "the logical name is free-form and the identifier is derived",
        field_path="Unit.name",
    ),
    MappingRow(
        "P2/B4/T87",
        "Unit names containing spaces — the `amazon_tech_audience` rollups",
        "the logical name may contain spaces",
        field_path="Unit.name",
    ),
    # ---- Part 2, section B5: the jinja layer -----------------------------
    MappingRow(
        "P2/B5/J03",
        "raises on a supplied-but-unused parameter",
        "declared parameters, checked structurally rather than by literal text match",
        field_path="ParameterSpec.name",
    ),
    MappingRow(
        "P2/B5/J12",
        "`validate_vf_suffix` enumerates",
        "the l2 / ts enumeration",
        field_path="ParameterSpec.enum",
    ),
    # ---- Part 2, section B6: audience_test -------------------------------
    MappingRow(
        "P2/B6/S05",
        "`limit_to_working_age: True` on all four settings entries",
        "the working-age boolean and the candidate-count ceiling are qualification predicates",
        field_path="Qualification.predicate",
    ),
    MappingRow(
        "P2/B6/S08",
        "`where_filters` referencing the `facts.*` alias",
        "the source.* namespace decouples the definition from the template's alias choice",
        field_path="Namespace.SOURCE",
    ),
    MappingRow(
        "P2/B6/S09",
        "31-clause parenthesised `OR` chain",
        "one authored filter entry is a disjunction, not an atom",
        field_path="Predicate.any_of",
    ),
    MappingRow(
        "P2/B6/S10",
        "leading `NOT` outside the parens",
        "negation over a disjunction",
        field_path="Predicate.negate",
    ),
    MappingRow(
        "P2/B6/S12",
        "A unit declared with **no** `facts_table` and no filters is legal at stage 1",
        "a bridge-only resolution with no predicate",
        field_path="Resolution.bridge",
    ),
    MappingRow(
        "P2/B6/S18",
        "confirms the parameter derivation empirically",
        "three distinct schemas derived from one release schema plus a suffix",
        field_path="TemplateDerivation.fallback",
    ),
    # ---- Part 2, section B7: audience_agent_test -------------------------
    MappingRow(
        "P2/B7/G03",
        "6-clause `OR` chain written on **one line**",
        "the lowercase one-line disjunction must parse identically",
        field_path="Predicate.any_of",
    ),
    MappingRow(
        "P2/B7/G07",
        "The fixture's target schema is `scratch`",
        "target_schema is a run parameter, confirmed varying across runs of one audience",
        field_path="ParameterSpec.parameter_type",
    ),
    # ---- Part 3, the Amazon audiences ------------------------------------
    MappingRow(
        "P3/1.1",
        "a named unit; the key the whole tool is organised around",
        "the authored unit name",
        field_path="Unit.name",
    ),
    MappingRow(
        "P3/1.6",
        "**A unit may have no fact source at all**",
        "omitting facts_table yields a bridge-only resolution",
        field_path="Resolution.source",
    ),
    MappingRow(
        "P3/1.7",
        "an **ordered list of raw SQL boolean fragments**",
        "an AND-combined ordered predicate list",
        field_path="Predicate.all_of",
    ),
    MappingRow(
        "P3/1.8",
        "multi-line disjunction of 60+ terms",
        "one filter entry nests a disjunction inside a negation",
        field_path="Predicate.negate",
    ),
    MappingRow(
        "P3/1.30",
        "Per-resolution alias drift",
        "empl vs facts for the same slot; the namespace decouples them",
        field_path="Namespace.SOURCE",
    ),
    MappingRow(
        "P3/1.31",
        "Predicates reference **four** distinct namespaces already",
        "bare unqualified references resolve by accident of FROM order and must be bound",
        field_path="Reference.namespace",
    ),
    MappingRow(
        "P3/3.2",
        "**The unit name comes from the filename**",
        "the unit name is bound by the model, never read out of the raw text",
        field_path="Unit.name",
    ),
    MappingRow(
        "P3/3.4",
        "**One authored file emits two labelled units**",
        "one unit projecting two labels",
        field_path="Unit.emits",
    ),
    MappingRow(
        "P3/3.12",
        "three-arm `UNION ALL` of the same query",
        "a unit whose source is itself a union",
        field_path="Unit.resolutions",
    ),
    MappingRow(
        "P3/3.15",
        "two `Unit`s, or one `Unit` with two `Resolution`s",
        "the omnibus overlap / non_overlap near-duplicates",
        decision=(
            "both shapes are expressible: Unit.resolutions is a list, so one unit with two "
            "resolutions unions them under one label, and two units keep two labels. the "
            "reconciliation record chooses per audience; the model does not"
        ),
    ),
    MappingRow(
        "P3/5.1",
        "a **per-unit qualification record**",
        "one settings entry per unit",
        field_path="Qualification.applies_to",
    ),
    MappingRow(
        "P3/5.2",
        "a per-unit ceiling on the bridge's `candidate_count`",
        "applied at qualification against the already-aggregated MIN",
        field_path="Namespace.RESOLVED",
    ),
    MappingRow(
        "P3/5.3",
        "a two-tier quality scheme, retuned wholesale",
        "the tiers are declared over a unit set, not flattened per unit",
        field_path="Qualification.applies_to",
    ),
    MappingRow(
        "P3/5.4",
        "a per-unit boolean gating a two-branch age window",
        "the working-age window is an ordinary predicate, unit-set scoped",
        field_path="Qualification.predicate",
    ),
    MappingRow(
        "P3/5.5",
        "**expressions on both sides**, mixing an entity attribute",
        "entity.*, param.*, and resolved.* in one comparison",
        field_path="Comparison.right",
    ),
    MappingRow(
        "P3/5.6",
        "is **derived** from the `--date` parameter by string slice",
        "the run year is a substring derivation",
        field_path="SubstringDerivation.start",
    ),
    MappingRow(
        "P3/5.7",
        "**The entity join at qualification is unconditional**",
        "the stage has joins, not just predicates",
        field_path="Qualification.relations",
    ),
    MappingRow(
        "P3/5.8",
        "Qualification is a **disjunction of per-unit arms**",
        "a unit not named in settings is silently dropped from the qualified set",
        field_path="units_without_qualification",
    ),
    MappingRow(
        "P3/9.2",
        "carries an **enumeration**",
        "vf_suffix is enumerated and validated in code",
        field_path="ParameterSpec.enum",
    ),
    MappingRow(
        "P3/9.3",
        "`match_schema` is **derived**",
        "release_schema plus vf_suffix, with a fallback when the suffix is absent",
        field_path="TemplateDerivation.fallback",
    ),
    MappingRow(
        "P3/9.4",
        "**derived from a derived parameter**",
        "analytics_schema derives from match_schema, which itself derives",
        field_path="validate_parameter_specs",
    ),
    MappingRow(
        "P3/9.5",
        "`year` is derived by **string slice**",
        "param.run_year",
        field_path="SubstringDerivation.source",
    ),
    MappingRow(
        "P3/9.6",
        "**The cross-parameter constraint lives in an argparse help string**",
        "nothing enforces it and violating it silently produces a smaller audience",
        field_path="ParameterConstraintViolated",
    ),
    MappingRow(
        "P3/10.1",
        "`record_year` has **four** distinct provenances in this slice",
        "three of the four are sentinels; sentinel domains must be declared",
        field_path="SentinelDomain",
    ),
    MappingRow(
        "P3/10.3",
        "`NULL::int record_year` — for standard units **without** a `facts_table`",
        "the age comparison becomes NULL and the row is dropped, silently",
        field_path="SentinelEffect.DROPS_ROW",
    ),
    MappingRow(
        "P3/10.4",
        "`-1 as record_year` (with the author's own comment",
        "the age ceiling becomes a no-op",
        field_path="SentinelEffect.WIDENS_PREDICATE",
    ),
    MappingRow(
        "P3/10.5",
        "a **hardcoded literal year** in every coworkers custom unit",
        "a frozen record_year literal widens the window by one year per rebuild",
        decision=(
            "a hardcoded vintage is neither a sentinel nor a parameter of this shard: it is a "
            "projected constant on the resolution, which lands with the source model in "
            "dsm-task-01d. recorded here so the row is not read as a ParameterSpec gap"
        ),
    ),
    MappingRow(
        "P3/10.6",
        "The age filter binds `record_year` **unqualified**",
        "the one-name-two-semantics trap the resolved.* namespace removes",
        field_path="Namespace.RESOLVED",
    ),
    MappingRow(
        "P3/10.7",
        "Three different working-age windows are implied",
        "working-age stays an ordinary predicate, never a framework primitive",
        decision=(
            "no working-age primitive exists in the model. three windows already run in "
            "production (21-69, <=75, 35-70), which settles it as Ripple binding policy "
            "expressed as an ordinary Qualification.predicate"
        ),
    ),
    MappingRow(
        "P3/13.1",
        "is authored **twice** with different predicates",
        "one unit, two resolutions",
        field_path="Unit.resolutions",
    ),
    MappingRow(
        "P3/13.3",
        "**But `audience_unit_dict` is last-wins**",
        "one name meaning the union of two in SQL and the second one in reporting",
        decision=(
            "neither is reproduced. a duplicate authored name is a parse error "
            "(validate_unique_unit_names); a genuinely plural unit is authored once with "
            "several entries in Unit.resolutions, which is what the emitted SQL already does"
        ),
    ),
    MappingRow(
        "P3/13.4",
        "`Unit.name` uniqueness must be enforced at authoring",
        "the wide pivot collapses duplicates, so the duplication is invisible downstream",
        field_path="validate_unique_unit_names",
    ),
    MappingRow(
        "P3/14.5",
        "Predicate operator vocabulary actually used in this slice",
        "LIKE / NOT LIKE / IN / NOT IN / comparisons / IS NULL / NOT / AND / OR / casts",
        field_path="ComparisonOperator",
    ),
    # ---- Part 4, the UHG audiences ---------------------------------------
    MappingRow(
        "P4/2",
        "Entity grain is `voterbase_id` throughout",
        "the entity column",
        field_path="GrainSpec.entity_column",
    ),
    MappingRow(
        "P4/3",
        "Grain **rename** at final delivery",
        "voterbase_id AS individual_id",
        field_path="GrainSpec.delivered_alias",
    ),
    MappingRow(
        "P4/15",
        "**Unit names contain spaces**",
        "three in the settings file and three in the units file",
        field_path="Unit.name",
    ),
    MappingRow(
        "P4/18",
        "**Duplicate unit name in one units file**",
        "journalists_health_policy declared twice with different sources, joins, and predicates",
        field_path="Unit.resolutions",
    ),
    MappingRow(
        "P4/19",
        "Duplicate name is **NOT last-wins in emission**",
        "both bodies render as two INSERTs under the same label",
        field_path="Unit.resolutions",
    ),
    MappingRow(
        "P4/20",
        "Duplicate-named unit collapses to **one** wide flag column",
        "emits defaults to the name and the identifier is derived once per unit",
        field_path="Unit.emitted_labels",
    ),
    MappingRow(
        "P4/21",
        "**One authored unit emitting two labelled units**",
        "knowwho_leg emits federal_legislators and state_legislators",
        field_path="Unit.emits",
    ),
    MappingRow(
        "P4/26",
        "has **no unit concept at all**",
        "one flat entity list with no unit column and no labels",
        decision=(
            "a definition with zero units is representable, and whether the container requires "
            "at least one Unit is a DatasetDefinition question owned by dsm-task-01d. Unit "
            "itself is unchanged: it requires a name and at least one resolution"
        ),
    ),
    MappingRow(
        "P4/28",
        "**Omitting `facts_table` is a bridge-only resolution**",
        "no fact join emitted",
        field_path="Resolution.source",
    ),
    MappingRow(
        "P4/34",
        "**`record_year` sentinel: `NULL::int`**",
        "declared sentinel domain rather than a compiler-emitted null-extension",
        field_path="SentinelKind.NULL",
    ),
    MappingRow(
        "P4/58",
        "**Authored filter order is preserved byte-for-byte in emission**",
        "all_of list ordering is semantic-preserving and must round-trip",
        field_path="Predicate.all_of",
    ),
    MappingRow(
        "P4/61",
        "**`ILIKE` appears zero times in this scope**",
        "the UHG audiences use LIKE plus lower() and POSIX regex",
        decision=(
            "a visible absence, recorded rather than dropped. the operator vocabulary carries "
            "ILIKE regardless, on the universal-2026 evidence"
        ),
    ),
    MappingRow(
        "P4/62",
        "POSIX regex `~` (case-sensitive)",
        "facts.employer ~ 'cabinet$'",
        field_path="ComparisonOperator.REGEX",
    ),
    MappingRow(
        "P4/63",
        "POSIX regex `~*` (case-insensitive), **2 sites in one unit**",
        "case-insensitive POSIX regex",
        field_path="ComparisonOperator.REGEX_CASE_INSENSITIVE",
    ),
    MappingRow(
        "P4/64",
        "**Mixed single and doubled backslash escapes inside one regex literal**",
        "the literal must round-trip byte-exactly",
        field_path="LiteralExpression.literal",
    ),
    MappingRow(
        "P4/69",
        "**Three distinct age windows in this scope alone**",
        "<= 75, <= 85, BETWEEN 35 and 70",
        decision=(
            "an ordinary predicate. BETWEEN itself is an Expression surface owned by "
            "dsm-task-01d; the two-sided window is expressible today as two Comparisons "
            "under Predicate.all_of"
        ),
    ),
    MappingRow(
        "P4/70",
        "Unqualified column references in authored predicates",
        "twelve distinct bare columns the namespace explicitly replaces",
        field_path="Reference",
    ),
    MappingRow(
        "P4/71",
        "**Alias drift is observable**",
        "the authored knowwho alias vanishes and is replaced by facts in the emitted SQL",
        field_path="Namespace.SOURCE",
    ),
    MappingRow(
        "P4/82",
        "**Qualification scoped to a named unit *set***",
        "six candidate-count tiers each over an explicit unit list",
        field_path="Qualification.applies_to",
    ),
    MappingRow(
        "P4/83",
        "**Two-branch age window, the second branch scoped to a 5-unit set**",
        "age <= 75 OR (age <= 85 AND unit in (five names))",
        field_path="Qualification.name",
    ),
    MappingRow(
        "P4/84",
        "Qualification predicates bind to the **materialized long row**",
        "candidate_count is already MIN and record_year already MAX at that stage",
        field_path="BindingStage.QUALIFICATION",
    ),
    MappingRow(
        "P4/85",
        "**A unit is silently annihilated by qualification**",
        "department_of_commerce is emitted and named in no arm, so every row is dropped",
        field_path="UnqualifiedUnits",
    ),
    MappingRow(
        "P4/127",
        "a schema-valued parameter resolved at render",
        "match_schema as a parameter",
        field_path="ParameterSpec",
    ),
    MappingRow(
        "P4/128",
        "The renderer requires five further parameters to be *referenced*",
        "a hard cross-parameter constraint stated only in a jinja comment",
        decision=(
            "not carried forward: this is a renderer-implementation leak, not a semantic. "
            "ParameterSpec declares parameters structurally, so a body that does not use a "
            "declared parameter is legal and needs no comment"
        ),
    ),
    MappingRow(
        "P4/130",
        "**Cross-parameter constraint (commercial-file flag / voter-file vendor) does not appear in this scope**",
        "verified absent from the UHG files; it lives in the renderer",
        decision=(
            "a visible absence, recorded rather than dropped. the constraint is modelled on "
            "the renderer evidence (P2/B1/R11, P3/9.6, P5/135)"
        ),
    ),
    # ---- Part 5, the universal 2026 audiences -----------------------------
    MappingRow(
        "P5/1",
        "Per-unit max match-candidate threshold",
        "a comparison on resolved.candidate_count with applies_to naming the unit",
        field_path="Qualification.predicate",
    ),
    MappingRow(
        "P5/5",
        "Threshold value differs per audience (core 10, expansion 5)",
        "literal in the qualification predicate, or promoted to a run parameter",
        decision=(
            "both are expressible and the choice is per definition: a literal lives in "
            "Comparison.right, and promoting it to a parameter makes it a ParameterSpec that "
            "ParameterSweep can then sweep. the sweep in P1/1.10 is the argument for promoting it"
        ),
    ),
    MappingRow(
        "P5/10",
        "`ILIKE` operator — 28 occurrences in this one file",
        "case-preserving on emit",
        decision=(
            "the operator KEYWORD is a closed vocabulary and is canonicalised to ILIKE on "
            "input, accepting the corpus's two lowercase spellings. what round-trips "
            "byte-exactly is the OPERAND literal, which is the thing a lossy re-emission "
            "would corrupt"
        ),
    ),
    MappingRow(
        "P5/11",
        "**POSIX case-insensitive regex `~*` with a parenthesised alternation**",
        "the ~* operator",
        field_path="ComparisonOperator.REGEX_CASE_INSENSITIVE",
    ),
    MappingRow(
        "P5/28",
        "`record_year` is a **GROUP BY key** here",
        "already a MAX or a group key depending on the resolution path",
        field_path="Namespace.RESOLVED",
    ),
    MappingRow(
        "P5/50",
        "a **different quality/age policy from core**",
        "qualification policy is per definition",
        field_path="Qualification",
    ),
    MappingRow(
        "P5/120",
        "**Hand-unrolled parameter sweep**",
        "ten near-identical arms differing only in a threshold literal",
        field_path="ParameterSpec.sweep",
    ),
    MappingRow(
        "P5/121",
        "The swept value is projected as a labelling literal column",
        "the sweep emits the value it swept",
        field_path="ParameterSweep.emit_column",
    ),
    MappingRow(
        "P5/133",
        "**Duplicate unit names → last-wins** in a dict comprehension",
        "uniqueness enforced at parse",
        field_path="validate_unique_unit_names",
    ),
    MappingRow(
        "P5/134",
        "Settings/units set-equality check",
        "a qualification must name a declared unit",
        field_path="validate_qualification_coverage",
    ),
    MappingRow(
        "P5/135",
        "**Cross-parameter constraint living in an argparse help string**",
        "tsmart_comm only applicable when using TargetSmart",
        field_path="ParameterConstraint.requires_one_of",
    ),
    MappingRow(
        "P5/136",
        '**Derived parameters**: `match_schema = f"{release_schema}_{vf_suffix}"`',
        "match_schema and analytics_schema",
        field_path="ParameterSpec.derivation",
    ),
    MappingRow(
        "P5/137",
        "**Derived parameter** `year = date[:4]`",
        "feeding the age window as param.run_year",
        field_path="SubstringDerivation",
    ),
    MappingRow(
        "P5/139",
        "**Qualification scoped to named unit sets**, OR-chained per unit",
        "authored per unit but declarable over a unit set",
        field_path="Qualification.applies_to",
    ),
    MappingRow(
        "P5/140",
        "**Namespace binding for post-aggregate long columns**",
        "binding them to source.* or bridge.* silently mis-binds the working-age filter",
        field_path="bindable_namespaces",
    ),
    MappingRow(
        "P5/169",
        "Unit names containing `>` or spaces",
        "no collision or truncation in this scope; the derivation is still a pure function",
        field_path="Unit.name",
    ),
    MappingRow(
        "P5/170",
        "`record_year` sentinels (`-1`, hardcoded literals)",
        "sentinel domains are declared, not implied",
        field_path="SentinelBinding.target",
    ),
    MappingRow(
        "P5/172",
        "One authored unit emitting **two labelled units**",
        "the model supports it; no instance in that scope",
        field_path="Unit.emits",
    ),
    # ---- Part 6, the sibling prototype agent ------------------------------
    MappingRow(
        "P6/A4",
        "is a *derived* parameter",
        "the derivation stated only in a JSON-schema description string",
        field_path="ParameterSpec.derivation",
    ),
    MappingRow(
        "P6/A5",
        "`vf_suffix` enum `{l2, ts}`",
        "the voter-file vendor selector",
        field_path="ParameterSpec.enum",
    ),
    MappingRow(
        "P6/A6",
        "a cross-parameter constraint living in a description string, unenforced",
        "the constraint is duplicated in the agent's tool schema too",
        field_path="ParameterSpec.constraints",
    ),
    MappingRow(
        "P6/D5",
        "Test asserts UTF-8 unit names round-trip",
        "non-ASCII in a unit name is an accepted input",
        field_path="Unit.name",
    ),
    MappingRow(
        "P6/E14",
        "**NEVER use `ILIKE` or regex**",
        "the prototype prompt rule",
        decision=(
            "deleted rather than carried forward. the rule was safety theatre over a "
            "string-paste emitter; production uses both, so the operator vocabulary carries "
            "ILIKE and POSIX ~ / ~* and the emitter owns escaping"
        ),
    ),
    MappingRow(
        "P6/E18",
        "Naming rules: lowercase, underscores, no spaces, max 100 chars",
        "the prompt's rule is narrower than production",
        field_path="Unit.name",
    ),
)


def _resolve(path: str) -> object:
    """resolve a dotted path against the definition package.

    :param path: dotted path, e.g. ``Unit.resolutions`` or ``bindable_namespaces``
    :ptype path: str
    :returns: the resolved object
    :rtype: object
    :raises AttributeError: path does not resolve
    """
    head, _, tail = path.partition(".")
    target: object = getattr(definition, head)
    for segment in tail.split(".") if tail else []:
        model_fields = getattr(target, "model_fields", None)
        if isinstance(model_fields, dict) and segment in model_fields:
            target = model_fields[segment]
        else:
            target = getattr(target, segment)
    return target


class TestRowTableShape:
    """the mapping's own gate, restated over this shard's slice."""

    def test_no_row_is_blank(self) -> None:
        blank = [row.row_id for row in IN_SCOPE_ROWS if not (row.field_path or row.decision)]
        assert blank == []

    def test_every_row_resolves_to_exactly_one_of_field_or_decision(self) -> None:
        both = [row.row_id for row in IN_SCOPE_ROWS if row.field_path and row.decision]
        assert both == []

    def test_row_ids_are_unique(self) -> None:
        row_ids = [row.row_id for row in IN_SCOPE_ROWS]
        assert len(row_ids) == len(set(row_ids))

    def test_every_row_carries_a_semantic(self) -> None:
        assert [row.row_id for row in IN_SCOPE_ROWS if not row.semantic.strip()] == []


class TestFieldReachability:
    """every field-resolved row names something that exists on the model."""

    @pytest.mark.parametrize("row", [row for row in IN_SCOPE_ROWS if row.field_path], ids=lambda row: row.row_id)
    def test_field_path_resolves(self, row: MappingRow) -> None:
        assert row.field_path is not None
        _resolve(row.field_path)

    def test_every_named_symbol_is_exported(self) -> None:
        exported = set(definition.__all__)
        heads = {row.field_path.partition(".")[0] for row in IN_SCOPE_ROWS if row.field_path is not None}
        assert heads <= exported


class TestDecisionRows:
    """a recorded decision is a sentence, not a shrug."""

    @pytest.mark.parametrize("row", [row for row in IN_SCOPE_ROWS if row.decision], ids=lambda row: row.row_id)
    def test_decision_is_substantive(self, row: MappingRow) -> None:
        assert row.decision is not None
        assert len(row.decision.split()) >= 12


def _corpus_mapping_path() -> Path | None:
    """locate ``corpus-mapping.md`` in the sibling ripple repo.

    :returns: path to the mapping document, or ``None`` when the sibling
        repo is not reachable from this checkout
    :rtype: pathlib.Path | None
    """
    override = os.environ.get("RIPPLE_DOCS_DIR")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override) / "corpus-mapping.md")
    repo_root = Path(__file__).resolve().parents[5]
    candidates.append(repo_root.parent / "14-eng-ai-bot-agent-ripple" / "docs" / "corpus-mapping.md")
    found = next((candidate for candidate in candidates if candidate.is_file()), None)
    return found


_MAPPING_PATH = _corpus_mapping_path()
_MAPPING_TEXT = _MAPPING_PATH.read_text(encoding="utf-8") if _MAPPING_PATH else ""

pytestmark_reason = "the sibling ripple repo carrying docs/corpus-mapping.md is not reachable"


@pytest.mark.skipif(_MAPPING_PATH is None, reason=pytestmark_reason)
class TestMappingDocumentCrossCheck:
    """re-verify the checked-in table against the live mapping document.

    3tears must not take a build dependency on the ripple repo, so the row
    table above is authoritative for CI. When the sibling repo IS present,
    every anchor is re-verified against it -- so a row renamed or deleted
    upstream fails here rather than rotting silently.
    """

    @pytest.mark.parametrize("row", IN_SCOPE_ROWS, ids=lambda row: row.row_id)
    def test_anchor_appears_in_the_mapping(self, row: MappingRow) -> None:
        assert row.anchor in _MAPPING_TEXT

    def test_the_mapping_document_gate_still_holds(self) -> None:
        empty_third_column = re.compile(r"^\|[^|\n]*\|[^|\n]*\|[ \t]*\|", re.MULTILINE)
        assert empty_third_column.search(_MAPPING_TEXT) is None
