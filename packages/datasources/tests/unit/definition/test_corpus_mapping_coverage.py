"""corpus-mapping coverage for the definition model.

This is the model shards' acceptance bar, not a fixture set. D3 is
explicit that ``dsm-task-01a``-``d`` are accepted against
``corpus-task-01``'s mapping rather than against fixtures drawn from the
design, because a fixture set drawn from the same document lets the model
pass by omission.

The module has TWO halves, and they check different things on purpose.

**The curated table** (:data:`IN_SCOPE_ROWS`) is the CI-authoritative
half. Each row names a semantic the corpus carries, an ``anchor`` that
must appear verbatim in ``docs/corpus-mapping.md``, and exactly one
resolution: a dotted path that must resolve against
:mod:`threetears.datasources.definition`, or a recorded decision. It grew
one shard at a time -- ``-01a`` seeded it, ``-01b`` and ``-01c`` widened
it, and ``-01d`` finished it.

**The whole-mapping half** (:class:`TestWholeMappingCoverage`) is
DSM-01D-20 and is built the other way round: it enumerates every row of
the mapping document MECHANICALLY and requires each to resolve to a model
symbol that exists, a recorded decision, or a recorded resolution class.
The curated table alone cannot discharge the bar, because a model can
pass a table it also wrote.

The mapping document lives in the sibling ripple repo, which 3tears must
not take a build dependency on. The curated table is therefore checked in
here and holds on its own; the document-driven half skips when that repo
is not reachable, and :class:`TestMappingDocumentCrossCheck` re-verifies
every curated anchor against the live document when it is.
"""

from __future__ import annotations

import collections
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
    # ---- dsm-task-01b: exclusions -----------------------------------------
    MappingRow(
        "P1/A9-exclusion-stages",
        "**`exclude_existing` lives here too, at a different stage from the residuals.**",
        "the residual units anti-join the resolved long table while exclude_existing anti-joins the qualified set",
        field_path="ExclusionSpec.stage",
    ),
    MappingRow(
        "P3/4.1",
        "**Self-referencing anti-join**",
        "a unit LEFT-joins the table it is inserting into and keeps only the non-matches",
        field_path="ExclusionSpec.all_prior",
    ),
    MappingRow(
        "P3/4.2",
        "**Six residual sites in this slice**",
        "six units carry the same anti-join predicate against their own accumulator",
        field_path="ExclusionSpec",
    ),
    MappingRow(
        "P3/4.3",
        "The anti-join key is the **entity alone**",
        "entity-keyed while the long grain is (unit, list_id, voterbase_id)",
        field_path="ExclusionSpec.key_columns",
    ),
    MappingRow(
        "P3/4.4",
        "The anti-join sits in `WHERE`, **before** the `GROUP BY`",
        "pre-aggregate, before the group-by computing MIN(candidate_count)",
        field_path="ExclusionLevel.PRE_AGGREGATE",
    ),
    MappingRow(
        "P3/4.5",
        "The subtracted set is the **resolved** rows",
        "the residual runs during stage 1 and qualification is stage 2",
        field_path="ArtifactStage.RESOLVED",
    ),
    MappingRow(
        "P3/4.6",
        "**The residual extent is order-dependent and the order is `os.listdir`**",
        "each custom unit excludes whatever the earlier ones inserted, in filesystem order",
        field_path="expand_all_prior",
    ),
    MappingRow(
        "P3/4.7",
        "**D7b instance**",
        "two units anti-join the same accumulator with a byte-identical predicate",
        decision=(
            "D7b chose non_overlap first, and this model expresses it by authored unit order alone: "
            "expand_all_prior reads that order, so the chosen direction lands in the definition and "
            "in the content hash rather than in an enumeration accident"
        ),
    ),
    MappingRow(
        "P3/4.8",
        "A second, weaker ordering dependency",
        "the two coworkers units have disjoint filters so their mutual order does not change membership",
        decision=(
            "recorded, not modelled separately. the pair is expressed by the same authored order as "
            "D7b, and parity must not score it as a second undecided direction because disjoint "
            "job_level filters make their mutual order immaterial"
        ),
    ),
    MappingRow(
        "P3/4.9",
        "The **long table is the accumulator**",
        "every custom unit reads the partially-built long table mid-build",
        field_path="ArtifactRef.stage",
    ),
    MappingRow(
        "P3/4.10",
        "A residual expressed as a **positive INNER join**",
        "an intersection and a difference in one unit",
        decision=(
            "two declarations composed, never one convenience: the INNER join is a resolution-stage "
            "intersect (ResolutionIntersect) and the anti-join is the unit's own ExclusionSpec, so "
            "the definition says which rows each removes"
        ),
    ),
    MappingRow(
        "P2/B6/S16",
        "**The residual mechanism, in full.**",
        "all four dimensions confirmed present and all four must be authored",
        field_path="ExclusionSpec",
    ),
    MappingRow(
        "P4/113",
        "**A post-composition exclusion against a computed cohort**",
        "a staff cohort built from the composed audience, then anti-joined out",
        field_path="ExclusionLevel.POST_AGGREGATE",
    ),
    MappingRow(
        "P4/114",
        "The staff cohort's own membership rule is 5 `LIKE` patterns",
        "the subtrahend is a computed relation, not a unit of this definition",
        field_path="ArtifactRef.table",
    ),
    MappingRow(
        "P4/135",
        "**Upstream audience reference, ×2, unioned into a single exclusion set**",
        "two upstream datasets subtracted as one exclusion",
        field_path="ArtifactRef.dataset",
    ),
    MappingRow(
        "P5/57",
        "Exclusion of upstream members",
        "NOT IN over an upstream wide artifact, entity-keyed and pre-aggregate",
        field_path="ExclusionSpec.subtrahends",
    ),
    MappingRow(
        "P5/175",
        "Chained residual exclusion",
        "not present in the universal audiences; all three subtract an upstream, never a sibling unit",
        decision=(
            "a visible absence, recorded rather than dropped. chained exclusion is exercised by the "
            "amazon audiences and the model carries it through expand_all_prior, whose expansion is "
            "what forces a materialized intermediate at every level"
        ),
    ),
    MappingRow(
        "P2/T41",
        "note the stage differs from `ExclusionSpec`",
        "exclude_existing anti-joins the qualified set while the residuals anti-join the resolved one",
        decision=(
            "Expansion.exclude_existing stays dsm-task-01d's field; what lands here is the reason it "
            "cannot share a default with the residuals, which is that the two committed stages differ "
            "and ExclusionSpec.stage is therefore required"
        ),
    ),
    # ---- dsm-task-01b: rollups --------------------------------------------
    MappingRow(
        "P3/2.1",
        "a per-unit label naming the group the unit belongs to",
        "the corpus authors membership on the member; the model authors it on the rollup",
        field_path="Rollup.members",
    ),
    MappingRow(
        "P3/2.2",
        "The rollup label is **stamped per long row**",
        "a rollup_unit column in the long DDL, not derived at delivery",
        field_path="RollupEmit.LONG_LABEL",
    ),
    MappingRow(
        "P3/2.3",
        "A rollup may have exactly one member",
        "single-member rollups are legal and a member may carry the rollup's own name",
        field_path="Rollup.members",
    ),
    MappingRow(
        "P3/2.5",
        "client-requested aggregation the tool does not support",
        "the rollup_unit column exists in the reference SQL and no committed template emits it",
        field_path="Rollup",
    ),
    MappingRow(
        "P3/2.6",
        "No `otherwise` / ELSE bucket exists anywhere in the corpus slice",
        "every amazon_tech_audience unit carries an explicit rollup_unit",
        field_path="Rollup.otherwise",
    ),
    MappingRow(
        "P2/T80",
        "**zero template support**",
        "20 authored rollup_unit occurrences over 7 distinct names and no template site",
        field_path="Rollup.emit",
    ),
    MappingRow(
        "P4/98",
        "**`federal_level` rollup**",
        "a named group of 14 units emitting a wide flag",
        field_path="RollupEmit.WIDE_FLAG",
    ),
    MappingRow(
        "P4/99",
        "**`state_level` rollup**",
        "a named group of 8 units emitting a wide flag",
        field_path="Rollup.members",
    ),
    MappingRow(
        "P4/100",
        "Rollups are **not a partition**",
        "14 + 8 of 23 units, and 83 entities are in both",
        decision=(
            "overlap ACROSS rollups is legal and is not validated away; only a member repeated WITHIN "
            "one rollup is refused, because first match wins makes the second copy dead. the residual "
            "unit is dsm-task-01a's validate_qualification_coverage, not Rollup.otherwise"
        ),
    ),
    MappingRow(
        "P4/101",
        "an ordered first-match-wins categorisation over two rollup flags",
        "government_level over federal_level and state_level, with an implicit NULL otherwise",
        field_path="Rollup.members",
    ),
    MappingRow(
        "P4/102",
        "Rollup label carried through to the final delivered artifact",
        "the label reaches the delivered projection, not only the wide flag",
        field_path="RollupEmit.PROVENANCE_LABEL",
    ),
    MappingRow(
        "P4/103",
        "the rollups are hand-written SQL in `full_uhg_audience.sql`",
        "the whole rollup layer sits outside the tool",
        field_path="Rollup",
    ),
    MappingRow(
        "P4/92",
        "The same ladder re-expressed as an **ordinal string**",
        "MAX() over ordinal-prefixed strings is a rollup encoded in string collation",
        decision=(
            "not carried forward as a Rollup: the ordinal prefix is an ordering smuggled into a label "
            "and MAX over collation is an Expression, not a membership rule. Rollup.members is ordered "
            "and first match wins, which states the same intent without the encoding"
        ),
    ),
    MappingRow(
        "P5/108",
        "**Ranked precedence categories with ordinal-prefixed labels**",
        "five categories over eight core units, stamped as a long label",
        field_path="Rollup.emit",
    ),
    MappingRow(
        "P5/109",
        "The same, over expansion units: 2 categories over 3 units",
        "a second rollup scoped to the other upstream",
        field_path="Rollup.over",
    ),
    MappingRow(
        "P5/110",
        "Multi-unit rollup members",
        "a rollup member list naming several units",
        field_path="Rollup.members",
    ),
    MappingRow(
        "P5/111",
        "catch-alls — both **unreachable**",
        "unmapped_core and unmapped_expansion cover 0 units and are still declared",
        field_path="Rollup.otherwise",
    ),
    MappingRow(
        "P5/113",
        "The rollup is computed **over the provenance artifact**",
        "not over long or wide",
        field_path="Rollup.over",
    ),
    MappingRow(
        "P5/157",
        "`rollup_unit` (the YAML key)",
        "absent from the universal audiences, which express a rollup in raw SQL instead",
        decision=(
            "a visible absence, recorded rather than dropped. Rollup is modelled on the amazon_tech "
            "and UHG evidence, and the universal audiences' raw-SQL rollups are the same semantic "
            "authored outside the tool rather than a different one"
        ),
    ),
    # ---- dsm-task-01b: set algebra ----------------------------------------
    MappingRow(
        "P3/6.1",
        "Composition is the **implicit union of every unit**",
        "every unit INSERTs into one table and nothing filters that union",
        field_path="SetExpr.is_default_union",
    ),
    MappingRow(
        "P2/A8-per-term",
        "So relations at the pivot are **per-composition-term**",
        "each composition term carries its own relations as well as its own projection",
        decision=(
            "per-term RelationRef is dsm-task-01c's element type and lands on SetTerm when it exists. "
            "what this shard settles is that the term is the unit of both, so a per-term relation has "
            "a term to hang off rather than a single spec over the whole composition"
        ),
    ),
    MappingRow(
        "P2/T52",
        "Wide pivot: one zero-filled column per relationship on the influencer branch",
        "one branch zero-fills a column the other computes",
        field_path="SetTerm.projection",
    ),
    MappingRow(
        "P4/109",
        "**Composition is a two-term union with per-term projection**",
        "the policymaker branch projects three rollup flags and the opinion-elite branch NULL for all three",
        field_path="SetTerm.projection",
    ),
    MappingRow(
        "P4/106",
        "is computed by two different mechanisms on the two union branches**",
        "one delivered column derived two incompatible ways, one per branch",
        field_path="TermColumn.value",
    ),
    MappingRow(
        "P5/84",
        "**Two composition terms, each with its own projection**",
        "the influencer branch projects unit flags and 0 householders; the relationship branch the reverse",
        field_path="SetTerm.projection",
    ),
    MappingRow(
        "P5/100",
        "Composition is `UNION ALL` of **two upstream wide artifacts**",
        "terms are dataset terms, here two upstream datasets",
        field_path="SetTerm.upstream",
    ),
    MappingRow(
        "P5/101",
        "The discriminator column is a **per-term literal**",
        "the term's identity made into data, then read by the precedence",
        field_path="TermColumn.value",
    ),
    MappingRow(
        "P5/122",
        "Composition of two upstream audiences by **`UNION` (not `UNION ALL`)**",
        "a union applying no precedence at all, unlike the ranked composition",
        field_path="SetOperator.UNION",
    ),
    MappingRow(
        "P5/173",
        "`intersect` composition",
        "absent from the universal audiences; the instance is in uhg_healthcare_providers",
        field_path="SetOperator.INTERSECT",
    ),
    MappingRow(
        "P6/G17",
        "`intersect` carries payload",
        "the intersection delivers aggregated columns drawn from both sides",
        field_path="IntersectColumn.term",
    ),
    MappingRow(
        "P4/110",
        "The union resolves overlap by **PM-wins precedence**",
        "an anti-join on the losing term, keyed on voterbase_id",
        field_path="SetExpr.dedup_order",
    ),
    MappingRow(
        "P4/111",
        "The **delivered category is not the dedup order**",
        "the category is computed inside each term by a different rule",
        field_path="SetExpr.category_order",
    ),
    MappingRow(
        "P2/T81",
        "Ranked precedence category over unit sets",
        "a CASE WHEN unit in (...) category over a named unit set",
        field_path="LabelArm.when",
    ),
    MappingRow(
        "P5/96",
        "**Dedup precedence order**",
        "ROW_NUMBER partitioned by voterbase_id, ordered core before expansion",
        field_path="SetExpr.dedup_key_columns",
    ),
    MappingRow(
        "P5/97",
        "**Delivered-label precedence order**",
        "householders before core before expansion, a different order with a third arm",
        field_path="SetExpr.category_order",
    ),
    MappingRow(
        "P5/98",
        "The label's **first arm tests a flag column, not a dataset term**",
        "WHEN householders = 1, which no list of term names can express",
        field_path="LabelArm.when",
    ),
    MappingRow(
        "P5/99",
        "The label is computed **before** the dedup filter",
        "the evaluation order relative to the dedup must be declared",
        field_path="SetExpr.category_position",
    ),
    MappingRow(
        "P5/103",
        "No tiebreak declared.",
        "one voterbase_id can appear twice within one term and ROW_NUMBER ties arbitrarily",
        field_path="SetExpr.tiebreak",
    ),
    MappingRow(
        "P3/6.18",
        "Ranked precedence / `RankedCategory` / ordered label concatenation",
        "absent from the amazon audiences: no ROW_NUMBER, no ordered category, no two-audience union",
        decision=(
            "a visible absence, recorded rather than dropped. ranked_precedence is modelled on the "
            "universal-2026 and UHG evidence, and this slice's silence is checked rather than "
            "overlooked, which are not the same claim"
        ),
    ),
    # ---- dsm-task-01c: relations, bridges, measures ----------------------
    MappingRow(
        "P1/1.5b",
        "A join is INNER or LEFT depending on whether the unit happens to declare an",
        "the join type flips on the presence of an unrelated field and nothing in the YAML says so",
        field_path="RelationRef.join",
    ),
    MappingRow(
        "P1/1.9",
        "this artifact has **no quality measure**",
        "expansion rows carry candidate_count NULL::int, so the wide MIN runs over an all-NULL column",
        field_path="QualityMeasure.unmeasured_is_null",
    ),
    MappingRow(
        "P2/B4/T06",
        "Quality measure is `MIN(candidate_count)` at grain",
        "the resolution aggregate is computed at an authored grain, not the surrounding group-by",
        field_path="Measure.grain",
    ),
    MappingRow(
        "P2/B4/T07",
        "at a **different** grain",
        "two grains for one named measure; the grain is never inherited",
        field_path="Measure.grain",
    ),
    MappingRow(
        "P2/B4/T09",
        "Presence-dependent join type: `INNER` iff `industries` is present",
        "adding an industry filter changes the audience twice, once by the predicate and once by the join",
        field_path="RelationRef.join",
    ),
    MappingRow(
        "P2/B4/T19",
        "the **join type is authored data**",
        "cat_join_type puts the join kind in the YAML while the target and condition stay fixed",
        field_path="JoinKind",
    ),
    MappingRow(
        "P2/B4/T20",
        "a boolean gating an entity-table join",
        "vf_join injects a join rather than substituting a scalar",
        field_path="RelationRef.when",
    ),
    MappingRow(
        "P2/B4/T21",
        "Per-unit `joins` list with `how`/`table`/`alias`/`on`",
        "four authored fields per join; the alias is what predicates reference",
        field_path="RelationRef.alias",
    ),
    MappingRow(
        "P2/B4/T22",
        "may be a parenthesised **derived-table body**",
        "the join target may be a whole query rather than a relation name",
        field_path="TypedDerivedTable",
    ),
    MappingRow(
        "P2/B4/T23",
        "is the join **condition**, distinct from the body",
        "on is authored separately from table and may reference an earlier alias",
        field_path="RelationRef.on",
    ),
    MappingRow(
        "P2/B4/T25",
        "injects a projection into the inner aggregate",
        "custom_aggregate_sql is a measure computed inside the resolution",
        field_path="MeasureScope.RESOLUTION",
    ),
    MappingRow(
        "P2/B4/T26",
        "injects an outer `WHERE` over the aggregate",
        "custom_aggregate_filters is not a HAVING and the two differ when the grains differ",
        field_path="Measure.filter_position",
    ),
    MappingRow(
        "P2/B4/T29",
        "structural join at 7 sites / 4 stages / 4 produced tables",
        "a parameter that gates structure, which no scalar substitution expresses",
        field_path="RelationRef.when",
    ),
    MappingRow(
        "P2/B4/T30",
        "applied to the expansion branch of the pivot only, never the influencer branch",
        "stage asymmetry: the gate is authored per relation per stage, never once per definition",
        field_path="RelationRef.is_conditional",
    ),
    MappingRow(
        "P2/B4/T36",
        "evaluated in `WHERE`, before the re-aggregation",
        "the threshold tests the pre-re-aggregate value",
        field_path="FilterPosition.OUTER_WHERE",
    ),
    MappingRow(
        "P2/B4/T37",
        "Qualified stage re-aggregates `MIN(candidate_count)` at grain",
        "a MIN of a MIN at a coarser grain than the one the threshold tested",
        field_path="MeasureScope.DELIVERY",
    ),
    MappingRow(
        "P2/B4/T43",
        "Expansion rows carry **no** quality measure",
        "candidate_count and source_name are both NULL, so an unmeasured band needs something to key on",
        field_path="QualityMeasure.unmeasured_is_null",
    ),
    MappingRow(
        "P2/B4/T55",
        "over an all-NULL column for expansion rows",
        "the wide artifact recomputes the measure at delivery scope",
        field_path="Measure.scope",
    ),
    MappingRow(
        "P2/B4/T79",
        "a **second quality measure on a different scale**",
        "source_match_tier and candidate_count cannot share one column",
        field_path="BridgeRef.quality_measures",
    ),
    MappingRow(
        "P2/B6/S11",
        "a **null test on a LEFT-joined column**",
        "the redundancy between the predicate and the join type must survive parity",
        field_path="RelationRef.optional",
    ),
    MappingRow(
        "P3/1.13",
        "a left join immediately narrowed to an inner one by an equality in WHERE",
        "the model must state whether unmatched rows are intended to survive",
        field_path="RelationRef.optional",
    ),
    MappingRow(
        "P3/1.15",
        "a list of **per-unit** FROM extensions, each with `table`, `alias`, `how`, `on`",
        "arbitrary per-unit FROM extension",
        field_path="RelationRef",
    ),
    MappingRow(
        "P3/1.16",
        "a parenthesised SELECT with its own LEFT JOIN to an inner aggregate subquery",
        "the body nests a join of its own, so it is a typed field rather than an escape hatch",
        field_path="TypedDerivedTable.relations",
    ),
    MappingRow(
        "P3/1.17",
        "can reference an alias declared by a *previous* join in the same list",
        "aliases scope left to right, as the emitted FROM clause does",
        field_path="validate_relation_aliases",
    ),
    MappingRow(
        "P3/1.18",
        "the join type is authored per join",
        "how: INNER is data, not an inference",
        field_path="JoinKind.INNER",
    ),
    MappingRow(
        "P3/1.19",
        "the name predicates reference",
        "the alias is a namespace key and cannot be optional",
        field_path="RelationRef.alias",
    ),
    MappingRow(
        "P3/1.20",
        "Nothing in the YAML says so",
        "LEFT versus INNER changes membership for rows with no job_title_fct match",
        decision=(
            "the presence-inference is not carried forward: RelationRef.join is authored per relation "
            "and no field derives it from another field's presence. whether the migration preserves "
            "the resulting row counts is a per-audience reconciliation record, not a model question"
        ),
    ),
    MappingRow(
        "P3/1.23",
        "an authored aggregate expression injected into the resolution's SELECT list",
        "the measure carries its own output alias",
        field_path="Measure.name",
    ),
    MappingRow(
        "P3/1.24",
        "explicit casts and division by a bridge column",
        "SUM(contribution::float * 1.0/mat.candidate_count::float) over source.* and bridge.*",
        field_path="Measure.expression",
    ),
    MappingRow(
        "P3/1.25",
        "a list of predicates over the measure alias",
        "having binds measure.<name> and nothing else",
        field_path="validate_having_measures",
    ),
    MappingRow(
        "P3/1.26",
        "but they render into an **outer `WHERE`** wrapping the aggregate subquery",
        "an outer where, not a having",
        field_path="FilterPosition.OUTER_WHERE",
    ),
    MappingRow(
        "P3/1.27",
        "The measure alias drifts between units",
        "contribution_sum and sum_of_contributions are one computation under two authored names",
        field_path="Measure.name",
    ),
    MappingRow(
        "P3/1.28",
        "a measure computed and then discarded",
        "bundlers declares an aggregate with no filter over it",
        decision=(
            "legal and unchanged: a Measure carries no filter of its own, so declaring one with no "
            "having is expressible and parity must not read the absence as a diff"
        ),
    ),
    MappingRow(
        "P3/1.29",
        "an aggregate the model must know is already aggregated downstream",
        "MIN(candidate_count) is the bridge's quality measure",
        field_path="QualityMeasure.column",
    ),
    MappingRow(
        "P3/5.9",
        "The qualification predicate re-aggregates",
        "MIN over GROUP BY 1,2,3 at a grain the threshold never saw",
        field_path="Measure.grain",
    ),
    MappingRow(
        "P3/9.7",
        "**`tsmart_comm` gates structure, not a scalar**",
        "it injects an INNER JOIN at four stages and omits it entirely otherwise",
        field_path="RelationRef.when",
    ),
    MappingRow(
        "P4/29",
        "Bridge is the match table, aliased `mat`",
        "the bridge is a declared relation with its own alias",
        field_path="BridgeRef.alias",
    ),
    MappingRow(
        "P4/32",
        "is `MIN`-aggregated at resolution",
        "the quality measure is computed inside the resolution, not at delivery",
        field_path="Measure.scope",
    ),
    MappingRow(
        "P4/39",
        "Per-unit `joins:` list, each with `table` / `alias` / `how`",
        "the relation target is authored beside its alias and kind",
        field_path="RelationRef.relation",
    ),
    MappingRow(
        "P4/40",
        "`how:` carries the join kind",
        "INNER and LEFT are both authored in one file",
        field_path="JoinKind",
    ),
    MappingRow(
        "P4/41",
        "**Join body as a derived table** (inline subquery in `table:`)",
        "three distinct inline bodies across two audiences",
        field_path="TypedDerivedTable",
    ),
    MappingRow(
        "P4/42",
        "**Join condition is authored separately from the body**",
        "on: is its own key",
        field_path="RelationRef.on",
    ),
    MappingRow(
        "P4/43",
        "Derived-table body carries its own `GROUP BY` **and `HAVING`**",
        "an aggregate relation, not a filter",
        field_path="TypedDerivedTable.having",
    ),
    MappingRow(
        "P4/44",
        "Join alias is the name predicates reference",
        "alias: edu is used as edu.institution in a predicate",
        field_path="RelationRef.alias",
    ),
    MappingRow(
        "P4/45",
        "may reference the **bridge** alias, not the fact source",
        "a join condition over bridge.*",
        field_path="BridgeRef.alias",
    ),
    MappingRow(
        "P4/46",
        "a per-unit key naming the category table's join kind",
        "cat_join_type names the kind of a governed relation's join",
        field_path="JoinKind",
    ),
    MappingRow(
        "P4/47",
        "declaring `cat_join_type` injects the `cat_union` join; omitting it omits the join entirely",
        "presence of a key decides whether a join exists at all",
        field_path="RelationRef.when",
    ),
    MappingRow(
        "P4/48",
        "a per-unit boolean injecting the entity attribute table",
        "vf_join at seven sites, one per unit",
        field_path="RelationRef.when",
    ),
    MappingRow(
        "P4/51",
        "Structural join at **four** stages for the same relation",
        "one relation reusable at resolution, qualification, and delivery",
        field_path="RelationRef",
    ),
    MappingRow(
        "P4/52",
        "Fact source joined to bridge on `list_id` in every case",
        "the bridge declares its own join path, so on is None",
        field_path="BridgeRef.on",
    ),
    MappingRow(
        "P4/54",
        "**The same LEFT join is used for both polarities**",
        "IS NULL is an anti-join and IS NOT NULL a semi-join off one relation",
        decision=(
            "one RelationRef with join=left and optional=True carries both polarities; which side is "
            "selected is the enclosing predicate's job, so the model does not force a second relation "
            "for the anti-join"
        ),
    ),
    MappingRow(
        "P4/89",
        "a *second*, differently-scaled quality measure on the same bridge",
        "source_match_tier <= 12 beside candidate_count",
        field_path="BridgeRef.quality_measures",
    ),
    MappingRow(
        "P4/90",
        "available only as a code comment",
        "the tier-to-CBSA meaning exists only in a SQL comment",
        decision=(
            "QualityMeasure carries no free-text semantic field on purpose: a caveat that must reach "
            "the model lands in a concept's caveats and then in the imperatives block of a tool "
            "return, never as a description beside the measure"
        ),
    ),
    MappingRow(
        "P4/91",
        "as a boolean relevance flag",
        "MAX(CASE ... END) over a 55-arm LIKE ladder is an ordinary measure expression",
        field_path="Measure.expression",
    ),
    MappingRow(
        "P4/96",
        "appear nowhere in this scope",
        "custom_aggregate_sql and custom_aggregate_filters are absent from the UHG files",
        decision=(
            "a visible absence, recorded rather than dropped. Measure and its filter_position are "
            "modelled on the Amazon evidence at P3/1.23 through P3/1.26 regardless"
        ),
    ),
    MappingRow(
        "P5/14",
        "**Per-unit `joins`**, 4 units, each `{table, alias, how, on}`",
        "relation, alias, join, and on, all authored",
        field_path="RelationRef",
    ),
    MappingRow(
        "P5/15",
        "authored as a free string, separate from the join target",
        "the condition is distinct from the relation",
        field_path="RelationRef.on",
    ),
    MappingRow(
        "P5/29",
        "as the quality measure",
        "min(mat.candidate_count) is match quality where lower is better",
        field_path="QualityMeasure.direction",
    ),
    MappingRow(
        "P5/30",
        "The commercial-file join is **absent from the body** and injected by the wrapper",
        "the join is not in the authored unit at all; a parameter puts it there",
        field_path="RelationRef.when",
    ),
    MappingRow(
        "P5/41",
        "**`tsmart_comm` as a structural join**, unconditional in the rendered output",
        "eight inserts carry the gated join in the rendered SQL",
        field_path="RelationRef.when",
    ),
    MappingRow(
        "P5/42",
        "a semi-join written as an inner join, able to fan out",
        "the joined alias is never referenced anywhere in the statement",
        decision=(
            "expressible today as an ordinary RelationRef whose alias no predicate references; "
            "whether the emitter renders EXISTS or preserves the inner join's fan-out is a compiler "
            "decision, not a model field"
        ),
    ),
    MappingRow(
        "P5/44",
        "Two aggregations of different kinds in one resolution",
        "one column per declared quality measure, never a single quality column",
        field_path="BridgeRef.long_columns",
    ),
    MappingRow(
        "P5/45",
        "the long grain is `(unit, source record, entity)`",
        "the grain is per source record, so one entity is summed once per record",
        field_path="Measure.grain",
    ),
    MappingRow(
        "P5/54",
        "aggregated under a `HAVING`",
        "one unit derives its allowlist from an upstream artifact under a having",
        field_path="validate_having_measures",
    ),
    MappingRow(
        "P5/58",
        "Same exclusion written into a **JOIN `ON`** in one unit and into a **`WHERE`** in the others",
        "for an inner join they are equivalent, for a left join they are not",
        field_path="RelationRef.on",
    ),
    MappingRow(
        "P5/59",
        "**The entire predicate body attached to a join's `ON` clause, with no `WHERE` at all**",
        "on carries a full predicate, not a key pair",
        field_path="RelationRef.on",
    ),
    MappingRow(
        "P5/64",
        "referencing a **select-list alias** defined in the same SELECT",
        "Redshift permits it, so a having naming no declared measure must fail at authoring",
        field_path="UndeclaredMeasure",
    ),
    MappingRow(
        "P5/85",
        "The commercial-file join is applied to the **relationship branch only**",
        "the gate is scoped per composition term",
        field_path="RelationRef.is_conditional",
    ),
    MappingRow(
        "P5/127",
        "**Quality predicate placed in a join `ON`**",
        "it binds the influencer's candidate_count while gating the householder's admission",
        decision=(
            "expressible as RelationRef.on over the influencer's quality column; whether gating an "
            "expanded member by the source member's match quality is intended stays a recorded "
            "decision, because the corpus authors it only by unqualified-name accident"
        ),
    ),
    MappingRow(
        "P5/128",
        "Relation joined **without an alias** and referenced by bare table name",
        "the alias is mandatory in the model",
        field_path="RelationRef.alias",
    ),
    MappingRow(
        "P5/162",
        "zero matches in scope; template at `1_generate_audience_units_table.sql.jinja2:84-86`",
        "cat_join_type is absent from this scope; the join kind is modelled anyway",
        field_path="JoinKind",
    ),
    MappingRow(
        "P5/163",
        "zero matches in scope; template at `1_generate_audience_units_table.sql.jinja2:88-90`",
        "vf_join is absent from this scope; the structural gate is modelled anyway",
        field_path="RelationRef.when",
    ),
    MappingRow(
        "P5/164",
        "**Presence-dependent join type** (`'INNER' if \"industries\" in dict else 'LEFT'`)",
        "no linkedin_audience_units.yaml in this scope, so the inference is unexercised here",
        decision=(
            "not expressible by design: a presence-derived join type is exactly what RelationRef.join "
            "refuses to carry, so the migration authors the join kind each unit actually ran with"
        ),
    ),
    # ---- dsm-task-01d: sources, provenance, delivery, and the hash -------
    MappingRow(
        "P1/1.1",
        "D9b resolves to 15, the template DDL",
        "the pinned provenance contract is the DDL's 15 columns, arbitrated by the positional INSERT",
        field_path="ProvenanceSpec.columns",
    ),
    MappingRow(
        "P1/1.3",
        "Three of the five parity artifacts have no committed reference",
        "which artifacts a definition emits is declared, because the renderer wrote two of five",
        field_path="ArtifactSpec.artifact",
    ),
    MappingRow(
        "P2/B1/R01",
        "Three unit sources concatenated in fixed template order",
        "one flat authored unit list; the three-file split is a prototype layout artifact",
        field_path="DatasetDefinition.units",
    ),
    MappingRow(
        "P2/B1/R17",
        "each require a prior run's table",
        "the dependency on a prior stage becomes a declared reference",
        field_path="ArtifactRef.scope",
    ),
    MappingRow(
        "P2/B2/H02",
        "5 literal entity sets, ~1700 ids total",
        "the hand-match units are a first-class source kind",
        field_path="LiteralEntities.entity_ids",
    ),
    MappingRow(
        "P2/B2/H03",
        "Whitespace-dirty entries — 5 ids with a leading space",
        "five committed ids carry a leading space and would silently match zero rows",
        field_path="LiteralEntities.normalization",
    ),
    MappingRow(
        "P2/B2/H04",
        "Duplicate ids within one set",
        "set semantics; the authored list is stored verbatim and de-duplicated on emission",
        field_path="LiteralEntities.normalized_ids",
    ),
    MappingRow(
        "P2/B2/H05",
        "needs no bri",
        "a literal entity set carries no bridge and no quality measure",
        field_path="EntityIdNormalization.changed_anything",
    ),
    MappingRow(
        "P2/B3/D02",
        "Provenance contract: 14 columns",
        "the readme's 14 was never executable; the pinned contract is the DDL's 15",
        field_path="ProvenanceContractViolation",
    ),
    MappingRow(
        "P2/B3/D06",
        "Relationship output contract: 8 fields",
        "the expansion-provenance artifact is the fifth artifact kind",
        field_path="ArtifactKind.RELATIONSHIP_UNION",
    ),
    MappingRow(
        "P2/B4/T01",
        "Long artifact DDL, 5 columns",
        "each artifact declares its own column set; neither the set nor the artifact list is universal",
        field_path="ArtifactSpec.columns",
    ),
    MappingRow(
        "P2/B4/T02",
        "`GRANT SELECT … TO GROUP INFLUENCERS` at every stage",
        "a warehouse grant to a group, explicitly not hashed",
        field_path="GrantSpec.grantee_kind",
    ),
    MappingRow(
        "P2/B4/T27",
        "Custom units spliced as a derived table",
        "the raw body is a source kind rather than a directory of unrenderable files",
        field_path="RawSelect.raw_sql",
    ),
    MappingRow(
        "P2/B4/T28",
        "Custom-unit provenance spliced with **no wrapper at all**",
        "the body must satisfy the pinned columns and the compiler verifies it",
        field_path="RawSelect.provenance",
    ),
    MappingRow(
        "P2/B4/T39",
        "Expansion union DDL, 8 columns",
        "the expansion provenance artifact declares its own eight columns",
        field_path="ArtifactSpec.grain",
    ),
    MappingRow(
        "P2/B4/T42",
        "Expansion provenance `fact` = `hh.influencer_voterbase_id`",
        "which member brought each expansion member in",
        field_path="Expansion.provenance",
    ),
    MappingRow(
        "P2/B4/T45",
        "Per-edge predicates scoped by unit",
        "the walk is scoped to a unit set, not to the whole audience",
        field_path="Expansion.applies_to",
    ),
    MappingRow(
        "P2/B4/T46",
        "Expansion edge is `{analytics}.household`",
        "the edge is a governed relation reference",
        field_path="Expansion.edge",
    ),
    MappingRow(
        "P2/B4/T49",
        "branches from a **non-grain key**",
        "one committed edge joins on list_id rather than on the entity",
        field_path="Expansion.covers",
    ),
    MappingRow(
        "P2/B4/T51",
        "Wide pivot: one `MAX(CASE WHEN unit = '…' THEN 1 ELSE 0 END)` column per settings entry",
        "the wide artifact is one of the five declared artifacts",
        field_path="ArtifactKind.WIDE",
    ),
    MappingRow(
        "P2/B4/T54",
        "Direct-vs-expansion flag: `1 as influencer` / `0 as influencer`",
        "a delivered derived column over a literal",
        field_path="DeliverySpec.columns",
    ),
    MappingRow(
        "P2/B4/T57",
        "Provenance DDL, 15 columns",
        "the authoritative contract, pinned once in the binding",
        field_path="ProvenanceSpec.column_names",
    ),
    MappingRow(
        "P2/B4/T58",
        "Provenance anchors on the long table",
        "every body inner-joins the long table on the entity and the unit literal",
        field_path="ProvenanceSpec.anchor",
    ),
    MappingRow(
        "P2/B4/T59",
        "a per-source co",
        "a per-source conditional projection, authored per column",
        field_path="ProvenanceColumn.expression",
    ),
    MappingRow(
        "P2/B4/T60",
        "Provenance `GROUP BY` computed by **positional arithmetic**",
        "every provenance body is an aggregate query and its grain is authored",
        field_path="ProvenanceSpec.grain",
    ),
    MappingRow(
        "P2/B4/T62",
        "Provenance column named `linkedin_industry` in the body but `linkedin_industries` in the DDL",
        "the positional INSERT hides the mismatch today and the projection check surfaces it",
        field_path="ProvenanceSpec.verify_columns_against",
    ),
    MappingRow(
        "P2/B4/T63",
        "Provenance `type` column is the `facts_table` value for standard units",
        "a body projects a derived label the unit never had",
        field_path="ProvenanceSpec.measures",
    ),
    MappingRow(
        "P2/B4/T72",
        "No template emits `DISTKEY`/`SORTKEY`/`DISTSTYLE`/`ENCODE`",
        "physical layout is a gap the prototype has, not a semantic to port",
        field_path="PhysicalLayout",
    ),
    MappingRow(
        "P2/B4/T73",
        "Physical layout hand-edited **into the tool's own generated output**",
        "layout attaches to any materialized artifact and is never hashed",
        field_path="ArtifactSpec.layout",
    ),
    MappingRow(
        "P2/B4/T74",
        "Full column-level physical spec",
        "per-column encodings, which Redshift's CTAS grammar cannot express",
        field_path="ColumnEncoding.AZ64",
    ),
    MappingRow(
        "P2/B4/T75",
        "Delivered width is part of the contract",
        "two delivered columns are sized to their exact category strings",
        field_path="DerivedColumn.sql_type",
    ),
    MappingRow(
        "P2/B4/T76",
        "`LISTAGG(DISTINCT …) WITHIN GROUP (ORDER BY …)` — 5 ordered sites",
        "ordered string aggregation, which needs an open expression",
        field_path="DerivedColumn.expression",
    ),
    MappingRow(
        "P2/B4/T78",
        "`is_unique_match` = `MAX(CASE WHEN candidate_count = 1 THEN 1 ELSE 0 END)`",
        "a conditional-aggregate flag computed over a named artifact",
        field_path="DerivedColumn.over",
    ),
    MappingRow(
        "P2/B4/T83",
        "S3 + `COPY` cross-cluster last mile",
        "a cross-warehouse materialization hop on the delivered table",
        decision=(
            "excluded by recorded decision: two datasource rows against DIFFERENT warehouses are "
            "not joinable in one statement, so a materialization hop is out of scope and is named "
            "as such rather than implied by a blanket per-definition rule"
        ),
    ),
    MappingRow(
        "P2/B5/J04",
        "Parameter-declaration-by-comment",
        "a raw body's parameters are a declared signature rather than a comment convention",
        field_path="RawSelect.parameters",
    ),
    MappingRow(
        "P2/B5/J05",
        "checks supplied→used but **never** used→supplied",
        "the signature is verified in both directions at authoring",
        field_path="ParameterSignatureViolation",
    ),
    MappingRow(
        "P3/3.14",
        "A custom unit reads an **out-of-band prior audience table** by name",
        "an upstream reference names the dataset and a resolution policy, never a run id",
        field_path="ArtifactRef.dataset",
    ),
    MappingRow(
        "P4/139",
        "**Same-day upstream/downstream pair**",
        "a draft reference pins that specific run; a policy reference to a draft is refused",
        field_path="UpstreamPin.policy",
    ),
    MappingRow(
        "P4/142",
        "no visibility or grant check is possible",
        "an upstream reference is authorized against visibility and grants, not merely named",
        decision=(
            "specified here and implemented Hub-side: this package cannot see visibility or grants, "
            "so the model carries the reference and the resolution check fails loud when no grant "
            "exists. cross-customer is permitted WHEN GRANTED, and the rule is the grant"
        ),
    ),
    MappingRow(
        "P5/142",
        "**The fifth artifact** `{audience}_relationship_union_{date}`, 8 columns",
        "the artifact enum needs a fifth member; it is the only place expansion rationale exists",
        field_path="ArtifactKind",
    ),
    MappingRow(
        "P6/G4",
        "This is a fourth count for the D9b provenance-contract question",
        "a fourth spelling of the provenance contract, in the prototype agent's own prompt",
        field_path="ProvenanceColumn.name",
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


# ---------------------------------------------------------------------------
# DSM-01D-20: coverage of the WHOLE mapping
#
# The table above is the CI-authoritative slice, hand-curated per shard. This
# section is the acceptance bar, and it is deliberately built the other way
# round: it enumerates every row of the mapping document MECHANICALLY and
# requires each one to resolve. A hand-typed row table cannot discharge the
# bar on its own, because a model can pass a table it also wrote -- which is
# exactly the "pass by omission" failure D3 names.
#
# Every mapping row resolves into one of three things:
#
#   1. a MODEL SYMBOL the document names, which must exist on this package.
#      484 of the 764 rows are this, and it is the anti-omission property:
#      the document says `ProvenanceSpec.columns` and the test fails until
#      that field exists.
#   2. a recorded DECISION -- the document's own `DECISION NEEDED`.
#   3. a recorded RESOLUTION CLASS -- a row resolving to something that is
#      real but is not a field of this model: a namespace binding, an
#      inspection surface, the run record, the dataset record, the compiler
#      or emitter, a parity fixture, or a platform surface in another repo.
#
# Anything outside those three is UNMAPPED and fails. The two reconciliation
# tables below are the only hand-written part, and both are asserted to carry
# no stale entry, so a spelling that stops appearing in the document cannot
# sit here forever pretending to cover something.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolReconciliation:
    """one document spelling that does not resolve on this package as written.

    :ivar symbol: the spelling the mapping document uses
    :ivar resolves_to: dotted path it resolves to here, or ``None`` when the
        symbol names something outside this package entirely
    :ivar rationale: why the two differ, in one sentence
    """

    symbol: str
    resolves_to: str | None
    rationale: str


SYMBOL_RECONCILIATIONS: tuple[SymbolReconciliation, ...] = (
    SymbolReconciliation(
        "UpstreamSource",
        "ArtifactRef",
        "design section 7 names the upstream source kind separately; the model carries ONE "
        "reference type and distinguishes it by ArtifactScope.DATASET, because an upstream "
        "artifact and this definition's own differ only in scope and resolve the same way",
    ),
    SymbolReconciliation(
        "RankedCategory",
        "SetExpr.category_order",
        "the delivered category is a declaration ON the ranked_precedence operator rather than a "
        "type beside it, because its arms are predicates and it is computed before or after the "
        "dedup, which no standalone type could express without repeating SetExpr",
    ),
    SymbolReconciliation(
        "SetExpr.union",
        "SetOperator.UNION",
        "the document spells operators as attributes; they are members of the operator enum",
    ),
    SymbolReconciliation(
        "SetExpr.intersect",
        "SetOperator.INTERSECT",
        "the document spells operators as attributes; they are members of the operator enum",
    ),
    SymbolReconciliation(
        "SetExpr.ranked_precedence",
        "SetOperator.RANKED_PRECEDENCE",
        "the document spells operators as attributes; they are members of the operator enum",
    ),
    SymbolReconciliation(
        "ExclusionSpec.key",
        "ExclusionSpec.key_columns",
        "tests/enforcement/test_secrets_typed.py reserves a bare `key` as a credential name that "
        "must be typed SecretStr, and the anti-join key is a list of column names, so the "
        "enforcement rule and the short spelling cannot both stand",
    ),
    SymbolReconciliation(
        "Measure.having",
        "Resolution.having",
        "a having is authored on the stage that computes the aggregate, and the measure declares "
        "where the filter lands through filter_position; putting a having on the measure would "
        "make one measure's filter position mean two things",
    ),
    SymbolReconciliation(
        "ConceptEntity.sql_fragment",
        "Predicate.concept",
        "the concept ENTITY lives in the knowledge layer, which this package does not depend on; "
        "the definition side of it is the predicate form that names one",
    ),
    SymbolReconciliation(
        "ToolResult",
        None,
        "the platform tool-return type, from the agent SDK; part 6 surveys the prototype AGENT "
        "layer and this row is about a tool surface rather than about definition content",
    ),
    SymbolReconciliation(
        "ToolResult.content",
        None,
        "the same platform tool-return type; the imperatives block lives there and is governed by "
        "the Hub's honesty design, not by the definition model",
    ),
)

_RECONCILED = {entry.symbol: entry for entry in SYMBOL_RECONCILIATIONS}

NON_SYMBOL_TOKENS: frozenset[str] = frozenset({"None", "True", "False", "Literal", "Downloads", "Pydantic", "Redshift"})
"""capitalised tokens inside a code span that are not model symbols.

``None`` / ``True`` / ``False`` / ``Literal`` are typing spellings inside a
quoted field declaration; ``Downloads`` is a filesystem path; ``Pydantic``
and ``Redshift`` are proper nouns.
"""


@dataclass(frozen=True)
class ProseResolution:
    """one mapping row whose resolution is prose the classifier cannot read.

    :ivar row_id: ``P<part>/<row>`` identifier
    :ivar resolves_to: what the row resolves to, in one phrase
    :ivar rationale: why it is not a model symbol
    """

    row_id: str
    resolves_to: str
    rationale: str


PROSE_RESOLUTIONS: tuple[ProseResolution, ...] = (
    ProseResolution(
        "P2/J01",
        "pydantic required and optional fields",
        "StrictUndefined's BEHAVIOUR -- fail loud on a missing field -- is preserved by required "
        "fields; the jinja mechanism itself is not carried forward",
    ),
    ProseResolution(
        "P3/6.1",
        "DatasetDefinition.composition, left None",
        "the default composition is the union of every unit, which is what an omitted composition "
        "means; SetExpr.is_default_union reports it",
    ),
    ProseResolution(
        "P3/6.9",
        "an absence row",
        "recorded as not exercised in the Amazon slice; the evidence for it comes from the UHG and "
        "universal audiences and is carried on their own rows",
    ),
    ProseResolution(
        "P3/14.2",
        "Predicate.concept",
        "the row is evidence that a shared fragment drifts when it is not first-class, which is "
        "the argument for a governed concept rather than a repeated literal",
    ),
    ProseResolution(
        "P4/111",
        "SetExpr.dedup_order beside SetExpr.category_order",
        "the two orders are separate declarations, which is exactly the field split the model carries",
    ),
    ProseResolution(
        "P4/139",
        "UpstreamPolicy.DRAFT_RUN",
        "a same-day pair means a draft reference has to be pinnable, and the pin is the run id "
        "that is deliberately excluded from the content hash",
    ),
    ProseResolution(
        "P4/140",
        "the dataset record's retention",
        "transitive retention closure is platform state on the dataset record (D1), not definition "
        "content; the definition supplies the reference graph it is computed over",
    ),
    ProseResolution(
        "P5/101",
        "SetTerm.projection",
        "the per-term literal is the term's identity made into data, which is what a term-level projection carries",
    ),
    ProseResolution(
        "P5/142",
        "ArtifactKind.RELATIONSHIP_UNION",
        "the fifth artifact kind, added by this shard; the row records that the enum had four",
    ),
    ProseResolution(
        "P6/F14",
        "a concept's caveats in the knowledge layer",
        "a three-valued-logic trap is a caveat the model must be told at the point of action, "
        "which is the knowledge layer's job and not the definition's",
    ),
    ProseResolution(
        "P6/F15",
        "a concept's caveats in the knowledge layer",
        "the same placement: semantics that must reach the model go in caveats, never in a definition field",
    ),
    ProseResolution(
        "P6/F16",
        "a concept's caveats in the knowledge layer",
        "the same placement, recorded per caveat so the gate has no blank",
    ),
    ProseResolution(
        "P6/G4",
        "ProvenanceSpec.columns",
        "a fourth spelling of the provenance contract, in the prototype agent's own prompt, which "
        "is more evidence for pinning the contract once rather than restating it",
    ),
    ProseResolution(
        "P6/G14",
        "a knowledge-layer playbook entry",
        "counting list_id counts source records rather than people, which is a denominator caveat "
        "for the knowledge layer rather than a definition field",
    ),
)

_PROSE_RESOLVED = {entry.row_id: entry for entry in PROSE_RESOLUTIONS}

RESOLUTION_CLASS_PHRASES: dict[str, tuple[str, ...]] = {
    "not_carried_forward": ("not carried forward", "no carry-forward"),
    "inspection_surface": ("inspection surface", "inspect surface", "inspect tool", "inspect capability"),
    "run_record": ("run record", "run-scoped", "run lifecycle", "run/stage", "run's"),
    "dataset_record": ("dataset record",),
    "compiler_or_emitter": (
        "emitter",
        "compiler",
        "ast",
        "topological order",
        "derived identifier",
        "logical→physical",
        "positional",
    ),
    "fixture_or_attributability": (
        "attributab",
        "fixture",
        "non-attributability",
        "d10 input",
        "not exercised",
        "see §",
    ),
    "platform_surface": (
        "new architecture",
        "schema/",
        "knowledge/",
        "agent.yaml",
        "datasources/",
        "claude.md",
        "design §",
        "design section",
    ),
    "cross_reference": ("same as", "corroborat", "see row", "why row"),
    "control_case": ("control case", "no decision"),
    "publish_policy": ("publish policy", "delivery/publish"),
}
"""resolution classes for a row that names no model symbol.

Each is a real destination rather than a shrug: the run record, the dataset
record, the compiler, an inspection surface, a parity fixture, or a surface
in another repo. A row matching none of them is UNMAPPED and fails.
"""

PROSE_SECTION_ROW_IDS: frozenset[str] = frozenset({"P2/A8-per-term"})
"""curated ids pointing at a PROSE section of the mapping rather than a row.

Part 1 and the parts' ``Part A`` sections answer questions in prose and
carry no resolution column, so the parser sees no row. Their authority is
the anchor, which :class:`TestMappingDocumentCrossCheck` verifies verbatim.
"""

_MAPPING_TABLE_HEADERS = frozenset({"Model field / decision", "Resolution"})
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_CODE_SPAN = re.compile(r"`([^`]+)`")
_SYMBOL_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:\.[a-z_][a-z_0-9]*)+|\b[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*\b")
_NAMESPACE_SPAN = re.compile(r"`(source|bridge|entity|resolved|param|measure|rel)\.")


@dataclass(frozen=True)
class DocumentRow:
    """one row parsed out of the mapping document.

    :ivar row_id: ``P<part>/<row>`` identifier
    :ivar section: heading the row sits under
    :ivar resolution: the row's own resolution cell
    :ivar line: line number in the document
    """

    row_id: str
    section: str
    resolution: str
    line: int


def _parse_mapping_rows(text: str) -> tuple[DocumentRow, ...]:
    """enumerate every mapping row in the document.

    A mapping table is one whose final header cell is ``Model field /
    decision`` or ``Resolution``; the document's other tables are counts,
    censuses, and column inventories and carry no resolution.

    :param text: the mapping document
    :ptype text: str
    :returns: every mapping row, in document order
    :rtype: tuple[DocumentRow, ...]
    """
    rows: list[DocumentRow] = []
    part = ""
    section = ""
    header: list[str] = []
    in_table = False
    for number, line in enumerate(text.splitlines(), 1):
        part_heading = re.match(r"^## Part (\d+)", line)
        if part_heading:
            part, section, header, in_table = part_heading.group(1), "", [], False
            continue
        section_heading = re.match(r"^#{3,5} (.+)", line)
        if section_heading:
            section, header, in_table = section_heading.group(1).strip(), [], False
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [cell.strip() for cell in _CELL_SPLIT.split(line.strip())[1:-1]]
        if cells and set("".join(cells)) <= set("-: "):
            in_table = True
            continue
        if not in_table:
            header = cells
            continue
        if header and header[-1] in _MAPPING_TABLE_HEADERS:
            rows.append(DocumentRow(f"P{part}/{cells[0]}", section, cells[-1], number))
    return tuple(rows)


def _named_symbols(resolution: str) -> tuple[str, ...]:
    """model symbols a resolution cell names.

    :param resolution: the row's resolution cell
    :ptype resolution: str
    :returns: symbols in appearance order
    :rtype: tuple[str, ...]
    """
    found: list[str] = []
    for span in _CODE_SPAN.findall(resolution):
        found.extend(token for token in _SYMBOL_TOKEN.findall(span) if token not in NON_SYMBOL_TOKENS)
    return tuple(found)


def _classify(row: DocumentRow) -> str:
    """the class a mapping row resolves into.

    :param row: parsed mapping row
    :ptype row: DocumentRow
    :returns: class name, or ``"unmapped"``
    :rtype: str
    """
    lowered = row.resolution.lower()
    classified = "unmapped"
    if "decision needed" in lowered:
        classified = "decision"
    elif _named_symbols(row.resolution):
        classified = "model_symbol"
    elif row.row_id in _PROSE_RESOLVED:
        classified = "prose_resolution"
    elif _NAMESPACE_SPAN.search(row.resolution):
        classified = "namespace"
    else:
        matched = next(
            (name for name, phrases in RESOLUTION_CLASS_PHRASES.items() if any(p in lowered for p in phrases)),
            None,
        )
        if matched is not None:
            classified = matched
    return classified


_DOCUMENT_ROWS = _parse_mapping_rows(_MAPPING_TEXT)


@pytest.mark.skipif(_MAPPING_PATH is None, reason=pytestmark_reason)
class TestWholeMappingCoverage:
    """DSM-01D-20, checked against the mapping rather than against a fixture."""

    def test_the_document_carries_the_expected_row_census(self) -> None:
        by_part = collections.Counter(row.row_id.partition("/")[0] for row in _DOCUMENT_ROWS)
        assert len(_DOCUMENT_ROWS) == 764
        assert dict(by_part) == {"P2": 179, "P3": 147, "P4": 150, "P5": 175, "P6": 113}

    def test_no_mapping_row_is_unmapped(self) -> None:
        unmapped = [(row.row_id, row.line, row.resolution) for row in _DOCUMENT_ROWS if _classify(row) == "unmapped"]
        assert unmapped == []

    @pytest.mark.parametrize(
        "row",
        [row for row in _DOCUMENT_ROWS if _classify(row) == "model_symbol"],
        ids=lambda row: row.row_id,
    )
    def test_every_named_model_symbol_exists(self, row: DocumentRow) -> None:
        for symbol in _named_symbols(row.resolution):
            entry = _RECONCILED.get(symbol)
            if entry is None:
                _resolve(symbol)
            elif entry.resolves_to is not None:
                _resolve(entry.resolves_to)

    def test_every_row_id_is_unique_in_the_document(self) -> None:
        row_ids = [row.row_id for row in _DOCUMENT_ROWS]
        repeated = sorted({row_id for row_id in row_ids if row_ids.count(row_id) > 1})
        assert repeated == []

    def test_no_reconciliation_entry_is_stale(self) -> None:
        named = {symbol for row in _DOCUMENT_ROWS for symbol in _named_symbols(row.resolution)}
        unused = sorted(entry.symbol for entry in SYMBOL_RECONCILIATIONS if entry.symbol not in named)
        assert unused == []

    def test_no_prose_resolution_is_stale(self) -> None:
        present = {row.row_id for row in _DOCUMENT_ROWS}
        unused = sorted(entry.row_id for entry in PROSE_RESOLUTIONS if entry.row_id not in present)
        assert unused == []

    def test_every_reconciliation_carries_a_substantive_rationale(self) -> None:
        thin = [entry.symbol for entry in SYMBOL_RECONCILIATIONS if len(entry.rationale.split()) < 12]
        thin += [entry.row_id for entry in PROSE_RESOLUTIONS if len(entry.rationale.split()) < 12]
        assert thin == []

    def test_the_checked_in_table_names_only_rows_the_document_carries(self) -> None:
        # the curated table uses a finer id (``P2/B4/T57``) than the parser's
        # ``P2/T57``; both must name a row that exists. ids pointing at a
        # PROSE section rather than a table row are listed explicitly, since
        # their authority is the anchor and the parser sees no row for them.
        present = {row.row_id for row in _DOCUMENT_ROWS} | PROSE_SECTION_ROW_IDS
        collapsed = {
            f"{row_id.partition('/')[0]}/{row_id.rpartition('/')[2]}"
            for row_id in (row.row_id for row in IN_SCOPE_ROWS)
        }
        missing = sorted(row_id for row_id in collapsed if row_id not in present and not row_id.startswith("P1/"))
        assert missing == []

    def test_the_model_symbol_class_is_the_bulk_of_the_mapping(self) -> None:
        # if this collapses, the classifier has started resolving rows by
        # phrase that used to resolve by field, which is a weakened gate.
        counts = collections.Counter(_classify(row) for row in _DOCUMENT_ROWS)
        assert counts["model_symbol"] >= 480
        assert counts["decision"] == 99
