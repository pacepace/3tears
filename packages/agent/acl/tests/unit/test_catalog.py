"""unit tests for :mod:`threetears.agent.acl.catalog`.

the catalog exists so a role's ``{resource_type: [action]}`` map can be
checked against a vocabulary an application actually declared, instead of
being free text that evaluates to silence. these tests pin the four
behaviours that make that worth having:

- a declared pair passes and an undeclared one is refused, by resource
  type and by action independently;
- the wildcard bucket is skipped DELIBERATELY rather than by falling
  through, because a wildcard role names no resource type and existing
  roles depend on it;
- two applications cannot both claim one resource type, because
  :meth:`threetears.agent.acl.Role.actions_for` looks a bucket up by
  namespace type alone and has no application dimension to tell them
  apart with;
- a parameterized action (the ``read_file_matching:`` shape the
  evaluator already ships) is declared once and matches every argument.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.agent.acl import (
    ActionDescriptor,
    CatalogViolationKind,
    PermissionCatalog,
    ResourceTypeDescriptor,
    UndeclaredPermission,
    WILDCARD_RESOURCE_TYPE,
    enforce_declared_permissions,
    validate_permissions,
)


def _survey_catalog() -> PermissionCatalog:
    """build the survey application's catalog as the admin surface declares it.

    :return: catalog with one ``survey`` resource type and three actions
    :rtype: PermissionCatalog
    """
    return PermissionCatalog(
        entries=(
            ResourceTypeDescriptor(
                application="survey",
                resource_type="survey",
                label="Survey",
                actions=(
                    ActionDescriptor(name="survey.create", label="Create survey"),
                    ActionDescriptor(name="collector.open", label="Open collector"),
                    ActionDescriptor(name="report.run", label="Run report"),
                ),
            ),
        ),
    )


class TestActionDescriptor:
    """one declared action, its label, and its parameterized form."""

    def test_plain_action_is_matched_exactly(self) -> None:
        """a non-parameterized action matches its own name and nothing else."""
        action = ActionDescriptor(name="collector.open", label="Open collector")
        assert action.matches("collector.open")
        assert not action.matches("collector.opened")
        assert not action.matches("collector")

    def test_parameterized_action_matches_any_argument(self) -> None:
        """a trailing-colon action is a prefix stem, per the evaluator's
        ``read_file_matching:`` shape."""
        action = ActionDescriptor(
            name="read_file_matching:",
            label="Read files matching",
            parameterized=True,
        )
        assert action.matches("read_file_matching:**/*.yaml")
        assert action.matches("read_file_matching:")
        assert not action.matches("write_file_matching:x")

    def test_parameterized_action_must_end_with_a_colon(self) -> None:
        """a stem with no separator would prefix-match unrelated actions."""
        with pytest.raises(ValidationError, match="parameterized"):
            ActionDescriptor(name="collector.open", label="x", parameterized=True)

    def test_plain_action_may_not_carry_a_colon(self) -> None:
        """a colon in a plain action is the parameterized shape declared without
        saying so, which then matches only the literal argument-free string."""
        with pytest.raises(ValidationError, match="parameterized"):
            ActionDescriptor(name="read_file_matching:", label="x")

    def test_label_is_required_and_non_empty(self) -> None:
        """shard 13 renders these; a blank label is a blank row in a role builder."""
        with pytest.raises(ValidationError, match="label"):
            ActionDescriptor(name="collector.open", label="   ")

    def test_action_name_may_not_contain_whitespace(self) -> None:
        """the action is compared by equality against the string a caller passes to
        ``authorize``; whitespace there is a typo that would never match."""
        with pytest.raises(ValidationError, match="whitespace"):
            ActionDescriptor(name="collector open", label="x")


class TestResourceTypeDescriptor:
    """one declaring application's statement about one namespace type."""

    def test_carries_its_declaring_application(self) -> None:
        """a catalog entry with no owner cannot be attributed or retired."""
        entry = ResourceTypeDescriptor(
            application="survey",
            resource_type="survey",
            label="Survey",
            actions=(ActionDescriptor(name="survey.create", label="Create survey"),),
        )
        assert entry.application == "survey"
        assert entry.resource_type == "survey"

    def test_declares_at_least_one_action(self) -> None:
        """a resource type granting nothing is a row no role can reference."""
        with pytest.raises(ValidationError, match="no actions"):
            ResourceTypeDescriptor(
                application="survey",
                resource_type="survey",
                label="Survey",
                actions=(),
            )

    def test_action_names_are_unique_within_a_resource_type(self) -> None:
        """two descriptors for one action means one label is silently dropped, and
        which one depends on iteration order at the consumer."""
        with pytest.raises(ValidationError, match="duplicate action"):
            ResourceTypeDescriptor(
                application="survey",
                resource_type="survey",
                label="Survey",
                actions=(
                    ActionDescriptor(name="survey.create", label="Create survey"),
                    ActionDescriptor(name="survey.create", label="Add survey"),
                ),
            )

    def test_wildcard_may_not_be_declared_as_a_resource_type(self) -> None:
        """``"*"`` is the evaluator's type-agnostic bucket, not a namespace type;
        declaring it would make every wildcard role validate against one
        application's action list."""
        with pytest.raises(ValidationError, match="wildcard"):
            ResourceTypeDescriptor(
                application="survey",
                resource_type=WILDCARD_RESOURCE_TYPE,
                label="Everything",
                actions=(ActionDescriptor(name="survey.create", label="Create survey"),),
            )

    def test_declares_reports_membership_by_action_string(self) -> None:
        """the lookup a role write path performs."""
        entry = ResourceTypeDescriptor(
            application="survey",
            resource_type="survey",
            label="Survey",
            actions=(ActionDescriptor(name="survey.create", label="Create survey"),),
        )
        assert entry.declares("survey.create")
        assert not entry.declares("survey.destroy")


class TestPermissionCatalogIndex:
    """resource-type lookup stays the exact-match dict hit ``actions_for`` is."""

    def test_resolves_a_declared_resource_type(self) -> None:
        """the entry comes back by its bare namespace-type key."""
        catalog = _survey_catalog()
        entry = catalog.entry_for("survey")
        assert entry is not None
        assert entry.application == "survey"

    def test_returns_none_for_an_undeclared_resource_type(self) -> None:
        """no entry rather than an exception; the caller decides what a miss means."""
        assert _survey_catalog().entry_for("workspace") is None

    def test_two_applications_may_not_claim_one_resource_type(self) -> None:
        """``Role.actions_for(namespace.namespace_type)`` takes the namespace type and
        nothing else, so two applications on one type cannot be told apart at
        evaluation time. refuse the catalog rather than merge the entries."""
        with pytest.raises(ValidationError, match="already declared by"):
            PermissionCatalog(
                entries=(
                    ResourceTypeDescriptor(
                        application="survey",
                        resource_type="report",
                        label="Report",
                        actions=(ActionDescriptor(name="report.run", label="Run"),),
                    ),
                    ResourceTypeDescriptor(
                        application="map",
                        resource_type="report",
                        label="Report",
                        actions=(ActionDescriptor(name="report.render", label="Render"),),
                    ),
                ),
            )

    def test_two_applications_declaring_distinct_types_do_not_collide(self) -> None:
        """the same short action verb under two namespace types stays separable,
        because the bucket key differs."""
        catalog = PermissionCatalog(
            entries=(
                ResourceTypeDescriptor(
                    application="survey",
                    resource_type="survey",
                    label="Survey",
                    actions=(ActionDescriptor(name="report.run", label="Run report"),),
                ),
                ResourceTypeDescriptor(
                    application="map",
                    resource_type="datasource",
                    label="Map data",
                    actions=(ActionDescriptor(name="report.run", label="Run report"),),
                ),
            ),
        )
        survey_entry = catalog.entry_for("survey")
        map_entry = catalog.entry_for("datasource")
        assert survey_entry is not None
        assert map_entry is not None
        assert survey_entry.application == "survey"
        assert map_entry.application == "map"
        assert validate_permissions({"survey": ["report.run"]}, catalog) == ()
        assert validate_permissions({"datasource": ["report.run"]}, catalog) == ()

    def test_an_empty_catalog_is_constructible(self) -> None:
        """a platform with nothing declared yet has an empty catalog, not an error;
        the refusal belongs to the role write path, which can then say so."""
        assert PermissionCatalog(entries=()).entry_for("survey") is None


class TestValidatePermissions:
    """the helper a role write path calls before persisting ``permissions``."""

    def test_a_fully_declared_map_produces_no_violations(self) -> None:
        """the passing case."""
        catalog = _survey_catalog()
        assert validate_permissions({"survey": ["survey.create", "collector.open"]}, catalog) == ()

    def test_an_undeclared_resource_type_is_refused(self) -> None:
        """no entry for the bucket key at all."""
        violations = validate_permissions({"widget": ["widget.poke"]}, _survey_catalog())
        assert len(violations) == 1
        assert violations[0].kind is CatalogViolationKind.UNDECLARED_RESOURCE_TYPE
        assert violations[0].resource_type == "widget"
        assert violations[0].action is None

    def test_an_undeclared_action_on_a_declared_type_is_refused(self) -> None:
        """the entry exists; the action is not in it."""
        violations = validate_permissions({"survey": ["survey.detonate"]}, _survey_catalog())
        assert len(violations) == 1
        assert violations[0].kind is CatalogViolationKind.UNDECLARED_ACTION
        assert violations[0].resource_type == "survey"
        assert violations[0].action == "survey.detonate"

    def test_an_undeclared_type_reports_once_not_once_per_action(self) -> None:
        """an operator fixing a typo'd bucket key wants one message, not five."""
        violations = validate_permissions({"widget": ["a", "b", "c"]}, _survey_catalog())
        assert len(violations) == 1

    def test_every_undeclared_action_is_reported(self) -> None:
        """a role builder shows all of them at once rather than one per round trip."""
        violations = validate_permissions(
            {"survey": ["survey.create", "survey.detonate", "survey.combust"]},
            _survey_catalog(),
        )
        assert {v.action for v in violations} == {"survey.detonate", "survey.combust"}

    def test_violations_are_ordered_deterministically(self) -> None:
        """the message an operator reads must not depend on set iteration order."""
        catalog = _survey_catalog()
        first = validate_permissions({"survey": ["z.act", "a.act"], "widget": ["x"]}, catalog)
        second = validate_permissions({"survey": ["z.act", "a.act"], "widget": ["x"]}, catalog)
        assert first == second
        assert [v.action for v in first if v.action is not None] == ["a.act", "z.act"]

    def test_a_parameterized_action_validates_with_its_argument(self) -> None:
        """the evaluator's ``read_file_matching:<glob>`` shape survives validation."""
        catalog = PermissionCatalog(
            entries=(
                ResourceTypeDescriptor(
                    application="platform",
                    resource_type="workspace",
                    label="Workspace",
                    actions=(
                        ActionDescriptor(
                            name="read_file_matching:",
                            label="Read files matching",
                            parameterized=True,
                        ),
                    ),
                ),
            ),
        )
        assert validate_permissions({"workspace": ["read_file_matching:**/*.yaml"]}, catalog) == ()

    def test_a_frozenset_valued_map_is_accepted(self) -> None:
        """``Role.permissions`` is ``Mapping[str, frozenset[str]]`` once loaded, and the
        Hub's write path holds ``list[str]``; both must validate."""
        catalog = _survey_catalog()
        assert validate_permissions({"survey": frozenset({"survey.create"})}, catalog) == ()

    def test_an_empty_action_list_is_not_a_violation(self) -> None:
        """an empty bucket grants nothing; it is pointless, not undeclared, and the
        catalog is not the place that opinion belongs."""
        assert validate_permissions({"survey": []}, _survey_catalog()) == ()


class TestWildcardIsSkippedDeliberately:
    """implementation note 6 and the matching anti-pattern, pinned."""

    def test_a_wildcard_only_role_produces_no_violations(self) -> None:
        """``{"*": ["read"]}`` is the shipped ``Reader`` shape; it names no resource
        type, so there is nothing to check it against."""
        assert validate_permissions({WILDCARD_RESOURCE_TYPE: ["read"]}, _survey_catalog()) == ()

    def test_a_wildcard_bucket_is_skipped_even_against_an_empty_catalog(self) -> None:
        """the skip is unconditional, not a side effect of the catalog happening to
        contain something."""
        assert validate_permissions({WILDCARD_RESOURCE_TYPE: ["anything"]}, PermissionCatalog(entries=())) == ()

    def test_declared_buckets_alongside_a_wildcard_are_still_checked(self) -> None:
        """the skip covers the wildcard bucket only; a role mixing both does not buy
        immunity for its typed buckets."""
        violations = validate_permissions(
            {WILDCARD_RESOURCE_TYPE: ["read"], "survey": ["survey.detonate"]},
            _survey_catalog(),
        )
        assert len(violations) == 1
        assert violations[0].resource_type == "survey"


class TestEnforceDeclaredPermissions:
    """the raising form a role write path uses when it has nothing to render."""

    def test_a_declared_map_passes_silently(self) -> None:
        """no exception, no return value."""
        assert enforce_declared_permissions({"survey": ["survey.create"]}, _survey_catalog()) is None

    def test_an_undeclared_pair_raises_with_its_violations_attached(self) -> None:
        """the caller can render the same detail the non-raising form returns."""
        with pytest.raises(UndeclaredPermission) as excinfo:
            enforce_declared_permissions({"survey": ["survey.detonate"]}, _survey_catalog())
        assert excinfo.value.violations[0].action == "survey.detonate"
        assert "survey.detonate" in str(excinfo.value)


class TestExistingRolesAreUnaffected:
    """success criterion: evaluation output identical before and after."""

    def test_the_catalog_does_not_touch_actions_for(self) -> None:
        """validation is a write-path concern; nothing here participates in
        evaluation, so a role that evaluated one way still does."""
        from uuid import uuid7

        from threetears.agent.acl import Role

        role = Role(
            id=uuid7(),
            name="Reader",
            permissions={WILDCARD_RESOURCE_TYPE: frozenset({"read"}), "survey": frozenset({"survey.create"})},
            is_built_in=True,
        )
        before = role.actions_for("survey")
        validate_permissions(dict(role.permissions), _survey_catalog())
        assert role.actions_for("survey") == before == frozenset({"read", "survey.create"})
