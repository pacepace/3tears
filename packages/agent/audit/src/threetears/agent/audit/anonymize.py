"""the one erasure rule for audit records: anonymize, never delete.

an audit record outlives the person it names. erasure (a GDPR request, a
principal anonymization, a customer offboarding) therefore never deletes an
audit row and never changes an id on it -- not the row id, not the actor id,
not an entity id in its details. what erasure does is anonymize the row's
content:

- :func:`anonymize_details` keeps every key of ``details`` and the structure
  beneath safe keys, keeps the value under a SAFE key, and replaces the whole
  value under every other key with :data:`ANONYMIZED_MARKER`.
- :func:`anonymize_ip` is the rule for an ``ip_address`` column.

**the safe list is explicit.** a key is kept only because someone decided its
value is structural (an entity id, a count, a status, a duration, a version, a
flag) for every producer that publishes it. a key nobody classified is masked,
so a new field fails safe rather than leaking. :data:`PERSONAL_DETAIL_KEYS`
records the keys that were looked at and found able to carry personal data (a
name, an email, an address, an IP, free or user-typed text, a path, exception
text, a credential or a fragment of one); it changes nothing at runtime -- an
unclassified key is masked exactly like a personal one -- and exists so the
classification is a record a gate can hold producers to
(:mod:`threetears.enforcement.audit_details`, which every producing repo runs
over its own sources).

**how nesting is judged.** the key a value sits under decides it:

- under an unsafe key the WHOLE value becomes the marker -- a leaf, a list, a
  dict and its keys alike. the key itself stays; what it held is the person's
  content, and that includes keys a user chose (``doc_set`` publishes the
  user's document under ``value``, whose field names can be an email address or
  a name). a nested ``status`` in such a subtree earns nothing.
- a dict under a safe key is judged key by key, by the same rule. safety does
  not flow into a dict, so a key added inside a structural map later is masked
  until someone classifies it. the cost is that a map keyed by data rather than
  by schema (``permissions_before``: group uuid -> actions) keeps its keys but
  loses each entry's value on erasure.
- a list or tuple under a safe key keeps its elements, and a dict inside it is
  judged by its own keys.
- ``None`` stays ``None`` under any key: there is nothing in it to anonymize.

no key at or above a safe level is removed or rewritten, and no key is ever
deleted: an unsafe key survives with the marker as its value.

**families.** some keys are structural only inside one event family --
``reason`` is a closed enum on ``identity.impersonation.stop`` and exception
text on ``identity.email.send_failure``. those are declared per event-type
prefix, in one registry with one lookup (:func:`safe_detail_keys_for`). the
families every platform producer publishes are declared here, because the
process that erases a record is usually not the one that produced it: the hub
anonymizes rows that identity-core and the survey engine published. a
:func:`declare_safe_detail_keys` call is visible only inside the process that
makes it, so it is the right tool for a service anonymizing its OWN local audit
store, and the wrong one for a family whose events land in the platform audit
table -- that family belongs in :data:`_BUILT_IN_FAMILY_SAFE_KEYS` below.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Final

from threetears.observe import get_logger

# the platform's one erasure marker, homed in 3tears-observe so the checkpoint saver shares
# its spelling without depending on this package; re-exported here as part of this module's
# erasure API.
from threetears.observe.erasure import ANONYMIZED_MARKER

__all__ = [
    "ANONYMIZED_MARKER",
    "PERSONAL_DETAIL_KEYS",
    "SAFE_DETAIL_KEYS",
    "anonymize_details",
    "anonymize_ip",
    "declare_safe_detail_keys",
    "is_classified_detail_key",
    "safe_detail_keys_for",
]


log = get_logger(__name__)


#: keys whose value is structural wherever the platform publishes them. one line
#: per key: what the value is, and where it is published (``hub``, ``identity``,
#: ``survey`` name the sibling repos; ``3tears`` this one).
SAFE_DETAIL_KEYS: Final[frozenset[str]] = frozenset(
    {
        # --- entity ids: a uuid (or an external platform's opaque id), never content.
        # the owner's rule is that erasure never changes an id.
        "absorbed_principal_id",  # identity.principal.merge (identity)
        "actor_id",  # uuid of the actor explained, rbac.introspect.explain (hub)
        "agent_id",  # rbac.introspect.effective (hub)
        "api_key_id",  # identity.apikey.* (identity)
        "candidate_id",  # knowledge.candidate.* (hub)
        "channel_ref",  # chat platform's channel id, engagement.channel_default.* (hub)
        "collector_id",  # collector.promoted (survey)
        "connection_id",  # identity.login.*, identity.connection.*, identity.principal.* (identity)
        "conversation_id",  # security.exploit.approval.* (hub)
        "customer_id",  # rbac.group.create, rbac.role.create (hub); identity.tenancy.platform_scope (identity)
        "datasource_id",  # the renamed datasource a grant followed, rbac.assignment.move (hub)
        "entity_id",  # knowledge.candidate.*, knowledge.promotion.* (hub)
        "external_realm_id",  # chat platform's workspace id, channel_realm.* (hub)
        "grant_id",  # admin.action mcp_tool_grant (hub)
        "group_id",  # rbac.* (hub)
        "identity_core_connection_id",  # channel_realm.create (hub)
        "identity_key_id",  # admin.action api_key revoke (hub)
        "member_id",  # uuid of the user, agent or group, rbac.group.member.* (hub)
        "membership_id",  # rbac.group.member.remove (hub)
        "model_id",  # gateway.credit_rate.create (hub)
        "namespace_id",  # rbac.introspect.*, rbac.assignment.* (hub)
        "partner_id",  # partner.* (survey)
        "pod_id",  # admin.action namespace rescope / reap (hub)
        "principal_id",  # identity.* (identity); admin.action mcp_tool_grant (hub)
        "provider_id",  # gateway.model.create (hub)
        "realm_id",  # channel.create (hub)
        "realm_ref",  # chat platform's workspace id, engagement.channel_default.* (hub)
        "retired_connection_ids",  # list of uuids, channel_realm.delete (hub)
        "reviewer_user_id",  # knowledge.promotion.* (hub)
        "role_id",  # rbac.assignment.* (hub)
        "scope_customer_id",  # rbac.assignment.* (hub)
        "scope_namespace_id",  # rbac.assignment.* (hub)
        "session_id",  # every survey event that names a session (survey)
        "source_template_id",  # template.promoted_to_platform (hub)
        "source_user_id",  # the absorbed user, user.merge (hub)
        "survey_id",  # every survey event that names a survey (survey)
        "survey_version_id",  # collector.promoted (survey)
        "survivor_principal_id",  # identity.principal.merge (identity)
        "target_id",  # engagement.target.* (hub)
        "target_principal_id",  # identity.principal.disable (identity)
        "template_id",  # template_column.*, table.bound (hub)
        "user_id",  # rbac.introspect.effective, rbac.mcp_ingress.*, security.tool.approver.* (hub)
        # --- counts, sizes and durations: integers or decimals, never content.
        "actor_count",  # rbac.introspect.namespace_access (hub)
        "agents_swept",  # user.merge (hub)
        "answer_length",  # length of an answer, never the answer, security.prompt_injection (survey)
        "api_keys_revoked",  # identity.principal.anonymize, identity.tenant.offboard (identity)
        "assignment_count",  # rbac.group.delete (hub)
        "batch_size",  # admin.action pii_rotation (hub)
        "bytes_after",  # workspace.fs_write / fs_edit / doc_set / doc_merge (3tears)
        "bytes_before",  # workspace.fs_write / fs_edit / doc_set / doc_merge (3tears)
        "carried_grants",  # admin.action namespace rescope (hub)
        "cascaded_assignments",  # agent.delete (hub)
        "cascaded_grants",  # admin.action namespace reap (hub)
        "columns_introspected",  # datasource.rollover (hub)
        "connections_disabled",  # identity.tenant.offboard (identity)
        "conversations_repointed",  # user.merge (hub)
        "credentials_anonymized",  # identity.principal.anonymize (identity)
        "credentials_removed",  # identity.mfa.second_factor_reset (identity)
        "credentials_repointed",  # identity.principal.merge (identity)
        "duration_ms",  # tool.call (3tears)
        "elapsed_ms",  # schema.introspect_refresh (hub)
        "error_count",  # admin.action pii_rotation (hub)
        "files_changed",  # workspace.create / reset / rollback (3tears)
        "gained_count",  # rbac.introspect.dry_run (hub)
        "group_memberships_deduped",  # user.merge (hub)
        "group_memberships_repointed",  # user.merge (hub)
        "lost_count",  # rbac.introspect.dry_run (hub)
        "member_count",  # rbac.group.delete (hub)
        "memories_repointed",  # user.merge (hub)
        "memory_alias_collisions_deleted",  # user.merge (hub)
        "method_count",  # identity.login.discover (identity)
        "mismatches",  # admin.action backup restore_dry_run (hub)
        "mutation_count",  # rbac.introspect.dry_run (hub)
        "occurrences",  # workspace.fs_edit (3tears)
        "owner_groups_reconciled",  # user.merge (hub)
        "principals_anonymized",  # identity.tenant.offboard (identity)
        "questions_count",  # partner.viewed_results / exported_results (survey)
        "removed_installs",  # agent.delete (hub)
        "result_count",  # namespace.discover (hub)
        "rotated_count",  # admin.action pii_rotation (hub)
        "rows_written",  # admin.action backup selective_restore (hub)
        "rule_index",  # position of a skip-logic rule, session.terminated (survey)
        "size_bytes",  # admin.action backup create (hub)
        "smtp_port",  # identity.email.send_failure (identity)
        "surveys_count",  # partner.listed_surveys (survey)
        "tables_added",  # schema.introspect_refresh (hub)
        "tables_changed",  # schema.introspect_refresh (hub)
        "tables_checked",  # schema.introspect_refresh, admin.action backup (hub)
        "tables_introspected",  # datasource.rollover (hub)
        "tables_removed",  # schema.introspect_refresh (hub)
        "tables_unchanged",  # schema.introspect_refresh (hub)
        "time_limit_seconds",  # admin.action index_build (hub)
        "ttl_seconds",  # admin.action customer impersonation (hub)
        # --- flags: booleans.
        "already_merged",  # identity.principal.merge (identity)
        "customer_admin",  # user.create (hub)
        "declaration_persisted",  # agent.update_*_access (hub)
        "enabled",  # identity.connection.*, identity.email_settings.change (identity); channel, capture (hub)
        "is_blocked",  # identity.principal.disable (identity)
        "is_break_glass",  # identity.login.success (identity)
        "knowledge_rebound",  # datasource.rollover (hub)
        "pilot_rows_migrated",  # collector.promoted (survey)
        "pod_notified",  # datasource.rollover (hub)
        "use_starttls",  # identity.email_settings.change (identity)
        # --- the platform's own vocabulary: enum values, code-supplied names, versions.
        "active_key_version",  # admin.action pii_rotation (hub)
        "actor_type",  # user / agent / group, rbac.introspect.explain (hub)
        "backup_type",  # admin.action backup create (hub)
        "change_kind",  # create / update / delete, workspace.materialize (3tears)
        "channel_type",  # slack / discord / ..., channel.*, channel_realm.*, engagement.* (hub)
        "connection_type",  # ConnectionType enum, identity.login.* (identity)
        "entity_type",  # knowledge.candidate.*, knowledge.promotion.* (hub)
        "error_code",  # an ApiError code, never its message, admin.request.refused, datasource.*, schema.* (hub)
        "error_type",  # an exception class name, never its text, datasource.*, schema.* (hub)
        "filter_type",  # a namespace type, namespace.discover (hub)
        "from_mode",  # CollectorMode enum, collector.promoted (survey)
        "from_status",  # engagement status enum, engagement.transition (hub)
        "interrupted_by",  # an exception class name, admin.action backup (hub)
        "limiting_side",  # enum, rbac.introspect.effective (hub)
        "managed_by",  # manual / ..., rbac.assignment.* (hub)
        "member_type",  # user / agent / group, rbac.group.member.* (hub)
        "model_name",  # a registered model's name, gateway.model.* (hub)
        "namespace_type",  # a namespace type, admin.action namespace (hub)
        "previous_row_scope",  # row-scope enum, admin.action namespace rescope (hub)
        "principal_type",  # admin.action mcp_tool_grant (hub)
        "restated",  # field NAMES restated at bind, never their values, identity.principal.pre_bind_claimed
        "revoked_via",  # fixed operation label, identity.apikey.revoke (identity)
        "row_scope",  # row-scope enum, admin.action namespace rescope (hub)
        "scope_namespace_type",  # a namespace type, rbac.assignment.* (hub)
        "scope_type",  # namespace / customer / platform, rbac.assignment.* (hub)
        "target_scope",  # knowledge scope enum, knowledge.* (hub)
        "target_type",  # ip / cidr / hostname / url -- the kind, never the target, engagement.target.* (hub)
        "to_mode",  # CollectorMode enum, collector.promoted (survey)
        "to_status",  # engagement status enum, engagement.transition (hub)
        "tool_name",  # a registered tool's name, tool.call (3tears); admin.action mcp_tool_grant (hub)
        "tool_names",  # registered tool names, security.exploit.approval.* (hub)
        "tool_version",  # tool.call (3tears)
        "validation_status",  # admin.action backup restore_dry_run (hub)
        "version",  # a workspace file version number, workspace.* (3tears)
        # --- timestamps.
        "date_terminated",  # session.terminated (survey)
        "paused_since",  # datasource.credential_pause_lifted (hub)
        # --- structural maps. the key is structural, but a dict beneath it is judged key by
        # key (see the module docstring), so leaves under data keys are still masked.
        "permissions",  # resource type -> actions, rbac.role.create (hub)
        "permissions_after",  # group or namespace uuid -> actions, rbac.* (hub)
        "permissions_before",  # group or namespace uuid -> actions, rbac.* (hub)
        "platform_repointed",  # table name -> row count, user.merge (hub)
        "role_permissions_after",  # resource type -> actions, rbac.role.update (hub)
        "role_permissions_before",  # resource type -> actions, rbac.role.update (hub)
    }
)


#: keys that were classified and found able to carry personal data somewhere they
#: are published. masked by :func:`anonymize_details` exactly as an unclassified key
#: is; recorded so the classification is complete and checkable. a family may still
#: declare one of these safe for its own events where the value there is structural
#: (``reason`` below).
PERSONAL_DETAIL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_key_masked",  # a masked provider key is still a credential fragment, gateway.provider.* (hub)
        "cause",  # optional free-text cause, channel.delete (hub)
        "column_name",  # operator-typed column name, admin.action pii_rotation (hub)
        "credentials_refs",  # secret-seam references for a channel's tokens, channel.create / update (hub)
        "description",  # user free text, rbac.group.create (hub)
        "description_after",  # user free text, rbac.group.update (hub)
        "description_before",  # user free text, rbac.group.update (hub)
        "domain",  # unvalidated caller or operator input, identity.login.discover, domain_allocation.change
        "email_domain",  # domain part of a person's email, identity.domain_allocation.refusal (identity)
        "error",  # str(exc), may echo input, admin.action backup, agent.update_*_access (hub)
        "error_message",  # driver or refusal text, admin.request.refused, datasource.*, schema.* (hub)
        "failure_reason",  # human-readable failure text, may echo tool input, tool.call (3tears)
        "from_address",  # an email address, identity.email_settings.change (identity)
        "granted",  # namespace names, which carry user-chosen segments, security.tool.approver.grant (hub)
        "group_name",  # user-supplied group name, rbac.assignment.delete (hub)
        "host",  # operator-typed hostname, identity.email_settings.change (identity)
        "hosted_domain",  # an IdP hosted domain, can identify a person's own domain (identity)
        "index",  # operator-supplied index name, admin.action index_build (hub)
        "jsonpath",  # a path into a user document, names its fields, workspace.doc_set (3tears)
        "key_prefix",  # displayed prefix of an api key: a credential fragment, identity.apikey.mint / rotate
        "message",  # author text shown to a respondent, session.terminated (survey)
        "name",  # a user-chosen name: workspace, group, role, engagement, job, realm (3tears, hub)
        "name_after",  # user-supplied group name, rbac.group.update (hub)
        "name_before",  # user-supplied group name, rbac.group.update (hub)
        "namespace_name",  # namespace names carry user-chosen segments, admin.action namespace (hub)
        "not_revoked",  # access bucket -> dropped entries, agent.update_*_access (hub)
        "partial_keys",  # the field names of a user's merge, workspace.doc_merge (3tears)
        "path",  # a request url path, can embed names and ids, admin.request.refused (hub)
        "reason",  # free text or str(exc) in most producers; declared safe per family where it is an enum
        "redirect_url",  # rendered from the respondent's answers, session.* (survey)
        "ref",  # a checkpoint label or claim-mapping name, typed by a user or admin (3tears, identity)
        "refused",  # str(exc), admin.action index_build (hub)
        "removed_datasources",  # datasource names carry user-declared table names, agent.delete (hub)
        "removed_namespaces",  # namespace names, agent.delete (hub)
        "replaced_schema",  # customer schema name, datasource.rollover (hub)
        "revoked",  # namespace names, security.tool.approver.revoke (hub)
        "rule_condition",  # author expression that can hold literal answer values, session.terminated (survey)
        "s3_key",  # object key path, admin.action backup create (hub)
        "schedule_config",  # an arbitrary caller-supplied dict, admin.action scheduled_job_update (hub)
        "schema_name",  # customer schema name, datasource.rollover (hub)
        "scope_namespace_name",  # user-supplied namespace name, rbac.assignment.create (hub)
        "scope_namespace_name_after",  # the grant's node name after its rename, rbac.assignment.move (hub)
        "scope_namespace_name_before",  # the grant's node name before its rename, rbac.assignment.move (hub)
        "sha256_after",  # digest of user content: pseudonymous, confirmable by guessing (3tears)
        "sha256_before",  # digest of user content: pseudonymous, confirmable by guessing (3tears)
        "sid",  # a raw session id, credential-adjacent; logs carry only its digest (identity)
        "smtp_host",  # operator-typed hostname, identity.email.send_failure (identity)
        "source",  # author-supplied media path or url, media.delivered (survey)
        "subject",  # an IdP subject (may be an email), user.create; index-build label (hub)
        "table",  # customer table name, admin.action backup / index_build (hub)
        "table_name",  # operator-typed table name, admin.action pii_rotation (hub)
        "template_name",  # a caller-chosen template name, workspace.create / reset (3tears)
        "tool_pattern",  # admin-typed tool glob, security.tool.approver.* (hub)
        "value",  # a user's document value (doc_set, 3tears) or an engagement target's ip / host / url (hub)
        "warehouse_group",  # warehouse group name, dataset.publish_principal (hub)
        "workspace_resource_id",  # "{workspace}/{relative path}", the path chosen by the user (3tears)
    }
)


#: keys structural only inside one event family, keyed by event-type prefix. a
#: prefix covers itself and every event type beneath it (``identity.connection``
#: covers ``identity.connection.created``). declared here rather than by each
#: producer because the hub erases records every producer published -- see the
#: module docstring.
_BUILT_IN_FAMILY_SAFE_KEYS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        # sandbox root named in operator configuration (default "bind") (3tears)
        "workspace.materialize": frozenset({"root_name"}),
        # "mfa_enforcement", and its MfaEnforcement enum value (identity)
        "identity.tenant_policy.change": frozenset({"field", "new_value"}),
        # ScopeReason enum, and a code-supplied operation label such as "principal.export" (identity)
        "identity.tenancy.platform_scope": frozenset({"reason", "operation"}),
        # fixed stop reasons: absolute_lifetime_exceeded, target_blocked, admin_blocked, gate_revoked (identity)
        "identity.impersonation.stop": frozenset({"reason"}),
        # refusal constants REFUSED_NO_TRUSTED_DOMAIN / REFUSED_EMAIL_DOMAIN_NOT_ALLOCATED (identity)
        "identity.domain_allocation.refusal": frozenset({"reason"}),
        # CustomerApprovalPolicy enum (identity)
        "identity.principal.approval_change": frozenset({"policy"}),
        # ConnectionType enum (identity)
        "identity.connection": frozenset({"type"}),
        # self_serve_recovery / self_serve_claim / admin (identity)
        "identity.reset": frozenset({"mode"}),
        # offboarding disposition enum (identity)
        "identity.tenant.offboard": frozenset({"disposition"}),
        # sid / sub / customer_id -- the kind of revocation, never the value (identity)
        "identity.token.revoke": frozenset({"scope"}),
        # an smtp port number (identity)
        "identity.email_settings.change": frozenset({"port"}),
        # the termination outcome and the configured redirect key, never the rendered url (survey)
        "session.completed": frozenset({"outcome", "redirect_key"}),
        "session.terminated": frozenset({"outcome", "redirect_key"}),
        # json / csv (survey)
        "partner.exported_results": frozenset({"format"}),
        # injection family names, never the answer (survey)
        "security.prompt_injection": frozenset({"families"}),
        # media type enum (survey)
        "media.delivered": frozenset({"type"}),
        # fixed "operator_forced_owner_deletion" (hub)
        "rbac.assignment.delete": frozenset({"reason"}),
        # fixed "datasource_renamed" (hub)
        "rbac.assignment.move": frozenset({"reason"}),
        # allow / deny (hub)
        "rbac.introspect": frozenset({"decision"}),
        # the approval verdict, and the uuid of the user who approved or denied the paused tool
        # call -- an id, which erasure never changes. declared for this family rather than in
        # SAFE_DETAIL_KEYS because the name does not say it holds an id: another producer
        # could put a display name under it, and there it stays masked (hub)
        "security.exploit.approval": frozenset({"decision", "decided_by"}),
        # a decimal score delta and a regression flag (hub)
        "knowledge.candidate": frozenset({"delta", "regression"}),
        # http verb and status code (hub)
        "admin.request.refused": frozenset({"method", "status"}),
        # backup, scheduled-job, mcp-grant and pii-rotation admin actions: enums, database names, counts;
        # data-space limit and reset-target actions: a limit's code-supplied name, integer limits and
        # data versions (hub)
        "admin.action": frozenset(
            {
                "applied_version",
                "databases",
                "database",
                "failed_databases",
                "identical",
                "inserts",
                "kind",
                "limit_after",
                "limit_before",
                "limit_name",
                "mode",
                "ok",
                "permission",
                "status",
                "target_version_before",
                "total_stale",
                "updates",
            }
        ),
        # federated / password (hub)
        "user.create": frozenset({"credential"}),
        # a fixed failure stage label (hub)
        "schema.introspect": frozenset({"stage"}),
        "schema.introspect_refresh": frozenset({"stage"}),
        # counts (hub)
        "schema.bulk_describe": frozenset({"matched", "unmatched", "total"}),
        # a flag (hub)
        "agent.context_capture_read": frozenset({"captured"}),
        # two flags: forced, and whether it finished an earlier delete's teardown (hub)
        "agent.delete": frozenset({"force", "finished_earlier_delete"}),
        # fixed labels grants_not_materialized / grants_not_revoked (hub)
        "agent.update_tool_access": frozenset({"failure", "warning"}),
        "agent.update_model_access": frozenset({"failure", "warning"}),
        "agent.update_access": frozenset({"failure", "warning"}),
        # a registered provider's name (hub)
        "gateway.provider.create": frozenset({"provider_name"}),
        # a tool pod changing an agent's data under its grant: the tables the statement
        # names (developer-declared names from the agent's table list, never customer
        # typed), the affected-row count, and the broker-minted transaction id (hub)
        "l3.write": frozenset({"tables", "row_count", "tx_id"}),
    }
)


#: one or more non-empty dotted segments, no whitespace.
_PREFIX_SHAPE = re.compile(r"[^.\s]+(\.[^.\s]+)*")

#: serializes declarations. readers take the current mapping without the lock: it is
#: replaced wholesale, never mutated in place.
_declare_lock = threading.Lock()
_family_safe_keys: dict[str, frozenset[str]] = dict(_BUILT_IN_FAMILY_SAFE_KEYS)


def declare_safe_detail_keys(event_type_prefix: str, keys: Iterable[str]) -> None:
    """
    declare keys safe for one event family, in this process.

    widens the safe set for every event type equal to ``event_type_prefix`` or
    beneath it by whole dotted segments (``survey`` covers ``survey.x``, never
    ``surveyor.x``). declarations accumulate and repeating one is harmless. a
    family may declare a key :data:`PERSONAL_DETAIL_KEYS` names, where the value
    under it is structural for that family alone; every other family still masks
    it. the declaration is visible only in the calling process -- a family whose
    events another process erases (the hub, for the platform audit table) is
    declared in this module instead.

    :param event_type_prefix: dotted event-type prefix the keys are safe under
    :ptype event_type_prefix: str
    :param keys: the detail keys whose values are structural in that family
    :ptype keys: Iterable[str]
    :return: nothing
    :rtype: None
    :raises TypeError: if ``keys`` is a bare string, which would declare its characters
    :raises ValueError: if the prefix is not dotted segments, or a key is blank
    """
    if isinstance(keys, str):
        raise TypeError(f"keys must be a collection of key names, not the bare string {keys!r}")
    if _PREFIX_SHAPE.fullmatch(event_type_prefix) is None:
        raise ValueError(f"event_type_prefix must be non-empty dotted segments; received {event_type_prefix!r}")
    declared = frozenset(keys)
    blank = sorted(key for key in declared if not key.strip())
    if blank:
        raise ValueError(f"detail keys must be non-blank; received {blank!r}")
    global _family_safe_keys
    with _declare_lock:
        updated = dict(_family_safe_keys)
        updated[event_type_prefix] = updated.get(event_type_prefix, frozenset()) | declared
        _family_safe_keys = updated
    log.info(
        "audit detail keys declared safe for an event family",
        extra={"extra_data": {"event_type_prefix": event_type_prefix, "keys": sorted(declared)}},
    )


def safe_detail_keys_for(event_type: str) -> frozenset[str]:
    """
    the complete safe set for one event type: the platform set plus its families.

    the single lookup behind :func:`anonymize_details` and
    :func:`is_classified_detail_key`. an event type no family covers gets
    :data:`SAFE_DETAIL_KEYS` alone.

    :param event_type: the audit event's dotted ``event_type``
    :ptype event_type: str
    :return: every key whose value is kept for that event type
    :rtype: frozenset[str]
    """
    families = _family_safe_keys
    segments = event_type.split(".")
    safe = SAFE_DETAIL_KEYS
    for depth in range(1, len(segments) + 1):
        declared = families.get(".".join(segments[:depth]))
        if declared is not None:
            safe = safe | declared
    return safe


def is_classified_detail_key(key: str, *, event_type: str) -> bool:
    """
    whether someone decided about ``key`` for this event type.

    classified means safe for the event type, or recorded as personal. an
    unclassified key is still masked on erasure. this is the in-process form of
    the question, for a producer or a test that holds this package.

    the enforcement gate (``threetears.enforcement.audit_details``) judges by
    this very function: the enforcement package cannot depend on this one, so a
    shell injects it as ``AuditDetailsConfig.is_classified`` and the gate calls it
    per event type. a change to what "classified" means is made here alone.

    :param key: a details key
    :ptype key: str
    :param event_type: the audit event's dotted ``event_type``
    :ptype event_type: str
    :return: ``True`` when the key is safe for the event type or personal
    :rtype: bool
    """
    return key in PERSONAL_DETAIL_KEYS or key in safe_detail_keys_for(event_type)


def anonymize_details(details: Mapping[str, Any], *, event_type: str) -> dict[str, Any]:
    """
    anonymize an audit record's details: same keys, safe skeleton, safe values only.

    keeps every key of ``details``. the value under a key safe for
    ``event_type`` is kept, with any dict inside it judged key by key; the
    whole value under any other key becomes :data:`ANONYMIZED_MARKER`, keys a
    user chose included. ``None`` stays ``None``. pure: no I/O, the input is not
    mutated and the result shares no container with it. idempotent: the marker
    under an unsafe key is replaced by itself, so a second pass changes nothing.

    :param details: the record's ``details`` mapping
    :ptype details: Mapping[str, Any]
    :param event_type: the record's dotted ``event_type``, which selects its families
    :ptype event_type: str
    :return: a new details dict with every unsafe value anonymized
    :rtype: dict[str, Any]
    """
    return _anonymize_mapping(details, safe=safe_detail_keys_for(event_type))


def _anonymize_mapping(mapping: Mapping[str, Any], *, safe: frozenset[str]) -> dict[str, Any]:
    """
    judge each key of one mapping: keep what a safe key holds, mask what any other holds.

    :param mapping: ``details`` itself, or a dict found under a safe key
    :ptype mapping: Mapping[str, Any]
    :param safe: the safe keys for the record's event type
    :ptype safe: frozenset[str]
    :return: a new dict with the same keys
    :rtype: dict[str, Any]
    """
    return {key: _keep(value, safe=safe) if key in safe else _mask(value) for key, value in mapping.items()}


def _keep(value: Any, *, safe: frozenset[str]) -> Any:
    """
    copy a value held by a safe key, judging any dict inside it afresh.

    :param value: the value under a safe key, or an element of a list or tuple there
    :ptype value: Any
    :param safe: the safe keys for the record's event type
    :ptype safe: frozenset[str]
    :return: an equal value, except where a nested dict holds unsafe keys
    :rtype: Any
    """
    result: Any
    if isinstance(value, Mapping):
        result = _anonymize_mapping(value, safe=safe)
    elif isinstance(value, list):
        result = [_keep(child, safe=safe) for child in value]
    elif isinstance(value, tuple):
        result = tuple(_keep(child, safe=safe) for child in value)
    else:
        result = value
    return result


def _mask(value: Any) -> str | None:
    """
    anonymize everything an unsafe key holds, subtree and keys alike.

    :param value: the value under an unsafe key
    :ptype value: Any
    :return: ``None`` when the value is ``None``, else :data:`ANONYMIZED_MARKER`
    :rtype: str | None
    """
    return None if value is None else ANONYMIZED_MARKER


def anonymize_ip(value: str | None) -> str | None:
    """
    the rule for an audit record's ``ip_address`` column: the value it holds once erased.

    an address is removed rather than masked or truncated, so the result is always
    ``None``. the marker cannot be stored in an address-typed column, and a truncated
    address (the hub already stores a /24 or /48 prefix at write time) is still personal
    data: with a timestamp it narrows to a household or an office. ``None`` is what the
    column holds for every event that never had an address, so an erased row reads as one
    with no address rather than as a special case. only the column's value changes; the
    row and its other columns stay.

    typed as the column (``str | None``), not as ``None``: the result is a value a caller
    assigns back to the column (``row.ip_address = anonymize_ip(row.ip_address)``), and
    binding the result of a function annotated ``-> None`` is a type error.

    :param value: the stored address, or ``None``
    :ptype value: str | None
    :return: the column's erased value, which is ``None`` for every input
    :rtype: str | None
    """
    del value
    return None
