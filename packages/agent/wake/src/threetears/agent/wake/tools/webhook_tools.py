"""Seven webhook-subscription CRUD tools.

Factory functions mint LangChain ``BaseTool`` instances bound to a
``(conversation_id, user_id, agent_id)`` actor triple plus the wake
Collections + a consumer-supplied :class:`WakeRegistryClient` for ACL
probes on ``default_skill_id`` attachments. The plaintext HMAC secret
is shown ONCE on create + ONCE on rotate; otherwise the entity carries
only the ciphertext per Implementation Note 8.

Spec ref: ``docs/agent-wake/shard-04-agent-tools-and-webhook-adapter.md``
Requirements TOOL-02 / TOOL-09 / TOOL-13 / TOOL-14 + PLACEMENT §1.1 /
§1.13.
"""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime
from typing import Any, Final, Literal
from uuid import UUID

from jinja2.exceptions import TemplateError
from jinja2.sandbox import SandboxedEnvironment
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from uuid_utils import uuid7

from threetears.agent.wake.collections import WebhookSubscriptionCollection
from threetears.agent.wake.entities import EncryptionService
from threetears.agent.wake.tools.resolve import parse_subscription_id
from threetears.agent.wake.tools.schedule_tools import (
    NAME_MAX_LEN,
    WakeRegistryClient,
    _tool_error,
    _validate_name,
)
from threetears.observe import get_logger

__all__ = [
    "PAYLOAD_TEMPLATE_MAX_BYTES",
    "SECRET_BYTE_LEN",
    "WebhookSubscriptionCreateInput",
    "WebhookSubscriptionIdInput",
    "WebhookSubscriptionUpdateInput",
    "load_webhook_subscription_create_tool",
    "load_webhook_subscription_delete_tool",
    "load_webhook_subscription_list_tool",
    "load_webhook_subscription_pause_tool",
    "load_webhook_subscription_resume_tool",
    "load_webhook_subscription_rotate_secret_tool",
    "load_webhook_subscription_update_tool",
]


log = get_logger(__name__)


# Per Implementation Note 8 + the spec's bounded-payload guidance.
PAYLOAD_TEMPLATE_MAX_BYTES: Final[int] = 4 * 1024  # 4 KB
SECRET_BYTE_LEN: Final[int] = 32  # 32 bytes -> 64 hex chars

_VALID_EXECUTION_MODES: frozenset[str] = frozenset({"inline", "spawn"})
_VALID_VERIFICATION_SCHEMES: frozenset[str] = frozenset({"generic_hmac_sha256"})

# SandboxedEnvironment is thread-safe + cheap to share. ``autoescape``
# defaults to False because the rendered output is consumed as plain
# text by the personality node, not HTML; turning it on would corrupt
# JSON / Markdown payloads downstream.
_jinja_env = SandboxedEnvironment(autoescape=False)


# ---------------------------------------------------------------------------
# Pydantic input schemas
# ---------------------------------------------------------------------------


class WebhookSubscriptionCreateInput(BaseModel):
    """Input schema for ``webhook_subscription_create``."""

    name: str | None = Field(
        default=None,
        description="Optional human-readable subscription name (max 256 chars).",
    )
    task_prompt_template: str = Field(
        description=(
            "Jinja2 sandbox template rendered with {{event}} (the payload). Max 4KB. Becomes the per-fire task prompt."
        ),
    )
    default_skill_id: str | None = Field(
        default=None,
        description="Optional [skill:<id>] loaded as the attached skill on each fire.",
    )
    execution_mode: Literal["inline", "spawn"] = Field(
        default="inline",
        description="'inline' fires in this conversation; 'spawn' creates a new conversation.",
    )
    allowed_source_pattern: str | None = Field(
        default=None,
        description="Optional regex matched against the inbound source IP.",
    )
    rate_limit_per_minute: int | None = Field(
        default=None,
        description="Optional override for the platform default per-subscription rate cap.",
    )


class WebhookSubscriptionUpdateInput(BaseModel):
    """Input schema for ``webhook_subscription_update``.

    Because LangChain ``@tool`` cannot distinguish "field absent" from
    "explicit null" at the JSON layer, detachment of nullable
    references uses explicit boolean fields:

    - ``detach_default_skill=true`` clears the default skill.
    - ``clear_name=true`` clears the optional human-readable name.
    - ``clear_allowed_source_pattern=true`` clears the source IP regex.

    Passing the attach value AND its detach flag together is rejected.
    """

    subscription_id: str = Field(description="[webhook:<id>] to update.")
    name: str | None = None
    clear_name: bool = Field(
        default=False,
        description="When true, clear the human-readable name. Must not be combined with name.",
    )
    task_prompt_template: str | None = None
    default_skill_id: str | None = Field(
        default=None,
        description="New [skill:<uuid>] or bare UUID to set as the default skill. Omit to leave unchanged. To detach, pass detach_default_skill=true (do not pass both).",
    )
    detach_default_skill: bool = Field(
        default=False,
        description="When true, clear the default_skill_id. Must not be combined with default_skill_id.",
    )
    execution_mode: Literal["inline", "spawn"] | None = None
    allowed_source_pattern: str | None = None
    clear_allowed_source_pattern: bool = Field(
        default=False,
        description="When true, clear allowed_source_pattern. Must not be combined with allowed_source_pattern.",
    )
    rate_limit_per_minute: int | None = None


class WebhookSubscriptionIdInput(BaseModel):
    """Shared input for pause / resume / delete / rotate."""

    subscription_id: str = Field(description="[webhook:<id>] of the target subscription.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_template(template: str | None) -> str | None:
    """Validate the Jinja2 template at create/update time.

    Bounded payload + syntax check via
    :meth:`SandboxedEnvironment.parse` so render-time errors that
    actually depend on the payload shape are the only failures the
    receiver surfaces at HTTP-handling time.
    """
    if template is None:
        return None
    if not isinstance(template, str):
        return "task_prompt_template must be a string"
    encoded = template.encode("utf-8")
    if len(encoded) > PAYLOAD_TEMPLATE_MAX_BYTES:
        return f"task_prompt_template exceeds {PAYLOAD_TEMPLATE_MAX_BYTES // 1024}KB cap"
    try:
        _jinja_env.parse(template)
    except TemplateError as exc:
        return f"task_prompt_template invalid: {exc}"
    return None


def _validate_source_pattern(pattern: str | None) -> str | None:
    """Validate ``allowed_source_pattern`` regex at create/update time."""
    if pattern is None:
        return None
    if not isinstance(pattern, str):
        return "allowed_source_pattern must be a string or null"
    try:
        re.compile(pattern)
    except re.error as exc:
        return f"allowed_source_pattern is not a valid regex: {exc}"
    return None


def _validate_rate_limit(value: int | None) -> str | None:
    """Validate optional rate-limit override."""
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        return f"rate_limit_per_minute must be a positive int; got {value!r}"
    return None


def _parse_skill_id_arg(raw: str) -> UUID | None:
    """Parse a ``[skill:<uuid>]`` or bare UUID string into UUID (or None)."""
    if not raw or not isinstance(raw, str):
        return None
    stripped = raw.strip()
    if stripped.startswith("[skill:") and stripped.endswith("]"):
        stripped = stripped[len("[skill:") : -1].strip()
    try:
        return UUID(stripped)
    except ValueError:
        # malformed UUID literal (typo, wrong format, etc.)
        return None
    except AttributeError:
        # defensive: handles unusual non-string candidates that
        # uuid.UUID rejects via attribute access on its input
        return None
    except TypeError:
        # defensive: non-str inputs (e.g. dict, list) that bypass the
        # earlier ``isinstance(raw, str)`` guard via duck-typing
        return None


def _format_subscription_line(
    entity: Any,
    *,
    skill_name: str | None,
) -> str:
    """Render a one-line catalog entry for a subscription row."""
    name = entity.name or "untitled"
    skill_segment = f" · skill: {skill_name}" if skill_name else ""
    last = entity.last_fired_at.isoformat() if entity.last_fired_at is not None else "never"
    return (
        f"[webhook:{entity.subscription_id}] · {name} · "
        f"mode: {entity.execution_mode} · {entity.status} · last_fired: {last}{skill_segment}"
    )


async def _check_skill_acl(
    *,
    registry: WakeRegistryClient,
    user_id: UUID,
    agent_id: UUID,
    skill_id: UUID,
    tool_name: str,
) -> str | None:
    """Probe ACL for one skill_id (mirrors schedule_tools._check_skill_acl)."""
    try:
        permitted = await registry.acl_permits_skill(
            user_id=user_id,
            agent_id=agent_id,
            skill_id=skill_id,
        )
    except Exception as exc:  # noqa: BLE001 - surface as tool error
        log.warning(
            "webhook subscription skill ACL probe raised",
            extra={"extra_data": {"skill_id": str(skill_id), "error": str(exc)}},
        )
        return f"default_skill_id {skill_id} ACL probe failed: {exc}"
    if not permitted:
        return f"default_skill_id {skill_id} not authorized for this user/agent"
    return None


# ---------------------------------------------------------------------------
# webhook_subscription_create
# ---------------------------------------------------------------------------


def load_webhook_subscription_create_tool(
    *,
    conversation_id: UUID,
    user_id: UUID,
    agent_id: UUID,
    subscriptions_collection: WebhookSubscriptionCollection,
    encryption_service: EncryptionService,
    registry: WakeRegistryClient,
    endpoint_base_url: str | None = None,
) -> list[BaseTool]:
    """Build a ``webhook_subscription_create`` tool.

    Generates a 32-byte secret, encrypts via the consumer's
    ``encryption_service``, persists the row, and returns the plaintext
    once for the user to copy. The plaintext is NEVER persisted; only
    the ciphertext lands on ``webhook_subscriptions.secret_ciphertext``.

    :param conversation_id: caller's conversation UUID
    :ptype conversation_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param subscriptions_collection: three-tier subscriptions collection
    :ptype subscriptions_collection: WebhookSubscriptionCollection
    :param encryption_service: consumer-supplied encryption service
        (Fernet wrapper or equivalent)
    :ptype encryption_service: EncryptionService
    :param registry: consumer-supplied registry for skill ACL
    :ptype registry: WakeRegistryClient
    :param endpoint_base_url: optional product-supplied URL prefix
        rendered in the response so the user can copy the receive URL
    :ptype endpoint_base_url: str | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("webhook_subscription_create", args_schema=WebhookSubscriptionCreateInput)
    async def webhook_subscription_create(
        task_prompt_template: str,
        name: str | None = None,
        default_skill_id: str | None = None,
        execution_mode: Literal["inline", "spawn"] = "inline",
        allowed_source_pattern: str | None = None,
        rate_limit_per_minute: int | None = None,
    ) -> str:
        """Create an inbound webhook subscription for this conversation."""
        for err in (
            _validate_name(name),
            _validate_template(task_prompt_template),
            _validate_source_pattern(allowed_source_pattern),
            _validate_rate_limit(rate_limit_per_minute),
        ):
            if err is not None:
                return _tool_error("webhook_subscription_create", err)

        if execution_mode not in _VALID_EXECUTION_MODES:
            return _tool_error(
                "webhook_subscription_create",
                f"execution_mode must be 'inline' or 'spawn'; got {execution_mode!r}",
            )

        attached_skill: UUID | None = None
        if default_skill_id is not None and default_skill_id != "":
            parsed_skill = _parse_skill_id_arg(default_skill_id)
            if parsed_skill is None:
                return _tool_error(
                    "webhook_subscription_create",
                    f"invalid default_skill_id {default_skill_id!r}",
                )
            err = await _check_skill_acl(
                registry=registry,
                user_id=user_id,
                agent_id=agent_id,
                skill_id=parsed_skill,
                tool_name="webhook_subscription_create",
            )
            if err is not None:
                return _tool_error("webhook_subscription_create", err)
            attached_skill = parsed_skill

        # Generate + encrypt the secret. The plaintext exists only in
        # this stack frame + the return string; the entity carries
        # ciphertext only.
        plaintext_secret = secrets.token_hex(SECRET_BYTE_LEN)
        try:
            ciphertext = encryption_service.encrypt(plaintext_secret.encode("utf-8"))
        except AttributeError:
            return _tool_error(
                "webhook_subscription_create",
                "encryption_service does not support encrypt(); cannot create subscription",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "webhook_subscription_create encrypt failed",
                extra={"extra_data": {"error": str(exc)}},
            )
            return _tool_error(
                "webhook_subscription_create",
                f"secret encryption failed: {exc}",
            )

        now = datetime.now(UTC)
        new_id = UUID(str(uuid7()))
        data: dict[str, Any] = {
            "subscription_id": new_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "default_skill_id": attached_skill,
            "name": name,
            "secret_ciphertext": bytes(ciphertext),
            "allowed_source_pattern": allowed_source_pattern,
            "execution_mode": execution_mode,
            "task_prompt_template": task_prompt_template,
            "verification_scheme": "generic_hmac_sha256",
            "status": "active",
            "rate_limit_per_minute": rate_limit_per_minute,
            "last_fired_at": None,
            "date_created": now,
            "date_updated": now,
        }
        entity = subscriptions_collection.create(data)
        try:
            await subscriptions_collection.save_entity(entity)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "webhook_subscription_create persist failed",
                extra={"extra_data": {"subscription_id": str(new_id), "error": str(exc)}},
            )
            return _tool_error("webhook_subscription_create", f"persist failed: {exc}")

        skill_name: str | None = None
        if attached_skill is not None:
            try:
                skill_name = await registry.skill_name_for_id(
                    user_id=user_id,
                    agent_id=agent_id,
                    skill_id=attached_skill,
                )
            except Exception:  # noqa: BLE001 - best-effort
                skill_name = None

        log.info(
            "webhook_subscription_create persisted",
            extra={
                "extra_data": {
                    "subscription_id": str(new_id),
                    "default_skill_id": str(attached_skill) if attached_skill else None,
                    "execution_mode": execution_mode,
                }
            },
        )

        endpoint_segment = ""
        if endpoint_base_url:
            base = endpoint_base_url.rstrip("/")
            endpoint_segment = f"\nendpoint: {base}/{new_id}"

        catalog = _format_subscription_line(entity, skill_name=skill_name)
        return f"{catalog}\nsecret (copy now; shown only once): {plaintext_secret}{endpoint_segment}"

    wake_schedule_create_desc = (
        "Create an inbound webhook subscription for THIS conversation.\n"
        "- task_prompt_template: Jinja2 sandbox; {{event}} = payload (max 4KB)\n"
        "- default_skill_id: optional skill loaded on each fire\n"
        "- allowed_source_pattern: optional regex against the source IP\n"
        "Returns [webhook:<id>] + the HMAC secret ONCE (copy it; cannot be retrieved later)."
    )
    webhook_subscription_create.description = wake_schedule_create_desc
    return [webhook_subscription_create]


# ---------------------------------------------------------------------------
# webhook_subscription_update
# ---------------------------------------------------------------------------


def load_webhook_subscription_update_tool(
    *,
    conversation_id: UUID,
    user_id: UUID,
    agent_id: UUID,
    subscriptions_collection: WebhookSubscriptionCollection,
    registry: WakeRegistryClient,
) -> list[BaseTool]:
    """Build a ``webhook_subscription_update`` tool with partial-update semantics.

    Cannot change ``secret_ciphertext`` -- use
    :func:`load_webhook_subscription_rotate_secret_tool` for that.

    :param conversation_id: caller's conversation UUID
    :ptype conversation_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param subscriptions_collection: three-tier subscriptions collection
    :ptype subscriptions_collection: WebhookSubscriptionCollection
    :param registry: consumer-supplied registry for skill ACL
    :ptype registry: WakeRegistryClient
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("webhook_subscription_update", args_schema=WebhookSubscriptionUpdateInput)
    async def webhook_subscription_update(
        subscription_id: str,
        name: str | None = None,
        clear_name: bool = False,
        task_prompt_template: str | None = None,
        default_skill_id: str | None = None,
        detach_default_skill: bool = False,
        execution_mode: Literal["inline", "spawn"] | None = None,
        allowed_source_pattern: str | None = None,
        clear_allowed_source_pattern: bool = False,
        rate_limit_per_minute: int | None = None,
    ) -> str:
        """Edit a webhook subscription in place. Pass only fields to change."""
        # Contradictory-input guards.
        if default_skill_id is not None and detach_default_skill:
            return _tool_error(
                "webhook_subscription_update",
                "default_skill_id and detach_default_skill=true cannot be combined; "
                "pass exactly one (or neither, to leave the attachment unchanged).",
            )
        if name is not None and clear_name:
            return _tool_error(
                "webhook_subscription_update",
                "name and clear_name=true cannot be combined; pass exactly one (or neither).",
            )
        if allowed_source_pattern is not None and clear_allowed_source_pattern:
            return _tool_error(
                "webhook_subscription_update",
                "allowed_source_pattern and clear_allowed_source_pattern=true "
                "cannot be combined; pass exactly one (or neither).",
            )

        parsed = parse_subscription_id(subscription_id)
        if parsed is None:
            return _tool_error(
                "webhook_subscription_update",
                f"invalid subscription_id {subscription_id!r}",
            )

        entity = await subscriptions_collection.get((conversation_id, parsed))
        if entity is None or entity.user_id != user_id:
            return _tool_error("webhook_subscription_update", "subscription not found")

        validation_errors: list[str | None] = [
            _validate_name(name) if name is not None else None,
            _validate_template(task_prompt_template),
            _validate_source_pattern(allowed_source_pattern),
            _validate_rate_limit(rate_limit_per_minute),
        ]
        for err in validation_errors:
            if err is not None:
                return _tool_error("webhook_subscription_update", err)

        if execution_mode is not None and execution_mode not in _VALID_EXECUTION_MODES:
            return _tool_error(
                "webhook_subscription_update",
                f"execution_mode must be 'inline' or 'spawn'; got {execution_mode!r}",
            )

        # default_skill_id handling via explicit attach/detach booleans.
        if detach_default_skill:
            entity.default_skill_id = None
        elif default_skill_id is not None:
            stripped = default_skill_id.strip()
            # Tag-confusion guard: webhook/schedule tags are not skill ids.
            if stripped.startswith("[webhook:") or stripped.startswith("[schedule:"):
                return _tool_error(
                    "webhook_subscription_update",
                    f"default_skill_id received a non-skill tag {default_skill_id!r}; "
                    "use a [skill:<uuid>] or bare UUID from skill_list/skill_get.",
                )
            parsed_skill = _parse_skill_id_arg(default_skill_id)
            if parsed_skill is None:
                return _tool_error(
                    "webhook_subscription_update",
                    f"invalid default_skill_id {default_skill_id!r}",
                )
            err = await _check_skill_acl(
                registry=registry,
                user_id=user_id,
                agent_id=agent_id,
                skill_id=parsed_skill,
                tool_name="webhook_subscription_update",
            )
            if err is not None:
                return _tool_error("webhook_subscription_update", err)
            entity.default_skill_id = parsed_skill

        if clear_name:
            entity.name = None
        elif name is not None:
            entity.name = name
        if task_prompt_template is not None:
            entity.task_prompt_template = task_prompt_template
        if execution_mode is not None:
            entity.execution_mode = execution_mode
        if clear_allowed_source_pattern:
            entity.allowed_source_pattern = None
        elif allowed_source_pattern is not None:
            entity.allowed_source_pattern = allowed_source_pattern
        if rate_limit_per_minute is not None:
            entity.rate_limit_per_minute = rate_limit_per_minute

        entity.date_updated = datetime.now(UTC)
        try:
            await subscriptions_collection.save_entity(entity)
        except Exception as exc:  # noqa: BLE001
            return _tool_error("webhook_subscription_update", f"persist failed: {exc}")

        skill_name: str | None = None
        if entity.default_skill_id is not None:
            try:
                skill_name = await registry.skill_name_for_id(
                    user_id=user_id,
                    agent_id=agent_id,
                    skill_id=entity.default_skill_id,
                )
            except Exception:  # noqa: BLE001 - best-effort
                skill_name = None
        return _format_subscription_line(entity, skill_name=skill_name)

    webhook_subscription_update.description = (
        "Edit a webhook subscription in place. Pass only fields to change.\n"
        "Attach a skill: pass default_skill_id=<uuid>. Detach: pass detach_default_skill=true.\n"
        "Clear name/source-pattern via clear_name=true / clear_allowed_source_pattern=true.\n"
        "Cannot change the HMAC secret -- use webhook_subscription_rotate_secret."
    )
    return [webhook_subscription_update]


# ---------------------------------------------------------------------------
# webhook_subscription_list
# ---------------------------------------------------------------------------


def load_webhook_subscription_list_tool(
    *,
    conversation_id: UUID,
    user_id: UUID,
    agent_id: UUID,
    subscriptions_collection: WebhookSubscriptionCollection,
    registry: WakeRegistryClient,
) -> list[BaseTool]:
    """Build a ``webhook_subscription_list`` tool scoped to the conversation."""

    class _ListInput(BaseModel):
        """No-arg list."""

    @tool("webhook_subscription_list", args_schema=_ListInput)
    async def webhook_subscription_list() -> str:
        """List inbound webhook subscriptions for this conversation."""
        try:
            rows = await subscriptions_collection.list_for_conversation(conversation_id)
        except Exception as exc:  # noqa: BLE001
            return _tool_error("webhook_subscription_list", f"list failed: {exc}")

        visible = [row for row in rows if row.user_id == user_id]
        if not visible:
            return "No webhook subscriptions in this conversation."

        lines: list[str] = [f"Found {len(visible)} subscriptions:"]
        for entity in visible:
            skill_name: str | None = None
            if entity.default_skill_id is not None:
                try:
                    skill_name = await registry.skill_name_for_id(
                        user_id=user_id,
                        agent_id=agent_id,
                        skill_id=entity.default_skill_id,
                    )
                except Exception:  # noqa: BLE001 - best-effort
                    skill_name = None
            lines.append("- " + _format_subscription_line(entity, skill_name=skill_name))
        return "\n".join(lines)

    webhook_subscription_list.description = (
        "List webhook subscriptions in THIS conversation. Returns [webhook:<id>] + name + status."
    )
    return [webhook_subscription_list]


# ---------------------------------------------------------------------------
# webhook_subscription_pause / resume / delete
# ---------------------------------------------------------------------------


def load_webhook_subscription_pause_tool(
    *,
    conversation_id: UUID,
    user_id: UUID,
    subscriptions_collection: WebhookSubscriptionCollection,
) -> list[BaseTool]:
    """Build a ``webhook_subscription_pause`` tool (status -> 'paused')."""

    @tool("webhook_subscription_pause", args_schema=WebhookSubscriptionIdInput)
    async def webhook_subscription_pause(subscription_id: str) -> str:
        """Pause a webhook subscription. Inbound webhooks return 404 until resumed."""
        parsed = parse_subscription_id(subscription_id)
        if parsed is None:
            return _tool_error(
                "webhook_subscription_pause",
                f"invalid subscription_id {subscription_id!r}",
            )
        entity = await subscriptions_collection.get((conversation_id, parsed))
        if entity is None or entity.user_id != user_id:
            return _tool_error("webhook_subscription_pause", "subscription not found")
        try:
            await subscriptions_collection.pause(conversation_id, parsed)
        except Exception as exc:  # noqa: BLE001
            return _tool_error("webhook_subscription_pause", f"persist failed: {exc}")
        return f"Paused [webhook:{parsed}]."

    webhook_subscription_pause.description = "Pause a webhook subscription. Inbound webhooks 404 until you resume it."
    return [webhook_subscription_pause]


def load_webhook_subscription_resume_tool(
    *,
    conversation_id: UUID,
    user_id: UUID,
    subscriptions_collection: WebhookSubscriptionCollection,
) -> list[BaseTool]:
    """Build a ``webhook_subscription_resume`` tool (status -> 'active')."""

    @tool("webhook_subscription_resume", args_schema=WebhookSubscriptionIdInput)
    async def webhook_subscription_resume(subscription_id: str) -> str:
        """Resume a paused webhook subscription."""
        parsed = parse_subscription_id(subscription_id)
        if parsed is None:
            return _tool_error(
                "webhook_subscription_resume",
                f"invalid subscription_id {subscription_id!r}",
            )
        entity = await subscriptions_collection.get((conversation_id, parsed))
        if entity is None or entity.user_id != user_id:
            return _tool_error("webhook_subscription_resume", "subscription not found")
        try:
            await subscriptions_collection.resume(conversation_id, parsed)
        except Exception as exc:  # noqa: BLE001
            return _tool_error("webhook_subscription_resume", f"persist failed: {exc}")
        return f"Resumed [webhook:{parsed}]."

    webhook_subscription_resume.description = "Resume a paused webhook subscription."
    return [webhook_subscription_resume]


def load_webhook_subscription_delete_tool(
    *,
    conversation_id: UUID,
    user_id: UUID,
    subscriptions_collection: WebhookSubscriptionCollection,
) -> list[BaseTool]:
    """Build a ``webhook_subscription_delete`` tool (hard delete)."""

    @tool("webhook_subscription_delete", args_schema=WebhookSubscriptionIdInput)
    async def webhook_subscription_delete(subscription_id: str) -> str:
        """Delete a webhook subscription permanently. Fire history unbinds (SET NULL)."""
        parsed = parse_subscription_id(subscription_id)
        if parsed is None:
            return _tool_error(
                "webhook_subscription_delete",
                f"invalid subscription_id {subscription_id!r}",
            )
        entity = await subscriptions_collection.get((conversation_id, parsed))
        if entity is None or entity.user_id != user_id:
            return _tool_error("webhook_subscription_delete", "subscription not found")
        try:
            await subscriptions_collection.delete((conversation_id, parsed))
        except Exception as exc:  # noqa: BLE001
            return _tool_error("webhook_subscription_delete", f"persist failed: {exc}")
        return f"Deleted [webhook:{parsed}] ({entity.name or 'untitled'})."

    webhook_subscription_delete.description = (
        "Delete a webhook subscription permanently. Fire history unbinds (SET NULL on FK)."
    )
    return [webhook_subscription_delete]


# ---------------------------------------------------------------------------
# webhook_subscription_rotate_secret
# ---------------------------------------------------------------------------


def load_webhook_subscription_rotate_secret_tool(
    *,
    conversation_id: UUID,
    user_id: UUID,
    subscriptions_collection: WebhookSubscriptionCollection,
    encryption_service: EncryptionService,
) -> list[BaseTool]:
    """Build a ``webhook_subscription_rotate_secret`` tool.

    Generates a new 32-byte secret, encrypts, replaces the row's
    ciphertext, and returns the plaintext ONCE. The previous secret is
    irrecoverable after rotation -- inbound webhooks signed with the
    old key will start failing HMAC verification at the next request.

    :param conversation_id: caller's conversation UUID
    :ptype conversation_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param subscriptions_collection: three-tier subscriptions collection
    :ptype subscriptions_collection: WebhookSubscriptionCollection
    :param encryption_service: consumer-supplied encryption service
    :ptype encryption_service: EncryptionService
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("webhook_subscription_rotate_secret", args_schema=WebhookSubscriptionIdInput)
    async def webhook_subscription_rotate_secret(subscription_id: str) -> str:
        """Rotate the HMAC secret. Returns the new plaintext ONCE."""
        parsed = parse_subscription_id(subscription_id)
        if parsed is None:
            return _tool_error(
                "webhook_subscription_rotate_secret",
                f"invalid subscription_id {subscription_id!r}",
            )
        entity = await subscriptions_collection.get((conversation_id, parsed))
        if entity is None or entity.user_id != user_id:
            return _tool_error(
                "webhook_subscription_rotate_secret",
                "subscription not found",
            )

        plaintext_secret = secrets.token_hex(SECRET_BYTE_LEN)
        try:
            ciphertext = encryption_service.encrypt(plaintext_secret.encode("utf-8"))
        except AttributeError:
            return _tool_error(
                "webhook_subscription_rotate_secret",
                "encryption_service does not support encrypt(); cannot rotate",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "webhook_subscription_rotate_secret encrypt failed",
                extra={"extra_data": {"subscription_id": str(parsed), "error": str(exc)}},
            )
            return _tool_error(
                "webhook_subscription_rotate_secret",
                f"secret encryption failed: {exc}",
            )

        try:
            await subscriptions_collection.rotate_secret(
                conversation_id,
                parsed,
                new_ciphertext=bytes(ciphertext),
            )
        except Exception as exc:  # noqa: BLE001
            return _tool_error(
                "webhook_subscription_rotate_secret",
                f"persist failed: {exc}",
            )

        log.info(
            "webhook_subscription_rotate_secret completed",
            extra={"extra_data": {"subscription_id": str(parsed)}},
        )
        return f"Rotated secret for [webhook:{parsed}]. New secret (copy now; shown only once): {plaintext_secret}"

    webhook_subscription_rotate_secret.description = (
        "Rotate the HMAC secret on a webhook subscription. Returns the new plaintext ONCE.\n"
        "Old secret stops working immediately -- update upstream callers."
    )
    return [webhook_subscription_rotate_secret]
