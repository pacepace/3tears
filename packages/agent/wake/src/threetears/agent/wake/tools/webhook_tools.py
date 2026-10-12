"""Seven webhook-subscription CRUD tools.

Factory functions mint LangChain ``BaseTool`` instances bound to a
``(conversation_id, user_id, agent_id)`` actor triple plus the wake
Collections + a consumer-supplied :class:`WakeRegistryClient` for ACL
probes on ``default_skill_id`` attachments. The plaintext HMAC secret
is shown ONCE on create + ONCE on rotate; otherwise the entity carries
only the ciphertext per Implementation Note 8.

As with the schedule tools, a subscription is the agent's: the tools reach
it from any of the agent's conversations, and never reach another agent's.
With the consumer's :class:`~threetears.agent.wake.types.WakeConversations`
a subscription is created into a wake conversation, like a wake.

Spec ref: ``docs/agent-wake/shard-04-agent-tools-and-webhook-adapter.md``
Requirements TOOL-02 / TOOL-09 / TOOL-13 / TOOL-14 + PLACEMENT §1.1 /
§1.13.
"""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime
from typing import Any, Final
from uuid import UUID

from jinja2.exceptions import TemplateError
from jinja2.sandbox import SandboxedEnvironment
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from uuid_utils import uuid7

from threetears.agent.wake.collections import WebhookSubscriptionCollection
from threetears.agent.wake.entities import EncryptionService
from threetears.agent.wake.tools.resolve import parse_conversation_id, parse_subscription_id
from threetears.agent.wake.types import WakeConversations
from threetears.core.collections import CallerTransaction
from threetears.core.exceptions import ConcurrentModificationError
from threetears.agent.wake.tools.schedule_tools import (
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
        description="Optional name, up to 256 characters.",
    )
    task_prompt_template: str = Field(
        description=(
            "The instructions you get each time it fires, up to 4KB. {{event}} is replaced with what was sent."
        ),
    )
    default_skill_id: str | None = Field(
        default=None,
        description="Optional skill to use each time it fires: a [skill:<id>] or its id.",
    )
    wake_conversation_id: str | None = Field(
        default=None,
        description=(
            "Optional [conversation:<id>] of one of your wake conversations, to add this subscription to it. "
            "Leave it out to start a new wake conversation."
        ),
    )
    allowed_source_pattern: str | None = Field(
        default=None,
        description="Optional regex. Only senders whose IP address matches are accepted.",
    )
    rate_limit_per_minute: int | None = Field(
        default=None,
        description="Optional limit on fires per minute. Leave out for the default.",
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

    subscription_id: str = Field(description="The [webhook:<id>] to change.")
    name: str | None = None
    clear_name: bool = Field(
        default=False,
        description="True removes the name. Do not also pass name.",
    )
    task_prompt_template: str | None = None
    default_skill_id: str | None = Field(
        default=None,
        description="A skill to attach: a [skill:<id>] or its id. To remove the skill, use detach_default_skill instead.",
    )
    detach_default_skill: bool = Field(
        default=False,
        description="True removes the attached skill. Do not also pass default_skill_id.",
    )
    allowed_source_pattern: str | None = None
    clear_allowed_source_pattern: bool = Field(
        default=False,
        description="True removes allowed_source_pattern. Do not also pass allowed_source_pattern.",
    )
    rate_limit_per_minute: int | None = None


class WebhookSubscriptionIdInput(BaseModel):
    """Shared input for pause / resume / delete / rotate."""

    subscription_id: str = Field(description="The [webhook:<id>].")


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
        # Malformed UUID literal (typo, wrong format, etc.). The caller turns None into a
        # tool-error the model sees; this line is for an operator asking why a lookup keeps missing.
        log.debug("skill id did not parse as a UUID", extra={"extra_data": {"candidate": stripped[:64]}})
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
        f"{entity.status} · in [conversation:{entity.conversation_id}] · last_fired: {last}{skill_segment}"
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
    wake_conversations: WakeConversations | None = None,
) -> list[BaseTool]:
    """Build a ``webhook_subscription_create`` tool.

    Generates a 32-byte secret, encrypts via the consumer's
    ``encryption_service``, persists the row, and returns the plaintext
    once for the user to copy. The plaintext is NEVER persisted; only
    the ciphertext lands on ``webhook_subscriptions.secret_ciphertext``.

    With ``wake_conversations`` the subscription lives in a wake
    conversation: the one the agent names, or a new one made in the same
    transaction as the row, whose parent is the caller's conversation.
    Without it the subscription lives in the caller's conversation.

    :param conversation_id: caller's conversation UUID; the parent of a new
        wake conversation
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
    :param wake_conversations: the consumer's wake-conversation hooks
    :ptype wake_conversations: WakeConversations | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("webhook_subscription_create", args_schema=WebhookSubscriptionCreateInput)
    async def webhook_subscription_create(
        task_prompt_template: str,
        name: str | None = None,
        default_skill_id: str | None = None,
        wake_conversation_id: str | None = None,
        allowed_source_pattern: str | None = None,
        rate_limit_per_minute: int | None = None,
    ) -> str:
        """Create an inbound webhook subscription."""
        for err in (
            _validate_name(name),
            _validate_template(task_prompt_template),
            _validate_source_pattern(allowed_source_pattern),
            _validate_rate_limit(rate_limit_per_minute),
        ):
            if err is not None:
                return _tool_error("webhook_subscription_create", err)

        target_conversation_id: UUID | None = conversation_id if wake_conversations is None else None
        if wake_conversation_id is not None:
            if wake_conversations is None:
                return _tool_error(
                    "webhook_subscription_create",
                    "subscriptions here live in this conversation; omit wake_conversation_id",
                )
            parsed_target = parse_conversation_id(wake_conversation_id)
            if parsed_target is None:
                return _tool_error(
                    "webhook_subscription_create", f"invalid wake_conversation_id {wake_conversation_id!r}"
                )
            if not await wake_conversations.is_wake_conversation(agent_id=agent_id, conversation_id=parsed_target):
                return _tool_error(
                    "webhook_subscription_create",
                    f"[conversation:{parsed_target}] is not one of your wake conversations",
                )
            target_conversation_id = parsed_target

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
            "conversation_id": target_conversation_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "default_skill_id": attached_skill,
            "name": name,
            "secret_ciphertext": bytes(ciphertext),
            "allowed_source_pattern": allowed_source_pattern,
            "execution_mode": "spawn",
            "task_prompt_template": task_prompt_template,
            "verification_scheme": "generic_hmac_sha256",
            "status": "active",
            "rate_limit_per_minute": rate_limit_per_minute,
            "last_fired_at": None,
            "date_created": now,
            "date_updated": now,
        }
        try:
            entity = await _persist_subscription(
                subscriptions_collection,
                data,
                wake_conversations if target_conversation_id is None else None,
                parent_conversation_id=conversation_id,
                name=name,
            )
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
                    "wake_conversation_id": str(entity.conversation_id),
                }
            },
        )

        endpoint_segment = ""
        if endpoint_base_url:
            base = endpoint_base_url.rstrip("/")
            endpoint_segment = f"\nendpoint: {base}/{new_id}"

        catalog = _format_subscription_line(entity, skill_name=skill_name)
        return f"{catalog}\nsecret (copy now; shown only once): {plaintext_secret}{endpoint_segment}"

    webhook_subscription_create.description = (
        "Give yourself an address that other systems can send events to. It lives in a wake "
        "conversation, and each event starts a new conversation with task_prompt_template filled in.\n"
        if wake_conversations is not None
        else "Give this conversation an address that other systems can send events to. Each event "
        "wakes you with task_prompt_template filled in.\n"
    ) + "Returns [webhook:<id>] and its secret. The secret is shown only once: copy it."
    return [webhook_subscription_create]


async def _persist_subscription(
    collection: WebhookSubscriptionCollection,
    data: dict[str, Any],
    wake_conversations: WakeConversations | None,
    *,
    parent_conversation_id: UUID,
    name: str | None,
) -> Any:
    """Insert a subscription, making its wake conversation in the same transaction when wanted.

    :param collection: three-tier subscriptions collection
    :ptype collection: WebhookSubscriptionCollection
    :param data: the new row; ``conversation_id`` is set here when a wake
        conversation is made
    :ptype data: dict[str, Any]
    :param wake_conversations: the hooks, when a new wake conversation is wanted
    :ptype wake_conversations: WakeConversations | None
    :param parent_conversation_id: the caller's conversation, the new one's parent
    :ptype parent_conversation_id: UUID
    :param name: the subscription's name
    :ptype name: str | None
    :return: the persisted subscription entity
    :rtype: WebhookSubscriptionEntity
    """
    if wake_conversations is None:
        entity = collection.create(data)
        await collection.save_entity(entity)
        return entity
    pool = collection.l3_pool
    if pool is None:
        raise RuntimeError("the subscription collection has no database to write to")
    # CallerTransaction, not conn.transaction(): the save joins it, and the row is evicted from
    # every cache tier once the transaction has committed or rolled back.
    async with pool.acquire() as conn, CallerTransaction(conn):
        row = dict(data)
        row["conversation_id"] = await wake_conversations.create(
            parent_conversation_id=parent_conversation_id,
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            name=name,
            conn=conn,
        )
        entity = collection.create(row)
        await collection.save_entity(entity, conn=conn)
    return entity


# ---------------------------------------------------------------------------
# webhook_subscription_update
# ---------------------------------------------------------------------------


def load_webhook_subscription_update_tool(
    *,
    user_id: UUID,
    agent_id: UUID,
    subscriptions_collection: WebhookSubscriptionCollection,
    registry: WakeRegistryClient,
) -> list[BaseTool]:
    """Build a ``webhook_subscription_update`` tool with partial-update semantics.

    Cannot change ``secret_ciphertext`` -- use
    :func:`load_webhook_subscription_rotate_secret_tool` for that.

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

        entity = await subscriptions_collection.find_for_agent(agent_id, parsed)
        if entity is None:
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
        if clear_allowed_source_pattern:
            entity.allowed_source_pattern = None
        elif allowed_source_pattern is not None:
            entity.allowed_source_pattern = allowed_source_pattern
        if rate_limit_per_minute is not None:
            entity.rate_limit_per_minute = rate_limit_per_minute

        entity.date_updated = datetime.now(UTC)
        try:
            await subscriptions_collection.save_entity(entity)
        except ConcurrentModificationError:
            # the save is fenced on the row this edit read; a fire, a pause or resume, a secret
            # rotation or another edit changed it since, and the whole-row save would write that
            # change away -- a rotated secret restored among them. The model re-reads instead.
            log.info(
                "webhook_subscription_update refused: subscription changed since it was read",
                extra={"extra_data": {"subscription_id": str(parsed)}},
            )
            return _tool_error(
                "webhook_subscription_update",
                "the subscription changed after this edit read it (it fired, was paused, resumed or "
                "rotated, or was edited elsewhere); nothing was saved. Read it again with "
                "webhook_subscription_list and reapply the change if it still makes sense.",
            )
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
        "Change a webhook subscription. Pass only what changes. To remove a skill, name or "
        "source pattern, pass detach_default_skill, clear_name or clear_allowed_source_pattern, "
        "not the value as well. To change the secret, use webhook_subscription_rotate_secret."
    )
    return [webhook_subscription_update]


# ---------------------------------------------------------------------------
# webhook_subscription_list
# ---------------------------------------------------------------------------


def load_webhook_subscription_list_tool(
    *,
    user_id: UUID,
    agent_id: UUID,
    subscriptions_collection: WebhookSubscriptionCollection,
    registry: WakeRegistryClient,
) -> list[BaseTool]:
    """Build a ``webhook_subscription_list`` tool listing the agent's subscriptions in every conversation."""

    class _ListInput(BaseModel):
        """No-arg list."""

    @tool("webhook_subscription_list", args_schema=_ListInput)
    async def webhook_subscription_list() -> str:
        """List your inbound webhook subscriptions."""
        try:
            visible = await subscriptions_collection.list_for_agent(agent_id)
        except Exception as exc:  # noqa: BLE001
            return _tool_error("webhook_subscription_list", f"list failed: {exc}")

        if not visible:
            return "You have no webhook subscriptions."

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
        "List your webhook subscriptions, in every conversation: id, name, status and the conversation each lives in."
    )
    return [webhook_subscription_list]


# ---------------------------------------------------------------------------
# webhook_subscription_pause / resume / delete
# ---------------------------------------------------------------------------


def load_webhook_subscription_pause_tool(
    *,
    agent_id: UUID,
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
        entity = await subscriptions_collection.find_for_agent(agent_id, parsed)
        if entity is None:
            return _tool_error("webhook_subscription_pause", "subscription not found")
        try:
            await subscriptions_collection.pause(entity.conversation_id, parsed)
        except Exception as exc:  # noqa: BLE001
            return _tool_error("webhook_subscription_pause", f"persist failed: {exc}")
        return f"Paused [webhook:{parsed}]."

    webhook_subscription_pause.description = (
        "Pause a webhook subscription. Events sent to it are refused until webhook_subscription_resume."
    )
    return [webhook_subscription_pause]


def load_webhook_subscription_resume_tool(
    *,
    agent_id: UUID,
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
        entity = await subscriptions_collection.find_for_agent(agent_id, parsed)
        if entity is None:
            return _tool_error("webhook_subscription_resume", "subscription not found")
        try:
            await subscriptions_collection.resume(entity.conversation_id, parsed)
        except Exception as exc:  # noqa: BLE001
            return _tool_error("webhook_subscription_resume", f"persist failed: {exc}")
        return f"Resumed [webhook:{parsed}]."

    webhook_subscription_resume.description = "Resume a paused webhook subscription."
    return [webhook_subscription_resume]


def load_webhook_subscription_delete_tool(
    *,
    agent_id: UUID,
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
        entity = await subscriptions_collection.find_for_agent(agent_id, parsed)
        if entity is None:
            return _tool_error("webhook_subscription_delete", "subscription not found")
        try:
            await subscriptions_collection.delete((entity.conversation_id, parsed))
        except Exception as exc:  # noqa: BLE001
            return _tool_error("webhook_subscription_delete", f"persist failed: {exc}")
        return f"Deleted [webhook:{parsed}] ({entity.name or 'untitled'})."

    webhook_subscription_delete.description = "Delete a webhook subscription for good. Its past fires are kept."
    return [webhook_subscription_delete]


# ---------------------------------------------------------------------------
# webhook_subscription_rotate_secret
# ---------------------------------------------------------------------------


def load_webhook_subscription_rotate_secret_tool(
    *,
    agent_id: UUID,
    subscriptions_collection: WebhookSubscriptionCollection,
    encryption_service: EncryptionService,
) -> list[BaseTool]:
    """Build a ``webhook_subscription_rotate_secret`` tool.

    Generates a new 32-byte secret, encrypts, replaces the row's
    ciphertext, and returns the plaintext ONCE. The previous secret is
    irrecoverable after rotation -- inbound webhooks signed with the
    old key will start failing HMAC verification at the next request.

    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
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
        entity = await subscriptions_collection.find_for_agent(agent_id, parsed)
        if entity is None:
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
                entity.conversation_id,
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
        "Replace a webhook subscription's secret. The old one stops working at once, so every "
        "sender needs the new one. The new secret is shown only once."
    )
    return [webhook_subscription_rotate_secret]
