"""the "repository advanced" broadcast: a writer pushed, so every other clone fetches.

After a push the writer publishes :class:`RepoAdvanced` on ``{ns}.git-store.advanced`` (the
namespace resolved when it is sent, so a process connected under another namespace never hears
it). A process with its own clone of that repository that is not its writer fetches and
fast-forwards, and reports the rows the new commits changed. Processes that share the writer's
working tree need none of this: they read the files the writer wrote.

The message is a trigger only. What changed is read from git, so a forged message costs a fetch
and nothing else. Delivery is best effort: a lost message leaves a clone behind until the next one
or its own periodic fetch; writes never depend on it.

One :class:`AdvancedListener` serves every repository a process opens on one client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from threetears.nats.errors import NatsClientError
from threetears.nats.subjects import Subjects
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.nats import NatsClient, Subscription

__all__ = ["AdvancedListener", "AdvancedReceiver", "RepoAdvanced", "publish_advanced"]

log = get_logger(__name__)


class RepoAdvanced(BaseModel):
    """a repository's branch moved on origin.

    :ivar key: the repository's key
    :ivar branch: the branch
    :ivar head: the commit it now names
    """

    key: str
    branch: str
    head: str


class AdvancedReceiver(Protocol):
    """a repository the listener hands its own messages to."""

    async def advanced(self, message: RepoAdvanced) -> None:
        """react to the repository's branch moving on origin.

        :param message: the message
        :ptype message: RepoAdvanced
        :return: None
        :rtype: None
        """
        ...


async def publish_advanced(nats: NatsClient, message: RepoAdvanced) -> None:
    """tell every process a repository's branch moved; a failure is logged, never raised.

    :param nats: the connected client
    :ptype nats: NatsClient
    :param message: the message
    :ptype message: RepoAdvanced
    :return: None
    :rtype: None
    """
    try:
        await nats.publish(subject=Subjects.git_store_advanced(), message=message)
    except NatsClientError as exc:
        log.warning(
            "git_store.advanced.unsent: other clones stay behind until the next push or their own fetch",
            extra={"extra_data": {"key": message.key, "branch": message.branch, "error": str(exc)}},
        )


class AdvancedListener:
    """one subscription to the broadcast, handing each message to the repository it names."""

    def __init__(self, nats: NatsClient) -> None:
        """bind to a client; nothing is subscribed until the first repository registers.

        :param nats: the connected client
        :ptype nats: NatsClient
        :return: None
        :rtype: None
        """
        self._nats = nats
        self._receivers: dict[tuple[str, str], AdvancedReceiver] = {}
        self._subscription: Subscription | None = None

    async def register(self, key: str, branch: str, receiver: AdvancedReceiver) -> None:
        """hand messages for ``(key, branch)`` to ``receiver``, subscribing on first use.

        :param key: the repository's key
        :ptype key: str
        :param branch: the branch
        :ptype branch: str
        :param receiver: the repository
        :ptype receiver: AdvancedReceiver
        :return: None
        :rtype: None
        """
        self._receivers[(key, branch)] = receiver
        if self._subscription is None:
            # sender-posture: trigger-only -- the message only prompts a fetch; what changed is read from git
            self._subscription = await self._nats.subscribe_typed(
                subject=Subjects.git_store_advanced(),
                cb=self._on_message,
                message_type=RepoAdvanced,
                deadletter_on_failure=False,
            )

    async def unregister(self, key: str, branch: str) -> None:
        """stop handing messages for ``(key, branch)``; the last one out unsubscribes.

        :param key: the repository's key
        :ptype key: str
        :param branch: the branch
        :ptype branch: str
        :return: None
        :rtype: None
        """
        self._receivers.pop((key, branch), None)
        if not self._receivers and self._subscription is not None:
            subscription, self._subscription = self._subscription, None
            await self._nats.unsubscribe(subscription)

    async def _on_message(self, message: RepoAdvanced) -> None:
        """hand a message to its repository, if this process has it open.

        :param message: the message
        :ptype message: RepoAdvanced
        :return: None
        :rtype: None
        """
        receiver = self._receivers.get((message.key, message.branch))
        if receiver is not None:
            await receiver.advanced(message)
