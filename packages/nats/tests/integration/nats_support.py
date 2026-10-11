"""a nats-server testcontainer started from a configuration given as text.

The configuration reaches the container in an environment variable, and the container's shell
writes it to a file before ``exec``-ing the server. It is never bind-mounted: a bind-mount source
is resolved on the Docker HOST, and where the daemon runs elsewhere (Docker Desktop reached through
a socket from a dev container, a remote ``DOCKER_HOST``) a config written to this machine's temp
directory is not there -- the server exits with ``open /etc/nats/nats.conf: no such file or
directory`` and the start times out waiting for a ready line that never comes.

That needs a shell in the image, so the default is an ``-alpine`` tag: plain ``nats:<version>`` is
built from scratch and has none. The alpine images' ``docker-entrypoint.sh`` runs a command that
does not start with ``-`` as given, so ``sh -c`` reaches the shell.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from testcontainers.nats import NatsContainer

NATS_IMAGE = "nats:2.14.2-alpine"

_CONF_ENV = "NATS_CONF"
_WRITE_CONF_AND_SERVE = f'printf "%s" "${_CONF_ENV}" > /tmp/nats.conf || exit 1; exec nats-server -c /tmp/nats.conf'


def nats_container(config_text: str, *, image: str = NATS_IMAGE) -> NatsContainer:
    """an unstarted nats-server container that will serve ``config_text``.

    :param config_text: the server configuration
    :ptype config_text: str
    :param image: the nats-server image; must carry a shell
    :ptype image: str
    :return: the container, not yet started
    :rtype: NatsContainer
    """
    from testcontainers.nats import NatsContainer  # noqa: PLC0415

    return (
        NatsContainer(image=image, jetstream=False)
        .with_env(_CONF_ENV, config_text)
        .with_command(["sh", "-c", _WRITE_CONF_AND_SERVE])
    )


@contextlib.contextmanager
def running_nats(config_text: str, *, image: str = NATS_IMAGE) -> Iterator[str]:
    """start a nats-server serving ``config_text``; yield its URI and stop it on exit.

    :param config_text: the server configuration
    :ptype config_text: str
    :param image: the nats-server image; must carry a shell
    :ptype image: str
    :return: the server's URI, yielded
    :rtype: Iterator[str]
    """
    container = nats_container(config_text, image=image)
    container.start()
    try:
        yield container.nats_uri()
    finally:
        container.stop()
