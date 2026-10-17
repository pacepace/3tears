"""a nats-server testcontainer started from a configuration given as text.

The configuration reaches the container in an environment variable and the container's shell
writes it out -- never a bind mount, whose source is resolved on the Docker HOST. Why, and why the
image must carry a shell: :func:`threetears.core.testing.fixtures.nats_server`.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from threetears.core.testing.fixtures import NATS_TEST_IMAGE, nats_server, nats_server_uri

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextlib.contextmanager
def running_nats(config_text: str, *, image: str = NATS_TEST_IMAGE) -> Iterator[str]:
    """start a nats-server serving ``config_text``; yield its URI and stop it on exit.

    :param config_text: the server configuration
    :ptype config_text: str
    :param image: the nats-server image; must carry a shell
    :ptype image: str
    :return: the server's URI, yielded
    :rtype: Iterator[str]
    """
    with nats_server(image, config=config_text) as container:
        yield nats_server_uri(container)
