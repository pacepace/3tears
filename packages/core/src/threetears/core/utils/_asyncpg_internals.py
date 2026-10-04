"""the one owner of every ``asyncpg`` private name ``threetears.core`` depends on.

:func:`~threetears.core.utils.pg_pool_kwargs.create_pool_with_startup_timeout` withholds the text
of an error from asyncpg's connection-parameter handling -- that text describes what the client
sent, and with a stray ``@`` or ``?`` in a password it quotes the password -- while every other
error keeps its own. asyncpg exposes no public boundary between parsing a DSN and its connect
options and the rest of a connect: both raise plain ``ValueError``. The boundary is the function
that does the parsing, so this module names it, and a release that renames or moves it fails
here, at import, by name -- not silently, by letting the parameter errors keep their text.

**Verified against asyncpg 0.31.0** (``asyncpg/connect_utils.py``) **and present in 0.30.0**, the
declared floor (``asyncpg>=0.30``). Moving to a new release: read ``connect_utils.py`` in it,
confirm the entry below still parses the DSN and its options and is still called from
``_parse_connect_arguments`` after the ``command_timeout`` and statement-cache checks, and run
``tests/unit/utils/test_asyncpg_internals.py`` on it.

The surface, and why each is used:

``asyncpg.connect_utils._parse_connect_dsn_and_args``
    :func:`raised_while_parsing_connect_parameters`. Turns a DSN and its connect arguments into
    addresses and parameters: the DSN's own parse (``urllib.parse``, ``int()`` of a port, the
    query string), the host list, ``sslmode`` and the other options. Every asyncpg error that can
    quote what was sent is raised inside it; ``ClientConfigurationError`` is raised nowhere else.
"""

from __future__ import annotations

import asyncpg.connect_utils

__all__ = ["raised_while_parsing_connect_parameters"]

_PARAMETER_PARSER_CODE = asyncpg.connect_utils._parse_connect_dsn_and_args.__code__


def raised_while_parsing_connect_parameters(error: BaseException) -> bool:
    """whether ``error`` was raised inside asyncpg's DSN and connect-option parsing.

    read off the error's traceback: true when one of its frames is running the parser.

    :param error: what a connect raised, with its traceback
    :ptype error: BaseException
    :return: ``True`` when the parser is on the error's traceback
    :rtype: bool
    """
    result = False
    frame = error.__traceback__
    while frame is not None and not result:
        result = frame.tb_frame.f_code is _PARAMETER_PARSER_CODE
        frame = frame.tb_next
    return result
