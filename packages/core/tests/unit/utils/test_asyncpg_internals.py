"""the asyncpg private surface the pool start depends on is still there, and still the boundary it was.

:mod:`threetears.core.utils._asyncpg_internals` names asyncpg's DSN and connect-option parser so a
pool start can withhold the text of the errors it raises. A release that renames the parser fails
the import of that module; a release that moves the DSN parse out of it, or the ``command_timeout``
check into it, fails here, against the installed asyncpg, before anything ships against it.
"""

from __future__ import annotations

import asyncpg
import pytest

from threetears.core.utils._asyncpg_internals import raised_while_parsing_connect_parameters


async def _connect_error(dsn: str, **kwargs: object) -> ValueError:
    """the ``ValueError`` ``asyncpg.connect`` raises for ``dsn``, before any network.

    :param dsn: a DSN asyncpg refuses
    :ptype dsn: str
    :param kwargs: further connect arguments
    :ptype kwargs: object
    :return: the error, traceback attached
    :rtype: ValueError
    """
    with pytest.raises(ValueError) as exc_info:
        await asyncpg.connect(dsn, timeout=1, **kwargs)
    return exc_info.value


class TestTheParserIsTheBoundary:
    """errors raised while parsing the DSN are the parser's; the earlier option checks are not."""

    async def test_a_dsn_whose_port_does_not_parse_is_the_parsers(self) -> None:
        error = await _connect_error("postgresql://u:se@cret:TAIL@127.0.0.1/d")
        assert raised_while_parsing_connect_parameters(error)

    async def test_a_dsn_whose_query_does_not_parse_is_the_parsers(self) -> None:
        error = await _connect_error("postgresql://u:pa?ssWORD@127.0.0.1:1/d")
        assert raised_while_parsing_connect_parameters(error)

    async def test_an_invalid_sslmode_is_the_parsers(self) -> None:
        error = await _connect_error("postgresql://u:p@127.0.0.1:1/d?sslmode=bogus")
        assert isinstance(error, asyncpg.exceptions.ClientConfigurationError)
        assert raised_while_parsing_connect_parameters(error)

    async def test_a_bad_command_timeout_is_not_the_parsers(self) -> None:
        """asyncpg checks ``command_timeout`` before it calls the parser; its message quotes only that value."""
        error = await _connect_error("postgresql://u:p@127.0.0.1:1/d", command_timeout=-1)
        assert "command_timeout" in str(error)
        assert not raised_while_parsing_connect_parameters(error)

    def test_an_error_with_no_traceback_is_not_the_parsers(self) -> None:
        assert not raised_while_parsing_connect_parameters(ValueError("never raised"))
