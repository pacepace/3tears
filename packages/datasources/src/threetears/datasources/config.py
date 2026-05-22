"""agent.yaml-facing datasource configuration model.

:class:`DatasourceConfig` is the developer-facing pydantic shape every
agent.yaml ``datasources:`` block validates against. it is framework-
level (not Hub-specific): future 3tears products consume the same
model so the developer-facing config shape stays consistent.

Hub admin DTOs (``DataSourceCreateRequest``, ``DataSourceResponse``,
etc.) explicitly STAY in Hub (``aibots/hub/datasources/hub_api.py``)
because they're API contracts, not framework primitives. only the
agent-yaml shape lives here.

NOTE: this module currently carries the flat shape (host / port /
database / username / password_env at the top level). the per-driver
discriminated-union ``ConnectionConfig`` refactor lands in
``datasource-task-08`` and replaces the flat fields with a nested
``connection_config:`` block. callers of THIS shard should plan for
the shape change.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

__all__ = ["DatasourceConfig"]


# regex pattern for valid environment variable names. duplicated from
# the SDK's ``agent_config.py`` so this module has no SDK-side import
# (the SDK depends on this package, not the other way around).
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# valid access mode values, mirrored from
# :class:`threetears.datasources.entities.DataSourceAccessMode`. kept
# as a literal set here so the config validator stays a pure pydantic
# field check without pulling the enum module at validation time.
_VALID_ACCESS_MODES = frozenset({"read", "write", "readwrite"})


class DatasourceConfig(BaseModel):
    """configuration for single external data source connection.

    :param name: human-readable name for this data source
    :ptype name: str
    :param datasource_type: type of data source (e.g. redshift,
        snowflake, bigquery, postgres). accepts the YAML alias
        ``type:`` for backward compatibility with the agent.yaml
        shape callers wrote against before this module existed.
    :ptype datasource_type: str
    :param host: hostname or endpoint for data source connection
    :ptype host: str
    :param database: database name to connect to
    :ptype database: str
    :param port: port number for data source connection
    :ptype port: int | None
    :param schemas: list of database schemas to expose to agents
    :ptype schemas: list[str]
    :param username: database username for connection
    :ptype username: str | None
    :param password_env: environment variable NAME holding the
        database password. NOT the secret value itself. driver
        resolves the env var at the last possible moment.
    :ptype password_env: str | None
    :param access_mode: tool registration mode (read, write, readwrite)
    :ptype access_mode: str
    """

    name: str
    datasource_type: str = Field(alias="type")
    host: str
    database: str
    port: int | None = None
    schemas: list[str] = Field(default_factory=list)
    username: str | None = None
    password_env: str | None = None
    access_mode: str = "readwrite"

    model_config = {"populate_by_name": True}

    @field_validator("password_env")
    @classmethod
    def password_env_must_be_valid_name(cls, value: str | None) -> str | None:
        """validates that password_env is a valid environment variable name.

        :param value: environment variable name or None
        :ptype value: str | None
        :return: validated environment variable name or None
        :rtype: str | None
        :raises ValueError: if name contains invalid characters
        """
        if value is not None and not _ENV_VAR_NAME_RE.match(value):
            msg = (
                f"invalid password_env {value!r}: "
                f"must be valid environment variable name "
                f"(letters, digits, underscores, starts with letter or underscore)"
            )
            raise ValueError(msg)
        return value

    @field_validator("access_mode")
    @classmethod
    def access_mode_must_be_valid(cls, value: str) -> str:
        """validates that access_mode is one of read, write, readwrite.

        :param value: access mode string to validate
        :ptype value: str
        :return: validated access mode string
        :rtype: str
        :raises ValueError: if access mode is not valid
        """
        if value not in _VALID_ACCESS_MODES:
            msg = f"invalid access_mode {value!r}: must be one of read, write, readwrite"
            raise ValueError(msg)
        return value
