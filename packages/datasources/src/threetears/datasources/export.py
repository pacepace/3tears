"""A warehouse's rows written to S3 as parquet by the warehouse itself, in place of paging them over the bus.

**Why.** Reading a large relation through the datasource rail costs a round trip per page, and the
page is bounded by the hub's row cap and the bus's message size: a few hundred thousand rows take
tens of minutes, almost all of it waiting. A warehouse that can write a query's result to object
storage (Redshift's ``UNLOAD``) does the same work in seconds, and the reader takes the files
straight from the bucket.

**What may be exported, and where.** The caller sends a plain ``SELECT``, one the read rail would
run for it, and a destination: a short relative path. It never sends the ``UNLOAD`` or a bucket. The
statement around the ``SELECT`` is fixed here (:func:`redshift_unload_statement`), its options
allow-listed: parquet, a manifest, the datasource's configured role, and no overwrite. The bucket,
the prefix under it and the role are the datasource's own configuration
(:class:`ExportConfig`, on its connection config), so a caller cannot name a location the operator
did not choose; the destination is checked against a grammar with no ``..``, no scheme and no
leading slash, and lands under the configured prefix.

**Embedding a ``SELECT`` in a string literal.** ``UNLOAD`` takes its query as a quoted literal, so
the ``SELECT`` is quoted once more here: every ``'`` doubled. Redshift also reads a backslash in a
literal as an escape, which would let a ``SELECT`` close the literal early; so a ``SELECT`` holding a
backslash, or a NUL, is refused rather than escaped (:class:`ExportRefusedError`). An export carries
no bind parameters, because ``UNLOAD`` cannot bind them; a value the ``SELECT`` filters by is written
into it as a literal (:func:`sql_string_literal`, with the same refusals).

**What comes back** (:class:`UnloadResult`): how many rows the warehouse wrote, and where: the
bucket, the key prefix every file sits under, and the manifest that lists them. The reader trusts
none of it on its own: :mod:`threetears.datasources.export_read` checks the files against the
manifest, the rows against that count, and the count against a fingerprint of the relation taken
before and after.

Imports nothing backend-specific, so the lazy-import contract of
:mod:`threetears.datasources.drivers` holds.
"""

from __future__ import annotations

import re
from typing import Final, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "DESTINATION_GRAMMAR",
    "DriverExportUnsupportedError",
    "ExportConfig",
    "ExportRefusedError",
    "UnloadResult",
    "check_destination",
    "check_embeddable",
    "redshift_unload_statement",
    "sql_string_literal",
]

#: a destination under the configured prefix: path segments of letters, digits, ``_`` and ``-``,
#: joined by single slashes. No dot at all, so no ``..``; no scheme, no leading or trailing slash.
DESTINATION_GRAMMAR: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{1,128}(/[A-Za-z0-9_-]{1,128}){0,7}$")

#: an S3 bucket name as AWS allows it (lower case, digits, dots and hyphens, 3 to 63 characters)
_BUCKET_GRAMMAR: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")

#: a key prefix: one or more path segments, each ending in a slash; no ``..`` (checked separately)
_PREFIX_GRAMMAR: Final[re.Pattern[str]] = re.compile(r"^([A-Za-z0-9_.-]+/)+$")

#: an IAM role ARN Redshift may assume for the write
_ROLE_ARN_GRAMMAR: Final[re.Pattern[str]] = re.compile(r"^arn:aws:iam::\d{12}:role/[A-Za-z0-9+=,.@_/-]{1,512}$")

#: the role spelling that names the cluster's default role
DEFAULT_ROLE: Final = "default"

#: characters a quoted literal must not hold: Redshift reads a backslash as an escape, and a NUL
#: ends the statement in some clients
_UNEMBEDDABLE: Final[tuple[str, ...]] = ("\\", "\x00")


class ExportRefusedError(ValueError):
    """an export asked for something this module will not write: a ``SELECT`` or value it cannot
    quote safely, or a destination outside the configured prefix."""


class DriverExportUnsupportedError(NotImplementedError):
    """the datasource cannot export: its engine has no export here, or it has no export configured."""


class ExportConfig(BaseModel):
    """where a datasource's exports go, set by the operator on its connection config.

    :param bucket: the S3 bucket the warehouse writes to
    :ptype bucket: str
    :param prefix: the key prefix every export lands under, ending in ``/``
    :ptype prefix: str
    :param iam_role: the role the warehouse writes as: ``default`` (the cluster's default role) or
        a role ARN attached to the cluster
    :ptype iam_role: str
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket: str = Field(description="the S3 bucket the warehouse writes exports to")
    prefix: str = Field(default="exports/", description="the key prefix every export lands under, ending in '/'")
    iam_role: str = Field(
        default=DEFAULT_ROLE,
        description="the role the warehouse writes as: 'default' for the cluster's default role, or a role ARN",
    )

    @field_validator("bucket")
    @classmethod
    def _bucket_is_a_bucket_name(cls, value: str) -> str:
        """refuse anything that is not an S3 bucket name.

        :param value: the bucket
        :ptype value: str
        :return: the value unchanged
        :rtype: str
        :raises ValueError: when it is not a bucket name
        """
        if not _BUCKET_GRAMMAR.match(value) or ".." in value:
            raise ValueError(f"export bucket {value!r} is not an S3 bucket name")
        return value

    @field_validator("prefix")
    @classmethod
    def _prefix_is_a_object_prefix(cls, value: str) -> str:
        """refuse a prefix that is not path segments ending in ``/``, or that climbs with ``..``.

        :param value: the prefix
        :ptype value: str
        :return: the value unchanged
        :rtype: str
        :raises ValueError: when it is not a plain key prefix
        """
        if not _PREFIX_GRAMMAR.match(value) or any(part in (".", "..") for part in value.split("/")):
            raise ValueError(f"export prefix {value!r} must be path segments each ending in '/', with no '.' or '..'")
        return value

    @field_validator("iam_role")
    @classmethod
    def _role_is_default_or_an_arn(cls, value: str) -> str:
        """refuse a role that is neither ``default`` nor a role ARN.

        :param value: the role
        :ptype value: str
        :return: the value unchanged
        :rtype: str
        :raises ValueError: when it is neither
        """
        if value != DEFAULT_ROLE and not _ROLE_ARN_GRAMMAR.match(value):
            raise ValueError(f"export iam_role must be {DEFAULT_ROLE!r} or an IAM role ARN, got {value!r}")
        return value


class UnloadResult(TypedDict):
    """what a warehouse reports about an export it wrote.

    :key row_count: rows the warehouse wrote
    :key bucket: the bucket the files are in
    :key object_prefix: the key prefix every file of this export sits under, ending in ``/``
    :key manifest_path: the object path of the manifest listing the files (named so rather than a key, which the secrets gate reads as a credential)
    """

    row_count: int
    bucket: str
    object_prefix: str
    manifest_path: str


def check_embeddable(text: str, what: str) -> str:
    """refuse text that cannot sit inside a quoted Redshift literal by doubling quotes alone.

    :param text: the text
    :ptype text: str
    :param what: what it is, for the refusal
    :ptype what: str
    :return: the text unchanged
    :rtype: str
    :raises ExportRefusedError: when it holds a backslash or a NUL
    """
    if any(character in text for character in _UNEMBEDDABLE):
        raise ExportRefusedError(
            f"{what} holds a backslash or a NUL, which a quoted literal cannot carry safely; nothing was exported"
        )
    return text


def sql_string_literal(value: str) -> str:
    """``value`` as a single-quoted SQL literal, for a ``SELECT`` an export carries (it binds no parameters).

    :param value: the value
    :ptype value: str
    :return: the literal, quotes included
    :rtype: str
    :raises ExportRefusedError: when the value holds a backslash or a NUL
    """
    check_embeddable(value, "a filter value")
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def check_destination(destination: str) -> str:
    """refuse a destination that is not a short relative path under the configured prefix.

    :param destination: the destination
    :ptype destination: str
    :return: the destination unchanged
    :rtype: str
    :raises ExportRefusedError: when it does not match :data:`DESTINATION_GRAMMAR`
    """
    if not DESTINATION_GRAMMAR.match(destination):
        raise ExportRefusedError(
            f"export destination {destination!r} must be a relative path of letters, digits, '_' and '-' "
            "segments joined by '/'; it lands under the datasource's configured prefix and may not leave it"
        )
    return destination


def redshift_unload_statement(select: str, config: ExportConfig, destination: str) -> tuple[str, UnloadResult]:
    """the one ``UNLOAD`` an export runs, and where its files will be.

    Allow-listed: the ``SELECT`` quoted as a literal, the configured bucket and prefix, the
    configured role, parquet, a verbose manifest (it lists each file's row count), and no
    ``ALLOWOVERWRITE``, so an export never replaces files already at its destination.

    :param select: the ``SELECT``, already admitted as a read the caller may run
    :ptype select: str
    :param config: the datasource's export configuration
    :ptype config: ExportConfig
    :param destination: the relative destination
    :ptype destination: str
    :return: the statement, and the result's location fields (``row_count`` zero until it runs)
    :rtype: tuple[str, UnloadResult]
    :raises ExportRefusedError: when the ``SELECT`` cannot be quoted or the destination is refused
    """
    check_embeddable(select, "the export's SELECT")
    check_destination(destination)
    object_prefix = f"{config.prefix}{destination}/"
    quoted = select.replace("'", "''")
    role = "default" if config.iam_role == DEFAULT_ROLE else f"'{config.iam_role}'"
    statement = (
        f"UNLOAD ('{quoted}') TO 's3://{config.bucket}/{object_prefix}' "
        f"IAM_ROLE {role} FORMAT AS PARQUET MANIFEST VERBOSE"
    )
    location = UnloadResult(
        row_count=0, bucket=config.bucket, object_prefix=object_prefix, manifest_path=f"{object_prefix}manifest"
    )
    return statement, location
