"""dataset-as-code: the name-keyed ``datasets/*.dataset.yaml`` envelope.

The file-first sibling of authoring a definition inside a chat. A
definition is the durable artifact, and without a file surface there is
no code review of an audience definition, no git history for one, and no
diff when one changes.

This module is the parse half and is deliberately OFFLINE: no Hub, no
NATS, no database. That is what makes it usable in CI and in a
pre-commit hook, and it is why the definition model lives in 3tears
rather than Hub-side. Nothing here opens a socket, and nothing here
deletes.

It mirrors knowledge-as-code
(``aibots_agents.devx.knowledge_file``) rather than inventing a second
shape for the same job, so an operator reading two reports does not have
to learn two vocabularies:

- files are **NAME-KEYED**, never UUID-keyed, so they travel across
  environments where dataset ids differ. A definition's identity is its
  ``name``, and a name repeated across two files in one directory is an
  error, because it makes fetch-match-apply ambiguous.
- **customer scope only.** ``scope: platform`` is a HARD error, rejected
  in the parser before pydantic sees the payload. Promotion is the only
  cross-customer channel, and a file that could mint platform scope would
  make "naming it" the same act as "authorizing it".
- **removed = drift, never auto-delete.** A definition present in the
  store and absent from the files is REPORTED (:func:`dataset_drift`).
  Deleting it would strand its runs' inventory and turn their tables into
  objects no inventory-keyed reaper can find.

**The envelope is an envelope, not a second schema.** It carries three
things -- the name, the scope, and the definition body -- and delegates
every field validation to
:class:`~threetears.datasources.definition.dataset.DatasetDefinition`. A
parallel schema would drift from the model within one release. It also
inherits the model's ``extra="forbid"``, which closes the gap
``aibots knowledge validate`` still carries: that validator cannot catch
a misspelled ``enforcement`` key because the field is an untyped dict.

**No visibility and no dataset grants.** Per D21 a dataset's audience is
platform state, validated at promotion and at every grant, and a file
that could set either would let anyone who can open a pull request grant
themselves cross-customer access. ``DatasetDefinition`` already refuses
both through ``extra="forbid"``; :func:`_reject_platform_state` refuses
them earlier with a message that names the field and points at the admin
path, rather than leaving an author to decode a generic extra-field
error. The warehouse ``GRANT`` under
:class:`~threetears.datasources.definition.delivery.DeliverySpec` is a
different thing entirely and stays authorable.

**Import is on demand, never on boot.** ``datasources/*.yaml``,
``schema/*.schema.yaml`` and ``knowledge/*.knowledge.yaml`` are imported
on EVERY agent start; ``datasets/*.dataset.yaml`` deliberately is not.
Those three are agent configuration, and re-applying them is idempotent.
A definition is content whose content hash IS its version, and versions
are immutable -- so an automatic import is an automatic version mint, and
minting one fires ``definition_drift`` over every historical run of the
definition. Both failure modes were weighed: on-boot means a malformed
file blocks agent boot for a capability the session may never use;
on-demand means a definition can lag its file until someone runs the
import. The second is visible (a run pins the version it used) and the
first is not, so import stays an explicit act.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from threetears.observe import get_logger

from threetears.datasources.definition.dataset import DatasetDefinition

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

__all__ = [
    "DATASETS_DIR_NAME",
    "DATASET_FILE_SUFFIX",
    "DatasetDriftReport",
    "DatasetFile",
    "DatasetFileError",
    "dataset_drift",
    "discover_dataset_files",
    "load_dataset_file",
    "load_dataset_files",
    "render_dataset_file",
    "unloaded_dataset_files",
    "warn_unloaded_dataset_files",
]

log = get_logger(__name__)

#: directory name dataset files live under, beside ``knowledge/`` and
#: ``schema/`` in an agent project.
DATASETS_DIR_NAME = "datasets"

#: filename suffix of one definition's file, parallel to
#: ``.knowledge.yaml`` and ``.schema.yaml``.
DATASET_FILE_SUFFIX = ".dataset.yaml"

#: the one scope a file may name. ``platform`` is a hard error and
#: ``user`` is not a file-authored scope, so ``customer`` is the only
#: value an author may write, and omitting the field means the same.
_FILE_SCOPE = "customer"

#: fields a file may never carry at the envelope or definition level.
#: Per D21 these are administered rows on the dataset record, not
#: definition content.
_PLATFORM_STATE_FIELDS: tuple[str, ...] = ("visibility", "grants")


class DatasetFileError(Exception):
    """raised when a dataset file fails to read, parse, or validate.

    One error type for the whole file surface, matching the knowledge
    layer's single ``KnowledgeFileError``: a caller distinguishing a
    malformed file from a platform-scope file by exception class would be
    coupling to a taxonomy that has no consumer.

    :param message: human-readable error description
    :ptype message: str
    """

    def __init__(self, message: str) -> None:
        """carry the message onto the base exception.

        :param message: human-readable error description
        :ptype message: str
        :returns: None
        :rtype: None
        """
        super().__init__(message)


class DatasetFile(BaseModel):
    """one ``*.dataset.yaml``: a name, a scope, and a delegated body.

    The name is carried ONCE, at the envelope, and threaded into the body
    at validation. A body repeating it would be a second source of truth
    for the natural key, and the two would disagree the first time one is
    edited alone.

    :ivar name: definition name; the natural key the importer matches on
    :ivar scope: file scope; ``None`` (the default) and ``customer`` are
        the only admissible values, and ``platform`` is a hard error
    :ivar definition: the delegated definition body, validated in full by
        :class:`~threetears.datasources.definition.dataset.DatasetDefinition`
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    scope: str | None = None
    definition: DatasetDefinition

    @model_validator(mode="before")
    @classmethod
    def _body_takes_the_file_name(cls, value: object) -> object:
        """thread the envelope's name into the delegated definition body.

        :param value: raw authored payload
        :ptype value: object
        :returns: payload whose body carries the envelope's name
        :rtype: object
        :raises ValueError: the body declares a name of its own
        """
        payload: object = value
        if isinstance(value, dict) and isinstance(value.get("definition"), dict) and "name" in value:
            body = value["definition"]
            if "name" in body:
                raise ValueError(
                    "the definition body declares name; a dataset file carries the name ONCE at the "
                    "file level, where it is the natural key the importer matches on. remove the "
                    "name under definition"
                )
            payload = {**value, "definition": {**body, "name": value["name"]}}
        return payload


class DatasetDriftReport(BaseModel):
    """definitions present in a store and absent from the files.

    Carries no deleted list because nothing is ever deleted. Absence is
    drift, and drift is reported and stopped on: a definition's runs own
    warehouse inventory, and the reaper finds those tables through that
    inventory alone, so dropping the definition strands them.

    :ivar drift: names present in the store and absent from the files,
        sorted
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    drift: list[str] = Field(default_factory=list)


def _reject_platform_scope(raw: dict[str, object], file_label: str) -> None:
    """reject any platform-scope declaration in the raw payload.

    The parser-side half of the hard guard, scanning the envelope's
    ``scope`` and the definition body's before pydantic sees either. A
    file MUST NOT be able to request platform scope: promotion is the only
    cross-customer channel, and per D21 it is a visibility transition on
    the dataset record plus a grant, not something a file declares.

    :param raw: parsed YAML mapping
    :ptype raw: dict[str, object]
    :param file_label: human-readable file identity for the message
    :ptype file_label: str
    :returns: None
    :rtype: None
    :raises DatasetFileError: platform scope appears at either level
    """
    body = raw.get("definition")
    candidates = [raw, body] if isinstance(body, dict) else [raw]
    for level in candidates:
        if str(level.get("scope", "")).lower() == "platform":
            raise DatasetFileError(
                f"dataset file {file_label} declares scope 'platform'; a file may not mint a "
                f"platform-scope definition (promotion is the only cross-customer channel). remove "
                f"the scope field (import is at the agent's customer scope) and promote through the "
                f"admin path -- per D21 promotion is a visibility transition on the dataset record "
                f"plus a grant, and neither is authored content."
            )


def _reject_platform_state(raw: dict[str, object], file_label: str) -> None:
    """reject an authored ``visibility`` or dataset ``grants`` (D21).

    ``DatasetDefinition`` already refuses both through ``extra="forbid"``,
    and this does not restate that rule -- it fails EARLIER with a message
    that names the field and points at the admin path, because a generic
    extra-field error leaves the author to guess that the field is
    forbidden by policy rather than misspelled.

    Scoped to the envelope and the definition body's top level.
    :attr:`~threetears.datasources.definition.delivery.DeliverySpec.grants`
    is the warehouse ``GRANT``, a delivered contract excluded from the
    content hash, and is untouched.

    :param raw: parsed YAML mapping
    :ptype raw: dict[str, object]
    :param file_label: human-readable file identity for the message
    :ptype file_label: str
    :returns: None
    :rtype: None
    :raises DatasetFileError: either field appears at either level
    """
    body = raw.get("definition")
    candidates = [raw, body] if isinstance(body, dict) else [raw]
    for level in candidates:
        for field in _PLATFORM_STATE_FIELDS:
            if field in level:
                raise DatasetFileError(
                    f"dataset file {file_label} declares {field!r}; per D21 a dataset's audience is "
                    f"platform state on the dataset record -- a visibility transition and a grant, "
                    f"each validated against the anti-laundering bound -- not definition content. a "
                    f"file that could set it would let anyone who can open a pull request grant "
                    f"themselves cross-customer access. remove the field and use the admin path. "
                    f"(the warehouse GRANT under definition.delivery.grants is a different thing and "
                    f"stays authorable.)"
                )


def _reject_non_customer_scope(dataset_file: DatasetFile, file_label: str) -> None:
    """reject a file-level scope that is not the customer default.

    :func:`_reject_platform_scope` already catches the literal
    ``platform``; this catches any OTHER non-customer scope a file might
    name, since a file authors the agent-owning customer's copy and
    nothing else.

    :param dataset_file: parsed envelope
    :ptype dataset_file: DatasetFile
    :param file_label: human-readable file identity for the message
    :ptype file_label: str
    :returns: None
    :rtype: None
    :raises DatasetFileError: scope is set to a non-customer value
    """
    scope = dataset_file.scope
    if scope is not None and scope.lower() != _FILE_SCOPE:
        raise DatasetFileError(
            f"dataset file {file_label} declares scope {scope!r}; a file imports at the agent's "
            f"customer scope only. remove the scope field (the default) or set it to 'customer'."
        )


def load_dataset_file(file_path: Path) -> DatasetFile:
    """read and validate one ``*.dataset.yaml``, offline.

    Reads YAML, enforces the platform-scope and platform-state guards,
    then validates the envelope -- which validates the whole definition,
    including its content hash's inputs. Nothing here reaches a Hub, a
    broker, or a database.

    :param file_path: path to a dataset file
    :ptype file_path: pathlib.Path
    :returns: validated envelope
    :rtype: DatasetFile
    :raises DatasetFileError: the file is unreadable or malformed, is not
        a mapping, declares platform scope or platform state, or fails the
        definition model
    """
    try:
        # parsed from the open handle rather than from a string, so PyYAML's
        # problem mark names the real path beside its line and column. read
        # into a string first and every syntax error is reported against
        # "<unicode string>", which is useless in a directory of files.
        with file_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except Exception as error:
        raise DatasetFileError(f"failed to read dataset file {file_path}: {error}") from error

    if not isinstance(raw, dict):
        raise DatasetFileError(f"dataset file {file_path} must be a YAML mapping, got {type(raw).__name__}")

    _reject_platform_scope(raw, str(file_path))
    _reject_platform_state(raw, str(file_path))

    try:
        loaded = DatasetFile.model_validate(raw)
    except Exception as error:
        raise DatasetFileError(f"invalid dataset file {file_path}: {error}") from error

    _reject_non_customer_scope(loaded, str(file_path))
    return loaded


def discover_dataset_files(datasets_path: Path) -> list[Path]:
    """resolve a path argument to the dataset files to act on.

    Accepts a directory, which globs ``*.dataset.yaml``, or a single file,
    which loads regardless of suffix so an explicitly named file is never
    silently skipped.

    :param datasets_path: directory or single dataset file
    :ptype datasets_path: pathlib.Path
    :returns: sorted dataset file paths
    :rtype: list[pathlib.Path]
    :raises DatasetFileError: the path does not exist, or a directory
        holds no dataset files
    """
    if datasets_path.is_dir():
        found = sorted(datasets_path.glob(f"*{DATASET_FILE_SUFFIX}"))
        if not found:
            raise DatasetFileError(f"no *{DATASET_FILE_SUFFIX} files found in {datasets_path}")
        resolved = found
    elif datasets_path.is_file():
        resolved = [datasets_path]
    else:
        raise DatasetFileError(f"dataset path not found: {datasets_path}")
    return resolved


def load_dataset_files(datasets_path: Path) -> dict[str, DatasetFile]:
    """load every dataset file under a path, keyed by definition name.

    The key is the natural key the importer matches on, so a name claimed
    by two files is refused here rather than resolved by directory order.
    Two files claiming one name would otherwise make which body reaches
    the store depend on the glob, and the glob is sorted by filename
    rather than by anything an author controls.

    :param datasets_path: directory or single dataset file
    :ptype datasets_path: pathlib.Path
    :returns: map of definition name to validated envelope
    :rtype: dict[str, DatasetFile]
    :raises DatasetFileError: a file fails to load, or two files claim one
        definition name
    """
    by_name: dict[str, DatasetFile] = {}
    origin: dict[str, Path] = {}
    for file_path in discover_dataset_files(datasets_path):
        loaded = load_dataset_file(file_path)
        claimed = origin.get(loaded.name)
        if claimed is not None:
            raise DatasetFileError(
                f"duplicate dataset name {loaded.name!r}: declared in {claimed.name} and "
                f"{file_path.name}. a definition name is the natural key the importer matches on, so "
                f"two files claiming one name make which body reaches the store depend on filename "
                f"order. rename one, or merge them"
            )
        by_name[loaded.name] = loaded
        origin[loaded.name] = file_path
    return by_name


def unloaded_dataset_files(datasets_path: Path) -> list[Path]:
    """return YAML files under a datasets dir the loader will NOT load.

    A directory load globs ``*.dataset.yaml``; any other ``*.yaml`` /
    ``*.yml`` sitting beside it is silently skipped and therefore
    invisibly dead. Callers warn on the returned list. A single-file path
    loads regardless of suffix and so has no un-loaded siblings to report.

    :param datasets_path: the path the command was invoked on
    :ptype datasets_path: pathlib.Path
    :returns: sorted ``*.yaml`` / ``*.yml`` files not matching the suffix
    :rtype: list[pathlib.Path]
    """
    stray: list[Path] = []
    if datasets_path.is_dir():
        loaded = set(datasets_path.glob(f"*{DATASET_FILE_SUFFIX}"))
        candidates = set(datasets_path.glob("*.yaml")) | set(datasets_path.glob("*.yml"))
        stray = sorted(candidates - loaded)
    return stray


def warn_unloaded_dataset_files(datasets_path: Path, warn: Callable[[str], None]) -> int:
    """emit a loud warning for every un-loaded dataset YAML file.

    :param datasets_path: the path the command was invoked on
    :ptype datasets_path: pathlib.Path
    :param warn: one-arg callable that renders a warning line
    :ptype warn: collections.abc.Callable[[str], None]
    :returns: count of un-loaded files warned about
    :rtype: int
    """
    stray = unloaded_dataset_files(datasets_path)
    for path in stray:
        warn(
            f"NOT LOADED: {path.name} -- dataset files MUST end in '{DATASET_FILE_SUFFIX}'. rename it "
            f"(or pass it explicitly); otherwise its definition never reaches the store."
        )
    return len(stray)


def dataset_drift(file_names: Iterable[str], store_names: Iterable[str]) -> DatasetDriftReport:
    """report store definitions absent from the files; never delete.

    Absence is drift. A definition present in the store and absent from
    the files is REPORTED and left alone, because its runs own warehouse
    inventory and the reaper finds those tables through that inventory
    alone -- dropping the definition strands them as objects nothing can
    key on. Deletion stays an explicit, separately-authorized act.

    :param file_names: definition names the files declare
    :ptype file_names: collections.abc.Iterable[str]
    :param store_names: customer-scope definition names the store holds
    :ptype store_names: collections.abc.Iterable[str]
    :returns: the drifted names, sorted
    :rtype: DatasetDriftReport
    """
    drifted = sorted(set(store_names) - set(file_names))
    for name in drifted:
        log.warning(
            "dataset drift: %r exists in the store (customer scope) but is absent from the dataset "
            "file(s); not deleted (deletion is explicit)",
            name,
        )
    return DatasetDriftReport(drift=drifted)


def render_dataset_file(dataset_file: DatasetFile) -> str:
    """render an envelope back to ``*.dataset.yaml`` text.

    The inverse of :func:`load_dataset_file`, and the reason the round
    trip is testable at all. Serialization goes through pydantic's JSON
    mode so every typed scalar keeps its tag -- notably
    :class:`~threetears.datasources.definition.expression.LiteralType`,
    without which a text ``'1.0'`` and a decimal ``1.0`` re-read as the
    same value and the edit between them mints no version. A
    serialization that drifts from the model mints versions silently, and
    a minted version is immutable.

    :param dataset_file: validated envelope
    :ptype dataset_file: DatasetFile
    :returns: YAML text, parseable by :func:`load_dataset_file`
    :rtype: str
    """
    dumped = dataset_file.definition.model_dump(mode="json")
    # the name lives at the envelope, once. it is stripped by comprehension
    # rather than by a mutating verb so this module holds no removal call of
    # any kind -- the structural half of "drift is never deleted".
    body = {key: value for key, value in dumped.items() if key != "name"}
    document: dict[str, object] = {"name": dataset_file.name}
    if dataset_file.scope is not None:
        document["scope"] = dataset_file.scope
    document["definition"] = body
    rendered: str = yaml.safe_dump(document, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100)
    return rendered
