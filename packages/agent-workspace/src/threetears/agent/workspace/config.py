"""workspace configuration pydantic models declared in agent.yaml."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from threetears.agent.workspace.bind_policy import BindConflictPolicy

__all__ = [
    "AllowConfig",
    "BindConfig",
    "ValidatorEntry",
    "WorkspaceConfig",
]


class BindConfig(BaseModel):
    """
    per-workspace bind-behavior configuration.

    :param on_conflict: policy governing L3 vs disk authority inside
        the bind window; see :class:`BindConflictPolicy` for semantics
    :ptype on_conflict: BindConflictPolicy
    """

    on_conflict: BindConflictPolicy = BindConflictPolicy.DISK_WINS


class ValidatorEntry(BaseModel):
    """
    maps relative-path glob to validator dotted import path.

    :param pattern: fnmatch glob such as "*.yaml" or "config/**"
    :ptype pattern: str
    :param validator: dotted import path resolving to callable
    :ptype validator: str
    """

    pattern: str
    validator: str


class AllowConfig(BaseModel):
    """
    per-mode glob allow-lists for workspace file access.

    :param read: globs that may be read; defaults to all paths
    :ptype read: list[str]
    :param write: globs that may be written; defaults to empty (fail-closed)
    :ptype write: list[str]
    """

    read: list[str] = Field(default_factory=lambda: ["**/*"])
    write: list[str] = Field(default_factory=list)


class WorkspaceConfig(BaseModel):
    """
    agent workspace configuration declared in agent.yaml.

    :param templates_dir: filesystem dir holding read-only image-shipped templates
    :ptype templates_dir: Path | None
    :param bind_root: filesystem dir under which bind() materializations land
    :ptype bind_root: Path | None
    :param allow: per-mode glob allow-lists for relative_path access
    :ptype allow: AllowConfig
    :param validators: per-pattern validator hooks invoked on every write
    :ptype validators: list[ValidatorEntry]
    :param bind: bind-behavior configuration (conflict policy, etc.)
    :ptype bind: BindConfig
    """

    templates_dir: Path | None = None
    bind_root: Path | None = None
    allow: AllowConfig = Field(default_factory=AllowConfig)
    validators: list[ValidatorEntry] = Field(default_factory=list)
    bind: BindConfig = Field(default_factory=BindConfig)

    @model_validator(mode="after")
    def _resolve_paths(self) -> WorkspaceConfig:
        """
        enforces absolute-path invariant on templates_dir and bind_root.

        :return: validated config instance
        :rtype: WorkspaceConfig
        :raises ValueError: if templates_dir or bind_root is relative
        """
        result: WorkspaceConfig = self
        for name, value in (
            ("templates_dir", self.templates_dir),
            ("bind_root", self.bind_root),
        ):
            if value is not None and not value.is_absolute():
                msg = f"{name} must be absolute path, got {value!r}"
                raise ValueError(msg)
        return result
