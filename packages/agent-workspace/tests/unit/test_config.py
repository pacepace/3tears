"""unit tests for WorkspaceConfig, AllowConfig, and ValidatorEntry pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.agent.workspace.config import AllowConfig, ValidatorEntry, WorkspaceConfig


def test_workspace_config_no_args_produces_valid_defaults() -> None:
    """WorkspaceConfig() with no args returns config with default AllowConfig and empty validators."""
    config = WorkspaceConfig()
    assert config.templates_dir is None
    assert config.bind_root is None
    assert isinstance(config.allow, AllowConfig)
    assert config.validators == []


def test_allow_config_defaults_read_all_write_empty() -> None:
    """AllowConfig defaults read to ['**/*'] and write to [] (fail-closed)."""
    allow = AllowConfig()
    assert allow.read == ["**/*"]
    assert allow.write == []


def test_allow_config_accepts_explicit_read_and_write_lists() -> None:
    """AllowConfig accepts explicit glob lists for read and write fields."""
    allow = AllowConfig(read=["docs/**"], write=["out/**", "*.txt"])
    assert allow.read == ["docs/**"]
    assert allow.write == ["out/**", "*.txt"]


def test_validator_entry_requires_pattern_and_validator() -> None:
    """ValidatorEntry requires both pattern and validator fields."""
    entry = ValidatorEntry(pattern="*.yaml", validator="pkg.module.callable")
    assert entry.pattern == "*.yaml"
    assert entry.validator == "pkg.module.callable"


def test_validator_entry_missing_validator_field_raises() -> None:
    """ValidatorEntry without validator field raises ValidationError."""
    with pytest.raises(ValidationError):
        ValidatorEntry.model_validate({"pattern": "*.yaml"})


def test_validator_entry_missing_pattern_field_raises() -> None:
    """ValidatorEntry without pattern field raises ValidationError."""
    with pytest.raises(ValidationError):
        ValidatorEntry.model_validate({"validator": "pkg.mod.fn"})


def test_workspace_config_round_trip_model_dump_validate() -> None:
    """WorkspaceConfig(...).model_dump() -> model_validate produces equivalent object."""
    original = WorkspaceConfig(
        templates_dir=Path("/srv/templates"),
        bind_root=Path("/srv/binds"),
        allow=AllowConfig(read=["docs/**"], write=["out/**"]),
        validators=[ValidatorEntry(pattern="*.yaml", validator="pkg.mod.fn")],
    )
    dumped = original.model_dump()
    restored = WorkspaceConfig.model_validate(dumped)
    assert restored == original


def test_workspace_config_relative_templates_dir_raises() -> None:
    """relative templates_dir raises ValidationError from model_validator."""
    with pytest.raises(ValidationError) as excinfo:
        WorkspaceConfig(templates_dir=Path("relative/templates"))
    assert "templates_dir" in str(excinfo.value)


def test_workspace_config_relative_bind_root_raises() -> None:
    """relative bind_root raises ValidationError from model_validator."""
    with pytest.raises(ValidationError) as excinfo:
        WorkspaceConfig(bind_root=Path("relative/bind"))
    assert "bind_root" in str(excinfo.value)


def test_workspace_config_absolute_paths_accepted() -> None:
    """absolute templates_dir and bind_root pass validation."""
    config = WorkspaceConfig(
        templates_dir=Path("/srv/templates"),
        bind_root=Path("/srv/binds"),
    )
    assert config.templates_dir == Path("/srv/templates")
    assert config.bind_root == Path("/srv/binds")


def test_workspace_config_yaml_like_dict_round_trip() -> None:
    """inline dict representing parsed YAML content validates into WorkspaceConfig cleanly.

    we avoid PyYAML as a hard test dep; parsed YAML yields a plain dict that
    model_validate handles identically to yaml.safe_load output.
    """
    yaml_like: dict[str, Any] = {
        "templates_dir": "/srv/templates",
        "bind_root": "/srv/binds",
        "allow": {
            "read": ["**/*"],
            "write": ["out/**"],
        },
        "validators": [
            {"pattern": "*.yaml", "validator": "pkg.mod.yaml_validator"},
            {"pattern": "config/**", "validator": "pkg.mod.config_validator"},
        ],
    }
    config = WorkspaceConfig.model_validate(yaml_like)
    assert config.templates_dir == Path("/srv/templates")
    assert config.bind_root == Path("/srv/binds")
    assert config.allow.read == ["**/*"]
    assert config.allow.write == ["out/**"]
    assert len(config.validators) == 2
    assert config.validators[0].pattern == "*.yaml"
    assert config.validators[0].validator == "pkg.mod.yaml_validator"
    assert config.validators[1].pattern == "config/**"
