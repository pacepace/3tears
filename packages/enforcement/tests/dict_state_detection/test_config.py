"""tests for ``DictStateConfig`` and ``DictStateAllowlistEntry``."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from threetears.enforcement.dict_state_detection import (
    AllowlistRationaleError,
    DictStateAllowlistEntry,
    DictStateConfig,
)


_VALID_RATIONALE = (
    "live LangChain ChatModel instances; non-serializable, "
    "process-local by design"
)


class TestDictStateConfigDefaults:
    def test_minimal_construction(self, tmp_path: Path) -> None:
        config = DictStateConfig(repo_root=tmp_path)
        assert config.repo_root == tmp_path
        assert config.src_roots is None
        assert config.exemptions_path is None
        assert config.mode_env_var == "DICT_STATE_ENFORCEMENT_MODE"
        assert config.allowlist == ()
        assert config.known_violations == ()

    def test_is_frozen(self, tmp_path: Path) -> None:
        config = DictStateConfig(repo_root=tmp_path)
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.repo_root = tmp_path / "other"  # type: ignore[misc]


class TestDictStateConfigOverrides:
    def test_explicit_src_roots(self, tmp_path: Path) -> None:
        roots = (tmp_path / "a", tmp_path / "b")
        config = DictStateConfig(repo_root=tmp_path, src_roots=roots)
        assert config.src_roots == roots

    def test_explicit_exemptions_path(self, tmp_path: Path) -> None:
        ex = tmp_path / "_dict_state_exemptions.txt"
        config = DictStateConfig(
            repo_root=tmp_path, exemptions_path=ex,
        )
        assert config.exemptions_path == ex

    def test_explicit_mode_env_var(self, tmp_path: Path) -> None:
        config = DictStateConfig(
            repo_root=tmp_path, mode_env_var="MY_MODE",
        )
        assert config.mode_env_var == "MY_MODE"

    def test_explicit_allowlist(self, tmp_path: Path) -> None:
        entry = DictStateAllowlistEntry(
            file="src/x.py",
            line=10,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        config = DictStateConfig(
            repo_root=tmp_path, allowlist=(entry,),
        )
        assert config.allowlist == (entry,)

    def test_explicit_known_violations(self, tmp_path: Path) -> None:
        entry = DictStateAllowlistEntry(
            file="src/x.py",
            line=10,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        config = DictStateConfig(
            repo_root=tmp_path, known_violations=(entry,),
        )
        assert config.known_violations == (entry,)

    def test_full_roundtrip(self, tmp_path: Path) -> None:
        roots = (tmp_path / "src",)
        ex = tmp_path / "ex.txt"
        allow = (
            DictStateAllowlistEntry(
                file="src/a.py",
                line=5,
                attr_name="_a",
                rationale=_VALID_RATIONALE,
            ),
        )
        known = (
            DictStateAllowlistEntry(
                file="src/b.py",
                line=7,
                attr_name="_b",
                rationale=_VALID_RATIONALE,
            ),
        )
        config = DictStateConfig(
            repo_root=tmp_path,
            src_roots=roots,
            exemptions_path=ex,
            mode_env_var="DICT_TEST_MODE",
            allowlist=allow,
            known_violations=known,
        )
        assert config.repo_root == tmp_path
        assert config.src_roots == roots
        assert config.exemptions_path == ex
        assert config.mode_env_var == "DICT_TEST_MODE"
        assert config.allowlist == allow
        assert config.known_violations == known


class TestAllowlistRationaleValidation:
    def test_sufficient_rationale_accepted(self) -> None:
        DictStateAllowlistEntry(
            file="src/x.py",
            line=10,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )

    def test_empty_rationale_rejected(self) -> None:
        with pytest.raises(AllowlistRationaleError, match="non-empty"):
            DictStateAllowlistEntry(
                file="src/x.py",
                line=10,
                attr_name="_cache",
                rationale="",
            )

    def test_whitespace_only_rationale_rejected(self) -> None:
        with pytest.raises(AllowlistRationaleError, match="non-empty"):
            DictStateAllowlistEntry(
                file="src/x.py",
                line=10,
                attr_name="_cache",
                rationale="    ",
            )

    def test_short_rationale_rejected(self) -> None:
        with pytest.raises(AllowlistRationaleError, match="at least"):
            DictStateAllowlistEntry(
                file="src/x.py",
                line=10,
                attr_name="_cache",
                rationale="too short",
            )

    def test_blanket_rationale_rejected(self) -> None:
        with pytest.raises(AllowlistRationaleError, match="blanket"):
            DictStateAllowlistEntry(
                file="src/x.py",
                line=10,
                attr_name="_cache",
                rationale="tests need this for the integration suite",
            )

    def test_blanket_rationale_with_punctuation_rejected(self) -> None:
        # rationale starts with blanket phrase + punctuation.
        with pytest.raises(AllowlistRationaleError, match="blanket"):
            DictStateAllowlistEntry(
                file="src/x.py",
                line=10,
                attr_name="_cache",
                rationale="temporary, until the next refactor lands",
            )

    def test_entry_is_frozen(self) -> None:
        entry = DictStateAllowlistEntry(
            file="src/x.py",
            line=10,
            attr_name="_cache",
            rationale=_VALID_RATIONALE,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.line = 20  # type: ignore[misc]
