"""tests for ModelType, ModelStatus, and ModelTier enums."""

from __future__ import annotations

from enum import StrEnum

from threetears.models.enums import ModelStatus, ModelTier, ModelType


class TestModelType:
    """tests for ModelType enum."""

    def test_model_type_is_str_enum(self) -> None:
        """ModelType inherits from StrEnum."""
        assert issubclass(ModelType, StrEnum)

    def test_model_type_values(self) -> None:
        """ModelType contains all expected members with correct values."""
        assert ModelType.CHAT == "chat"
        assert ModelType.EMBEDDING == "embedding"
        assert ModelType.TRANSCRIPTION == "transcription"
        assert ModelType.IMAGE_GENERATION == "image_generation"
        assert ModelType.SPEECH == "speech"
        assert ModelType.RERANKING == "reranking"

    def test_model_type_member_count(self) -> None:
        """ModelType has exactly six members."""
        assert len(ModelType) == 6

    def test_model_type_string_comparison(self) -> None:
        """ModelType members compare equal to their string values."""
        assert ModelType.CHAT == "chat"
        assert "embedding" == ModelType.EMBEDDING

    def test_model_type_is_string_instance(self) -> None:
        """ModelType members are string instances."""
        assert isinstance(ModelType.CHAT, str)
        assert isinstance(ModelType.IMAGE_GENERATION, str)
        assert isinstance(ModelType.SPEECH, str)
        assert isinstance(ModelType.RERANKING, str)


class TestModelStatus:
    """tests for ModelStatus enum."""

    def test_model_status_is_str_enum(self) -> None:
        """ModelStatus inherits from StrEnum."""
        assert issubclass(ModelStatus, StrEnum)

    def test_model_status_values(self) -> None:
        """ModelStatus contains all expected members with correct values."""
        assert ModelStatus.ACTIVE == "active"
        assert ModelStatus.DEPRECATED == "deprecated"
        assert ModelStatus.DISABLED == "disabled"

    def test_model_status_member_count(self) -> None:
        """ModelStatus has exactly three members."""
        assert len(ModelStatus) == 3

    def test_model_status_string_comparison(self) -> None:
        """ModelStatus members compare equal to their string values."""
        assert ModelStatus.ACTIVE == "active"
        assert "deprecated" == ModelStatus.DEPRECATED

    def test_model_status_is_string_instance(self) -> None:
        """ModelStatus members are string instances."""
        assert isinstance(ModelStatus.ACTIVE, str)
        assert isinstance(ModelStatus.DISABLED, str)


class TestModelTier:
    """tests for ModelTier enum."""

    def test_model_tier_is_str_enum(self) -> None:
        """ModelTier inherits from StrEnum."""
        assert issubclass(ModelTier, StrEnum)

    def test_model_tier_values(self) -> None:
        """ModelTier contains all expected members with correct values."""
        assert ModelTier.SMALL == "small"
        assert ModelTier.MEDIUM == "medium"
        assert ModelTier.LARGE == "large"

    def test_model_tier_member_count(self) -> None:
        """ModelTier has exactly three members."""
        assert len(ModelTier) == 3

    def test_model_tier_string_comparison(self) -> None:
        """ModelTier members compare equal to their string values."""
        assert ModelTier.SMALL == "small"
        assert "large" == ModelTier.LARGE

    def test_model_tier_is_string_instance(self) -> None:
        """ModelTier members are string instances."""
        assert isinstance(ModelTier.SMALL, str)
        assert isinstance(ModelTier.LARGE, str)
