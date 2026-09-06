"""Unit tests for SchedulerConfig model."""

from typing import Any

import pytest
from pydantic import ValidationError

from inferopt.scheduler.config import SchedulerConfig


class TestSchedulerConfig:
    """Test suite for SchedulerConfig validation and immutability."""

    def test_default_configuration(self) -> None:
        """Verify default configuration parameters."""
        config = SchedulerConfig()
        assert config.max_concurrency == 4
        assert config.max_queue_size == 100
        assert config.max_history_size == 1000

    def test_custom_valid_configuration(self) -> None:
        """Verify explicit valid configuration parameters."""
        config = SchedulerConfig(
            max_concurrency=8,
            max_queue_size=250,
            max_history_size=5000,
        )
        assert config.max_concurrency == 8
        assert config.max_queue_size == 250
        assert config.max_history_size == 5000

    def test_invalid_max_concurrency_zero(self) -> None:
        """Verify max_concurrency = 0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SchedulerConfig(max_concurrency=0)
        assert "max_concurrency" in str(exc_info.value)

    def test_invalid_max_concurrency_negative(self) -> None:
        """Verify negative max_concurrency is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SchedulerConfig(max_concurrency=-1)
        assert "max_concurrency" in str(exc_info.value)

    def test_valid_max_queue_size_zero(self) -> None:
        """Verify max_queue_size = 0 is permitted (unbuffered queue)."""
        config = SchedulerConfig(max_queue_size=0)
        assert config.max_queue_size == 0

    def test_invalid_max_queue_size_negative(self) -> None:
        """Verify negative max_queue_size is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SchedulerConfig(max_queue_size=-5)
        assert "max_queue_size" in str(exc_info.value)

    def test_invalid_max_history_size_negative(self) -> None:
        """Verify negative max_history_size is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SchedulerConfig(max_history_size=-1)
        assert "max_history_size" in str(exc_info.value)

    def test_config_immutability(self) -> None:
        """Verify SchedulerConfig is frozen and immutable."""
        config = SchedulerConfig()
        config_any: Any = config
        with pytest.raises(ValidationError):
            config_any.max_concurrency = 16

    def test_extra_fields_forbidden(self) -> None:
        """Verify unexpected config fields are rejected."""
        with pytest.raises(ValidationError):
            SchedulerConfig(unsupported_param=123)  # type: ignore[call-arg]
