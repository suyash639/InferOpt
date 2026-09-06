"""Unit tests for InferenceBatch domain model and BatchConfig."""

from typing import Any

import pytest
from pydantic import ValidationError

from inferopt.core.models import InferenceBatch, InferenceRequest
from inferopt.scheduler.config import BatchConfig, SchedulerConfig


class TestInferenceBatch:
    """Test suite for InferenceBatch domain model."""

    def test_valid_batch_creation(self) -> None:
        """Verify valid batch creation with multiple requests."""
        req1 = InferenceRequest(request_id="req-1", model="m1", prompt="Prompt 1", max_tokens=64)
        req2 = InferenceRequest(request_id="req-2", model="m1", prompt="Prompt 2", max_tokens=128)

        batch = InferenceBatch(requests=(req1, req2))

        assert len(batch.batch_id) > 0
        assert batch.size == 2
        assert batch.request_ids == ["req-1", "req-2"]
        assert batch.total_max_tokens == 192
        assert batch.created_at > 0

    def test_single_request_batch(self) -> None:
        """Verify valid batch creation with a single request."""
        req = InferenceRequest(request_id="req-single", model="m1", prompt="Prompt")
        batch = InferenceBatch(requests=(req,))

        assert batch.size == 1
        assert batch.request_ids == ["req-single"]
        assert batch.total_max_tokens == req.max_tokens

    def test_empty_requests_rejected(self) -> None:
        """Verify empty requests sequence raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InferenceBatch(requests=())
        assert "InferenceBatch cannot be empty" in str(exc_info.value)

    def test_batch_immutability(self) -> None:
        """Verify InferenceBatch is frozen and immutable."""
        req = InferenceRequest(request_id="req-1", model="m1", prompt="Prompt")
        batch = InferenceBatch(requests=(req,))
        batch_any: Any = batch

        with pytest.raises(ValidationError):
            batch_any.requests = ()

    def test_extra_fields_forbidden(self) -> None:
        """Verify extra attributes are forbidden on InferenceBatch."""
        req = InferenceRequest(request_id="req-1", model="m1", prompt="Prompt")
        with pytest.raises(ValidationError):
            InferenceBatch(requests=(req,), unexpected_param=123)  # type: ignore[call-arg]


class TestBatchConfig:
    """Test suite for BatchConfig validation and defaults."""

    def test_default_batch_config(self) -> None:
        """Verify default batch configuration values."""
        cfg = BatchConfig()
        assert cfg.max_batch_size == 4
        assert cfg.batch_wait_ms == 0.0

    def test_custom_valid_batch_config(self) -> None:
        """Verify explicit valid batch configuration values."""
        cfg = BatchConfig(max_batch_size=8, batch_wait_ms=25.0)
        assert cfg.max_batch_size == 8
        assert cfg.batch_wait_ms == 25.0

    def test_invalid_max_batch_size_zero_or_negative(self) -> None:
        """Verify max_batch_size <= 0 is rejected."""
        with pytest.raises(ValidationError):
            BatchConfig(max_batch_size=0)
        with pytest.raises(ValidationError):
            BatchConfig(max_batch_size=-2)

    def test_invalid_batch_wait_ms_negative(self) -> None:
        """Verify negative batch_wait_ms is rejected."""
        with pytest.raises(ValidationError):
            BatchConfig(batch_wait_ms=-1.0)

    def test_batch_config_immutability(self) -> None:
        """Verify BatchConfig is frozen and immutable."""
        cfg = BatchConfig()
        cfg_any: Any = cfg
        with pytest.raises(ValidationError):
            cfg_any.max_batch_size = 16

    def test_scheduler_config_integration(self) -> None:
        """Verify SchedulerConfig embeds BatchConfig correctly."""
        sched_cfg = SchedulerConfig(
            max_concurrency=2,
            batch_config=BatchConfig(max_batch_size=6, batch_wait_ms=15.0),
        )
        assert sched_cfg.max_concurrency == 2
        assert sched_cfg.batch_config.max_batch_size == 6
        assert sched_cfg.batch_config.batch_wait_ms == 15.0
