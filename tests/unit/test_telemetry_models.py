"""Unit tests for telemetry domain models, metric records, and snapshot structures."""

import pytest
from pydantic import ValidationError

from inferopt.scheduler.lifecycle import RequestStatus
from inferopt.telemetry.models import (
    BatchMetrics,
    BatchStats,
    MetricsSnapshot,
    QueueStats,
    RequestMetrics,
    RequestStats,
    ThroughputStats,
)


class TestRequestMetrics:
    """Tests for RequestMetrics validation, constraints, and immutability."""

    def test_valid_request_metrics(self) -> None:
        metrics = RequestMetrics(
            request_id="req-123",
            priority=10,
            queue_wait_ms=15.5,
            execution_ms=50.0,
            total_latency_ms=65.5,
            status=RequestStatus.COMPLETED,
            input_tokens=12,
            output_tokens=34,
            max_tokens=64,
            backend_name="mock-engine",
            batch_id="batch-001",
        )
        assert metrics.request_id == "req-123"
        assert metrics.priority == 10
        assert metrics.queue_wait_ms == 15.5
        assert metrics.execution_ms == 50.0
        assert metrics.total_latency_ms == 65.5
        assert metrics.status == RequestStatus.COMPLETED
        assert metrics.input_tokens == 12
        assert metrics.output_tokens == 34
        assert metrics.max_tokens == 64
        assert metrics.backend_name == "mock-engine"
        assert metrics.batch_id == "batch-001"
        assert metrics.error_message is None
        assert metrics.recorded_at > 0

    def test_immutability(self) -> None:
        metrics = RequestMetrics(
            request_id="req-123",
            priority=0,
            queue_wait_ms=0.0,
            execution_ms=10.0,
            total_latency_ms=10.0,
            status=RequestStatus.COMPLETED,
            max_tokens=32,
            backend_name="mock",
        )
        with pytest.raises(ValidationError):
            metrics.execution_ms = 20.0

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            RequestMetrics.model_validate(
                {
                    "request_id": "req-123",
                    "priority": 0,
                    "queue_wait_ms": 0.0,
                    "execution_ms": 10.0,
                    "total_latency_ms": 10.0,
                    "status": RequestStatus.COMPLETED,
                    "max_tokens": 32,
                    "backend_name": "mock",
                    "unsupported_extra_field": 123,
                }
            )

    def test_negative_timings_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RequestMetrics(
                request_id="req-123",
                priority=0,
                queue_wait_ms=-1.0,
                execution_ms=10.0,
                total_latency_ms=9.0,
                status=RequestStatus.COMPLETED,
                max_tokens=32,
                backend_name="mock",
            )
        with pytest.raises(ValidationError):
            RequestMetrics(
                request_id="req-123",
                priority=0,
                queue_wait_ms=0.0,
                execution_ms=-5.0,
                total_latency_ms=0.0,
                status=RequestStatus.COMPLETED,
                max_tokens=32,
                backend_name="mock",
            )

    def test_invalid_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RequestMetrics(
                request_id="req-123",
                priority=0,
                queue_wait_ms=0.0,
                execution_ms=10.0,
                total_latency_ms=10.0,
                status=RequestStatus.COMPLETED,
                input_tokens=-1,
                max_tokens=32,
                backend_name="mock",
            )
        with pytest.raises(ValidationError):
            RequestMetrics(
                request_id="req-123",
                priority=0,
                queue_wait_ms=0.0,
                execution_ms=10.0,
                total_latency_ms=10.0,
                status=RequestStatus.COMPLETED,
                max_tokens=0,
                backend_name="mock",
            )


class TestBatchMetrics:
    """Tests for BatchMetrics validation, constraints, and immutability."""

    def test_valid_batch_metrics(self) -> None:
        batch_m = BatchMetrics(
            batch_id="b-999",
            size=3,
            batch_formation_wait_ms=12.0,
            execution_ms=45.0,
            total_max_tokens=256,
            backend_name="mock-batch",
            request_ids=("req-1", "req-2", "req-3"),
            completed_request_count=3,
            failed_request_count=0,
        )
        assert batch_m.batch_id == "b-999"
        assert batch_m.size == 3
        assert batch_m.batch_formation_wait_ms == 12.0
        assert batch_m.execution_ms == 45.0
        assert batch_m.total_max_tokens == 256
        assert batch_m.backend_name == "mock-batch"
        assert len(batch_m.request_ids) == 3
        assert batch_m.completed_request_count == 3
        assert batch_m.failed_request_count == 0

    def test_immutability(self) -> None:
        batch_m = BatchMetrics(
            batch_id="b-999",
            size=1,
            batch_formation_wait_ms=0.0,
            execution_ms=10.0,
            total_max_tokens=32,
            backend_name="mock",
            request_ids=("req-1",),
            completed_request_count=1,
            failed_request_count=0,
        )
        with pytest.raises(ValidationError):
            batch_m.size = 5

    def test_invalid_size_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BatchMetrics(
                batch_id="b-999",
                size=0,
                batch_formation_wait_ms=0.0,
                execution_ms=10.0,
                total_max_tokens=32,
                backend_name="mock",
                request_ids=(),
                completed_request_count=0,
                failed_request_count=0,
            )

    def test_negative_formation_wait_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BatchMetrics(
                batch_id="b-999",
                size=1,
                batch_formation_wait_ms=-2.0,
                execution_ms=10.0,
                total_max_tokens=32,
                backend_name="mock",
                request_ids=("req-1",),
                completed_request_count=1,
                failed_request_count=0,
            )


class TestSnapshotModels:
    """Tests for aggregate stats and snapshot serialization."""

    def test_default_snapshot(self) -> None:
        snapshot = MetricsSnapshot()
        assert isinstance(snapshot.requests, RequestStats)
        assert isinstance(snapshot.batches, BatchStats)
        assert isinstance(snapshot.throughput, ThroughputStats)
        assert isinstance(snapshot.queue, QueueStats)
        assert snapshot.requests.total_requests == 0
        assert snapshot.requests.avg_total_latency_ms == 0.0
        assert snapshot.batches.total_batches == 0
        assert snapshot.throughput.requests_per_sec == 0.0
        assert snapshot.queue.current_queue_depth == 0

    def test_snapshot_immutability(self) -> None:
        snapshot = MetricsSnapshot()
        with pytest.raises(ValidationError):
            snapshot.requests = RequestStats()

    def test_snapshot_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            MetricsSnapshot.model_validate({"unknown_field": 123})
