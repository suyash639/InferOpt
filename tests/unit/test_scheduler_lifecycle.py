"""Unit tests for request lifecycle state tracking and timing metrics."""

import pytest

from inferopt.scheduler.lifecycle import RequestRecord, RequestStatus


class TestRequestStatus:
    """Test suite for RequestStatus enum."""

    def test_status_values(self) -> None:
        """Verify all lifecycle state string values."""
        assert RequestStatus.QUEUED == "QUEUED"
        assert RequestStatus.RUNNING == "RUNNING"
        assert RequestStatus.COMPLETED == "COMPLETED"
        assert RequestStatus.FAILED == "FAILED"
        assert RequestStatus.CANCELLED == "CANCELLED"

    def test_is_terminal_property(self) -> None:
        """Verify terminal state detection."""
        assert not RequestStatus.QUEUED.is_terminal
        assert not RequestStatus.RUNNING.is_terminal
        assert RequestStatus.COMPLETED.is_terminal
        assert RequestStatus.FAILED.is_terminal
        assert RequestStatus.CANCELLED.is_terminal


class TestRequestRecord:
    """Test suite for RequestRecord tracking and metrics."""

    def test_default_record_creation(self) -> None:
        """Verify initial record state."""
        record = RequestRecord(request_id="req-1", queued_at=100.0)
        assert record.request_id == "req-1"
        assert record.status == RequestStatus.QUEUED
        assert record.queued_at == 100.0
        assert record.started_at is None
        assert record.completed_at is None
        assert record.error_message is None
        assert record.queue_wait_ms is None
        assert record.execution_ms is None
        assert record.total_latency_ms is None

    def test_happy_path_transitions_and_timing(self) -> None:
        """Verify QUEUED -> RUNNING -> COMPLETED transitions and calculated durations."""
        record = RequestRecord(request_id="req-1", queued_at=100.0)

        # Transition to RUNNING at t=100.050 (50ms queue wait)
        record.mark_running(started_at=100.050)
        assert record.status.value == RequestStatus.RUNNING.value
        assert record.started_at == 100.050
        assert record.queue_wait_ms == pytest.approx(50.0)
        assert record.execution_ms is None
        assert record.total_latency_ms is None

        # Transition to COMPLETED at t=100.150 (100ms execution, 150ms total)
        record.mark_completed(completed_at=100.150)
        assert record.status.value == RequestStatus.COMPLETED.value

        assert record.completed_at == 100.150
        assert record.queue_wait_ms == pytest.approx(50.0)
        assert record.execution_ms == pytest.approx(100.0)
        assert record.total_latency_ms == pytest.approx(150.0)

    def test_failure_transition(self) -> None:
        """Verify failure transition from RUNNING to FAILED."""
        record = RequestRecord(request_id="req-2", queued_at=200.0)
        record.mark_running(started_at=200.010)
        record.mark_failed(completed_at=200.030, error_message="Engine timeout")

        assert record.status == RequestStatus.FAILED
        assert record.error_message == "Engine timeout"
        assert record.queue_wait_ms == pytest.approx(10.0)
        assert record.execution_ms == pytest.approx(20.0)
        assert record.total_latency_ms == pytest.approx(30.0)

    def test_cancellation_transition_from_queued(self) -> None:
        """Verify cancellation directly from QUEUED state."""
        record = RequestRecord(request_id="req-3", queued_at=300.0)
        record.mark_cancelled(cancelled_at=300.005, error_message="Client disconnected")

        assert record.status == RequestStatus.CANCELLED
        assert record.error_message == "Client disconnected"
        assert record.queue_wait_ms is None
        assert record.execution_ms is None
        assert record.total_latency_ms == pytest.approx(5.0)

    def test_invalid_transitions_rejected(self) -> None:
        """Verify invalid state transitions raise ValueError."""
        record = RequestRecord(request_id="req-4", queued_at=100.0)
        record.mark_running(started_at=100.010)
        record.mark_completed(completed_at=100.020)

        # Cannot transition terminal COMPLETED record to RUNNING, COMPLETED, FAILED, or CANCELLED
        with pytest.raises(ValueError):
            record.mark_running(started_at=100.030)

        with pytest.raises(ValueError):
            record.mark_completed(completed_at=100.030)

        with pytest.raises(ValueError):
            record.mark_failed(completed_at=100.030, error_message="late failure")

        with pytest.raises(ValueError):
            record.mark_cancelled(cancelled_at=100.030)
