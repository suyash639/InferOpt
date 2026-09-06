"""Unit tests verifying telemetry integration in Scheduler."""

import asyncio
from unittest.mock import MagicMock

import pytest

from inferopt.backends.mock import MockBackend
from inferopt.core.exceptions import InferenceError
from inferopt.core.models import InferenceRequest, InferenceResponse
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.scheduler.lifecycle import RequestStatus
from inferopt.scheduler.scheduler import Scheduler
from inferopt.telemetry.collector import MetricsCollector


class NonBatchBackend:
    """Backend implementing only the standard InferenceBackend protocol."""

    def __init__(self, backend_name: str = "non-batch-mock") -> None:
        self.backend_name = backend_name

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            generated_text=f"Response to {request.prompt}",
            input_tokens=5,
            output_tokens=10,
            latency_ms=1.0,
            backend_name=self.backend_name,
        )


class FailingBackend:
    """Backend that intentionally fails request execution."""

    def __init__(self) -> None:
        self.backend_name = "failing-backend"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        raise InferenceError("Synthetic backend hardware failure")


@pytest.mark.asyncio
class TestSchedulerTelemetry:
    """Tests for scheduler telemetry capture, lifecycle tracking, and fault isolation."""

    async def test_single_request_telemetry(self) -> None:
        backend = MockBackend(default_latency_sec=0.001)
        collector = MetricsCollector()
        config = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=1, batch_wait_ms=0.0),
        )

        async with Scheduler(backend=backend, config=config, collector=collector) as scheduler:
            req = InferenceRequest(model="test-model", prompt="Hello telemetry", max_tokens=32)
            resp = await scheduler.submit(req)
            assert resp.request_id == req.request_id

        snapshot = collector.snapshot()
        assert snapshot.requests.total_requests == 1
        assert snapshot.requests.completed_requests == 1
        assert snapshot.requests.failed_requests == 0
        assert snapshot.batches.total_batches == 1
        assert snapshot.batches.completed_batches == 1

        recent_reqs = collector.get_recent_requests()
        assert len(recent_reqs) == 1
        r_m = recent_reqs[0]
        assert r_m.request_id == req.request_id
        assert r_m.status == RequestStatus.COMPLETED
        assert r_m.backend_name == backend.backend_name
        assert r_m.input_tokens == resp.input_tokens
        assert r_m.output_tokens == resp.output_tokens
        assert r_m.batch_id is not None
        # Timing invariants
        assert r_m.total_latency_ms >= r_m.queue_wait_ms
        assert r_m.total_latency_ms >= r_m.execution_ms

    async def test_batched_requests_telemetry(self) -> None:
        backend = MockBackend(default_latency_sec=0.001)
        collector = MetricsCollector()
        config = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=3, batch_wait_ms=50.0),
        )

        async with Scheduler(backend=backend, config=config, collector=collector) as scheduler:
            reqs = [
                InferenceRequest(model="m", prompt=f"Batch prompt {i}", max_tokens=16)
                for i in range(3)
            ]
            responses = await asyncio.gather(*[scheduler.submit(r) for r in reqs])
            assert len(responses) == 3

        snapshot = collector.snapshot()
        assert snapshot.requests.total_requests == 3
        assert snapshot.requests.completed_requests == 3
        assert snapshot.batches.total_batches == 1
        assert snapshot.batches.avg_batch_size == 3.0

        batch_m = collector.get_recent_batches()[0]
        assert batch_m.size == 3
        assert batch_m.completed_request_count == 3
        assert batch_m.failed_request_count == 0
        assert set(batch_m.request_ids) == {r.request_id for r in reqs}

    async def test_backend_failure_telemetry(self) -> None:
        backend = FailingBackend()
        collector = MetricsCollector()
        config = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=1, batch_wait_ms=0.0),
        )

        async with Scheduler(backend=backend, config=config, collector=collector) as scheduler:
            req = InferenceRequest(model="m", prompt="Fail me")
            with pytest.raises(InferenceError):
                await scheduler.submit(req)

        snapshot = collector.snapshot()
        assert snapshot.requests.total_requests == 1
        assert snapshot.requests.completed_requests == 0
        assert snapshot.requests.failed_requests == 1
        assert snapshot.batches.failed_batches == 1

        r_m = collector.get_recent_requests()[0]
        assert r_m.status == RequestStatus.FAILED
        assert "Synthetic backend hardware failure" in (r_m.error_message or "")

    async def test_telemetry_fault_isolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that telemetry exceptions NEVER disrupt inference execution."""
        backend = MockBackend(default_latency_sec=0.001)
        collector = MetricsCollector()

        # Monkeypatch record_request to raise an error
        mock_record_req = MagicMock(side_effect=RuntimeError("Telemetry store unavailable"))
        mock_record_batch = MagicMock(side_effect=RuntimeError("Telemetry store unavailable"))

        monkeypatch.setattr(collector, "record_request", mock_record_req)
        monkeypatch.setattr(collector, "record_batch", mock_record_batch)

        config = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=1, batch_wait_ms=0.0),
        )

        async with Scheduler(backend=backend, config=config, collector=collector) as scheduler:
            req = InferenceRequest(model="m", prompt="Should succeed despite telemetry crash")
            resp = await scheduler.submit(req)
            assert resp.request_id == req.request_id

    async def test_cancellation_during_queue_wait_telemetry(self) -> None:
        backend = MockBackend(default_latency_sec=0.1)
        collector = MetricsCollector()
        # Single worker busy
        config = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=1, batch_wait_ms=0.0),
        )

        async with Scheduler(backend=backend, config=config, collector=collector) as scheduler:
            # First request occupies worker
            task1 = asyncio.create_task(
                scheduler.submit(InferenceRequest(model="m", prompt="Occupying worker"))
            )
            await asyncio.sleep(0.01)

            # Second request sits in queue
            task2 = asyncio.create_task(
                scheduler.submit(InferenceRequest(model="m", prompt="Queued request"))
            )
            await asyncio.sleep(0.01)

            task2.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task2

            await task1

        snapshot = collector.snapshot()
        assert snapshot.requests.total_requests == 2
        assert snapshot.requests.completed_requests == 1
        assert snapshot.requests.cancelled_requests == 1

    async def test_non_batch_backend_telemetry(self) -> None:
        backend = NonBatchBackend()
        collector = MetricsCollector()
        config = SchedulerConfig(
            max_concurrency=2,
            batch_config=BatchConfig(max_batch_size=2, batch_wait_ms=0.0),
        )

        async with Scheduler(backend=backend, config=config, collector=collector) as scheduler:
            req1 = InferenceRequest(model="m", prompt="Non-batch 1")
            req2 = InferenceRequest(model="m", prompt="Non-batch 2")
            resps = await asyncio.gather(scheduler.submit(req1), scheduler.submit(req2))
            assert len(resps) == 2

        snapshot = collector.snapshot()
        assert snapshot.requests.total_requests == 2
        assert snapshot.requests.completed_requests == 2
        assert snapshot.batches.total_batches == 1
        assert snapshot.batches.avg_batch_size == 2.0
