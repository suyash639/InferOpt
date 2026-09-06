"""Comprehensive deterministic unit tests for InferOpt batching subsystem."""

import asyncio
import time
from typing import Any

import pytest

from inferopt.backends.base import BatchInferenceBackend, InferenceBackend
from inferopt.core.exceptions import (
    InferenceError,
    QueueFullError,
)
from inferopt.core.models import InferenceBatch, InferenceRequest, InferenceResponse
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.scheduler.lifecycle import RequestStatus
from inferopt.scheduler.scheduler import Scheduler


class BatchTrackingBackend(BatchInferenceBackend, InferenceBackend):
    """Instrumented backend tracking formed batches and controlling execution timing."""

    def __init__(self, backend_name: str = "batch-tracking-test") -> None:
        self._name = backend_name
        self.formed_batches: list[InferenceBatch] = []
        self.dispatched_request_ids: list[str] = []
        self.completed_request_ids: list[str] = []
        self.active_batch_concurrency: int = 0
        self.max_observed_batch_concurrency: int = 0
        self.failing_request_ids: set[str] = set()
        self.failing_batch_ids: set[str] = set()

        self.pause_event: asyncio.Event = asyncio.Event()
        self.pause_event.set()
        self.batch_started_event: asyncio.Event = asyncio.Event()

    @property
    def backend_name(self) -> str:
        return self._name

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.dispatched_request_ids.append(request.request_id)
        if request.request_id in self.failing_request_ids:
            raise InferenceError(f"Simulated failure for {request.request_id}")
        self.completed_request_ids.append(request.request_id)
        return InferenceResponse(
            request_id=request.request_id,
            generated_text=f"Response for {request.request_id}",
            input_tokens=10,
            output_tokens=20,
            latency_ms=5.0,
            backend_name=self.backend_name,
        )

    async def generate_batch(self, batch: InferenceBatch) -> list[InferenceResponse]:
        self.formed_batches.append(batch)
        self.dispatched_request_ids.extend(batch.request_ids)
        self.active_batch_concurrency += 1
        self.max_observed_batch_concurrency = max(
            self.max_observed_batch_concurrency, self.active_batch_concurrency
        )
        self.batch_started_event.set()

        try:
            await self.pause_event.wait()

            if batch.batch_id in self.failing_batch_ids:
                raise InferenceError(f"Simulated batch failure for {batch.batch_id}")

            responses: list[InferenceResponse] = []
            for req in batch.requests:
                if req.request_id in self.failing_request_ids:
                    raise InferenceError(f"Simulated request failure for {req.request_id}")
                self.completed_request_ids.append(req.request_id)
                responses.append(
                    InferenceResponse(
                        request_id=req.request_id,
                        generated_text=f"Batch response for {req.request_id}",
                        input_tokens=10,
                        output_tokens=20,
                        latency_ms=5.0,
                        backend_name=self.backend_name,
                    )
                )
            return responses
        finally:
            self.active_batch_concurrency -= 1


class IndividualBackend(InferenceBackend):
    """Backend implementing only InferenceBackend (non-batch-native)."""

    def __init__(self, backend_name: str = "individual-test") -> None:
        self._name = backend_name
        self.dispatched_requests: list[str] = []
        self.failing_request_ids: set[str] = set()

    @property
    def backend_name(self) -> str:
        return self._name

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.dispatched_requests.append(request.request_id)
        if request.request_id in self.failing_request_ids:
            raise InferenceError(f"Individual failure for {request.request_id}")
        return InferenceResponse(
            request_id=request.request_id,
            generated_text=f"Individual response for {request.request_id}",
            input_tokens=8,
            output_tokens=16,
            latency_ms=4.0,
            backend_name=self.backend_name,
        )


def make_request(
    request_id: str,
    priority: int = 0,
    prompt: str = "Test prompt",
    metadata: dict[str, Any] | None = None,
) -> InferenceRequest:
    """Helper factory for test requests."""
    return InferenceRequest(
        request_id=request_id,
        model="test-model",
        prompt=prompt,
        priority=priority,
        metadata=metadata or {},
    )


class TestBatchFormation:
    """Test suite for deterministic batch formation and wait window dynamics."""

    async def test_single_request_batch(self) -> None:
        """Verify a single request forms an InferenceBatch of size 1."""
        backend = BatchTrackingBackend()
        config = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=0.0),
        )
        scheduler = Scheduler(backend=backend, config=config)

        async with scheduler:
            req = make_request("req-1")
            resp = await scheduler.submit(req)

            assert resp.request_id == "req-1"
            assert len(backend.formed_batches) == 1
            assert backend.formed_batches[0].size == 1
            assert backend.formed_batches[0].request_ids == ["req-1"]
            assert scheduler.get_status("req-1") == RequestStatus.COMPLETED

    async def test_multiple_requests_form_full_batch(self) -> None:
        """Verify multiple queued requests fill a batch up to max_batch_size."""
        backend = BatchTrackingBackend()
        backend.pause_event.clear()

        config = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=3, batch_wait_ms=0.0),
        )
        scheduler = Scheduler(backend=backend, config=config)

        async with scheduler:
            tasks = [
                asyncio.create_task(scheduler.submit(make_request(f"b-{i}"))) for i in range(3)
            ]
            await asyncio.sleep(0)
            assert scheduler.queued_count == 3

            backend.pause_event.set()
            responses = await asyncio.gather(*tasks)

            assert len(responses) == 3
            assert len(backend.formed_batches) == 1
            assert backend.formed_batches[0].size == 3
            assert backend.formed_batches[0].request_ids == ["b-0", "b-1", "b-2"]

    async def test_batch_fills_before_timeout_dispatches_immediately(self) -> None:
        """Verify batch dispatches immediately when full without waiting for window expiry."""
        backend = BatchTrackingBackend()
        # Large wait window: 2000ms. If it waited for timeout, test would take >2s.
        config = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=3, batch_wait_ms=2000.0),
        )
        scheduler = Scheduler(backend=backend, config=config)

        async with scheduler:
            start_time = time.perf_counter()
            tasks = [
                asyncio.create_task(scheduler.submit(make_request(f"quick-{i}"))) for i in range(3)
            ]
            await asyncio.sleep(0)

            responses = await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - start_time

            # Should complete almost immediately (< 0.5s), not waiting 2.0s
            assert elapsed < 0.5
            assert len(responses) == 3
            assert len(backend.formed_batches) == 1
            assert backend.formed_batches[0].size == 3

    async def test_batch_wait_window_partial_batch_dispatch(self) -> None:
        """Verify partial batch is dispatched when wait window expires."""
        backend = BatchTrackingBackend()
        config = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=30.0),
        )
        scheduler = Scheduler(backend=backend, config=config)

        async with scheduler:
            # Submit only 2 requests (less than max_batch_size 4)
            tasks = [
                asyncio.create_task(scheduler.submit(make_request("part-0"))),
                asyncio.create_task(scheduler.submit(make_request("part-1"))),
            ]
            await asyncio.sleep(0)

            responses = await asyncio.gather(*tasks)

            assert len(responses) == 2
            assert len(backend.formed_batches) == 1
            assert backend.formed_batches[0].size == 2
            assert backend.formed_batches[0].request_ids == ["part-0", "part-1"]

    async def test_priority_and_fifo_order_across_batches(self) -> None:
        """Verify batches preserve priority ordering and FIFO tie-breaking.

        Requests:
        A (p=1), B (p=5), C (p=5), D (p=2), E (p=10), F (p=2)
        With max_batch_size = 3:
        Batch 1: [E(10), B(5), C(5)]
        Batch 2: [D(2), F(2), A(1)]
        """
        backend = BatchTrackingBackend()
        backend.pause_event.clear()

        config = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=3, batch_wait_ms=0.0),
        )
        scheduler = Scheduler(backend=backend, config=config)

        async with scheduler:
            reqs = [
                make_request("A", priority=1),
                make_request("B", priority=5),
                make_request("C", priority=5),
                make_request("D", priority=2),
                make_request("E", priority=10),
                make_request("F", priority=2),
            ]
            tasks = [asyncio.create_task(scheduler.submit(r)) for r in reqs]
            await asyncio.sleep(0)
            assert scheduler.queued_count == 6

            backend.pause_event.set()
            await asyncio.gather(*tasks)

            assert len(backend.formed_batches) == 2
            assert backend.formed_batches[0].request_ids == ["E", "B", "C"]
            assert backend.formed_batches[1].request_ids == ["D", "F", "A"]


class TestBatchConcurrencyAndBackpressure:
    """Test concurrency semantics and backpressure with batching."""

    async def test_concurrency_bounds_active_batches(self) -> None:
        """Verify max_concurrency strictly bounds the number of concurrently executing batches."""
        backend = BatchTrackingBackend()
        config = SchedulerConfig(
            max_concurrency=2,
            batch_config=BatchConfig(max_batch_size=2, batch_wait_ms=0.0),
        )
        scheduler = Scheduler(backend=backend, config=config)

        async with scheduler:
            requests = [make_request(f"conc-{i}") for i in range(8)]
            await asyncio.gather(*[scheduler.submit(r) for r in requests])

            # 8 requests in batches of 2 -> 4 batches total
            assert len(backend.formed_batches) == 4
            assert backend.max_observed_batch_concurrency <= 2

    async def test_backpressure_rejection_with_batching(self) -> None:
        """Verify queue saturation rejects submissions without corrupting formed batches."""
        backend = BatchTrackingBackend()
        backend.pause_event.clear()

        # max_concurrency=1, max_queue_size=2, max_batch_size=2
        config = SchedulerConfig(
            max_concurrency=1,
            max_queue_size=2,
            batch_config=BatchConfig(max_batch_size=2, batch_wait_ms=0.0),
        )
        scheduler = Scheduler(backend=backend, config=config)

        async with scheduler:
            # First batch fills worker
            t_active1 = asyncio.create_task(scheduler.submit(make_request("act-1")))
            t_active2 = asyncio.create_task(scheduler.submit(make_request("act-2")))
            await backend.batch_started_event.wait()

            # Fill queue to capacity (2 items)
            t_q1 = asyncio.create_task(scheduler.submit(make_request("q-1")))
            t_q2 = asyncio.create_task(scheduler.submit(make_request("q-2")))
            await asyncio.sleep(0)
            assert scheduler.queued_count == 2

            # Next submission must raise QueueFullError
            with pytest.raises(QueueFullError):
                await scheduler.submit(make_request("rejected-item"))

            backend.pause_event.set()
            await asyncio.gather(t_active1, t_active2, t_q1, t_q2)

            assert len(backend.formed_batches) == 2
            assert all("rejected-item" not in b.request_ids for b in backend.formed_batches)


class TestBatchFailureIsolation:
    """Test error handling and isolation across batch executions."""

    async def test_individual_backend_failure_isolation(self) -> None:
        """With standard InferenceBackend, a failed request isolates from siblings in batch."""
        backend = IndividualBackend()

        backend.failing_request_ids.add("fail-req")

        config = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=3, batch_wait_ms=0.0),
        )
        scheduler = Scheduler(backend=backend, config=config)

        async with scheduler:
            t1 = asyncio.create_task(scheduler.submit(make_request("ok-1")))
            t2 = asyncio.create_task(scheduler.submit(make_request("fail-req")))
            t3 = asyncio.create_task(scheduler.submit(make_request("ok-2")))
            await asyncio.sleep(0)

            resp1 = await t1
            assert resp1.request_id == "ok-1"
            assert scheduler.get_status("ok-1") == RequestStatus.COMPLETED

            with pytest.raises(InferenceError):
                await t2
            assert scheduler.get_status("fail-req") == RequestStatus.FAILED

            resp3 = await t3
            assert resp3.request_id == "ok-2"
            assert scheduler.get_status("ok-2") == RequestStatus.COMPLETED

    async def test_batch_backend_failure(self) -> None:
        """With BatchInferenceBackend, batch failure marks all requests in batch FAILED."""
        backend = BatchTrackingBackend()
        backend.pause_event.clear()
        config = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=2, batch_wait_ms=0.0),
        )
        scheduler = Scheduler(backend=backend, config=config)

        async with scheduler:
            # Cause first batch to fail
            t1 = asyncio.create_task(scheduler.submit(make_request("f-1")))
            t2 = asyncio.create_task(scheduler.submit(make_request("f-2")))
            await backend.batch_started_event.wait()

            # Mark batch ID as failing
            assert len(backend.formed_batches) == 1
            backend.failing_batch_ids.add(backend.formed_batches[0].batch_id)
            backend.pause_event.set()

            with pytest.raises(InferenceError):
                await t1
            with pytest.raises(InferenceError):
                await t2

            assert scheduler.get_status("f-1") == RequestStatus.FAILED
            assert scheduler.get_status("f-2") == RequestStatus.FAILED

            # Scheduler remains healthy and processes next batch
            resp_ok = await scheduler.submit(make_request("recovery-req"))
            assert resp_ok.request_id == "recovery-req"
            assert scheduler.get_status("recovery-req") == RequestStatus.COMPLETED


class TestBatchShutdown:
    """Test shutdown handling during batch wait windows."""

    async def test_shutdown_during_batch_wait_window(self) -> None:
        """Verify scheduler shutdown while waiting in batching window terminates cleanly."""
        backend = BatchTrackingBackend()
        # 5 second wait window
        config = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=5000.0),
        )
        scheduler = Scheduler(backend=backend, config=config)
        await scheduler.start()

        # Submit 1 request so worker enters the 5-second wait window
        task = asyncio.create_task(scheduler.submit(make_request("in-window")))
        await asyncio.sleep(0.01)

        # Trigger shutdown
        await scheduler.shutdown(wait_running=False, cancel_queued=True, timeout=2.0)

        with pytest.raises((InferenceError, asyncio.CancelledError, Exception)):
            await task

        assert not scheduler.is_running
        assert len(scheduler._workers) == 0
