"""Comprehensive deterministic unit tests for the InferOpt request scheduler."""

import asyncio
from typing import Any

import pytest

from inferopt.backends.base import InferenceBackend
from inferopt.core.exceptions import (
    InferenceError,
    QueueFullError,
    SchedulerNotRunningError,
    SchedulerShutdownError,
)
from inferopt.core.models import InferenceRequest, InferenceResponse
from inferopt.scheduler.config import SchedulerConfig
from inferopt.scheduler.lifecycle import RequestStatus
from inferopt.scheduler.scheduler import Scheduler


class ControlledBackend(InferenceBackend):
    """Deterministic, instrumented backend for synchronizing and asserting test conditions."""

    def __init__(self, backend_name: str = "controlled-test") -> None:
        self._name = backend_name
        self.active_concurrency: int = 0
        self.max_observed_concurrency: int = 0
        self.dispatched_requests: list[str] = []
        self.completed_requests: list[str] = []
        self.failing_request_ids: set[str] = set()

        # Synchronization primitives
        self.pause_event: asyncio.Event = asyncio.Event()
        self.pause_event.set()  # Unpaused by default
        self.request_started_events: dict[str, asyncio.Event] = {}

    @property
    def backend_name(self) -> str:
        return self._name

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.dispatched_requests.append(request.request_id)
        self.active_concurrency += 1
        self.max_observed_concurrency = max(self.max_observed_concurrency, self.active_concurrency)

        # Notify any waiters that this request has started
        if request.request_id in self.request_started_events:
            self.request_started_events[request.request_id].set()

        try:
            # Block execution if pause_event is cleared
            await self.pause_event.wait()

            if request.request_id in self.failing_request_ids:
                raise InferenceError(f"Simulated execution failure for {request.request_id}")

            self.completed_requests.append(request.request_id)
            return InferenceResponse(
                request_id=request.request_id,
                generated_text=f"Response for {request.request_id}",
                input_tokens=10,
                output_tokens=20,
                latency_ms=15.0,
                backend_name=self.backend_name,
            )
        finally:
            self.active_concurrency -= 1


def make_request(
    request_id: str = "req-1",
    priority: int = 0,
    prompt: str = "Test prompt",
    model: str = "test-model",
    metadata: dict[str, Any] | None = None,
) -> InferenceRequest:
    """Helper factory for creating InferenceRequest instances."""
    return InferenceRequest(
        request_id=request_id,
        model=model,
        prompt=prompt,
        priority=priority,
        metadata=metadata or {},
    )


class TestSchedulerBasics:
    """Test basic scheduler submission, execution, and lifecycle."""

    async def test_unstarted_scheduler_rejects_submissions(self) -> None:
        """Verify submitting to an unstarted scheduler raises SchedulerNotRunningError."""
        backend = ControlledBackend()
        scheduler = Scheduler(backend=backend)

        req = make_request()
        with pytest.raises(SchedulerNotRunningError):
            await scheduler.submit(req)

    async def test_single_request_success(self) -> None:
        """Verify a single request is queued, dispatched, executed, and completed."""
        backend = ControlledBackend()
        config = SchedulerConfig(max_concurrency=2)
        scheduler = Scheduler(backend=backend, config=config)

        await scheduler.start()
        try:
            req = make_request(request_id="req-single", prompt="Hello world")
            response = await scheduler.submit(req)

            assert response.request_id == "req-single"
            assert response.backend_name == "controlled-test"
            assert scheduler.get_status("req-single") == RequestStatus.COMPLETED

            record = scheduler.get_record("req-single")
            assert record is not None
            assert record.status == RequestStatus.COMPLETED
            assert record.queue_wait_ms is not None
            assert record.execution_ms is not None
            assert record.total_latency_ms is not None
        finally:
            await scheduler.shutdown()

    async def test_multiple_requests_success(self) -> None:
        """Verify multiple concurrent requests complete and return matching responses."""
        backend = ControlledBackend()
        scheduler = Scheduler(backend=backend, config=SchedulerConfig(max_concurrency=3))

        async with scheduler:
            requests = [make_request(request_id=f"multi-{i}") for i in range(5)]
            responses = await asyncio.gather(*[scheduler.submit(r) for r in requests])

            assert len(responses) == 5
            for i, resp in enumerate(responses):
                assert resp.request_id == f"multi-{i}"
                assert scheduler.get_status(f"multi-{i}") == RequestStatus.COMPLETED


class TestSchedulerPriorityAndFIFO:
    """Test priority-aware dispatching and deterministic FIFO tie-breaking."""

    async def test_priority_ordering(self) -> None:
        """Verify requests with higher numeric priority are dispatched before lower priority ones.

        Scenario:
        1. Backend is paused with max_concurrency=1.
        2. First blocker request occupies the 1 running worker slot.
        3. Requests arrive in order:
           A (priority 1), B (priority 5), C (priority 5), D (priority 2).
        4. Release backend.
        5. Expected dispatch order: Blocker -> B -> C -> D -> A.
        """
        backend = ControlledBackend()
        backend.pause_event.clear()  # Hold execution

        scheduler = Scheduler(backend=backend, config=SchedulerConfig(max_concurrency=1))
        async with scheduler:
            blocker = make_request(request_id="blocker", priority=0)
            backend.request_started_events["blocker"] = asyncio.Event()

            # Submit blocker and wait until it starts executing
            blocker_task = asyncio.create_task(scheduler.submit(blocker))
            await backend.request_started_events["blocker"].wait()

            # Queue remaining requests while worker is occupied
            req_a = make_request(request_id="req-A", priority=1)
            req_b = make_request(request_id="req-B", priority=5)
            req_c = make_request(request_id="req-C", priority=5)
            req_d = make_request(request_id="req-D", priority=2)

            task_a = asyncio.create_task(scheduler.submit(req_a))
            task_b = asyncio.create_task(scheduler.submit(req_b))
            task_c = asyncio.create_task(scheduler.submit(req_c))
            task_d = asyncio.create_task(scheduler.submit(req_d))

            # Allow all tasks to execute submit() and enter the priority queue
            await asyncio.sleep(0)
            assert scheduler.queued_count == 4

            # Release backend so all requests execute sequentially
            backend.pause_event.set()

            await asyncio.gather(blocker_task, task_a, task_b, task_c, task_d)

            # Expected dispatch order: blocker first, then highest priority (B=5, C=5), D=2, A=1
            assert backend.dispatched_requests == [
                "blocker",
                "req-B",
                "req-C",
                "req-D",
                "req-A",
            ]

    async def test_fifo_ordering_equal_priority(self) -> None:
        """Verify strict FIFO order is preserved among requests with identical priorities."""
        backend = ControlledBackend()
        backend.pause_event.clear()

        scheduler = Scheduler(backend=backend, config=SchedulerConfig(max_concurrency=1))
        async with scheduler:
            blocker = make_request(request_id="blocker", priority=0)
            backend.request_started_events["blocker"] = asyncio.Event()

            blocker_task = asyncio.create_task(scheduler.submit(blocker))
            await backend.request_started_events["blocker"].wait()

            equal_tasks = [
                asyncio.create_task(
                    scheduler.submit(make_request(request_id=f"fifo-{i}", priority=10))
                )
                for i in range(5)
            ]

            await asyncio.sleep(0)
            assert scheduler.queued_count == 5

            backend.pause_event.set()
            await asyncio.gather(blocker_task, *equal_tasks)

            expected = ["blocker"] + [f"fifo-{i}" for i in range(5)]
            assert backend.dispatched_requests == expected

    async def test_negative_and_zero_priorities(self) -> None:
        """Verify negative, zero, and positive priorities are dispatched in exact relative order."""
        backend = ControlledBackend()
        backend.pause_event.clear()

        scheduler = Scheduler(backend=backend, config=SchedulerConfig(max_concurrency=1))
        async with scheduler:
            blocker = make_request(request_id="blocker", priority=0)
            backend.request_started_events["blocker"] = asyncio.Event()

            blocker_task = asyncio.create_task(scheduler.submit(blocker))
            await backend.request_started_events["blocker"].wait()

            req_neg = make_request(request_id="neg", priority=-5)
            req_zero = make_request(request_id="zero", priority=0)
            req_pos = make_request(request_id="pos", priority=100)

            t_neg = asyncio.create_task(scheduler.submit(req_neg))
            t_zero = asyncio.create_task(scheduler.submit(req_zero))
            t_pos = asyncio.create_task(scheduler.submit(req_pos))

            await asyncio.sleep(0)
            assert scheduler.queued_count == 3

            backend.pause_event.set()
            await asyncio.gather(blocker_task, t_neg, t_zero, t_pos)

            assert backend.dispatched_requests == ["blocker", "pos", "zero", "neg"]


class TestSchedulerConcurrency:
    """Test strict bounded concurrency enforcement."""

    async def test_concurrency_limit_one(self) -> None:
        """With max_concurrency=1, verify at no point is active concurrency > 1."""
        backend = ControlledBackend()
        scheduler = Scheduler(backend=backend, config=SchedulerConfig(max_concurrency=1))

        async with scheduler:
            requests = [make_request(request_id=f"c1-{i}") for i in range(6)]
            await asyncio.gather(*[scheduler.submit(r) for r in requests])

            assert backend.max_observed_concurrency == 1
            assert len(backend.completed_requests) == 6

    async def test_concurrency_limit_two(self) -> None:
        """With max_concurrency=2, verify at no point is active concurrency > 2."""
        backend = ControlledBackend()
        scheduler = Scheduler(backend=backend, config=SchedulerConfig(max_concurrency=2))

        async with scheduler:
            requests = [make_request(request_id=f"c2-{i}") for i in range(8)]
            await asyncio.gather(*[scheduler.submit(r) for r in requests])

            assert backend.max_observed_concurrency <= 2
            assert len(backend.completed_requests) == 8


class TestSchedulerBackpressure:
    """Test queue capacity bounding and immediate backpressure rejection."""

    async def test_queue_full_rejection(self) -> None:
        """Verify submissions exceeding max_queue_size immediately raise QueueFullError."""
        backend = ControlledBackend()
        backend.pause_event.clear()

        # max_concurrency=1, max_queue_size=2: can hold 1 active + 2 queued = 3 total in flight
        scheduler = Scheduler(
            backend=backend,
            config=SchedulerConfig(max_concurrency=1, max_queue_size=2),
        )

        async with scheduler:
            req_active = make_request(request_id="active")
            backend.request_started_events["active"] = asyncio.Event()

            t_active = asyncio.create_task(scheduler.submit(req_active))
            await backend.request_started_events["active"].wait()

            req_q1 = make_request(request_id="q1")
            req_q2 = make_request(request_id="q2")
            t_q1 = asyncio.create_task(scheduler.submit(req_q1))
            t_q2 = asyncio.create_task(scheduler.submit(req_q2))

            # Allow submit tasks to enqueue before testing backpressure
            await asyncio.sleep(0)
            assert scheduler.queued_count == 2

            # 4th submission should immediately fail with QueueFullError
            req_rejected = make_request(request_id="rejected")
            with pytest.raises(QueueFullError) as exc_info:
                await scheduler.submit(req_rejected)

            assert "queue is full" in str(exc_info.value).lower()
            assert scheduler.get_status("rejected") is None

            # Unpause backend and verify accepted requests finish cleanly
            backend.pause_event.set()
            await asyncio.gather(t_active, t_q1, t_q2)

            assert len(backend.completed_requests) == 3

    async def test_unbuffered_queue_rejection(self) -> None:
        """Verify max_queue_size=0 rejects when worker capacity is occupied."""
        backend = ControlledBackend()
        backend.pause_event.clear()

        scheduler = Scheduler(
            backend=backend,
            config=SchedulerConfig(max_concurrency=1, max_queue_size=0),
        )

        async with scheduler:
            req1 = make_request(request_id="req1")
            backend.request_started_events["req1"] = asyncio.Event()

            t1 = asyncio.create_task(scheduler.submit(req1))
            await backend.request_started_events["req1"].wait()

            # Since max_queue_size=0, any attempt to queue must fail
            req2 = make_request(request_id="req2")
            with pytest.raises(QueueFullError):
                await scheduler.submit(req2)

            backend.pause_event.set()
            await t1


class TestSchedulerFailureIsolation:
    """Test backend error handling and worker isolation."""

    async def test_backend_failure_propagates_and_isolates(self) -> None:
        """Verify a backend failure marks the request FAILED and keeps the scheduler healthy."""
        backend = ControlledBackend()
        backend.failing_request_ids.add("failing-req")

        scheduler = Scheduler(backend=backend, config=SchedulerConfig(max_concurrency=2))

        async with scheduler:
            req1 = make_request(request_id="success-1")
            req2 = make_request(request_id="failing-req")
            req3 = make_request(request_id="success-2")

            resp1 = await scheduler.submit(req1)
            assert resp1.request_id == "success-1"
            assert scheduler.get_status("success-1") == RequestStatus.COMPLETED

            # Failed request should raise the backend error
            with pytest.raises(InferenceError) as exc_info:
                await scheduler.submit(req2)
            assert "Simulated execution failure" in str(exc_info.value)

            assert scheduler.get_status("failing-req") == RequestStatus.FAILED
            record2 = scheduler.get_record("failing-req")
            assert record2 is not None
            assert record2.error_message is not None
            assert "Simulated execution failure" in record2.error_message

            # Subsequent request still processes successfully
            resp3 = await scheduler.submit(req3)
            assert resp3.request_id == "success-2"
            assert scheduler.get_status("success-2") == RequestStatus.COMPLETED


class TestSchedulerLifecycleAndTracking:
    """Test request state tracking, timing metrics, and bounded history."""

    async def test_status_transitions(self) -> None:
        """Verify request lifecycle state transitions from QUEUED -> RUNNING -> COMPLETED."""
        backend = ControlledBackend()
        backend.pause_event.clear()

        scheduler = Scheduler(backend=backend, config=SchedulerConfig(max_concurrency=1))

        async with scheduler:
            req = make_request(request_id="tracked-req")
            backend.request_started_events["tracked-req"] = asyncio.Event()

            task = asyncio.create_task(scheduler.submit(req))

            # Wait until request is dispatched and marked RUNNING
            await backend.request_started_events["tracked-req"].wait()
            assert scheduler.get_status("tracked-req") == RequestStatus.RUNNING

            backend.pause_event.set()
            await task

            assert scheduler.get_status("tracked-req") == RequestStatus.COMPLETED

            record = scheduler.get_record("tracked-req")
            assert record is not None
            assert record.queue_wait_ms is not None
            assert record.queue_wait_ms >= 0.0
            assert record.execution_ms is not None
            assert record.execution_ms >= 0.0
            assert record.total_latency_ms is not None
            assert record.total_latency_ms >= 0.0

    async def test_bounded_history_pruning(self) -> None:
        """Verify terminal records are pruned when exceeding max_history_size."""
        backend = ControlledBackend()
        scheduler = Scheduler(
            backend=backend,
            config=SchedulerConfig(max_concurrency=1, max_history_size=2),
        )

        async with scheduler:
            for i in range(5):
                await scheduler.submit(make_request(request_id=f"hist-{i}"))

            # Only the most recent 2 terminal records should remain
            assert scheduler.get_record("hist-0") is None
            assert scheduler.get_record("hist-1") is None
            assert scheduler.get_record("hist-2") is None
            assert scheduler.get_record("hist-3") is not None
            assert scheduler.get_record("hist-4") is not None


class TestSchedulerCancellationAndShutdown:
    """Test cancellation semantics and clean shutdown."""

    async def test_cancel_queued_request_before_dispatch(self) -> None:
        """Verify cancelling a queued submission task marks it CANCELLED and skips execution."""
        backend = ControlledBackend()
        backend.pause_event.clear()

        scheduler = Scheduler(backend=backend, config=SchedulerConfig(max_concurrency=1))

        async with scheduler:
            blocker = make_request(request_id="blocker")
            backend.request_started_events["blocker"] = asyncio.Event()

            blocker_task = asyncio.create_task(scheduler.submit(blocker))
            await backend.request_started_events["blocker"].wait()

            req_cancel = make_request(request_id="to-cancel")
            cancel_task = asyncio.create_task(scheduler.submit(req_cancel))

            await asyncio.sleep(0)
            assert scheduler.queued_count == 1

            # Cancel while in queue
            cancel_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancel_task

            assert scheduler.get_status("to-cancel") == RequestStatus.CANCELLED

            backend.pause_event.set()
            await blocker_task

            # Verify the cancelled request was never dispatched to the backend
            assert "to-cancel" not in backend.dispatched_requests

    async def test_shutdown_drains_and_cancels_queued_requests(self) -> None:
        """Verify scheduler shutdown rejects queued requests with SchedulerShutdownError."""
        backend = ControlledBackend()
        backend.pause_event.clear()

        scheduler = Scheduler(backend=backend, config=SchedulerConfig(max_concurrency=1))
        await scheduler.start()

        blocker = make_request(request_id="blocker")
        backend.request_started_events["blocker"] = asyncio.Event()

        blocker_task = asyncio.create_task(scheduler.submit(blocker))
        await backend.request_started_events["blocker"].wait()

        queued_req = make_request(request_id="queued-shut")
        queued_task = asyncio.create_task(scheduler.submit(queued_req))

        await asyncio.sleep(0)
        assert scheduler.queued_count == 1

        # Start shutdown in background
        shutdown_task = asyncio.create_task(
            scheduler.shutdown(wait_running=True, cancel_queued=True, timeout=5.0)
        )

        # Queued request should be cancelled/rejected with SchedulerShutdownError
        with pytest.raises(SchedulerShutdownError):
            await queued_task

        assert scheduler.get_status("queued-shut") == RequestStatus.CANCELLED

        # Release active request so shutdown can complete
        backend.pause_event.set()
        await blocker_task
        await shutdown_task

        assert not scheduler.is_running
        assert not scheduler.is_shutting_down

        # Submissions after shutdown should fail
        with pytest.raises(SchedulerNotRunningError):
            await scheduler.submit(make_request(request_id="after-shutdown"))

    async def test_no_orphaned_worker_tasks_after_shutdown(self) -> None:
        """Verify all internal worker tasks are joined and cleaned up after shutdown."""
        backend = ControlledBackend()
        scheduler = Scheduler(backend=backend, config=SchedulerConfig(max_concurrency=4))

        await scheduler.start()
        assert len(scheduler._workers) == 4

        await scheduler.shutdown()
        assert len(scheduler._workers) == 0
        assert not scheduler.is_running
