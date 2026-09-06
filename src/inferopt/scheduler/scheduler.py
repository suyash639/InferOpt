"""Asynchronous request scheduler with priority, concurrency bounding, and backpressure."""

import asyncio
import itertools
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Final

from inferopt.backends.base import InferenceBackend
from inferopt.core.exceptions import (
    QueueFullError,
    SchedulerNotRunningError,
    SchedulerShutdownError,
)
from inferopt.core.models import InferenceRequest, InferenceResponse
from inferopt.scheduler.config import SchedulerConfig
from inferopt.scheduler.lifecycle import RequestRecord, RequestStatus

_SENTINEL: Final[object] = object()


@dataclass
class _QueueItem:
    """Internal wrapper for queued requests awaiting worker dispatch."""

    request: InferenceRequest
    future: asyncio.Future[InferenceResponse]
    record: RequestRecord


class Scheduler:
    """Asynchronous request scheduler for LLM inference serving.

    Enforces bounded concurrency, priority-based dispatch with FIFO tie-breaking,
    queue capacity backpressure, request lifecycle tracking, and execution timing metrics.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        config: SchedulerConfig | None = None,
    ) -> None:
        """Initialize the scheduler with a backend implementation and configuration.

        Args:
            backend: The inference engine executing requests (e.g., MockBackend, MLXBackend).
            config: Scheduler configuration settings. Defaults to SchedulerConfig().
        """
        self._backend = backend
        self._config = config or SchedulerConfig()

        self._queue: asyncio.PriorityQueue[tuple[int, int, Any]] = asyncio.PriorityQueue()
        self._sequence_counter = itertools.count()

        self._workers: list[asyncio.Task[None]] = []
        self._is_running: bool = False
        self._is_shutting_down: bool = False

        self._records: OrderedDict[str, RequestRecord] = OrderedDict()
        self._active_requests: set[str] = set()
        self._active_changed_event: asyncio.Event = asyncio.Event()

    @property
    def config(self) -> SchedulerConfig:
        """Current scheduler configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Return True if the scheduler worker pool is active and accepting work."""
        return self._is_running and not self._is_shutting_down

    @property
    def is_shutting_down(self) -> bool:
        """Return True if the scheduler is in the process of shutting down."""
        return self._is_shutting_down

    @property
    def active_count(self) -> int:
        """Number of requests currently executing on the backend."""
        return len(self._active_requests)

    @property
    def queued_count(self) -> int:
        """Number of pending requests waiting in the priority queue."""
        return self._queue.qsize()

    async def start(self) -> None:
        """Start the scheduler worker pool.

        Initializes the internal priority queue and spawns `max_concurrency` worker tasks.
        """
        if self._is_running:
            return

        self._is_running = True
        self._is_shutting_down = False
        self._queue = asyncio.PriorityQueue()
        self._active_changed_event = asyncio.Event()

        self._workers = [
            asyncio.create_task(self._worker_loop(worker_id=i), name=f"inferopt-worker-{i}")
            for i in range(self._config.max_concurrency)
        ]

    async def shutdown(
        self,
        wait_running: bool = True,
        cancel_queued: bool = True,
        timeout: float | None = 10.0,
    ) -> None:
        """Gracefully shut down the scheduler.

        Args:
            wait_running: If True, wait for currently executing requests to finish.
            cancel_queued: If True, cancel all pending requests remaining in the queue.
            timeout: Maximum time in seconds to wait for active requests to finish.
        """
        if not self._is_running or self._is_shutting_down:
            return

        self._is_shutting_down = True

        # 1. Drain and cancel pending queued requests if requested
        if cancel_queued:
            while not self._queue.empty():
                try:
                    _, _, raw_item = self._queue.get_nowait()
                    if raw_item is not _SENTINEL and isinstance(raw_item, _QueueItem):
                        if not raw_item.future.done():
                            raw_item.future.set_exception(
                                SchedulerShutdownError(
                                    "Request cancelled due to scheduler shutdown."
                                )
                            )
                        if not raw_item.record.status.is_terminal:
                            raw_item.record.mark_cancelled(
                                time.perf_counter(),
                                error_message="Scheduler shutdown cancelled pending request.",
                            )
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    break

        # 2. Wait for active requests to complete if requested
        if wait_running and self._active_requests:
            try:
                start_wait = time.perf_counter()
                while self._active_requests:
                    if timeout is not None and (time.perf_counter() - start_wait) > timeout:
                        break
                    self._active_changed_event.clear()
                    try:
                        wait_slice = 0.05
                        if timeout is not None:
                            remaining = timeout - (time.perf_counter() - start_wait)
                            if remaining <= 0:
                                break
                            wait_slice = min(wait_slice, remaining)
                        await asyncio.wait_for(
                            self._active_changed_event.wait(), timeout=wait_slice
                        )
                    except TimeoutError:
                        continue
            except Exception:
                pass

        # 3. Terminate and clean up all worker tasks
        self._is_running = False
        for worker in self._workers:
            if not worker.done():
                worker.cancel()

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._is_shutting_down = False

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        """Submit an inference request to the scheduler for execution.

        Args:
            request: Validated domain inference request.

        Returns:
            InferenceResponse generated by the backend.

        Raises:
            SchedulerShutdownError: If the scheduler is shutting down.
            SchedulerNotRunningError: If the scheduler has not been started.
            QueueFullError: If the scheduler queue is at maximum capacity.
            InferenceError: If execution on the backend fails.
        """
        if self._is_shutting_down:
            raise SchedulerShutdownError(
                "Scheduler is shutting down. New submissions are rejected."
            )
        if not self._is_running:
            raise SchedulerNotRunningError(
                "Scheduler is not running. Call start() before submitting requests."
            )

        if self._config.max_queue_size == 0:
            if self.active_count >= self._config.max_concurrency or self._queue.qsize() > 0:
                raise QueueFullError(
                    "Scheduler queue capacity is 0 and all workers are busy. Request rejected."
                )
        elif self._queue.qsize() >= self._config.max_queue_size:
            raise QueueFullError(
                f"Scheduler queue is full (capacity={self._config.max_queue_size}, "
                f"current={self._queue.qsize()}). Request rejected."
            )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[InferenceResponse] = loop.create_future()

        record = RequestRecord(
            request_id=request.request_id,
            status=RequestStatus.QUEUED,
            queued_at=time.perf_counter(),
        )
        self._add_record(record)

        seq_id = next(self._sequence_counter)
        item = _QueueItem(request=request, future=future, record=record)

        # Invert priority so min-heap dequeues highest numeric priority first.
        # seq_id ensures deterministic FIFO ordering among identical priorities.
        self._queue.put_nowait((-request.priority, seq_id, item))

        try:
            return await future
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            if not record.status.is_terminal:
                record.mark_cancelled(
                    time.perf_counter(),
                    error_message="Request submission cancelled by caller.",
                )
            raise

    def get_status(self, request_id: str) -> RequestStatus | None:
        """Get the current lifecycle state of a request by its ID."""
        record = self._records.get(request_id)
        return record.status if record is not None else None

    def get_record(self, request_id: str) -> RequestRecord | None:
        """Get the full lifecycle tracking record and timing metrics for a request."""
        return self._records.get(request_id)

    async def _worker_loop(self, worker_id: int) -> None:
        """Internal worker task processing queued requests concurrently."""
        while self._is_running:
            try:
                _, _, raw_item = await self._queue.get()
            except asyncio.CancelledError:
                break

            if raw_item is _SENTINEL or not isinstance(raw_item, _QueueItem):
                self._queue.task_done()
                break

            item: _QueueItem = raw_item

            # Check if caller cancelled before dispatch
            if item.future.cancelled():
                if not item.record.status.is_terminal:
                    item.record.mark_cancelled(
                        time.perf_counter(),
                        error_message="Request cancelled prior to dispatch.",
                    )
                self._queue.task_done()
                self._prune_history()
                continue

            self._active_requests.add(item.request.request_id)
            self._active_changed_event.set()
            item.record.mark_running(time.perf_counter())

            try:
                response = await self._backend.generate(item.request)
                item.record.mark_completed(time.perf_counter())
                if not item.future.done():
                    item.future.set_result(response)
            except asyncio.CancelledError:
                if not item.record.status.is_terminal:
                    item.record.mark_cancelled(
                        time.perf_counter(),
                        error_message="Execution cancelled.",
                    )
                if not item.future.done():
                    item.future.cancel()
                raise
            except Exception as exc:
                if not item.record.status.is_terminal:
                    item.record.mark_failed(
                        time.perf_counter(),
                        error_message=str(exc),
                    )
                if not item.future.done():
                    item.future.set_exception(exc)
            finally:
                self._active_requests.discard(item.request.request_id)
                self._active_changed_event.set()
                self._queue.task_done()
                self._prune_history()

    def _add_record(self, record: RequestRecord) -> None:
        """Store a request record and apply retention bounds."""
        self._records[record.request_id] = record
        self._prune_history()

    def _prune_history(self) -> None:
        """Prune oldest terminal records when record count exceeds history limit."""
        if self._config.max_history_size <= 0:
            terminal_keys = [k for k, r in self._records.items() if r.status.is_terminal]
            for k in terminal_keys:
                del self._records[k]
            return

        terminal_count = sum(1 for r in self._records.values() if r.status.is_terminal)
        while terminal_count > self._config.max_history_size:
            for k, r in list(self._records.items()):
                if r.status.is_terminal:
                    del self._records[k]
                    terminal_count -= 1
                    break

    async def __aenter__(self) -> "Scheduler":
        """Async context manager entry: starts the scheduler."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit: gracefully shuts down the scheduler."""
        await self.shutdown()


# Alias for explicit clarity
AsyncScheduler = Scheduler
