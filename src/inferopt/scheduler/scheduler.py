"""Asynchronous request scheduler with priority, concurrency, batching, and telemetry."""

import asyncio
import contextlib
import itertools
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Final

from inferopt.backends.base import BatchInferenceBackend, InferenceBackend
from inferopt.core.exceptions import (
    InferenceError,
    QueueFullError,
    SchedulerNotRunningError,
    SchedulerShutdownError,
)
from inferopt.core.models import InferenceBatch, InferenceRequest, InferenceResponse
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.scheduler.lifecycle import RequestRecord, RequestStatus
from inferopt.telemetry.collector import MetricsCollector
from inferopt.telemetry.models import BatchMetrics, RequestMetrics

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
    dynamic batch formation with wait windows, queue capacity backpressure,
    request lifecycle tracking, execution timing metrics, and non-intrusive telemetry.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        config: SchedulerConfig | None = None,
        collector: MetricsCollector | None = None,
    ) -> None:
        """Initialize the scheduler with a backend implementation, configuration, and telemetry.

        Args:
            backend: The inference engine executing requests (e.g., MockBackend, MLXBackend).
            config: Scheduler configuration settings. Defaults to SchedulerConfig().
            collector: Optional metrics collector. Defaults to a new MetricsCollector().
        """
        self._backend = backend
        self._config = config or SchedulerConfig()
        self._collector = collector if collector is not None else MetricsCollector()

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
    def batch_config(self) -> BatchConfig:
        """Current batching configuration."""
        return self._config.batch_config

    @property
    def collector(self) -> MetricsCollector:
        """Active telemetry and metrics collector."""
        return self._collector

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

    def _safe_record(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        """Safely execute a telemetry collection callback, strictly isolating any exceptions."""
        with contextlib.suppress(Exception):
            fn(*args, **kwargs)

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
        backend_name = getattr(self._backend, "backend_name", "unknown")

        # 1. Drain and cancel pending queued requests if requested
        if cancel_queued:
            cancelled_time = time.perf_counter()
            while not self._queue.empty():
                try:
                    _, _, raw_item = self._queue.get_nowait()
                    if raw_item is not _SENTINEL and isinstance(raw_item, _QueueItem):
                        if not raw_item.record.status.is_terminal:
                            raw_item.record.mark_cancelled(
                                cancelled_time,
                                error_message="Scheduler shutdown cancelled pending request.",
                            )
                        self._prune_history()
                        self._safe_record(
                            self._collector.record_request,
                            RequestMetrics(
                                request_id=raw_item.request.request_id,
                                priority=raw_item.request.priority,
                                queue_wait_ms=raw_item.record.queue_wait_ms or 0.0,
                                execution_ms=0.0,
                                total_latency_ms=raw_item.record.total_latency_ms or 0.0,
                                status=RequestStatus.CANCELLED,
                                max_tokens=raw_item.request.max_tokens,
                                backend_name=backend_name,
                                error_message="Scheduler shutdown cancelled pending request.",
                            ),
                        )
                        if not raw_item.future.done():
                            raw_item.future.set_exception(
                                SchedulerShutdownError(
                                    "Request cancelled due to scheduler shutdown."
                                )
                            )
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    break
            self._safe_record(self._collector.record_queue_depth, 0)

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
        self._safe_record(self._collector.record_queue_depth, self._queue.qsize())

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
                self._prune_history()
                self._safe_record(
                    self._collector.record_request,
                    RequestMetrics(
                        request_id=request.request_id,
                        priority=request.priority,
                        queue_wait_ms=record.queue_wait_ms or 0.0,
                        execution_ms=0.0,
                        total_latency_ms=record.total_latency_ms or 0.0,
                        status=RequestStatus.CANCELLED,
                        max_tokens=request.max_tokens,
                        backend_name=getattr(self._backend, "backend_name", "unknown"),
                        error_message="Request submission cancelled by caller.",
                    ),
                )
                self._safe_record(self._collector.record_queue_depth, self._queue.qsize())
            raise

    def get_status(self, request_id: str) -> RequestStatus | None:
        """Get the current lifecycle state of a request by its ID."""
        record = self._records.get(request_id)
        return record.status if record is not None else None

    def get_record(self, request_id: str) -> RequestRecord | None:
        """Get the full lifecycle tracking record and timing metrics for a request."""
        return self._records.get(request_id)

    async def _worker_loop(self, worker_id: int) -> None:
        """Internal worker task processing batches of queued requests concurrently."""
        batch_cfg = self._config.batch_config
        max_batch_size = batch_cfg.max_batch_size
        batch_wait_sec = batch_cfg.batch_wait_ms / 1000.0
        backend_name = getattr(self._backend, "backend_name", "unknown")

        while self._is_running:
            try:
                _, _, raw_item = await self._queue.get()
            except asyncio.CancelledError:
                break

            if raw_item is _SENTINEL or not isinstance(raw_item, _QueueItem):
                self._queue.task_done()
                break

            # Handle cancelled first item
            if raw_item.future.cancelled():
                if not raw_item.record.status.is_terminal:
                    raw_item.record.mark_cancelled(
                        time.perf_counter(),
                        error_message="Request cancelled prior to dispatch.",
                    )
                    self._safe_record(
                        self._collector.record_request,
                        RequestMetrics(
                            request_id=raw_item.request.request_id,
                            priority=raw_item.request.priority,
                            queue_wait_ms=raw_item.record.queue_wait_ms or 0.0,
                            execution_ms=0.0,
                            total_latency_ms=raw_item.record.total_latency_ms or 0.0,
                            status=RequestStatus.CANCELLED,
                            max_tokens=raw_item.request.max_tokens,
                            backend_name=backend_name,
                            error_message="Request cancelled prior to dispatch.",
                        ),
                    )
                self._queue.task_done()
                self._prune_history()
                continue

            batch_formation_start = time.perf_counter()
            items: list[_QueueItem] = [raw_item]

            # 1. Drain any items already waiting in the queue up to max_batch_size
            while len(items) < max_batch_size and not self._queue.empty():
                try:
                    _, _, extra_raw = self._queue.get_nowait()
                    if extra_raw is _SENTINEL or not isinstance(extra_raw, _QueueItem):
                        self._queue.task_done()
                        break
                    if extra_raw.future.cancelled():
                        if not extra_raw.record.status.is_terminal:
                            extra_raw.record.mark_cancelled(
                                time.perf_counter(),
                                error_message="Request cancelled prior to dispatch.",
                            )
                            self._safe_record(
                                self._collector.record_request,
                                RequestMetrics(
                                    request_id=extra_raw.request.request_id,
                                    priority=extra_raw.request.priority,
                                    queue_wait_ms=extra_raw.record.queue_wait_ms or 0.0,
                                    execution_ms=0.0,
                                    total_latency_ms=extra_raw.record.total_latency_ms or 0.0,
                                    status=RequestStatus.CANCELLED,
                                    max_tokens=extra_raw.request.max_tokens,
                                    backend_name=backend_name,
                                    error_message="Request cancelled prior to dispatch.",
                                ),
                            )
                        self._queue.task_done()
                        self._prune_history()
                        continue
                    items.append(extra_raw)
                except asyncio.QueueEmpty:
                    break

            # 2. If batch is not full and batch_wait_sec > 0, wait up to the window limit
            if len(items) < max_batch_size and batch_wait_sec > 0:
                deadline = batch_formation_start + batch_wait_sec
                while len(items) < max_batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        _, _, next_raw = await asyncio.wait_for(
                            self._queue.get(), timeout=remaining
                        )
                        if next_raw is _SENTINEL or not isinstance(next_raw, _QueueItem):
                            self._queue.task_done()
                            break
                        if next_raw.future.cancelled():
                            if not next_raw.record.status.is_terminal:
                                next_raw.record.mark_cancelled(
                                    time.perf_counter(),
                                    error_message="Request cancelled prior to dispatch.",
                                )
                                self._safe_record(
                                    self._collector.record_request,
                                    RequestMetrics(
                                        request_id=next_raw.request.request_id,
                                        priority=next_raw.request.priority,
                                        queue_wait_ms=next_raw.record.queue_wait_ms or 0.0,
                                        execution_ms=0.0,
                                        total_latency_ms=next_raw.record.total_latency_ms or 0.0,
                                        status=RequestStatus.CANCELLED,
                                        max_tokens=next_raw.request.max_tokens,
                                        backend_name=backend_name,
                                        error_message="Request cancelled prior to dispatch.",
                                    ),
                                )
                            self._queue.task_done()
                            self._prune_history()
                            continue
                        items.append(next_raw)
                    except TimeoutError:
                        break
                    except asyncio.CancelledError:
                        cancelled_time = time.perf_counter()
                        for it in items:
                            if not it.record.status.is_terminal:
                                it.record.mark_cancelled(
                                    cancelled_time,
                                    error_message="Scheduler shutdown cancelled batch formation.",
                                )
                            self._prune_history()
                            self._safe_record(
                                self._collector.record_request,
                                RequestMetrics(
                                    request_id=it.request.request_id,
                                    priority=it.request.priority,
                                    queue_wait_ms=it.record.queue_wait_ms or 0.0,
                                    execution_ms=0.0,
                                    total_latency_ms=it.record.total_latency_ms or 0.0,
                                    status=RequestStatus.CANCELLED,
                                    max_tokens=it.request.max_tokens,
                                    backend_name=backend_name,
                                    error_message="Scheduler shutdown cancelled batch formation.",
                                ),
                            )
                            if not it.future.done():
                                it.future.set_exception(
                                    SchedulerShutdownError(
                                        "Request cancelled due to scheduler shutdown."
                                    )
                                )
                            self._queue.task_done()
                        raise

            if not items:
                continue

            # Mark all items as RUNNING
            dispatch_time = time.perf_counter()
            formation_wait_ms = max(0.0, (dispatch_time - batch_formation_start) * 1000.0)

            for it in items:
                self._active_requests.add(it.request.request_id)
                it.record.mark_running(dispatch_time)
            self._active_changed_event.set()
            self._safe_record(self._collector.record_queue_depth, self._queue.qsize())
            self._safe_record(self._collector.record_active_concurrency, len(self._active_requests))

            batch = InferenceBatch(requests=tuple(it.request for it in items))

            try:
                if isinstance(self._backend, BatchInferenceBackend):
                    # Backend natively supports batched execution
                    responses = await self._backend.generate_batch(batch)
                    completed_time = time.perf_counter()
                    exec_duration_ms = max(0.0, (completed_time - dispatch_time) * 1000.0)
                    resp_map = {r.request_id: r for r in responses}

                    completed_count = 0
                    failed_count = 0

                    for it in items:
                        resp = resp_map.get(it.request.request_id)
                        if resp is not None:
                            it.record.mark_completed(completed_time)
                            completed_count += 1
                            self._prune_history()
                            self._safe_record(
                                self._collector.record_request,
                                RequestMetrics(
                                    request_id=it.request.request_id,
                                    priority=it.request.priority,
                                    queue_wait_ms=it.record.queue_wait_ms or 0.0,
                                    execution_ms=it.record.execution_ms or exec_duration_ms,
                                    total_latency_ms=it.record.total_latency_ms or 0.0,
                                    status=RequestStatus.COMPLETED,
                                    input_tokens=resp.input_tokens,
                                    output_tokens=resp.output_tokens,
                                    max_tokens=it.request.max_tokens,
                                    backend_name=resp.backend_name,
                                    batch_id=batch.batch_id,
                                ),
                            )
                            if not it.future.done():
                                it.future.set_result(resp)
                        else:
                            err_msg = (
                                f"Backend omitted response for request {it.request.request_id}"
                            )
                            it.record.mark_failed(completed_time, err_msg)
                            failed_count += 1
                            self._prune_history()
                            self._safe_record(
                                self._collector.record_request,
                                RequestMetrics(
                                    request_id=it.request.request_id,
                                    priority=it.request.priority,
                                    queue_wait_ms=it.record.queue_wait_ms or 0.0,
                                    execution_ms=it.record.execution_ms or exec_duration_ms,
                                    total_latency_ms=it.record.total_latency_ms or 0.0,
                                    status=RequestStatus.FAILED,
                                    max_tokens=it.request.max_tokens,
                                    backend_name=backend_name,
                                    batch_id=batch.batch_id,
                                    error_message=err_msg,
                                ),
                            )
                            if not it.future.done():
                                it.future.set_exception(InferenceError(err_msg))

                    self._safe_record(
                        self._collector.record_batch,
                        BatchMetrics(
                            batch_id=batch.batch_id,
                            size=batch.size,
                            batch_formation_wait_ms=formation_wait_ms,
                            execution_ms=exec_duration_ms,
                            total_max_tokens=batch.total_max_tokens,
                            backend_name=backend_name,
                            request_ids=tuple(batch.request_ids),
                            completed_request_count=completed_count,
                            failed_request_count=failed_count,
                        ),
                    )
                else:
                    # Backend executes requests individually
                    completed_count = 0
                    failed_count = 0

                    async def _execute_single(
                        item_to_exec: _QueueItem,
                        assigned_batch_id: str,
                        engine_name: str,
                    ) -> None:
                        nonlocal completed_count, failed_count
                        try:
                            resp = await self._backend.generate(item_to_exec.request)
                            done_time = time.perf_counter()
                            item_to_exec.record.mark_completed(done_time)
                            completed_count += 1
                            self._prune_history()
                            self._safe_record(
                                self._collector.record_request,
                                RequestMetrics(
                                    request_id=item_to_exec.request.request_id,
                                    priority=item_to_exec.request.priority,
                                    queue_wait_ms=item_to_exec.record.queue_wait_ms or 0.0,
                                    execution_ms=item_to_exec.record.execution_ms or 0.0,
                                    total_latency_ms=item_to_exec.record.total_latency_ms or 0.0,
                                    status=RequestStatus.COMPLETED,
                                    input_tokens=resp.input_tokens,
                                    output_tokens=resp.output_tokens,
                                    max_tokens=item_to_exec.request.max_tokens,
                                    backend_name=resp.backend_name,
                                    batch_id=assigned_batch_id,
                                ),
                            )
                            if not item_to_exec.future.done():
                                item_to_exec.future.set_result(resp)
                        except asyncio.CancelledError:
                            if not item_to_exec.record.status.is_terminal:
                                item_to_exec.record.mark_cancelled(
                                    time.perf_counter(),
                                    error_message="Execution cancelled.",
                                )
                            self._prune_history()
                            self._safe_record(
                                self._collector.record_request,
                                RequestMetrics(
                                    request_id=item_to_exec.request.request_id,
                                    priority=item_to_exec.request.priority,
                                    queue_wait_ms=item_to_exec.record.queue_wait_ms or 0.0,
                                    execution_ms=0.0,
                                    total_latency_ms=item_to_exec.record.total_latency_ms or 0.0,
                                    status=RequestStatus.CANCELLED,
                                    max_tokens=item_to_exec.request.max_tokens,
                                    backend_name=engine_name,
                                    batch_id=assigned_batch_id,
                                    error_message="Execution cancelled.",
                                ),
                            )
                            if not item_to_exec.future.done():
                                item_to_exec.future.cancel()
                            raise
                        except Exception as exc:
                            if not item_to_exec.record.status.is_terminal:
                                item_to_exec.record.mark_failed(
                                    time.perf_counter(),
                                    error_message=str(exc),
                                )
                            failed_count += 1
                            self._prune_history()
                            self._safe_record(
                                self._collector.record_request,
                                RequestMetrics(
                                    request_id=item_to_exec.request.request_id,
                                    priority=item_to_exec.request.priority,
                                    queue_wait_ms=item_to_exec.record.queue_wait_ms or 0.0,
                                    execution_ms=0.0,
                                    total_latency_ms=item_to_exec.record.total_latency_ms or 0.0,
                                    status=RequestStatus.FAILED,
                                    max_tokens=item_to_exec.request.max_tokens,
                                    backend_name=engine_name,
                                    batch_id=assigned_batch_id,
                                    error_message=str(exc),
                                ),
                            )
                            if not item_to_exec.future.done():
                                item_to_exec.future.set_exception(exc)

                    exec_start = time.perf_counter()
                    await asyncio.gather(
                        *[_execute_single(it, batch.batch_id, backend_name) for it in items],
                        return_exceptions=True,
                    )
                    exec_end = time.perf_counter()
                    exec_duration_ms = max(0.0, (exec_end - exec_start) * 1000.0)

                    self._safe_record(
                        self._collector.record_batch,
                        BatchMetrics(
                            batch_id=batch.batch_id,
                            size=batch.size,
                            batch_formation_wait_ms=formation_wait_ms,
                            execution_ms=exec_duration_ms,
                            total_max_tokens=batch.total_max_tokens,
                            backend_name=backend_name,
                            request_ids=tuple(batch.request_ids),
                            completed_request_count=completed_count,
                            failed_request_count=failed_count,
                        ),
                    )

            except asyncio.CancelledError:
                cancelled_time = time.perf_counter()
                for it in items:
                    if not it.record.status.is_terminal:
                        it.record.mark_cancelled(
                            cancelled_time, error_message="Execution cancelled."
                        )
                    self._prune_history()
                    self._safe_record(
                        self._collector.record_request,
                        RequestMetrics(
                            request_id=it.request.request_id,
                            priority=it.request.priority,
                            queue_wait_ms=it.record.queue_wait_ms or 0.0,
                            execution_ms=0.0,
                            total_latency_ms=it.record.total_latency_ms or 0.0,
                            status=RequestStatus.CANCELLED,
                            max_tokens=it.request.max_tokens,
                            backend_name=backend_name,
                            batch_id=batch.batch_id,
                            error_message="Execution cancelled.",
                        ),
                    )
                    if not it.future.done():
                        it.future.cancel()
                raise
            except Exception as exc:
                failed_time = time.perf_counter()
                exec_duration_ms = max(0.0, (failed_time - dispatch_time) * 1000.0)
                for it in items:
                    if not it.record.status.is_terminal:
                        it.record.mark_failed(failed_time, error_message=str(exc))
                    self._prune_history()
                    self._safe_record(
                        self._collector.record_request,
                        RequestMetrics(
                            request_id=it.request.request_id,
                            priority=it.request.priority,
                            queue_wait_ms=it.record.queue_wait_ms or 0.0,
                            execution_ms=it.record.execution_ms or exec_duration_ms,
                            total_latency_ms=it.record.total_latency_ms or 0.0,
                            status=RequestStatus.FAILED,
                            max_tokens=it.request.max_tokens,
                            backend_name=backend_name,
                            batch_id=batch.batch_id,
                            error_message=str(exc),
                        ),
                    )
                    if not it.future.done():
                        it.future.set_exception(exc)

                self._safe_record(
                    self._collector.record_batch,
                    BatchMetrics(
                        batch_id=batch.batch_id,
                        size=batch.size,
                        batch_formation_wait_ms=formation_wait_ms,
                        execution_ms=exec_duration_ms,
                        total_max_tokens=batch.total_max_tokens,
                        backend_name=backend_name,
                        request_ids=tuple(batch.request_ids),
                        completed_request_count=0,
                        failed_request_count=len(items),
                    ),
                )
            finally:
                for it in items:
                    self._active_requests.discard(it.request.request_id)
                    self._queue.task_done()
                self._active_changed_event.set()
                self._safe_record(
                    self._collector.record_active_concurrency, len(self._active_requests)
                )
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
