"""In-memory telemetry and metrics collector for inference scheduling and batching."""

import math
import threading
import time
from typing import Any

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


def _compute_percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list of float values using linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * (p / 100.0)
    floor_k = math.floor(k)
    ceil_k = math.ceil(k)
    if floor_k == ceil_k:
        return values[int(k)]
    d0 = values[floor_k] * (ceil_k - k)
    d1 = values[ceil_k] * (k - floor_k)
    return d0 + d1


class MetricsCollector:
    """Thread-safe, in-process metrics aggregator and telemetry recorder.

    Records request lifecycle metrics, batch formations, queue depths,
    and active concurrency counts. Produces immutable point-in-time snapshots
    with computed averages, percentiles, and throughput estimates.
    """

    def __init__(self, max_history: int = 10000) -> None:
        """Initialize the metrics collector.

        Args:
            max_history: Maximum number of request and batch metrics retained in memory.
        """
        self._max_history = max_history
        self._lock = threading.Lock()

        self._requests: list[RequestMetrics] = []
        self._batches: list[BatchMetrics] = []

        self._current_queue_depth: int = 0
        self._peak_queue_depth: int = 0
        self._current_active_requests: int = 0
        self._peak_active_requests: int = 0

        self._created_at: float = time.perf_counter()

    @property
    def total_recorded_requests(self) -> int:
        """Total number of request metric records currently stored."""
        with self._lock:
            return len(self._requests)

    @property
    def total_recorded_batches(self) -> int:
        """Total number of batch metric records currently stored."""
        with self._lock:
            return len(self._batches)

    def record_request(self, metrics: RequestMetrics) -> None:
        """Record a completed, failed, or cancelled request metric.

        Args:
            metrics: Validated immutable RequestMetrics record.
        """
        with self._lock:
            self._requests.append(metrics)
            if len(self._requests) > self._max_history:
                # Retain the newest max_history items
                self._requests = self._requests[-self._max_history :]

    def record_batch(self, metrics: BatchMetrics) -> None:
        """Record an executed batch metric.

        Args:
            metrics: Validated immutable BatchMetrics record.
        """
        with self._lock:
            self._batches.append(metrics)
            if len(self._batches) > self._max_history:
                self._batches = self._batches[-self._max_history :]

    def record_queue_depth(self, depth: int) -> None:
        """Record an instantaneous queue depth observation.

        Args:
            depth: Current number of requests waiting in the scheduler queue.
        """
        with self._lock:
            self._current_queue_depth = max(0, depth)
            if self._current_queue_depth > self._peak_queue_depth:
                self._peak_queue_depth = self._current_queue_depth

    def record_active_concurrency(self, count: int) -> None:
        """Record an instantaneous active request concurrency observation.

        Args:
            count: Current number of requests actively executing on backends.
        """
        with self._lock:
            self._current_active_requests = max(0, count)
            if self._current_active_requests > self._peak_active_requests:
                self._peak_active_requests = self._current_active_requests

    def get_recent_requests(self, limit: int = 100) -> tuple[RequestMetrics, ...]:
        """Return a tuple of recent request metrics up to the specified limit."""
        with self._lock:
            return tuple(self._requests[-limit:])

    def get_recent_batches(self, limit: int = 100) -> tuple[BatchMetrics, ...]:
        """Return a tuple of recent batch metrics up to the specified limit."""
        with self._lock:
            return tuple(self._batches[-limit:])

    def reset(self) -> None:
        """Reset all recorded metrics, peaks, and observation counters."""
        with self._lock:
            self._requests.clear()
            self._batches.clear()
            self._current_queue_depth = 0
            self._peak_queue_depth = 0
            self._current_active_requests = 0
            self._peak_active_requests = 0
            self._created_at = time.perf_counter()

    def snapshot(self, metadata: dict[str, Any] | None = None) -> MetricsSnapshot:
        """Compute and return an immutable point-in-time metrics snapshot.

        Calculates aggregate request statistics, batch formation and execution metrics,
        throughput rates, and peak concurrency observations.

        Args:
            metadata: Optional user-supplied tags or context attached to the snapshot.

        Returns:
            Immutable MetricsSnapshot instance.
        """
        with self._lock:
            requests = list(self._requests)
            batches = list(self._batches)
            curr_q = self._current_queue_depth
            peak_q = self._peak_queue_depth
            curr_act = self._current_active_requests
            peak_act = self._peak_active_requests
            created_at = self._created_at

        now = time.perf_counter()
        elapsed_sec = max(0.0, now - created_at)

        # 1. Request Statistics
        total_requests = len(requests)
        completed_requests = sum(1 for r in requests if r.status == RequestStatus.COMPLETED)
        failed_requests = sum(1 for r in requests if r.status == RequestStatus.FAILED)
        cancelled_requests = sum(1 for r in requests if r.status == RequestStatus.CANCELLED)

        avg_queue_wait = (
            sum(r.queue_wait_ms for r in requests) / total_requests if total_requests > 0 else 0.0
        )
        avg_exec = (
            sum(r.execution_ms for r in requests) / total_requests if total_requests > 0 else 0.0
        )
        avg_total = (
            sum(r.total_latency_ms for r in requests) / total_requests
            if total_requests > 0
            else 0.0
        )

        latencies = sorted([r.total_latency_ms for r in requests])
        min_total = latencies[0] if latencies else 0.0
        max_total = latencies[-1] if latencies else 0.0
        p50_total = _compute_percentile(latencies, 50.0)
        p95_total = _compute_percentile(latencies, 95.0)
        p99_total = _compute_percentile(latencies, 99.0)

        req_stats = RequestStats(
            total_requests=total_requests,
            completed_requests=completed_requests,
            failed_requests=failed_requests,
            cancelled_requests=cancelled_requests,
            avg_queue_wait_ms=avg_queue_wait,
            avg_execution_ms=avg_exec,
            avg_total_latency_ms=avg_total,
            min_total_latency_ms=min_total,
            max_total_latency_ms=max_total,
            p50_total_latency_ms=p50_total,
            p95_total_latency_ms=p95_total,
            p99_total_latency_ms=p99_total,
        )

        # 2. Batch Statistics
        total_batches = len(batches)
        completed_batches = sum(1 for b in batches if b.failed_request_count == 0)
        failed_batches = sum(1 for b in batches if b.failed_request_count > 0)

        avg_batch_size = sum(b.size for b in batches) / total_batches if total_batches > 0 else 0.0
        min_batch_size = min((b.size for b in batches), default=0)
        max_batch_size = max((b.size for b in batches), default=0)
        avg_batch_wait = (
            sum(b.batch_formation_wait_ms for b in batches) / total_batches
            if total_batches > 0
            else 0.0
        )
        avg_batch_exec = (
            sum(b.execution_ms for b in batches) / total_batches if total_batches > 0 else 0.0
        )

        batch_stats = BatchStats(
            total_batches=total_batches,
            completed_batches=completed_batches,
            failed_batches=failed_batches,
            avg_batch_size=avg_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            avg_batch_formation_wait_ms=avg_batch_wait,
            avg_batch_execution_ms=avg_batch_exec,
        )

        # 3. Throughput Statistics
        req_per_sec = (completed_requests / elapsed_sec) if elapsed_sec > 0 else 0.0
        batch_per_sec = (total_batches / elapsed_sec) if elapsed_sec > 0 else 0.0

        total_input_tokens = sum(r.input_tokens or 0 for r in requests)
        total_output_tokens = sum(r.output_tokens or 0 for r in requests)
        total_tokens = total_input_tokens + total_output_tokens
        tokens_per_sec = (total_tokens / elapsed_sec) if elapsed_sec > 0 else 0.0

        throughput_stats = ThroughputStats(
            elapsed_sec=elapsed_sec,
            requests_per_sec=req_per_sec,
            batches_per_sec=batch_per_sec,
            tokens_per_sec=tokens_per_sec,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )

        # 4. Queue Statistics
        queue_stats = QueueStats(
            current_queue_depth=curr_q,
            peak_queue_depth=peak_q,
            current_active_requests=curr_act,
            peak_active_requests=peak_act,
        )

        return MetricsSnapshot(
            timestamp=now,
            requests=req_stats,
            batches=batch_stats,
            throughput=throughput_stats,
            queue=queue_stats,
            metadata=dict(metadata or {}),
        )
