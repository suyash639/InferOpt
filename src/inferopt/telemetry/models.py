"""Domain models for inference telemetry, metrics collection, and snapshots."""

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from inferopt.scheduler.lifecycle import RequestStatus


class RequestMetrics(BaseModel):
    """Immutable metric record for an individual inference request.

    Captures lifecycle durations, queue waiting times, execution latency,
    token counts, and terminal status.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the measured request.",
    )
    priority: int = Field(
        default=0,
        description="Priority level assigned to the request at submission time.",
    )
    queue_wait_ms: float = Field(
        ...,
        ge=0.0,
        description="Time spent waiting in queue before execution dispatch (ms).",
    )
    execution_ms: float = Field(
        ...,
        ge=0.0,
        description="Time spent executing on the backend (ms).",
    )
    total_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="End-to-end turnaround latency from queue submission to completion (ms).",
    )
    status: RequestStatus = Field(
        ...,
        description="Terminal lifecycle status of the request (COMPLETED, FAILED, CANCELLED).",
    )
    input_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Estimated or measured input prompt token count, if available.",
    )
    output_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Generated output token count, if available.",
    )
    max_tokens: int = Field(
        ...,
        gt=0,
        description="Configured max_tokens parameter for this request.",
    )
    backend_name: str = Field(
        ...,
        min_length=1,
        description="Name of the backend engine that processed or attempted the request.",
    )
    batch_id: str | None = Field(
        default=None,
        description="Identifier of the batch this request executed in, if batched.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error detail message if the request failed or was cancelled.",
    )
    recorded_at: float = Field(
        default_factory=time.perf_counter,
        description="Monotonic timestamp recorded via time.perf_counter() at metric capture.",
    )


class BatchMetrics(BaseModel):
    """Immutable metric record for an executed batch of inference requests.

    Captures batch formation wait window, backend execution duration, batch size,
    and completion/failure counts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the formed batch.",
    )
    size: int = Field(
        ...,
        gt=0,
        description="Number of requests included in this batch.",
    )
    batch_formation_wait_ms: float = Field(
        ...,
        ge=0.0,
        description="Time spent waiting in the batch formation window (ms).",
    )
    execution_ms: float = Field(
        ...,
        ge=0.0,
        description="Time spent executing the batch on the backend (ms).",
    )
    total_max_tokens: int = Field(
        ...,
        ge=0,
        description="Sum of max_tokens limits across all requests in the batch.",
    )
    backend_name: str = Field(
        ...,
        min_length=1,
        description="Name of the backend engine executing the batch.",
    )
    request_ids: tuple[str, ...] = Field(
        ...,
        description="Identifiers of requests comprising this batch.",
    )
    completed_request_count: int = Field(
        ...,
        ge=0,
        description="Number of requests in the batch that completed successfully.",
    )
    failed_request_count: int = Field(
        ...,
        ge=0,
        description="Number of requests in the batch that failed during execution.",
    )
    recorded_at: float = Field(
        default_factory=time.perf_counter,
        description="Monotonic timestamp recorded via time.perf_counter() at metric capture.",
    )


class RequestStats(BaseModel):
    """Aggregated request performance statistics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_requests: int = Field(default=0, ge=0)
    completed_requests: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)
    cancelled_requests: int = Field(default=0, ge=0)
    avg_queue_wait_ms: float = Field(default=0.0, ge=0.0)
    avg_execution_ms: float = Field(default=0.0, ge=0.0)
    avg_total_latency_ms: float = Field(default=0.0, ge=0.0)
    min_total_latency_ms: float = Field(default=0.0, ge=0.0)
    max_total_latency_ms: float = Field(default=0.0, ge=0.0)
    p50_total_latency_ms: float = Field(default=0.0, ge=0.0)
    p95_total_latency_ms: float = Field(default=0.0, ge=0.0)
    p99_total_latency_ms: float = Field(default=0.0, ge=0.0)


class BatchStats(BaseModel):
    """Aggregated batch formation and execution statistics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_batches: int = Field(default=0, ge=0)
    completed_batches: int = Field(default=0, ge=0)
    failed_batches: int = Field(default=0, ge=0)
    avg_batch_size: float = Field(default=0.0, ge=0.0)
    min_batch_size: int = Field(default=0, ge=0)
    max_batch_size: int = Field(default=0, ge=0)
    avg_batch_formation_wait_ms: float = Field(default=0.0, ge=0.0)
    avg_batch_execution_ms: float = Field(default=0.0, ge=0.0)


class ThroughputStats(BaseModel):
    """Throughput and token performance metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    elapsed_sec: float = Field(default=0.0, ge=0.0)
    requests_per_sec: float = Field(default=0.0, ge=0.0)
    batches_per_sec: float = Field(default=0.0, ge=0.0)
    tokens_per_sec: float = Field(default=0.0, ge=0.0)
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)


class QueueStats(BaseModel):
    """Real-time and peak queue and concurrency observations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    current_queue_depth: int = Field(default=0, ge=0)
    peak_queue_depth: int = Field(default=0, ge=0)
    current_active_requests: int = Field(default=0, ge=0)
    peak_active_requests: int = Field(default=0, ge=0)


class MetricsSnapshot(BaseModel):
    """Point-in-time immutable snapshot of all collected telemetry metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float = Field(
        default_factory=time.perf_counter,
        description="Monotonic timestamp when this snapshot was computed.",
    )
    requests: RequestStats = Field(default_factory=RequestStats)
    batches: BatchStats = Field(default_factory=BatchStats)
    throughput: ThroughputStats = Field(default_factory=ThroughputStats)
    queue: QueueStats = Field(default_factory=QueueStats)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata or tags attached to the snapshot.",
    )
