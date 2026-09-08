"""Domain models for synthetic workload generation, arrival patterns, and benchmark results."""

import time
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from inferopt.core.models import InferenceRequest, InferenceResponse
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.telemetry.models import MetricsSnapshot


class ArrivalPattern(StrEnum):
    """Supported traffic arrival patterns for benchmarking."""

    SEQUENTIAL = "SEQUENTIAL"
    CONCURRENT = "CONCURRENT"
    FIXED_RATE = "FIXED_RATE"
    BURST = "BURST"


class PromptCategory(StrEnum):
    """Categorization of synthetic prompts by complexity and length."""

    SHORT = "SHORT"
    MEDIUM = "MEDIUM"
    LONG = "LONG"
    REASONING = "REASONING"
    SUMMARIZATION = "SUMMARIZATION"
    FACTUAL = "FACTUAL"


class WorkloadRequestSpec(BaseModel):
    """Specification of a single benchmark request to be submitted to InferOpt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the benchmark request.",
    )
    model: str = Field(
        default="benchmark-model",
        min_length=1,
        description="Target model identifier.",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="Prompt text to generate completions for.",
    )
    priority: int = Field(
        default=0,
        description="Priority level for scheduler dispatch.",
    )
    max_tokens: int = Field(
        default=128,
        gt=0,
        description="Maximum tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Generation temperature.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata or annotations.",
    )
    scheduled_delay_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Scheduled relative arrival delay in milliseconds from workload start.",
    )

    def to_inference_request(self) -> InferenceRequest:
        """Convert specification into a valid InferOpt domain InferenceRequest."""
        return InferenceRequest(
            request_id=self.request_id,
            model=self.model,
            prompt=self.prompt,
            priority=self.priority,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            metadata=dict(self.metadata),
        )


class WorkloadConfig(BaseModel):
    """Configuration governing the generation of synthetic benchmark workloads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_name: str = Field(
        ...,
        min_length=1,
        description="Identifier of the scenario (e.g. 'light', 'medium', 'burst').",
    )
    description: str = Field(
        default="",
        description="Human-readable summary of the workload purpose.",
    )
    num_requests: int = Field(
        default=10,
        gt=0,
        description="Total number of requests generated in the scenario.",
    )
    arrival_pattern: ArrivalPattern = Field(
        default=ArrivalPattern.CONCURRENT,
        description="Traffic arrival pattern (SEQUENTIAL, CONCURRENT, FIXED_RATE, BURST).",
    )
    arrival_rate_rps: float | None = Field(
        default=None,
        gt=0.0,
        description="Target request arrival rate in requests per second for FIXED_RATE pattern.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for deterministic reproducible prompt generation.",
    )
    prompt_categories: tuple[PromptCategory, ...] = Field(
        default=(PromptCategory.SHORT,),
        description="Set of prompt categories sampled during generation.",
    )
    max_tokens: int = Field(
        default=128,
        gt=0,
        description="Maximum generation tokens per request.",
    )
    priority_levels: tuple[int, ...] = Field(
        default=(0,),
        description="Set of priority values sampled during generation.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Generation temperature for sampled requests "
            "(defaults to 0.0 for deterministic benchmarking)."
        ),
    )
    concurrency: int | None = Field(
        default=None,
        gt=0,
        description="Optional client-side concurrency limit for submission.",
    )


class WorkloadScenario(BaseModel):
    """Immutable container holding a complete, reproducible sequence of benchmark requests."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_name: str = Field(
        ...,
        min_length=1,
        description="Scenario identifier.",
    )
    description: str = Field(
        default="",
        description="Scenario description.",
    )
    config: WorkloadConfig = Field(
        ...,
        description="Configuration used to generate this scenario.",
    )
    requests: tuple[WorkloadRequestSpec, ...] = Field(
        ...,
        description="Ordered sequence of request specifications comprising the workload.",
    )
    created_at: float = Field(
        default_factory=time.time,
        description="Timestamp when the scenario was generated.",
    )

    @property
    def total_requests(self) -> int:
        """Count of requests in this scenario."""
        return len(self.requests)

    def to_json(self, indent: int = 2) -> str:
        """Serialize scenario to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "WorkloadScenario":
        """Deserialize scenario from JSON string."""
        return cls.model_validate_json(json_str)

    def save_json(self, file_path: str | Path) -> None:
        """Persist scenario specification to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load_json(cls, file_path: str | Path) -> "WorkloadScenario":
        """Load and validate scenario specification from a JSON file."""
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            return cls.from_json(f.read())


class BenchmarkResult(BaseModel):
    """Structured immutable results of an executed benchmark scenario."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this benchmark execution run.",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Wall-clock timestamp when the benchmark was executed.",
    )
    scenario_name: str = Field(
        ...,
        min_length=1,
        description="Name of the evaluated workload scenario.",
    )
    workload_config: WorkloadConfig = Field(
        ...,
        description="Configuration parameters of the evaluated workload.",
    )
    backend_name: str = Field(
        ...,
        min_length=1,
        description="Name of the backend engine executed against.",
    )
    scheduler_config: SchedulerConfig = Field(
        ...,
        description="Scheduler configuration used during the benchmark.",
    )
    batch_config: BatchConfig = Field(
        ...,
        description="Batching configuration used during the benchmark.",
    )
    telemetry_snapshot: MetricsSnapshot = Field(
        ...,
        description="Full telemetry snapshot captured at the end of the benchmark run.",
    )
    duration_sec: float = Field(
        ...,
        ge=0.0,
        description="Total elapsed benchmark execution duration in seconds.",
    )
    total_requests: int = Field(
        ...,
        ge=0,
        description="Total requests submitted during the run.",
    )
    completed_requests: int = Field(
        ...,
        ge=0,
        description="Count of requests that completed successfully.",
    )
    failed_requests: int = Field(
        ...,
        ge=0,
        description="Count of requests that failed during execution.",
    )
    cancelled_requests: int = Field(
        ...,
        ge=0,
        description="Count of requests cancelled during execution.",
    )
    requests_per_sec: float = Field(
        ...,
        ge=0.0,
        description="Throughput in completed requests per second.",
    )
    batches_per_sec: float = Field(
        ...,
        ge=0.0,
        description="Throughput in executed batches per second.",
    )
    tokens_per_sec: float = Field(
        ...,
        ge=0.0,
        description="Aggregate token throughput per second.",
    )
    avg_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Mean turnaround latency in milliseconds.",
    )
    p50_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Median (50th percentile) latency in milliseconds.",
    )
    p95_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="95th percentile latency in milliseconds.",
    )
    p99_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="99th percentile latency in milliseconds.",
    )
    avg_queue_wait_ms: float = Field(
        ...,
        ge=0.0,
        description="Average queue wait duration in milliseconds.",
    )
    avg_execution_ms: float = Field(
        ...,
        ge=0.0,
        description="Average backend execution latency in milliseconds.",
    )
    peak_queue_depth: int = Field(
        ...,
        ge=0,
        description="Peak queue depth observed during the run.",
    )
    peak_active_requests: int = Field(
        ...,
        ge=0,
        description="Peak concurrent requests executing on the backend.",
    )
    total_batches: int = Field(
        ...,
        ge=0,
        description="Total number of formed batches executed.",
    )
    avg_batch_size: float = Field(
        ...,
        ge=0.0,
        description="Average batch size formed during the run.",
    )
    max_batch_size: int = Field(
        ...,
        ge=0,
        description="Maximum batch size formed during the run.",
    )
    responses: tuple[InferenceResponse, ...] = Field(
        default_factory=tuple,
        description="Completed inference response objects captured during the run.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution environment or reproducibility metadata.",
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialize benchmark result to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "BenchmarkResult":
        """Deserialize benchmark result from JSON string."""
        return cls.model_validate_json(json_str)

    def save_json(self, file_path: str | Path) -> None:
        """Persist benchmark result to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load_json(cls, file_path: str | Path) -> "BenchmarkResult":
        """Load and validate benchmark result from a JSON file."""
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            return cls.from_json(f.read())
