"""Domain models for parameter spaces, objectives, constraints, and optimization results."""

import itertools
import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from inferopt.benchmarks.models import BenchmarkResult
from inferopt.scheduler.config import BatchConfig, SchedulerConfig


class TunableConfig(BaseModel):
    """Immutable representation of tunable scheduler and dynamic batching parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_concurrency: int = Field(
        default=4,
        gt=0,
        description="Maximum concurrent worker tasks executing batches.",
    )
    max_batch_size: int = Field(
        default=4,
        gt=0,
        description="Maximum number of requests grouped into a single batch.",
    )
    batch_wait_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Maximum batch formation wait window in milliseconds.",
    )

    def to_scheduler_config(
        self, max_queue_size: int = 100, max_history_size: int = 1000
    ) -> SchedulerConfig:
        """Convert tunable parameters into a complete SchedulerConfig instance."""
        return SchedulerConfig(
            max_concurrency=self.max_concurrency,
            max_queue_size=max_queue_size,
            max_history_size=max_history_size,
            batch_config=BatchConfig(
                max_batch_size=self.max_batch_size,
                batch_wait_ms=self.batch_wait_ms,
            ),
        )

    @classmethod
    def from_scheduler_config(cls, config: SchedulerConfig) -> "TunableConfig":
        """Extract tunable parameters from a SchedulerConfig."""
        return cls(
            max_concurrency=config.max_concurrency,
            max_batch_size=config.batch_config.max_batch_size,
            batch_wait_ms=config.batch_config.batch_wait_ms,
        )


class CandidateSpace(BaseModel):
    """Specification of candidate parameter values forming a grid-search space."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    concurrencies: tuple[int, ...] = Field(
        default=(1, 2, 4),
        description="Allowed concurrency values to evaluate.",
    )
    batch_sizes: tuple[int, ...] = Field(
        default=(1, 2, 4, 8),
        description="Allowed max_batch_size values to evaluate.",
    )
    batch_waits_ms: tuple[float, ...] = Field(
        default=(0.0, 5.0, 10.0, 25.0),
        description="Allowed batch_wait_ms values to evaluate.",
    )

    @field_validator("concurrencies")
    @classmethod
    def validate_concurrencies(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        """Ensure concurrency candidates are non-empty and strictly positive."""
        if not v:
            raise ValueError("concurrencies cannot be empty.")
        if any(c <= 0 for c in v):
            raise ValueError("All concurrency values must be > 0.")
        return tuple(sorted(set(v)))

    @field_validator("batch_sizes")
    @classmethod
    def validate_batch_sizes(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        """Ensure batch size candidates are non-empty and strictly positive."""
        if not v:
            raise ValueError("batch_sizes cannot be empty.")
        if any(b <= 0 for b in v):
            raise ValueError("All batch size values must be > 0.")
        return tuple(sorted(set(v)))

    @field_validator("batch_waits_ms")
    @classmethod
    def validate_batch_waits(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        """Ensure batch wait candidates are non-empty and non-negative."""
        if not v:
            raise ValueError("batch_waits_ms cannot be empty.")
        if any(w < 0.0 for w in v):
            raise ValueError("All batch wait values must be >= 0.0.")
        return tuple(sorted(set(v)))

    @property
    def total_candidates(self) -> int:
        """Total number of configurations in the Cartesian product search space."""
        return len(self.concurrencies) * len(self.batch_sizes) * len(self.batch_waits_ms)

    def generate_candidates(self) -> tuple[TunableConfig, ...]:
        """Generate deterministic Cartesian product of all valid parameter candidates."""
        candidates: list[TunableConfig] = [
            TunableConfig(
                max_concurrency=c,
                max_batch_size=b,
                batch_wait_ms=w,
            )
            for c, b, w in itertools.product(
                self.concurrencies, self.batch_sizes, self.batch_waits_ms
            )
        ]
        return tuple(candidates)


class OptimizationObjectiveType(StrEnum):
    """Supported optimization goals."""

    THROUGHPUT = "THROUGHPUT"
    LATENCY = "LATENCY"
    BALANCED = "BALANCED"


class ObjectiveConfig(BaseModel):
    """Specification of the optimization objective and associated weights."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_type: OptimizationObjectiveType = Field(
        default=OptimizationObjectiveType.THROUGHPUT,
        description="Target objective function (THROUGHPUT, LATENCY, BALANCED).",
    )
    throughput_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight assigned to throughput in BALANCED objective.",
    )
    latency_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight assigned to latency in BALANCED objective.",
    )
    target_p95_latency_ms: float | None = Field(
        default=None,
        gt=0.0,
        description="Reference p95 latency normalizer in milliseconds.",
    )
    target_throughput_rps: float | None = Field(
        default=None,
        gt=0.0,
        description="Reference throughput normalizer in requests per second.",
    )

    @field_validator("latency_weight")
    @classmethod
    def validate_weights(cls, v: float, info: Any) -> float:
        """Ensure sum of weights is positive for BALANCED objective."""
        t_weight = info.data.get("throughput_weight", 0.5)
        if t_weight + v <= 0.0:
            raise ValueError("Sum of throughput_weight and latency_weight must be > 0.0.")
        return v


class OptimizationConstraints(BaseModel):
    """Explicit SLA and resource boundaries that candidates must satisfy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_p95_latency_ms: float | None = Field(
        default=None,
        gt=0.0,
        description="Upper bound on allowable p95 turnaround latency in milliseconds.",
    )
    min_throughput_rps: float | None = Field(
        default=None,
        ge=0.0,
        description="Lower bound on allowable throughput in requests per second.",
    )
    max_batch_size: int | None = Field(
        default=None,
        gt=0,
        description="Upper bound on allowable max_batch_size configuration.",
    )
    max_concurrency: int | None = Field(
        default=None,
        gt=0,
        description="Upper bound on allowable max_concurrency configuration.",
    )
    max_batch_wait_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Upper bound on allowable batch_wait_ms configuration.",
    )
    max_failed_requests: int | None = Field(
        default=0,
        ge=0,
        description="Maximum allowed failed requests during benchmark run (default 0).",
    )

    def evaluate(self, config: TunableConfig, result: BenchmarkResult) -> tuple[str, ...]:
        """Evaluate candidate against all active constraints and return violation descriptions."""
        violations: list[str] = []

        if self.max_p95_latency_ms is not None and result.p95_latency_ms > self.max_p95_latency_ms:
            violations.append(
                f"p95 latency {result.p95_latency_ms:.2f}ms exceeds "
                f"max {self.max_p95_latency_ms:.2f}ms"
            )

        if (
            self.min_throughput_rps is not None
            and result.requests_per_sec < self.min_throughput_rps
        ):
            violations.append(
                f"Throughput {result.requests_per_sec:.2f} rps is below "
                f"min {self.min_throughput_rps:.2f} rps"
            )

        if self.max_batch_size is not None and config.max_batch_size > self.max_batch_size:
            violations.append(
                f"Batch size {config.max_batch_size} exceeds maximum {self.max_batch_size}"
            )

        if self.max_concurrency is not None and config.max_concurrency > self.max_concurrency:
            violations.append(
                f"Concurrency {config.max_concurrency} exceeds maximum {self.max_concurrency}"
            )

        if self.max_batch_wait_ms is not None and config.batch_wait_ms > self.max_batch_wait_ms:
            violations.append(
                f"Batch wait {config.batch_wait_ms:.2f}ms exceeds "
                f"max {self.max_batch_wait_ms:.2f}ms"
            )

        if (
            self.max_failed_requests is not None
            and result.failed_requests > self.max_failed_requests
        ):
            violations.append(
                f"Failed requests count {result.failed_requests} exceeds "
                f"limit {self.max_failed_requests}"
            )

        return tuple(violations)


class CandidateEvaluation(BaseModel):
    """Evaluation score, feasibility status, and metric attribution for a candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: TunableConfig = Field(
        ...,
        description="Evaluated candidate scheduler configuration.",
    )
    is_feasible: bool = Field(
        ...,
        description="True if candidate satisfies all constraints.",
    )
    objective_score: float = Field(
        ...,
        description="Computed objective score (higher is always better).",
    )
    constraint_violations: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Descriptions of any violated constraints.",
    )
    metrics: BenchmarkResult | None = Field(
        default=None,
        description="Underlying measured benchmark result, if evaluated historically.",
    )
    explanation: str = Field(
        default="",
        description="Human-readable rationale for score and feasibility outcome.",
    )


class OptimizationResult(BaseModel):
    """Immutable result of an optimization run including recommendation and evaluation ledger."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    optimization_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this optimization execution.",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Timestamp when optimization was performed.",
    )
    objective: ObjectiveConfig = Field(
        ...,
        description="Objective configuration used for evaluation.",
    )
    constraints: OptimizationConstraints = Field(
        ...,
        description="Constraints applied during evaluation.",
    )
    recommended_config: TunableConfig | None = Field(
        default=None,
        description="Optimal feasible configuration recommended by engine, or None if infeasible.",
    )
    best_score: float | None = Field(
        default=None,
        description="Objective score of the recommended configuration.",
    )
    is_feasible: bool = Field(
        ...,
        description="True if at least one candidate satisfied all constraints.",
    )
    total_candidates: int = Field(
        ...,
        ge=0,
        description="Total candidate configurations evaluated.",
    )
    feasible_candidates: int = Field(
        ...,
        ge=0,
        description="Count of evaluated candidates that satisfied all constraints.",
    )
    evaluations: tuple[CandidateEvaluation, ...] = Field(
        ...,
        description="Detailed evaluation ledger for every tested candidate configuration.",
    )
    summary_explanation: str = Field(
        default="",
        description="Summary explanation of why the winner was selected or why run failed.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution context or environment metadata.",
    )
