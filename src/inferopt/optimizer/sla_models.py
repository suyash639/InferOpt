"""Domain models for SLA-aware online adaptive control and Pareto latency guardrailing."""

import time
import uuid
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inferopt.benchmarks.vllm_validation import VLLMEnvironmentMetadata
from inferopt.optimizer.models import TunableConfig

DEFAULT_HEADROOM_RATIO: Final[float] = 0.75
DEFAULT_MITIGATION_DWELL_SEC: Final[float] = 0.5


class Step14SLAAdaptationEventRecord(BaseModel):
    """Immutable record of an individual online SLA adaptation event during execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(description="Unique adaptation event identifier")
    timestamp_offset_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Elapsed seconds from experiment start when adaptation triggered",
    )
    phase_index: int = Field(
        default=0,
        ge=0,
        description="Workload phase index during which event occurred",
    )
    phase_name: str = Field(default="", description="Name of active workload phase")
    mode: str = Field(description="SLA mode triggering adaptation (EXPAND, MITIGATE, HOLD)")
    old_config: str = Field(description="Configuration prior to adaptation")
    new_config: str = Field(description="Newly applied configuration")
    observed_p95_ms: float = Field(
        ge=0.0,
        description="Observed sliding-window p95 latency at adaptation",
    )
    target_slo_p95_ms: float = Field(gt=0.0, description="Target p95 latency SLO threshold in ms")
    observed_queue_wait_ms: float = Field(
        default=0.0, ge=0.0, description="Observed queue wait time at adaptation"
    )
    reason: str = Field(description="Deterministic reason justifying the adaptation")
    time_to_detect_ms: float = Field(
        default=0.0, ge=0.0, description="Time from condition onset to detection in ms"
    )
    time_to_adapt_ms: float = Field(
        default=0.0, ge=0.0, description="Time to apply configuration change in ms"
    )
    in_flight_requests_at_change: int = Field(
        default=0, ge=0, description="Active requests in execution when configuration changed"
    )
    queue_depth_at_change: int = Field(
        default=0, ge=0, description="Pending queue depth when configuration changed"
    )


class SLAMode(StrEnum):
    """Operational mode of the SLA-constrained adaptive controller."""

    EXPAND = "EXPAND"  # Headroom available (p95 < alpha * SLO), safe to scale throughput
    HOLD = "HOLD"  # Within deadband (alpha * SLO <= p95 <= SLO), steady state
    MITIGATE = "MITIGATE"  # Approaching or violating SLO (p95 > SLO), throttle down


class TargetSLO(BaseModel):
    """Target Service Level Objective specification for runtime latency guardrailing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    p95_latency_ms: float = Field(
        gt=0.0,
        description="Target maximum allowable p95 total latency in milliseconds.",
    )
    p99_latency_ms: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional target maximum allowable p99 total latency in milliseconds.",
    )
    max_queue_wait_ms: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional maximum allowable average queue wait time in milliseconds.",
    )
    headroom_ratio: float = Field(
        default=DEFAULT_HEADROOM_RATIO,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of target SLO below which capacity expansion is permitted (e.g. 0.75)."
        ),
    )

    @model_validator(mode="after")
    def validate_slo_hierarchy(self) -> "TargetSLO":
        """Verify that p99 latency target is greater than or equal to p95 if specified."""
        if self.p99_latency_ms is not None and self.p99_latency_ms < self.p95_latency_ms:
            raise ValueError(
                f"p99_latency_ms ({self.p99_latency_ms}ms) must be >= "
                f"p95_latency_ms ({self.p95_latency_ms}ms)"
            )
        return self


class SLAStatusRecord(BaseModel):
    """Real-time telemetry evaluation against target Service Level Objective."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this SLA evaluation event.",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when this SLA evaluation occurred.",
    )
    target_p95_ms: float = Field(
        gt=0.0,
        description="Target p95 latency SLO threshold in ms.",
    )
    observed_p95_ms: float = Field(
        ge=0.0,
        description="Observed sliding-window p95 total latency in ms.",
    )
    observed_queue_wait_ms: float = Field(
        ge=0.0,
        description="Observed sliding-window average queue wait time in ms.",
    )
    headroom_ms: float = Field(
        description="Absolute latency headroom in ms (target_p95 - observed_p95).",
    )
    mode: SLAMode = Field(
        description="Resulting SLA controller operational mode.",
    )
    slo_violated: bool = Field(
        description="True if observed p95 exceeds target p95 SLO.",
    )
    active_config: TunableConfig = Field(
        description="Active scheduler tunable configuration at time of evaluation.",
    )
    recommended_config: TunableConfig = Field(
        description="Recommended tunable configuration produced by SLA controller.",
    )
    reason: str = Field(
        description="Human-readable rationale for the SLA mode classification and decision.",
    )


class Step14PhaseMetricRecord(BaseModel):
    """Comprehensive performance and SLA telemetry for an individual benchmark phase."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase_index: int = Field(ge=0, description="0-indexed phase position")
    phase_name: str = Field(description="Descriptive identifier for the workload phase")
    target_slo_p95_ms: float = Field(gt=0.0, description="Target p95 SLO in ms")
    active_config: str = Field(description="Scheduler configuration string (e.g., 'c=1, b=2')")
    scheduled_requests: int = Field(ge=0, description="Requests scheduled in this phase")
    total_requests: int = Field(ge=0, description="Total requests planned in phase")
    completed_requests: int = Field(ge=0, description="Requests successfully completed")
    failed_requests: int = Field(default=0, ge=0, description="Requests failed")
    measured_requests: int = Field(ge=0, description="Requests measured for latency")
    backend_generate_calls: int = Field(ge=0, description="Single generate calls to backend")
    backend_generate_batch_calls: int = Field(ge=0, description="Batched generate calls to backend")
    duration_sec: float = Field(
        gt=0.0, description="Phase wall-clock execution duration in seconds"
    )
    throughput_rps: float = Field(ge=0.0, description="Phase throughput in req/s")
    output_tokens_per_sec: float = Field(ge=0.0, description="Generated tokens throughput")
    total_tokens_per_sec: float = Field(ge=0.0, description="Total tokens throughput")
    mean_latency_ms: float = Field(ge=0.0, description="Mean total end-to-end latency in ms")
    p50_latency_ms: float = Field(ge=0.0, description="p50 total latency in ms")
    p90_latency_ms: float = Field(ge=0.0, description="p90 total latency in ms")
    p95_latency_ms: float = Field(ge=0.0, description="p95 total latency in ms")
    p99_latency_ms: float = Field(ge=0.0, description="p99 total latency in ms")
    avg_queue_wait_ms: float = Field(ge=0.0, description="Average queue wait duration in ms")
    avg_backend_execution_ms: float = Field(ge=0.0, description="Average backend execution in ms")
    peak_queue_depth: int = Field(ge=0, description="Peak queue backlog observed in phase")
    total_batches: int = Field(ge=0, description="Total batches executed in phase")
    avg_batch_size: float = Field(ge=0.0, description="Average batch size in phase")
    sla_violation_count: int = Field(ge=0, description="Count of requests exceeding target p95 SLO")
    sla_violation_rate_pct: float = Field(ge=0.0, le=100.0, description="SLA violation percentage")
    adaptation_count_in_phase: int = Field(ge=0, description="Adaptations applied during phase")
    integrity_valid: bool = Field(description="True if scheduled == completed == measured")


class Step14ConditionSummary(BaseModel):
    """Aggregated performance summary across all workload phases for a test condition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition_name: str = Field(description="Condition identifier (e.g. SLA_AWARE_ADAPTIVE)")
    condition_type: str = Field(
        description="Type: 'static_conservative', 'static_aggressive', 'sla_adaptive'"
    )
    target_slo_p95_ms: float = Field(gt=0.0, description="Evaluated target p95 SLO in ms")
    scheduled_requests: int = Field(ge=0, description="Total scheduled requests")
    total_requests: int = Field(ge=0, description="Total requests")
    completed_requests: int = Field(ge=0, description="Total completed requests")
    failed_requests: int = Field(default=0, ge=0, description="Total failed requests")
    measured_requests: int = Field(ge=0, description="Total measured requests")
    backend_generate_calls: int = Field(ge=0, description="Total single generate backend calls")
    backend_generate_batch_calls: int = Field(
        ge=0, description="Total batched generate backend calls"
    )
    total_duration_sec: float = Field(gt=0.0, description="Total wall-clock duration in seconds")
    overall_throughput_rps: float = Field(ge=0.0, description="Overall throughput in req/s")
    overall_p95_latency_ms: float = Field(ge=0.0, description="Overall p95 total latency in ms")
    overall_p99_latency_ms: float = Field(ge=0.0, description="Overall p99 total latency in ms")
    mean_queue_wait_ms: float = Field(ge=0.0, description="Mean queue wait time in ms")
    mean_backend_execution_ms: float = Field(
        ge=0.0, description="Mean backend execution time in ms"
    )
    total_sla_violations: int = Field(ge=0, description="Total requests exceeding target p95 SLO")
    overall_sla_violation_rate_pct: float = Field(
        ge=0.0, le=100.0, description="Overall SLA violation %"
    )
    phase_metrics: tuple[Step14PhaseMetricRecord, ...] = Field(
        description="Per-phase metric records"
    )
    adaptation_events: tuple[Step14SLAAdaptationEventRecord, ...] = Field(
        default_factory=tuple, description="Chronological adaptation events recorded"
    )
    total_adaptations: int = Field(default=0, ge=0, description="Total adaptations executed")
    oscillation_count: int = Field(default=0, ge=0, description="Count of direction reversals")
    engine_initialization_count: int = Field(default=1, description="Engine initializations")
    engine_teardown_count: int = Field(default=0, description="Engine teardowns")
    engine_instance_id: str = Field(default="unknown", description="Engine instance identifier")
    integrity_valid: bool = Field(description="True if 100% integrity gate passed")


class Step14BaselineComparison(BaseModel):
    """Comparative evaluation between SLA-Adaptive InferOpt and Static Baselines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_slo_p95_ms: float = Field(description="Evaluated target p95 SLO in ms")
    static_conservative_tput: float = Field(description="Conservative baseline throughput")
    static_conservative_p95: float = Field(description="Conservative baseline p95 latency")
    static_conservative_sla_violation_pct: float = Field(
        description="Conservative baseline SLA violation %"
    )
    static_aggressive_tput: float = Field(description="Aggressive baseline throughput")
    static_aggressive_p95: float = Field(description="Aggressive baseline p95 latency")
    static_aggressive_sla_violation_pct: float = Field(
        description="Aggressive baseline SLA violation %"
    )
    sla_adaptive_tput: float = Field(description="SLA-Adaptive InferOpt throughput")
    sla_adaptive_p95: float = Field(description="SLA-Adaptive InferOpt p95 latency")
    sla_adaptive_sla_violation_pct: float = Field(
        description="SLA-Adaptive InferOpt SLA violation %"
    )
    throughput_improvement_vs_conservative_pct: float = Field(
        description="Throughput delta % vs Static Conservative"
    )
    sla_violation_reduction_vs_aggressive_pct: float = Field(
        description="SLA violation rate reduction vs Static Aggressive"
    )
    total_adaptations: int = Field(ge=0, description="Total adaptations executed in adaptive run")
    avg_time_to_detect_ms: float = Field(ge=0.0, description="Average time-to-detect in ms")
    avg_time_to_adapt_ms: float = Field(ge=0.0, description="Average time-to-adapt in ms")


class Step14SLAReport(BaseModel):
    """Complete, standalone, machine-readable Step 14 SLA-Aware Adaptive Control report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Unique Step 14 experiment identifier")
    timestamp: float = Field(description="Experiment execution timestamp")
    git_commit: str = Field(description="Git commit hash")
    model_id: str = Field(description="Evaluated model ID")
    backend: str = Field(default="vllm", description="Inference backend implementation identifier")
    backend_execution_confirmed: bool = Field(
        default=False, description="True ONLY if verified execution occurred on real GPU engine"
    )
    engine_instance_id: str = Field(
        default="unknown", description="Unique backend engine instance identifier"
    )
    engine_initialization_count: int = Field(
        default=1, description="Engine initializations across run"
    )
    engine_teardown_count: int = Field(default=1, description="Engine teardowns across run")
    scheduled_requests: int = Field(default=0, ge=0, description="Total scheduled requests in run")
    completed_requests: int = Field(default=0, ge=0, description="Total completed requests in run")
    failed_requests: int = Field(default=0, ge=0, description="Total failed requests in run")
    measured_requests: int = Field(default=0, ge=0, description="Total measured requests in run")
    backend_generate_calls: int = Field(
        default=0, ge=0, description="Total single-request backend generate calls in run"
    )
    backend_generate_batch_calls: int = Field(
        default=0, ge=0, description="Total batched backend generate_batch calls in run"
    )
    environment: VLLMEnvironmentMetadata = Field(description="Hardware and runtime environment")
    target_slo: TargetSLO = Field(description="Target SLO used for closed-loop control")
    phase_sequence: tuple[str, ...] = Field(description="Evaluated phase sequence names")
    num_requests_per_phase: int = Field(ge=1, description="Requests per phase")
    conditions: dict[str, Step14ConditionSummary] = Field(
        description="Per-condition evaluation summaries"
    )
    comparison: Step14BaselineComparison = Field(
        description="Comparison across SLA-adaptive and static conditions"
    )
    findings: dict[str, tuple[str, ...]] = Field(
        default_factory=dict, description="Categorized scientific findings"
    )
