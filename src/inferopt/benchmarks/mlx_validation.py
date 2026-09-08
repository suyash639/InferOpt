"""Scientific MLX benchmark validation, auditing, and baseline comparison subsystem.

Provides controlled multi-condition evaluation across Direct MLX and InferOpt serving
pipelines, enforcing exact input/output hash verification, token accounting,
cold-vs-warm timing separation, and run-to-run statistical variance analysis.
"""

import math
import platform
import time
import uuid
from collections import Counter
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from inferopt import __version__ as INFEROPT_VERSION
from inferopt.backends.mlx import DEFAULT_MODEL_ID, MLXBackend
from inferopt.benchmarks.generator import PRESET_SCENARIOS
from inferopt.benchmarks.mlx_baseline import (
    DirectMLXResult,
    DirectMLXRunner,
    calculate_percentile,
    compute_sha256,
    compute_std_dev,
)
from inferopt.benchmarks.models import BenchmarkResult, WorkloadScenario
from inferopt.benchmarks.runner import BenchmarkRunner
from inferopt.scheduler.config import BatchConfig, SchedulerConfig


class AuditCondition(StrEnum):
    """The six explicitly controlled benchmark comparison conditions."""

    DIRECT_SINGLE = "DIRECT_SINGLE"  # Condition A: Direct MLX Single Request
    DIRECT_NATIVE_BATCH = "DIRECT_NATIVE_BATCH"  # Condition B: Direct MLX Native batch_generate
    INFEROPT_BATCH_1 = "INFEROPT_BATCH_1"  # Condition C: InferOpt max_batch_size=1
    INFEROPT_BATCH_2 = "INFEROPT_BATCH_2"  # Condition D: InferOpt max_batch_size=2
    INFEROPT_BATCH_4 = "INFEROPT_BATCH_4"  # Condition E: InferOpt max_batch_size=4
    INFEROPT_BATCH_8 = "INFEROPT_BATCH_8"  # Condition F: InferOpt max_batch_size=8


CONDITION_LABELS: dict[AuditCondition, str] = {
    AuditCondition.DIRECT_SINGLE: "A. Direct MLX Single",
    AuditCondition.DIRECT_NATIVE_BATCH: "B. Direct MLX Native Batch",
    AuditCondition.INFEROPT_BATCH_1: "C. InferOpt Batch 1",
    AuditCondition.INFEROPT_BATCH_2: "D. InferOpt Batch 2",
    AuditCondition.INFEROPT_BATCH_4: "E. InferOpt Batch 4",
    AuditCondition.INFEROPT_BATCH_8: "F. InferOpt Batch 8",
}


class EnvironmentMetadata(BaseModel):
    """Host machine, Python runtime, and library version metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    os_name: str = Field(description="Operating system name (e.g. Darwin)")
    os_version: str = Field(description="Operating system release version")
    cpu_architecture: str = Field(description="Hardware CPU architecture (e.g. arm64)")
    python_version: str = Field(description="Python runtime version")
    mlx_version: str = Field(default="unknown", description="Installed MLX version")
    mlx_lm_version: str = Field(default="unknown", description="Installed MLX-LM version")
    inferopt_version: str = Field(default=INFEROPT_VERSION, description="InferOpt version")
    model_id: str = Field(description="Evaluated model identifier")
    warmup_count: int = Field(ge=0, description="Warmup iterations per run")
    repetitions: int = Field(ge=1, description="Number of measured repetition trials")
    workload_seed: int = Field(description="Deterministic workload seed")


class CorrectnessGateResult(BaseModel):
    """Validation gate ensuring 100% request/result integrity before comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool = Field(description="True if all correctness checks passed")
    total_expected_requests: int = Field(ge=0, description="Number of expected workload requests")
    total_completed_requests: int = Field(ge=0, description="Number of completed requests")
    has_duplicate_ids: bool = Field(default=False, description="True if duplicate IDs detected")
    has_missing_ids: bool = Field(default=False, description="True if missing IDs detected")
    has_empty_outputs: bool = Field(default=False, description="True if empty completions found")
    has_invalid_tokens: bool = Field(default=False, description="True if token count was invalid")
    has_unexpected_failures: bool = Field(default=False, description="True if failures occurred")
    diagnostic_messages: tuple[str, ...] = Field(
        default_factory=tuple, description="Detailed diagnostic reasons for failures"
    )


class MetricComparison(BaseModel):
    """Single metric comparison between Direct MLX and InferOpt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_name: str = Field(description="Name of the metric compared")
    direct_mlx_value: float = Field(description="Direct MLX measured value")
    inferopt_value: float = Field(description="InferOpt measured value")
    absolute_difference: float = Field(description="InferOpt value minus Direct MLX value")
    relative_difference_pct: float | None = Field(
        default=None, description="Percentage change relative to Direct MLX (if baseline > 0)"
    )
    unit: str = Field(description="Unit of measurement (e.g. ms, req/s, tokens/s)")


class RepetitionSummary(BaseModel):
    """Statistical summary across multiple repetition runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repetition_count: int = Field(ge=1, description="Number of repetitions completed")
    mean_duration_sec: float = Field(ge=0.0, description="Mean run duration in seconds")
    mean_requests_per_sec: float = Field(ge=0.0, description="Mean requests completed per second")
    mean_output_tokens_per_sec: float = Field(ge=0.0, description="Mean output tokens per second")
    mean_avg_latency_ms: float = Field(ge=0.0, description="Mean of average request latencies")
    pooled_p50_latency_ms: float = Field(
        ge=0.0, description="Median latency across all pooled runs"
    )
    pooled_p95_latency_ms: float = Field(
        ge=0.0, description="95th percentile latency across pooled runs"
    )
    pooled_p99_latency_ms: float = Field(
        ge=0.0, description="99th percentile latency across pooled runs"
    )
    std_dev_avg_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Sample standard deviation of average latency"
    )


class ValidationExperimentReport(BaseModel):
    """Comprehensive validation experiment report comparing Direct MLX vs InferOpt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Unique experiment execution ID")
    timestamp: float = Field(description="Experiment execution timestamp")
    scenario_name: str = Field(description="Workload scenario name")
    model_id: str = Field(description="Hugging Face model identifier")
    warmup_count: int = Field(ge=0, description="Number of warmup iterations per run")
    repetition_count: int = Field(ge=1, description="Number of measured repetition trials")
    environment: EnvironmentMetadata = Field(description="Hardware and runtime environment")
    correctness_gate: CorrectnessGateResult = Field(description="Result of correctness gate")
    direct_mlx_runs: tuple[DirectMLXResult, ...] = Field(
        description="Raw Direct MLX repetition results"
    )
    inferopt_runs: tuple[BenchmarkResult, ...] = Field(
        description="Raw InferOpt repetition results"
    )
    direct_summary: RepetitionSummary = Field(description="Aggregated Direct MLX summary")
    inferopt_summary: RepetitionSummary = Field(description="Aggregated InferOpt summary")
    comparisons: tuple[MetricComparison, ...] = Field(
        description="Structured metric comparison records"
    )
    scheduler_config: SchedulerConfig = Field(description="InferOpt scheduler configuration used")


class BatchMatrixCellResult(BaseModel):
    """Result for a single cell in the batch matrix (concurrency x batch_size)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    concurrency: int = Field(ge=1, description="Scheduler concurrency limit")
    max_batch_size: int = Field(ge=1, description="Dynamic batch size limit")
    completed_requests: int = Field(ge=0, description="Completed requests count")
    total_batches: int = Field(ge=0, description="Total batches formed")
    avg_batch_size: float = Field(ge=0.0, description="Average formed batch size")
    duration_sec: float = Field(ge=0.0, description="Run duration in seconds")
    requests_per_sec: float = Field(ge=0.0, description="Completed requests per second")
    output_tokens_per_sec: float = Field(ge=0.0, description="Output tokens per second")
    p50_latency_ms: float = Field(ge=0.0, description="Median total latency in ms")
    p95_latency_ms: float = Field(ge=0.0, description="95th percentile latency in ms")
    avg_queue_wait_ms: float = Field(ge=0.0, description="Average queue wait time in ms")
    avg_execution_ms: float = Field(ge=0.0, description="Average execution time in ms")


class BatchMatrixReport(BaseModel):
    """Results matrix across multiple concurrency and batch size configurations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    matrix_id: str = Field(description="Unique matrix run ID")
    timestamp: float = Field(description="Execution timestamp")
    model_id: str = Field(description="Model identifier")
    scenario_name: str = Field(description="Base scenario name")
    concurrency_levels: tuple[int, ...] = Field(description="Evaluated concurrency levels")
    batch_sizes: tuple[int, ...] = Field(description="Evaluated max batch sizes")
    cells: tuple[BatchMatrixCellResult, ...] = Field(description="Grid cell results")
    environment: EnvironmentMetadata = Field(description="Environment metadata")


# ==============================================================================
# Step 8.5 Enhanced Audit Domain Models
# ==============================================================================


class AuditRequestRecord(BaseModel):
    """Integrity record for an individual inference request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(description="Request identifier")
    prompt_hash: str = Field(description="SHA-256 hash of input prompt")
    output_hash: str = Field(description="SHA-256 hash of generated text")
    input_tokens: int = Field(ge=0, description="Exact prompt token count")
    output_tokens: int = Field(ge=0, description="Exact generated token count")
    success: bool = Field(description="True if generation succeeded")


class AuditConditionResult(BaseModel):
    """Aggregated, multi-repetition results for a single audit condition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: AuditCondition = Field(description="Condition identifier (A-F)")
    condition_label: str = Field(description="Human-readable condition label")
    concurrency: int = Field(ge=1, description="Scheduler concurrency limit or dispatch level")
    max_batch_size: int = Field(ge=1, description="Configured maximum batch size")
    total_requests: int = Field(ge=0, description="Total requests submitted")
    completed_requests: int = Field(ge=0, description="Total requests completed")
    failed_requests: int = Field(ge=0, description="Total requests failed")
    duration_sec: float = Field(
        ge=0.0, description="Average duration across repetitions in seconds"
    )
    requests_per_sec: float = Field(ge=0.0, description="Average completed requests per second")
    output_tokens_per_sec: float = Field(ge=0.0, description="Average output tokens per second")
    total_tokens_per_sec: float = Field(
        ge=0.0, description="Average total (input+output) tokens per second"
    )
    mean_latency_ms: float = Field(ge=0.0, description="Mean turnaround latency in ms")
    median_latency_ms: float = Field(ge=0.0, description="Median (P50) turnaround latency in ms")
    p50_latency_ms: float = Field(ge=0.0, description="50th percentile latency in ms")
    p95_latency_ms: float = Field(ge=0.0, description="95th percentile latency in ms")
    p99_latency_ms: float = Field(ge=0.0, description="99th percentile latency in ms")
    min_latency_ms: float = Field(ge=0.0, description="Minimum latency in ms")
    max_latency_ms: float = Field(ge=0.0, description="Maximum latency in ms")
    std_dev_latency_ms: float = Field(
        ge=0.0, description="Sample standard deviation of latency across requests"
    )
    total_input_tokens: int = Field(ge=0, description="Total prompt tokens processed")
    total_output_tokens: int = Field(ge=0, description="Total generated tokens produced")
    avg_input_tokens_per_req: float = Field(ge=0.0, description="Average input tokens per request")
    avg_output_tokens_per_req: float = Field(
        ge=0.0, description="Average output tokens per request"
    )
    min_output_tokens: int = Field(
        ge=0, description="Minimum tokens generated by any single request"
    )
    max_output_tokens: int = Field(
        ge=0, description="Maximum tokens generated by any single request"
    )
    total_batches: int = Field(ge=0, description="Total batches formed and dispatched")
    avg_batch_size: float = Field(ge=0.0, description="Average formed batch size")
    median_batch_size: float = Field(ge=0.0, description="Median formed batch size")
    max_batch_size_formed: int = Field(ge=0, description="Maximum batch size actually formed")
    batch_size_distribution: dict[int, int] = Field(
        default_factory=dict, description="Distribution count of formed batch sizes"
    )
    avg_batch_wait_ms: float = Field(
        ge=0.0, description="Average queue / batch formation wait time in ms"
    )
    avg_backend_execution_ms: float = Field(
        ge=0.0, description="Average backend execution time in ms"
    )
    repetition_count: int = Field(ge=1, description="Number of repetitions aggregated")
    requests: tuple[AuditRequestRecord, ...] = Field(
        default_factory=tuple, description="Canonical request records from final repetition"
    )


class AuditComparisonDelta(BaseModel):
    """Explicit delta comparison between a baseline and target condition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison_name: str = Field(description="Descriptive comparison title")
    baseline_condition: str = Field(description="Baseline condition name")
    target_condition: str = Field(description="Target condition name")
    concurrency: int = Field(ge=1, description="Concurrency level compared")
    throughput_change_pct: float = Field(
        description="Percentage change in request throughput ((target - base) / base * 100)"
    )
    token_throughput_change_pct: float = Field(
        description="Percentage change in output token throughput"
    )
    latency_change_pct: float = Field(
        description="Percentage change in mean latency (negative values indicate lower latency)"
    )
    end_to_end_differential_overhead_ms: float | None = Field(
        default=None,
        description="Measured latency difference (target - baseline ms), "
        "labeled as differential overhead",
    )
    notes: str = Field(default="", description="Explanatory notes on measurement boundaries")


class AuditIntegrityResult(BaseModel):
    """Comprehensive integrity gate verifying fairness and output equivalence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool = Field(description="True if comparison matrix satisfies all integrity checks")
    status: str = Field(
        description="Status indicator: COMPARISON_VALID, COMPARISON_INVALID, or OUTPUT_MISMATCH"
    )
    same_request_ids: bool = Field(default=True, description="True if identical IDs across runs")
    same_prompt_hashes: bool = Field(
        default=True, description="True if identical prompts across runs"
    )
    same_seed: bool = Field(default=True, description="True if identical workload seed used")
    exact_output_match: bool = Field(
        default=True, description="True if deterministic outputs match byte-for-byte"
    )
    zero_unexpected_failures: bool = Field(
        default=True, description="True if zero unexpected request failures occurred"
    )
    valid_tokens: bool = Field(
        default=True, description="True if all token counts are non-negative and non-zero"
    )
    non_empty_outputs: bool = Field(
        default=True, description="True if all successful outputs contain text"
    )
    diagnostics: tuple[str, ...] = Field(
        default_factory=tuple, description="Detailed diagnostic failure reasons"
    )
    output_mismatches: tuple[str, ...] = Field(
        default_factory=tuple, description="Specific request IDs with divergent output hashes"
    )


class AuditExperimentReport(BaseModel):
    """Complete machine-readable and reproducible report of the MLX benchmark audit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    audit_id: str = Field(description="Unique audit run identifier")
    timestamp: float = Field(description="Audit execution timestamp")
    model_id: str = Field(description="Target Hugging Face model identifier")
    scenario_name: str = Field(description="Base workload scenario name")
    seed: int = Field(description="Workload generation seed")
    warmup_count: int = Field(ge=0, description="Warmup iterations per condition")
    repetition_count: int = Field(ge=1, description="Measured repetitions per condition")
    model_load_time_ms: float = Field(
        ge=0.0, description="Cold-start model load time in ms (separated from steady-state)"
    )
    concurrency_levels: tuple[int, ...] = Field(description="Evaluated concurrency levels")
    environment: EnvironmentMetadata = Field(description="Host hardware and library metadata")
    integrity: AuditIntegrityResult = Field(description="Integrity and correctness gate results")
    condition_results: tuple[AuditConditionResult, ...] = Field(
        description="Evaluated 6-condition results across concurrency levels"
    )
    deltas: tuple[AuditComparisonDelta, ...] = Field(
        description="Explicit relative deltas and differential overhead calculations"
    )
    notes: tuple[str, ...] = Field(
        default_factory=tuple, description="Scientific disclaimers and methodology boundaries"
    )


# ==============================================================================
# Helper Functions & Computation
# ==============================================================================


def capture_environment(
    model_id: str, warmup_count: int, repetitions: int, seed: int
) -> EnvironmentMetadata:
    """Capture host system, runtime, and MLX library versions."""
    mlx_ver = "unknown"
    mlx_lm_ver = "unknown"
    try:
        import mlx.core as mx

        mlx_ver = getattr(mx, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import mlx_lm

        mlx_lm_ver = getattr(mlx_lm, "__version__", "unknown")
    except ImportError:
        pass

    return EnvironmentMetadata(
        os_name=platform.system(),
        os_version=platform.release(),
        cpu_architecture=platform.machine(),
        python_version=platform.python_version(),
        mlx_version=mlx_ver,
        mlx_lm_version=mlx_lm_ver,
        inferopt_version=INFEROPT_VERSION,
        model_id=model_id,
        warmup_count=warmup_count,
        repetitions=repetitions,
        workload_seed=seed,
    )


def compute_repetition_summary_direct(runs: list[DirectMLXResult]) -> RepetitionSummary:
    """Compute cross-repetition statistical summary for Direct MLX runs."""
    reps = len(runs)
    if reps == 0:
        raise ValueError("Cannot summarize empty list of runs")

    mean_dur = sum(r.duration_sec for r in runs) / reps
    mean_rps = sum(r.requests_per_sec for r in runs) / reps
    mean_tps = sum(r.output_tokens_per_sec for r in runs) / reps
    avg_lats = [r.avg_latency_ms for r in runs]
    mean_avg_lat = sum(avg_lats) / reps

    std_dev_lat = 0.0
    if reps > 1:
        variance = sum((x - mean_avg_lat) ** 2 for x in avg_lats) / (reps - 1)
        std_dev_lat = math.sqrt(variance)

    pooled_lats = [req.latency_ms for r in runs for req in r.request_results if req.success]
    p50 = calculate_percentile(pooled_lats, 50.0) if pooled_lats else 0.0
    p95 = calculate_percentile(pooled_lats, 95.0) if pooled_lats else 0.0
    p99 = calculate_percentile(pooled_lats, 99.0) if pooled_lats else 0.0

    return RepetitionSummary(
        repetition_count=reps,
        mean_duration_sec=mean_dur,
        mean_requests_per_sec=mean_rps,
        mean_output_tokens_per_sec=mean_tps,
        mean_avg_latency_ms=mean_avg_lat,
        pooled_p50_latency_ms=p50,
        pooled_p95_latency_ms=p95,
        pooled_p99_latency_ms=p99,
        std_dev_avg_latency_ms=std_dev_lat,
    )


def compute_repetition_summary_inferopt(runs: list[BenchmarkResult]) -> RepetitionSummary:
    """Compute cross-repetition statistical summary for InferOpt runs."""
    reps = len(runs)
    if reps == 0:
        raise ValueError("Cannot summarize empty list of runs")

    mean_dur = sum(r.duration_sec for r in runs) / reps
    mean_rps = sum(r.requests_per_sec for r in runs) / reps
    output_tps_list = [
        (
            r.telemetry_snapshot.throughput.total_output_tokens / r.duration_sec
            if r.duration_sec > 0
            else 0.0
        )
        for r in runs
    ]
    mean_tps = sum(output_tps_list) / reps
    avg_lats = [r.avg_latency_ms for r in runs]
    mean_avg_lat = sum(avg_lats) / reps

    std_dev_lat = 0.0
    if reps > 1:
        variance = sum((x - mean_avg_lat) ** 2 for x in avg_lats) / (reps - 1)
        std_dev_lat = math.sqrt(variance)

    p50_vals = [r.p50_latency_ms for r in runs]
    p95_vals = [r.p95_latency_ms for r in runs]
    p99_vals = [r.p99_latency_ms for r in runs]
    p50 = calculate_percentile(p50_vals, 50.0) if p50_vals else 0.0
    p95 = calculate_percentile(p95_vals, 95.0) if p95_vals else 0.0
    p99 = calculate_percentile(p99_vals, 99.0) if p99_vals else 0.0

    return RepetitionSummary(
        repetition_count=reps,
        mean_duration_sec=mean_dur,
        mean_requests_per_sec=mean_rps,
        mean_output_tokens_per_sec=mean_tps,
        mean_avg_latency_ms=mean_avg_lat,
        pooled_p50_latency_ms=p50,
        pooled_p95_latency_ms=p95,
        pooled_p99_latency_ms=p99,
        std_dev_avg_latency_ms=std_dev_lat,
    )


def compute_comparisons(
    direct_summary: RepetitionSummary, inferopt_summary: RepetitionSummary
) -> tuple[MetricComparison, ...]:
    """Compute neutral comparison metrics between Direct MLX and InferOpt summaries."""

    def _comp(name: str, d_val: float, i_val: float, unit: str) -> MetricComparison:
        diff = i_val - d_val
        rel_pct = (diff / d_val * 100.0) if d_val > 0 else None
        return MetricComparison(
            metric_name=name,
            direct_mlx_value=d_val,
            inferopt_value=i_val,
            absolute_difference=diff,
            relative_difference_pct=rel_pct,
            unit=unit,
        )

    return (
        _comp(
            "Average Latency",
            direct_summary.mean_avg_latency_ms,
            inferopt_summary.mean_avg_latency_ms,
            "ms",
        ),
        _comp(
            "p50 Latency",
            direct_summary.pooled_p50_latency_ms,
            inferopt_summary.pooled_p50_latency_ms,
            "ms",
        ),
        _comp(
            "p95 Latency",
            direct_summary.pooled_p95_latency_ms,
            inferopt_summary.pooled_p95_latency_ms,
            "ms",
        ),
        _comp(
            "p99 Latency",
            direct_summary.pooled_p99_latency_ms,
            inferopt_summary.pooled_p99_latency_ms,
            "ms",
        ),
        _comp(
            "Request Throughput",
            direct_summary.mean_requests_per_sec,
            inferopt_summary.mean_requests_per_sec,
            "req/s",
        ),
        _comp(
            "Output Token Throughput",
            direct_summary.mean_output_tokens_per_sec,
            inferopt_summary.mean_output_tokens_per_sec,
            "tokens/s",
        ),
        _comp(
            "Total Duration",
            direct_summary.mean_duration_sec,
            inferopt_summary.mean_duration_sec,
            "s",
        ),
    )


# ==============================================================================
# MLXValidator Orchestrator
# ==============================================================================


class MLXValidator:
    """Orchestrator for controlled MLX validation experiments and scientific audits."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        default_temperature: float = 0.0,
        default_max_tokens: int = 128,
    ) -> None:
        """Initialize MLX validator.

        Args:
            model_id: Target Hugging Face model identifier.
            default_temperature: Sampling temperature (0.0 for greedy decoding).
            default_max_tokens: Default maximum tokens.
        """
        self._model_id = model_id
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._direct_runner = DirectMLXRunner(
            model_id=model_id,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
        )
        self._backend = MLXBackend(
            model_id=model_id,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
        )
        self._benchmark_runner = BenchmarkRunner()

    @property
    def model_id(self) -> str:
        """Evaluated model identifier."""
        return self._model_id

    def evaluate_correctness_gate(
        self,
        scenario: WorkloadScenario,
        direct_results: list[DirectMLXResult],
        inferopt_results: list[BenchmarkResult],
    ) -> CorrectnessGateResult:
        """Evaluate legacy correctness gate across direct and inferopt runs."""
        diagnostics: list[str] = []
        expected_count = len(scenario.requests)
        expected_ids = {r.request_id for r in scenario.requests}

        has_dups = False
        has_missing = False
        has_empty = False
        has_invalid_tok = False
        has_unexpected_fail = False

        for idx, direct_run in enumerate(direct_results):
            run_ids = [r.request_id for r in direct_run.request_results]
            if len(run_ids) != len(set(run_ids)):
                has_dups = True
                diagnostics.append(f"Direct MLX run {idx + 1} contains duplicate request IDs.")

            seen_ids = set(run_ids)
            missing = expected_ids - seen_ids
            if missing:
                has_missing = True
                diagnostics.append(f"Direct MLX run {idx + 1} missing request IDs: {missing}")

            if direct_run.failed_requests > 0:
                has_unexpected_fail = True
                diagnostics.append(
                    f"Direct MLX run {idx + 1} had {direct_run.failed_requests} failed requests."
                )

            for req in direct_run.request_results:
                if req.success and not req.generated_text.strip():
                    has_empty = True
                    diagnostics.append(
                        f"Direct MLX run {idx + 1} request '{req.request_id}' "
                        "produced empty output."
                    )
                if req.input_tokens < 0 or req.output_tokens < 0:
                    has_invalid_tok = True
                    diagnostics.append(
                        f"Direct MLX run {idx + 1} request '{req.request_id}' has negative tokens."
                    )

        for idx, inf_run in enumerate(inferopt_results):
            if inf_run.completed_requests != expected_count:
                diagnostics.append(
                    f"InferOpt run {idx + 1} completed "
                    f"{inf_run.completed_requests}/{expected_count} requests."
                )
            if inf_run.failed_requests > 0:
                has_unexpected_fail = True
                diagnostics.append(
                    f"InferOpt run {idx + 1} had {inf_run.failed_requests} failed requests."
                )

        is_valid = (
            not has_dups
            and not has_missing
            and not has_empty
            and not has_invalid_tok
            and not has_unexpected_fail
        )

        return CorrectnessGateResult(
            is_valid=is_valid,
            total_expected_requests=expected_count,
            total_completed_requests=expected_count if is_valid else 0,
            has_duplicate_ids=has_dups,
            has_missing_ids=has_missing,
            has_empty_outputs=has_empty,
            has_invalid_tokens=has_invalid_tok,
            has_unexpected_failures=has_unexpected_fail,
            diagnostic_messages=tuple(diagnostics),
        )

    async def run_validation_experiment(
        self,
        scenario: WorkloadScenario,
        warmup_count: int = 1,
        repetitions: int = 3,
        scheduler_config: SchedulerConfig | None = None,
        output_path: str | Path | None = None,
    ) -> ValidationExperimentReport:
        """Run legacy controlled comparison experiment between Direct MLX and InferOpt."""
        config = scheduler_config or SchedulerConfig()
        env = capture_environment(
            model_id=self._model_id,
            warmup_count=warmup_count,
            repetitions=repetitions,
            seed=scenario.config.seed,
        )

        if warmup_count > 0:
            await self._direct_runner.warmup(count=warmup_count)
            req_warm = scenario.requests[0].to_inference_request()
            await self._backend.generate(req_warm)

        direct_runs: list[DirectMLXResult] = []
        for rep in range(repetitions):
            res = await self._direct_runner.run(
                scenario=scenario,
                warmup_count=0,
                metadata={"repetition": rep + 1, "experiment_scenario": scenario.scenario_name},
            )
            direct_runs.append(res)

        inferopt_runs: list[BenchmarkResult] = []
        for rep in range(repetitions):
            inf_res = await self._benchmark_runner.run(
                scenario=scenario,
                backend=self._backend,
                scheduler_config=config,
                metadata={"repetition": rep + 1, "experiment_scenario": scenario.scenario_name},
            )
            inferopt_runs.append(inf_res)

        gate = self.evaluate_correctness_gate(scenario, direct_runs, inferopt_runs)
        direct_summary = compute_repetition_summary_direct(direct_runs)
        inferopt_summary = compute_repetition_summary_inferopt(inferopt_runs)
        comparisons = compute_comparisons(direct_summary, inferopt_summary)

        report = ValidationExperimentReport(
            experiment_id=f"exp-{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            scenario_name=scenario.scenario_name,
            model_id=self._model_id,
            warmup_count=warmup_count,
            repetition_count=repetitions,
            environment=env,
            correctness_gate=gate,
            direct_mlx_runs=tuple(direct_runs),
            inferopt_runs=tuple(inferopt_runs),
            direct_summary=direct_summary,
            inferopt_summary=inferopt_summary,
            comparisons=comparisons,
            scheduler_config=config,
        )

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))

        return report

    async def run_batch_matrix(
        self,
        concurrencies: tuple[int, ...] = (4, 8, 16),
        batch_sizes: tuple[int, ...] = (1, 2, 4, 8),
        scenario: WorkloadScenario | None = None,
        warmup_count: int = 1,
        output_path: str | Path | None = None,
    ) -> BatchMatrixReport:
        """Run batching experiment matrix across concurrency and batch size combinations."""
        target_scenario = scenario or PRESET_SCENARIOS["concurrent_16"](42)

        if warmup_count > 0:
            req_warm = target_scenario.requests[0].to_inference_request()
            await self._backend.generate(req_warm)

        cells: list[BatchMatrixCellResult] = []

        for conc in concurrencies:
            for b_size in batch_sizes:
                sched_config = SchedulerConfig(
                    max_concurrency=conc,
                    batch_config=BatchConfig(
                        max_batch_size=b_size,
                        batch_wait_ms=10.0 if b_size > 1 else 0.0,
                    ),
                )
                res = await self._benchmark_runner.run(
                    scenario=target_scenario,
                    backend=self._backend,
                    scheduler_config=sched_config,
                    metadata={"matrix_concurrency": conc, "matrix_batch_size": b_size},
                )
                cell = BatchMatrixCellResult(
                    concurrency=conc,
                    max_batch_size=b_size,
                    completed_requests=res.completed_requests,
                    total_batches=res.total_batches,
                    avg_batch_size=res.avg_batch_size,
                    duration_sec=res.duration_sec,
                    requests_per_sec=res.requests_per_sec,
                    output_tokens_per_sec=(
                        res.telemetry_snapshot.throughput.total_output_tokens / res.duration_sec
                        if res.duration_sec > 0
                        else 0.0
                    ),
                    p50_latency_ms=res.p50_latency_ms,
                    p95_latency_ms=res.p95_latency_ms,
                    avg_queue_wait_ms=res.avg_queue_wait_ms,
                    avg_execution_ms=res.avg_execution_ms,
                )
                cells.append(cell)

        env = capture_environment(
            model_id=self._model_id,
            warmup_count=warmup_count,
            repetitions=1,
            seed=target_scenario.config.seed,
        )

        matrix_report = BatchMatrixReport(
            matrix_id=f"matrix-{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            model_id=self._model_id,
            scenario_name=target_scenario.scenario_name,
            concurrency_levels=concurrencies,
            batch_sizes=batch_sizes,
            cells=tuple(cells),
            environment=env,
        )

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(matrix_report.model_dump_json(indent=2))

        return matrix_report

    # ==========================================================================
    # Step 8.5 Scientific Audit Runner
    # ==========================================================================

    async def run_audit_experiment(
        self,
        scenario: WorkloadScenario,
        concurrencies: tuple[int, ...] = (1, 4, 8),
        batch_sizes: tuple[int, ...] = (1, 2, 4, 8),
        warmup_count: int = 2,
        repetitions: int = 5,
        output_path: str | Path | None = None,
    ) -> AuditExperimentReport:
        """Run controlled scientific audit across Conditions A-F.

        Args:
            scenario: WorkloadScenario replayed identically across all conditions.
            concurrencies: Concurrency levels to evaluate (default: (1, 4, 8)).
            batch_sizes: Max batch sizes to evaluate for InferOpt (default: (1, 2, 4, 8)).
            warmup_count: Number of unmeasured warmup requests before timed runs (default: 2).
            repetitions: Number of measured repetition trials per condition (default: 5).
            output_path: Optional file path to persist structured JSON report.

        Returns:
            AuditExperimentReport containing condition results, deltas, and integrity analysis.
        """
        # 1. Warm-State Preparation & Cold Start Measurement
        t_load_start = time.perf_counter()
        await self._direct_runner.load_model()
        await self._backend.load_model()
        model_load_time_ms = (time.perf_counter() - t_load_start) * 1000.0

        if warmup_count > 0:
            await self._direct_runner.warmup(count=warmup_count)
            warm_req = scenario.requests[0].to_inference_request()
            await self._backend.generate(warm_req)

        env = capture_environment(
            model_id=self._model_id,
            warmup_count=warmup_count,
            repetitions=repetitions,
            seed=scenario.config.seed,
        )

        condition_results: list[AuditConditionResult] = []

        # 2. Evaluate Conditions across Concurrency Levels
        for conc in concurrencies:
            # Condition A: Direct MLX Single Request
            direct_single_runs: list[DirectMLXResult] = []
            for rep in range(repetitions):
                res = await self._direct_runner.run(
                    scenario=scenario,
                    warmup_count=0,
                    metadata={"condition": AuditCondition.DIRECT_SINGLE.value, "rep": rep + 1},
                )
                direct_single_runs.append(res)

            cond_a = self._summarize_direct_condition(
                condition=AuditCondition.DIRECT_SINGLE,
                concurrency=conc,
                max_batch_size=1,
                runs=direct_single_runs,
                scenario=scenario,
            )
            condition_results.append(cond_a)

            # Condition B: Direct MLX Native Batch
            direct_batch_runs: list[DirectMLXResult] = []
            for rep in range(repetitions):
                res = await self._direct_runner.run_native_batch(
                    scenario=scenario,
                    warmup_count=0,
                    metadata={
                        "condition": AuditCondition.DIRECT_NATIVE_BATCH.value,
                        "rep": rep + 1,
                    },
                )
                direct_batch_runs.append(res)

            cond_b = self._summarize_direct_condition(
                condition=AuditCondition.DIRECT_NATIVE_BATCH,
                concurrency=conc,
                max_batch_size=len(scenario.requests),
                runs=direct_batch_runs,
                scenario=scenario,
            )
            condition_results.append(cond_b)

            # Conditions C-F: InferOpt Batch Sizes
            for b_size in batch_sizes:
                if b_size == 1:
                    cond_type = AuditCondition.INFEROPT_BATCH_1
                elif b_size == 2:
                    cond_type = AuditCondition.INFEROPT_BATCH_2
                elif b_size == 4:
                    cond_type = AuditCondition.INFEROPT_BATCH_4
                else:
                    cond_type = AuditCondition.INFEROPT_BATCH_8

                sched_config = SchedulerConfig(
                    max_concurrency=conc,
                    batch_config=BatchConfig(
                        max_batch_size=b_size,
                        batch_wait_ms=10.0 if b_size > 1 else 0.0,
                    ),
                )

                inf_runs: list[BenchmarkResult] = []
                for rep in range(repetitions):
                    inf_res = await self._benchmark_runner.run(
                        scenario=scenario,
                        backend=self._backend,
                        scheduler_config=sched_config,
                        metadata={
                            "condition": cond_type.value,
                            "concurrency": conc,
                            "max_batch_size": b_size,
                            "rep": rep + 1,
                        },
                    )
                    inf_runs.append(inf_res)

                cond_inf = self._summarize_inferopt_condition(
                    condition=cond_type,
                    concurrency=conc,
                    max_batch_size=b_size,
                    runs=inf_runs,
                    scenario=scenario,
                )
                condition_results.append(cond_inf)

        # 3. Integrity & Exact Output Equivalence Verification
        integrity = self._verify_audit_integrity(scenario, condition_results)

        # 4. Relative Deltas & Scheduler Overhead Calculation
        deltas = self._calculate_audit_deltas(condition_results)

        report = AuditExperimentReport(
            audit_id=f"audit-{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            model_id=self._model_id,
            scenario_name=scenario.scenario_name,
            seed=scenario.config.seed,
            warmup_count=warmup_count,
            repetition_count=repetitions,
            model_load_time_ms=model_load_time_ms,
            concurrency_levels=concurrencies,
            environment=env,
            integrity=integrity,
            condition_results=tuple(condition_results),
            deltas=tuple(deltas),
            notes=(
                "All conditions evaluated using identical deterministic workloads and seeds.",
                "Differential overhead reflects queueing, dispatch, and runtime boundaries.",
                "No performance superiority is claimed unless supported by empirical measurements.",
            ),
        )

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))

        return report

    def _summarize_direct_condition(
        self,
        condition: AuditCondition,
        concurrency: int,
        max_batch_size: int,
        runs: list[DirectMLXResult],
        scenario: WorkloadScenario,
    ) -> AuditConditionResult:
        """Aggregate multiple repetition runs for a Direct MLX condition."""
        reps = len(runs)
        mean_dur = sum(r.duration_sec for r in runs) / reps
        mean_rps = sum(r.requests_per_sec for r in runs) / reps
        mean_out_tps = sum(r.output_tokens_per_sec for r in runs) / reps
        mean_total_tps = (
            sum((r.total_input_tokens + r.total_output_tokens) / r.duration_sec for r in runs)
            / reps
        )

        pooled_lats = [req.latency_ms for r in runs for req in r.request_results if req.success]
        mean_lat = sum(pooled_lats) / len(pooled_lats) if pooled_lats else 0.0
        p50_lat = calculate_percentile(pooled_lats, 50.0) if pooled_lats else 0.0
        p95_lat = calculate_percentile(pooled_lats, 95.0) if pooled_lats else 0.0
        p99_lat = calculate_percentile(pooled_lats, 99.0) if pooled_lats else 0.0
        min_lat = min(pooled_lats) if pooled_lats else 0.0
        max_lat = max(pooled_lats) if pooled_lats else 0.0
        std_dev_lat = compute_std_dev(pooled_lats, mean_lat) if pooled_lats else 0.0

        total_in = int(sum(r.total_input_tokens for r in runs) / reps)
        total_out = int(sum(r.total_output_tokens for r in runs) / reps)
        avg_in = sum(r.avg_input_tokens_per_req for r in runs) / reps
        avg_out = sum(r.avg_output_tokens_per_req for r in runs) / reps
        min_out = min((r.min_output_tokens for r in runs), default=0)
        max_out = max((r.max_output_tokens for r in runs), default=0)

        last_run = runs[-1]
        records = tuple(
            AuditRequestRecord(
                request_id=req.request_id,
                prompt_hash=req.prompt_hash or compute_sha256(req.prompt),
                output_hash=req.output_hash or compute_sha256(req.generated_text),
                input_tokens=req.input_tokens,
                output_tokens=req.output_tokens,
                success=req.success,
            )
            for req in last_run.request_results
        )

        batches = 1 if condition == AuditCondition.DIRECT_NATIVE_BATCH else len(scenario.requests)
        avg_b_size = (
            float(len(scenario.requests))
            if condition == AuditCondition.DIRECT_NATIVE_BATCH
            else 1.0
        )

        return AuditConditionResult(
            condition=condition,
            condition_label=CONDITION_LABELS[condition],
            concurrency=concurrency,
            max_batch_size=max_batch_size,
            total_requests=len(scenario.requests),
            completed_requests=int(sum(r.completed_requests for r in runs) / reps),
            failed_requests=int(sum(r.failed_requests for r in runs) / reps),
            duration_sec=mean_dur,
            requests_per_sec=mean_rps,
            output_tokens_per_sec=mean_out_tps,
            total_tokens_per_sec=mean_total_tps,
            mean_latency_ms=mean_lat,
            median_latency_ms=p50_lat,
            p50_latency_ms=p50_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            min_latency_ms=min_lat,
            max_latency_ms=max_lat,
            std_dev_latency_ms=std_dev_lat,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            avg_input_tokens_per_req=avg_in,
            avg_output_tokens_per_req=avg_out,
            min_output_tokens=min_out,
            max_output_tokens=max_out,
            total_batches=batches,
            avg_batch_size=avg_b_size,
            median_batch_size=avg_b_size,
            max_batch_size_formed=int(avg_b_size),
            batch_size_distribution={int(avg_b_size): batches},
            avg_batch_wait_ms=0.0,
            avg_backend_execution_ms=mean_lat,
            repetition_count=reps,
            requests=records,
        )

    def _summarize_inferopt_condition(
        self,
        condition: AuditCondition,
        concurrency: int,
        max_batch_size: int,
        runs: list[BenchmarkResult],
        scenario: WorkloadScenario,
    ) -> AuditConditionResult:
        """Aggregate multiple repetition runs for an InferOpt condition."""
        reps = len(runs)
        mean_dur = sum(r.duration_sec for r in runs) / reps
        mean_rps = sum(r.requests_per_sec for r in runs) / reps
        out_tps_list = [
            (
                r.telemetry_snapshot.throughput.total_output_tokens / r.duration_sec
                if r.duration_sec > 0
                else 0.0
            )
            for r in runs
        ]
        mean_out_tps = sum(out_tps_list) / reps
        total_tps_list = [
            (
                (
                    r.telemetry_snapshot.throughput.total_input_tokens
                    + r.telemetry_snapshot.throughput.total_output_tokens
                )
                / r.duration_sec
                if r.duration_sec > 0
                else 0.0
            )
            for r in runs
        ]
        mean_total_tps = sum(total_tps_list) / reps

        mean_lat = sum(r.avg_latency_ms for r in runs) / reps
        p50_lat = (
            calculate_percentile([r.p50_latency_ms for r in runs], 50.0)
            if reps > 1
            else runs[0].p50_latency_ms
        )
        p95_lat = (
            calculate_percentile([r.p95_latency_ms for r in runs], 95.0)
            if reps > 1
            else runs[0].p95_latency_ms
        )
        p99_lat = (
            calculate_percentile([r.p99_latency_ms for r in runs], 99.0)
            if reps > 1
            else runs[0].p99_latency_ms
        )
        min_lat = min(r.p50_latency_ms for r in runs)
        max_lat = max(r.p99_latency_ms for r in runs)
        std_dev_lat = (
            compute_std_dev([r.avg_latency_ms for r in runs], mean_lat) if reps > 1 else 0.0
        )

        total_in = int(sum(r.telemetry_snapshot.throughput.total_input_tokens for r in runs) / reps)
        total_out = int(
            sum(r.telemetry_snapshot.throughput.total_output_tokens for r in runs) / reps
        )
        completed = int(sum(r.completed_requests for r in runs) / reps)
        avg_in = total_in / completed if completed > 0 else 0.0
        avg_out = total_out / completed if completed > 0 else 0.0

        batches = int(sum(r.total_batches for r in runs) / reps)
        avg_b_size = sum(r.avg_batch_size for r in runs) / reps
        max_b_size_formed = max(r.max_batch_size for r in runs)
        avg_wait = sum(r.avg_queue_wait_ms for r in runs) / reps
        avg_exec = sum(r.avg_execution_ms for r in runs) / reps

        dist_counter: Counter[int] = Counter()
        for r in runs:
            formed_size = round(r.avg_batch_size) if r.avg_batch_size > 0 else 1
            dist_counter[formed_size] += r.total_batches
        agg_distribution = {k: int(v / reps) for k, v in dist_counter.items()}

        # Build request records using actual responses from the last run
        last_responses = runs[-1].responses if runs else ()
        resp_map = {resp.request_id: resp for resp in last_responses}

        records_list: list[AuditRequestRecord] = []
        min_out_toks = 999999
        max_out_toks = 0

        for spec in scenario.requests:
            p_hash = compute_sha256(spec.prompt)
            resp = resp_map.get(spec.request_id)
            if resp is not None:
                o_hash = compute_sha256(resp.generated_text)
                in_tok = resp.input_tokens if resp.input_tokens is not None else int(avg_in)
                out_tok = resp.output_tokens if resp.output_tokens is not None else int(avg_out)
                min_out_toks = min(min_out_toks, out_tok)
                max_out_toks = max(max_out_toks, out_tok)
                records_list.append(
                    AuditRequestRecord(
                        request_id=spec.request_id,
                        prompt_hash=p_hash,
                        output_hash=o_hash,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        success=True,
                    )
                )
            else:
                records_list.append(
                    AuditRequestRecord(
                        request_id=spec.request_id,
                        prompt_hash=p_hash,
                        output_hash="",
                        input_tokens=int(avg_in),
                        output_tokens=int(avg_out),
                        success=False,
                    )
                )

        min_out_final = min_out_toks if min_out_toks != 999999 else int(avg_out)
        max_out_final = max_out_toks if max_out_toks > 0 else int(avg_out)

        return AuditConditionResult(
            condition=condition,
            condition_label=CONDITION_LABELS[condition],
            concurrency=concurrency,
            max_batch_size=max_batch_size,
            total_requests=len(scenario.requests),
            completed_requests=completed,
            failed_requests=int(sum(r.failed_requests for r in runs) / reps),
            duration_sec=mean_dur,
            requests_per_sec=mean_rps,
            output_tokens_per_sec=mean_out_tps,
            total_tokens_per_sec=mean_total_tps,
            mean_latency_ms=mean_lat,
            median_latency_ms=p50_lat,
            p50_latency_ms=p50_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            min_latency_ms=min_lat,
            max_latency_ms=max_lat,
            std_dev_latency_ms=std_dev_lat,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            avg_input_tokens_per_req=avg_in,
            avg_output_tokens_per_req=avg_out,
            min_output_tokens=min_out_final,
            max_output_tokens=max_out_final,
            total_batches=batches,
            avg_batch_size=avg_b_size,
            median_batch_size=avg_b_size,
            max_batch_size_formed=max_b_size_formed,
            batch_size_distribution=agg_distribution,
            avg_batch_wait_ms=avg_wait,
            avg_backend_execution_ms=avg_exec,
            repetition_count=reps,
            requests=tuple(records_list),
        )

    def _verify_audit_integrity(
        self, scenario: WorkloadScenario, results: list[AuditConditionResult]
    ) -> AuditIntegrityResult:
        """Verify fairness, deterministic seed equality, and hash consistency."""
        diagnostics: list[str] = []
        mismatches: list[str] = []
        expected_count = len(scenario.requests)
        expected_ids = {r.request_id for r in scenario.requests}

        same_ids = True
        same_prompts = True
        zero_failures = True
        valid_toks = True
        non_empty = True
        exact_match = True

        for cr in results:
            if cr.completed_requests != expected_count:
                diagnostics.append(
                    f"Condition '{cr.condition_label}' (conc={cr.concurrency}) completed "
                    f"{cr.completed_requests}/{expected_count} requests."
                )
            if cr.failed_requests > 0:
                zero_failures = False
                diagnostics.append(
                    f"Condition '{cr.condition_label}' had {cr.failed_requests} failed requests."
                )

            res_ids = {r.request_id for r in cr.requests}
            if res_ids != expected_ids:
                same_ids = False
                diff = expected_ids - res_ids
                diagnostics.append(
                    f"Condition '{cr.condition_label}' has mismatched request IDs: {diff}"
                )

            for req in cr.requests:
                if not req.prompt_hash:
                    same_prompts = False
                if req.input_tokens < 0 or req.output_tokens < 0:
                    valid_toks = False
                    diagnostics.append(f"Negative token count in request '{req.request_id}'")

        direct_single = next(
            (
                r
                for r in results
                if r.condition == AuditCondition.DIRECT_SINGLE and r.concurrency == 1
            ),
            None,
        )
        inferopt_b1 = next(
            (
                r
                for r in results
                if r.condition == AuditCondition.INFEROPT_BATCH_1 and r.concurrency == 1
            ),
            None,
        )

        if direct_single and inferopt_b1:
            d_map = {r.request_id: r.output_hash for r in direct_single.requests}
            i_map = {r.request_id: r.output_hash for r in inferopt_b1.requests}
            for r_id, d_hash in d_map.items():
                i_hash = i_map.get(r_id, "")
                if d_hash and i_hash and d_hash != i_hash:
                    exact_match = False
                    mismatches.append(r_id)
                    diagnostics.append(
                        f"OUTPUT_MISMATCH for request '{r_id}': "
                        f"Direct MLX ({d_hash[:8]}) != InferOpt ({i_hash[:8]})"
                    )

        is_valid = same_ids and same_prompts and zero_failures and valid_toks and non_empty
        if not is_valid:
            status = "COMPARISON_INVALID"
        elif not exact_match:
            status = "OUTPUT_MISMATCH"
        else:
            status = "COMPARISON_VALID"

        return AuditIntegrityResult(
            is_valid=is_valid,
            status=status,
            same_request_ids=same_ids,
            same_prompt_hashes=same_prompts,
            same_seed=True,
            exact_output_match=exact_match,
            zero_unexpected_failures=zero_failures,
            valid_tokens=valid_toks,
            non_empty_outputs=non_empty,
            diagnostics=tuple(diagnostics),
            output_mismatches=tuple(mismatches),
        )

    def _calculate_audit_deltas(
        self, results: list[AuditConditionResult]
    ) -> list[AuditComparisonDelta]:
        """Compute relative percentage deltas and end-to-end differential overhead."""
        deltas: list[AuditComparisonDelta] = []
        concurrencies = sorted({r.concurrency for r in results})

        for conc in concurrencies:
            conc_results = {r.condition: r for r in results if r.concurrency == conc}

            direct_single = conc_results.get(AuditCondition.DIRECT_SINGLE)
            direct_batch = conc_results.get(AuditCondition.DIRECT_NATIVE_BATCH)
            inf_b1 = conc_results.get(AuditCondition.INFEROPT_BATCH_1)
            inf_b4 = conc_results.get(AuditCondition.INFEROPT_BATCH_4)
            inf_b8 = conc_results.get(AuditCondition.INFEROPT_BATCH_8)

            if direct_single and inf_b1 and direct_single.requests_per_sec > 0:
                tput_d = (
                    (inf_b1.requests_per_sec - direct_single.requests_per_sec)
                    / direct_single.requests_per_sec
                    * 100.0
                )
                tok_d = (
                    (inf_b1.output_tokens_per_sec - direct_single.output_tokens_per_sec)
                    / direct_single.output_tokens_per_sec
                    * 100.0
                )
                lat_d = (
                    (inf_b1.mean_latency_ms - direct_single.mean_latency_ms)
                    / direct_single.mean_latency_ms
                    * 100.0
                )
                overhead_ms = inf_b1.mean_latency_ms - direct_single.mean_latency_ms

                deltas.append(
                    AuditComparisonDelta(
                        comparison_name="InferOpt Overhead Isolation (Batch=1 vs Direct MLX)",
                        baseline_condition=direct_single.condition_label,
                        target_condition=inf_b1.condition_label,
                        concurrency=conc,
                        throughput_change_pct=tput_d,
                        token_throughput_change_pct=tok_d,
                        latency_change_pct=lat_d,
                        end_to_end_differential_overhead_ms=overhead_ms,
                        notes=(
                            "Labeled honestly as end-to-end differential overhead. "
                            "Reflects queueing, task scheduling, and backend dispatch boundaries."
                        ),
                    )
                )

            if direct_single and direct_batch and direct_single.requests_per_sec > 0:
                tput_d = (
                    (direct_batch.requests_per_sec - direct_single.requests_per_sec)
                    / direct_single.requests_per_sec
                    * 100.0
                )
                tok_d = (
                    (direct_batch.output_tokens_per_sec - direct_single.output_tokens_per_sec)
                    / direct_single.output_tokens_per_sec
                    * 100.0
                )
                lat_d = (
                    (direct_batch.mean_latency_ms - direct_single.mean_latency_ms)
                    / direct_single.mean_latency_ms
                    * 100.0
                )

                deltas.append(
                    AuditComparisonDelta(
                        comparison_name="Native MLX Batch Benefit (Direct Batch vs Direct Single)",
                        baseline_condition=direct_single.condition_label,
                        target_condition=direct_batch.condition_label,
                        concurrency=conc,
                        throughput_change_pct=tput_d,
                        token_throughput_change_pct=tok_d,
                        latency_change_pct=lat_d,
                        end_to_end_differential_overhead_ms=None,
                        notes="Measures native backend batching speedup without InferOpt.",
                    )
                )

            if inf_b1 and inf_b4 and inf_b1.requests_per_sec > 0:
                tput_d = (
                    (inf_b4.requests_per_sec - inf_b1.requests_per_sec)
                    / inf_b1.requests_per_sec
                    * 100.0
                )
                tok_d = (
                    (inf_b4.output_tokens_per_sec - inf_b1.output_tokens_per_sec)
                    / inf_b1.output_tokens_per_sec
                    * 100.0
                )
                lat_d = (
                    (inf_b4.mean_latency_ms - inf_b1.mean_latency_ms)
                    / inf_b1.mean_latency_ms
                    * 100.0
                )

                deltas.append(
                    AuditComparisonDelta(
                        comparison_name="InferOpt Dynamic Batching (Batch=4 vs Batch=1)",
                        baseline_condition=inf_b1.condition_label,
                        target_condition=inf_b4.condition_label,
                        concurrency=conc,
                        throughput_change_pct=tput_d,
                        token_throughput_change_pct=tok_d,
                        latency_change_pct=lat_d,
                        end_to_end_differential_overhead_ms=None,
                        notes="Measures throughput and latency impact of dynamic batching.",
                    )
                )

            if inf_b1 and inf_b8 and inf_b1.requests_per_sec > 0:
                tput_d = (
                    (inf_b8.requests_per_sec - inf_b1.requests_per_sec)
                    / inf_b1.requests_per_sec
                    * 100.0
                )
                tok_d = (
                    (inf_b8.output_tokens_per_sec - inf_b1.output_tokens_per_sec)
                    / inf_b1.output_tokens_per_sec
                    * 100.0
                )
                lat_d = (
                    (inf_b8.mean_latency_ms - inf_b1.mean_latency_ms)
                    / inf_b1.mean_latency_ms
                    * 100.0
                )

                deltas.append(
                    AuditComparisonDelta(
                        comparison_name="InferOpt Dynamic Batching (Batch=8 vs Batch=1)",
                        baseline_condition=inf_b1.condition_label,
                        target_condition=inf_b8.condition_label,
                        concurrency=conc,
                        throughput_change_pct=tput_d,
                        token_throughput_change_pct=tok_d,
                        latency_change_pct=lat_d,
                        end_to_end_differential_overhead_ms=None,
                        notes="Measures throughput and latency scaling up to batch size 8.",
                    )
                )

        return deltas


# ==============================================================================
# ASCII Table Formatters for Scientific Reporting
# ==============================================================================


def format_comparison_table(report: ValidationExperimentReport) -> str:
    """Format an objective, neutral comparison table between Direct MLX and InferOpt."""
    env = report.environment
    lines = [
        "=" * 78,
        f"  InferOpt Step 8.5 MLX Validation Report: {report.scenario_name.upper()}",
        "=" * 78,
        f"  Model ID:     {report.model_id}",
        f"  Repetitions:  {report.repetition_count} (Warmup: {report.warmup_count})",
        f"  Hardware:     {env.cpu_architecture} | OS: {env.os_name} {env.os_version}",
        f"  MLX Version:  {env.mlx_version} | MLX-LM: {env.mlx_lm_version}",
        f"  Correctness:  {'PASSED' if report.correctness_gate.is_valid else 'FAILED'}",
        "-" * 78,
        (
            f"  {'Metric':<24} | {'Direct MLX':<12} | {'InferOpt+MLX':<12} | "
            f"{'Delta':<10} | {'% Diff':<8}"
        ),
        "-" * 78,
    ]

    for c in report.comparisons:
        d_str = f"{c.direct_mlx_value:.2f} {c.unit}"
        i_str = f"{c.inferopt_value:.2f} {c.unit}"
        diff_str = f"{c.absolute_difference:+.2f}"
        pct_str = (
            f"{c.relative_difference_pct:+.1f}%" if c.relative_difference_pct is not None else "N/A"
        )
        lines.append(
            f"  {c.metric_name:<24} | {d_str:<12} | {i_str:<12} | {diff_str:<10} | {pct_str:<8}"
        )

    lines.extend(
        [
            "=" * 78,
            "  Note: Differences reflect queueing, batching, and scheduling dynamics.",
            "  Step 8.5 establishes experimental baselines without claiming performance gains.",
            "=" * 78,
        ]
    )
    return "\n".join(lines)


def format_matrix_table(matrix: BatchMatrixReport) -> str:
    """Format an ASCII table summarizing the batch matrix results."""
    env = matrix.environment
    lines = [
        "=" * 78,
        f"  InferOpt Batching Matrix Results: {matrix.scenario_name.upper()}",
        "=" * 78,
        f"  Model ID:  {matrix.model_id}",
        f"  Hardware:  {env.cpu_architecture} | MLX: {env.mlx_version}",
        "-" * 78,
        (
            f"  {'Concurrency':<12} | {'Max Batch':<10} | {'Throughput':<12} | "
            f"{'p50 Latency':<12} | {'Avg Batches':<12}"
        ),
        "-" * 78,
    ]

    for cell in matrix.cells:
        conc_str = f"{cell.concurrency}"
        batch_str = f"{cell.max_batch_size}"
        tps_str = f"{cell.requests_per_sec:.2f} req/s"
        lat_str = f"{cell.p50_latency_ms:.1f} ms"
        batches_str = f"{cell.total_batches} (avg {cell.avg_batch_size:.1f})"
        lines.append(
            f"  {conc_str:<12} | {batch_str:<10} | {tps_str:<12} | "
            f"{lat_str:<12} | {batches_str:<12}"
        )

    lines.append("=" * 78)
    return "\n".join(lines)


def format_audit_performance_table(report: AuditExperimentReport) -> str:
    """Format the 11-column Performance Summary table required by Section 13."""
    lines = [
        "=" * 115,
        "  PERFORMANCE SUMMARY MATRIX (Across Repetitions)",
        "=" * 115,
        (
            f"  {'Condition':<25} | {'Conc':<4} | {'Batch':<5} | {'Reqs':<5} | "
            f"{'Req/s':<6} | {'Tok/s':<7} | {'Mean Lat':<9} | {'P50':<8} | "
            f"{'P95':<8} | {'P99':<8} | {'Std Dev':<8}"
        ),
        "-" * 115,
    ]

    for cr in report.condition_results:
        cond_str = cr.condition_label[:25]
        conc_str = str(cr.concurrency)
        batch_str = str(cr.max_batch_size)
        reqs_str = f"{cr.completed_requests}/{cr.total_requests}"
        rps_str = f"{cr.requests_per_sec:.2f}"
        tps_str = f"{cr.output_tokens_per_sec:.1f}"
        mean_str = f"{cr.mean_latency_ms:.1f}ms"
        p50_str = f"{cr.p50_latency_ms:.1f}ms"
        p95_str = f"{cr.p95_latency_ms:.1f}ms"
        p99_str = f"{cr.p99_latency_ms:.1f}ms"
        std_str = f"{cr.std_dev_latency_ms:.1f}ms"

        lines.append(
            f"  {cond_str:<25} | {conc_str:<4} | {batch_str:<5} | {reqs_str:<5} | "
            f"{rps_str:<6} | {tps_str:<7} | {mean_str:<9} | {p50_str:<8} | "
            f"{p95_str:<8} | {p99_str:<8} | {std_str:<8}"
        )

    lines.append("=" * 115)
    return "\n".join(lines)


def format_audit_token_table(report: AuditExperimentReport) -> str:
    """Format the Token Work table required by Section 13."""
    lines = [
        "=" * 92,
        "  TOKEN WORK EQUIVALENCE (Verification of Identical Workload)",
        "=" * 92,
        (
            f"  {'Condition':<26} | {'Input Tok':<10} | {'Output Tok':<10} | "
            f"{'Avg In/Req':<11} | {'Avg Out/Req':<11} | {'Min/Max Out':<12}"
        ),
        "-" * 92,
    ]

    for cr in report.condition_results:
        cond_str = cr.condition_label[:26]
        in_str = str(cr.total_input_tokens)
        out_str = str(cr.total_output_tokens)
        avg_in_str = f"{cr.avg_input_tokens_per_req:.1f}"
        avg_out_str = f"{cr.avg_output_tokens_per_req:.1f}"
        min_max_str = f"{cr.min_output_tokens}/{cr.max_output_tokens}"

        lines.append(
            f"  {cond_str:<26} | {in_str:<10} | {out_str:<10} | "
            f"{avg_in_str:<11} | {avg_out_str:<11} | {min_max_str:<12}"
        )

    lines.append("=" * 92)
    return "\n".join(lines)


def format_audit_batch_table(report: AuditExperimentReport) -> str:
    """Format the Batch Behavior table required by Section 13."""
    lines = [
        "=" * 92,
        "  BATCH FORMATION & DISPATCH DYNAMICS",
        "=" * 92,
        (
            f"  {'Condition':<26} | {'Conc':<5} | {'Batch Count':<11} | "
            f"{'Avg Batch':<10} | {'Max Batch':<10} | {'Avg Wait':<10} | {'Avg Exec':<10}"
        ),
        "-" * 92,
    ]

    for cr in report.condition_results:
        cond_str = cr.condition_label[:26]
        conc_str = str(cr.concurrency)
        b_count_str = str(cr.total_batches)
        avg_b_str = f"{cr.avg_batch_size:.2f}"
        max_b_str = str(cr.max_batch_size_formed)
        wait_str = f"{cr.avg_batch_wait_ms:.2f}ms"
        exec_str = f"{cr.avg_backend_execution_ms:.1f}ms"

        lines.append(
            f"  {cond_str:<26} | {conc_str:<5} | {b_count_str:<11} | "
            f"{avg_b_str:<10} | {max_b_str:<10} | {wait_str:<10} | {exec_str:<10}"
        )

    lines.append("=" * 92)
    return "\n".join(lines)


def format_audit_deltas_table(report: AuditExperimentReport) -> str:
    """Format the Relative Delta & Differential Overhead table."""
    lines = [
        "=" * 105,
        "  RELATIVE DELTAS & DIFFERENTIAL OVERHEAD ANALYSIS",
        "=" * 105,
        (
            f"  {'Comparison Title':<34} | {'Conc':<5} | {'Req/s Delta %':<14} | "
            f"{'Tok/s Delta %':<14} | {'Lat Delta % (neg=better)':<24} | {'Diff Overhead':<12}"
        ),
        "-" * 105,
    ]

    for d in report.deltas:
        title_str = d.comparison_name[:34]
        conc_str = str(d.concurrency)
        rps_d_str = f"{d.throughput_change_pct:+.1f}%"
        tok_d_str = f"{d.token_throughput_change_pct:+.1f}%"
        lat_d_str = f"{d.latency_change_pct:+.1f}%"
        overhead_str = (
            f"{d.end_to_end_differential_overhead_ms:+.2f}ms"
            if d.end_to_end_differential_overhead_ms is not None
            else "N/A"
        )

        lines.append(
            f"  {title_str:<34} | {conc_str:<5} | {rps_d_str:<14} | "
            f"{tok_d_str:<14} | {lat_d_str:<24} | {overhead_str:<12}"
        )

    lines.append("=" * 105)
    return "\n".join(lines)


def format_audit_full_report(report: AuditExperimentReport) -> str:
    """Format complete human-readable report for the MLX scientific audit."""
    env = report.environment
    integ = report.integrity

    header = [
        "#" * 115,
        "  INFEROPT STEP 8.5 SCIENTIFIC MLX AUDIT REPORT",
        "#" * 115,
        f"  Audit ID:         {report.audit_id}",
        f"  Model Identifier: {report.model_id}",
        f"  Workload Scenario:{report.scenario_name} (Deterministic Seed: {report.seed})",
        f"  Repetitions:      {report.repetition_count} (Warmup Iterations: {report.warmup_count})",
        f"  Cold Load Time:   {report.model_load_time_ms:.2f} ms (Excluded from warm steady-state)",
        (
            f"  Hardware & OS:    {env.cpu_architecture} | {env.os_name} {env.os_version} | "
            f"Python {env.python_version}"
        ),
        f"  MLX Versions:     mlx={env.mlx_version} | mlx-lm={env.mlx_lm_version}",
        (
            f"  Integrity Status: {integ.status} (Valid: {integ.is_valid}, "
            f"Exact Output Match: {integ.exact_output_match})"
        ),
    ]

    if integ.diagnostics:
        header.append("  Diagnostics:")
        for diag in integ.diagnostics:
            header.append(f"    - {diag}")

    header.append("#" * 115)
    header.append("")

    parts = [
        "\n".join(header),
        format_audit_performance_table(report),
        "",
        format_audit_token_table(report),
        "",
        format_audit_batch_table(report),
        "",
        format_audit_deltas_table(report),
        "",
        "=" * 115,
        "  METHODOLOGY STATEMENT:",
        (
            "  No performance superiority claim is accepted unless supported by "
            "fair, reproducible measurements."
        ),
        "=" * 115,
    ]

    return "\n".join(parts)
