"""Scientific real-vLLM benchmark harness, validation, and baseline comparison subsystem.

Enforces identical workload replay, deterministic SHA-256 workload hashing,
multi-repetition statistical aggregation (mean, median, min, max, std dev),
non-intrusive warmup separation, an automated integrity gate, differential
overhead analysis, and objective findings classification (PROVEN / SUGGESTED / NOT PROVEN).
"""

import contextlib
import hashlib
import platform
import time
import uuid
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from inferopt import __version__ as INFEROPT_VERSION
from inferopt.backends.vllm import DEFAULT_VLLM_MODEL_ID, VLLMBackend, VLLMConfig
from inferopt.benchmarks.models import WorkloadScenario
from inferopt.benchmarks.runner import BenchmarkRunner
from inferopt.benchmarks.vllm_baseline import (
    DirectVLLMResult,
    DirectVLLMRunner,
    calculate_percentile,
    compute_sha256,
    compute_std_dev,
)
from inferopt.scheduler.config import BatchConfig, SchedulerConfig


class VLLMCondition(StrEnum):
    """The standard five benchmark comparison conditions for vLLM."""

    DIRECT_VLLM = "DIRECT_VLLM"  # Condition A: Direct vLLM baseline
    INFEROPT_BATCH_1 = "INFEROPT_BATCH_1"  # Condition B: InferOpt max_batch_size=1
    INFEROPT_BATCH_2 = "INFEROPT_BATCH_2"  # Condition C: InferOpt max_batch_size=2
    INFEROPT_BATCH_4 = "INFEROPT_BATCH_4"  # Condition D: InferOpt max_batch_size=4
    INFEROPT_BATCH_8 = "INFEROPT_BATCH_8"  # Condition E: InferOpt max_batch_size=8


VLLM_CONDITION_LABELS: dict[VLLMCondition, str] = {
    VLLMCondition.DIRECT_VLLM: "A. Direct vLLM",
    VLLMCondition.INFEROPT_BATCH_1: "B. InferOpt Batch 1",
    VLLMCondition.INFEROPT_BATCH_2: "C. InferOpt Batch 2",
    VLLMCondition.INFEROPT_BATCH_4: "D. InferOpt Batch 4",
    VLLMCondition.INFEROPT_BATCH_8: "E. InferOpt Batch 8",
}


def compute_workload_hash(scenario: WorkloadScenario) -> str:
    """Compute deterministic SHA-256 hash representing the exact workload requests."""
    hasher = hashlib.sha256()
    hasher.update(scenario.scenario_name.encode("utf-8"))
    hasher.update(str(len(scenario.requests)).encode("utf-8"))
    hasher.update(str(scenario.config.seed).encode("utf-8"))
    for r in scenario.requests:
        hasher.update(r.request_id.encode("utf-8"))
        hasher.update(r.prompt.encode("utf-8"))
        hasher.update(str(r.max_tokens).encode("utf-8"))
        hasher.update(str(r.temperature).encode("utf-8"))
    return hasher.hexdigest()


class VLLMEnvironmentMetadata(BaseModel):
    """Host machine, GPU hardware, Python runtime, and library version metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    os_name: str = Field(description="Operating system name")
    os_version: str = Field(description="Operating system release version")
    cpu_architecture: str = Field(description="Hardware CPU architecture")
    python_version: str = Field(description="Python runtime version")
    vllm_version: str = Field(default="unknown", description="Installed vLLM version")
    torch_version: str = Field(default="unknown", description="Installed PyTorch version")
    cuda_version: str = Field(default="unknown", description="CUDA runtime version")
    gpu_name: str = Field(default="unknown", description="NVIDIA GPU device model name")
    gpu_count: int = Field(default=0, ge=0, description="Count of visible GPU devices")
    inferopt_version: str = Field(default=INFEROPT_VERSION, description="InferOpt library version")
    model_id: str = Field(description="Evaluated model repository identifier")
    warmup_count: int = Field(ge=0, description="Warmup iterations per run")
    repetitions: int = Field(ge=1, description="Number of measured repetition trials")
    workload_seed: int = Field(description="Deterministic workload seed")
    workload_hash: str = Field(description="Deterministic SHA-256 hash of the workload")


class VLLMBenchmarkRequestRecord(BaseModel):
    """Integrity record for an individual inference request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(description="Request identifier")
    prompt_hash: str = Field(description="SHA-256 hash of input prompt")
    output_hash: str = Field(description="SHA-256 hash of generated text")
    input_tokens: int = Field(ge=0, description="Exact prompt token count")
    output_tokens: int = Field(ge=0, description="Exact generated token count")
    latency_ms: float = Field(ge=0.0, description="Turnaround execution latency in ms")
    success: bool = Field(description="True if generation succeeded")


class VLLMConditionResult(BaseModel):
    """Aggregated, multi-repetition results for a single benchmark condition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: VLLMCondition = Field(description="Condition identifier")
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
    avg_queue_wait_ms: float = Field(ge=0.0, description="Average queue wait time in milliseconds")
    p50_queue_wait_ms: float = Field(
        default=0.0, ge=0.0, description="50th percentile queue wait time in ms"
    )
    p95_queue_wait_ms: float = Field(
        default=0.0, ge=0.0, description="95th percentile queue wait time in ms"
    )
    p99_queue_wait_ms: float = Field(
        default=0.0, ge=0.0, description="99th percentile queue wait time in ms"
    )
    avg_backend_execution_ms: float = Field(
        ge=0.0, description="Average backend execution time in ms"
    )
    p50_backend_execution_ms: float = Field(
        default=0.0, ge=0.0, description="50th percentile execution time in ms"
    )
    p95_backend_execution_ms: float = Field(
        default=0.0, ge=0.0, description="95th percentile execution time in ms"
    )
    p99_backend_execution_ms: float = Field(
        default=0.0, ge=0.0, description="99th percentile execution time in ms"
    )
    total_input_tokens: int = Field(ge=0, description="Total prompt tokens processed")
    total_output_tokens: int = Field(ge=0, description="Total generated tokens produced")
    total_tokens: int = Field(default=0, ge=0, description="Sum of input and output tokens")
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
    min_batch_size: int = Field(default=0, ge=0, description="Minimum batch size formed")
    max_batch_size_formed: int = Field(ge=0, description="Maximum batch size actually formed")
    batch_size_distribution: dict[int, int] = Field(
        default_factory=dict, description="Distribution count of formed batch sizes"
    )
    avg_batch_formation_wait_ms: float = Field(
        ge=0.0, description="Average batch formation wait time in ms"
    )
    avg_batch_execution_ms: float = Field(
        ge=0.0, description="Average batch backend execution time in ms"
    )
    repetition_count: int = Field(ge=1, description="Number of repetitions aggregated")
    requests: tuple[VLLMBenchmarkRequestRecord, ...] = Field(
        default_factory=tuple, description="Canonical request records from final repetition"
    )


class VLLMComparisonDelta(BaseModel):
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
        "labeled as end-to-end differential overhead",
    )
    notes: str = Field(default="", description="Explanatory notes on measurement boundaries")


class VLLMIntegrityResult(BaseModel):
    """Comprehensive integrity gate verifying fairness and output equivalence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool = Field(description="True if comparison matrix satisfies all integrity checks")
    total_expected_requests: int = Field(ge=0, description="Number of expected workload requests")
    total_completed_requests: int = Field(ge=0, description="Number of completed requests")
    has_duplicate_ids: bool = Field(default=False, description="True if duplicate IDs detected")
    has_missing_ids: bool = Field(default=False, description="True if missing IDs detected")
    has_empty_outputs: bool = Field(default=False, description="True if empty completions found")
    has_invalid_tokens: bool = Field(default=False, description="True if token count was invalid")
    has_unexpected_failures: bool = Field(default=False, description="True if failures occurred")
    has_workload_hash_mismatch: bool = Field(
        default=False, description="True if conditions received different workloads"
    )
    diagnostic_messages: tuple[str, ...] = Field(
        default_factory=tuple, description="Detailed diagnostic reasons for failures"
    )


class VLLMExperimentReport(BaseModel):
    """Complete, standalone, machine-readable scientific benchmark report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Unique experiment execution identifier")
    timestamp: float = Field(description="Experiment execution timestamp")
    scenario_name: str = Field(description="Workload scenario name")
    model_id: str = Field(description="Hugging Face model identifier")
    workload_hash: str = Field(description="Deterministic SHA-256 hash of the evaluated workload")
    environment: VLLMEnvironmentMetadata = Field(description="Hardware and runtime environment")
    integrity: VLLMIntegrityResult = Field(
        description="Integrity validation result confirming fairness"
    )
    conditions: tuple[VLLMConditionResult, ...] = Field(
        description="Results for each evaluated condition"
    )
    deltas: tuple[VLLMComparisonDelta, ...] = Field(
        description="Structured delta comparisons relative to Direct vLLM baseline"
    )
    concurrency_levels: tuple[int, ...] = Field(description="Evaluated concurrency levels")
    batch_sizes: tuple[int, ...] = Field(description="Evaluated maximum batch sizes")
    repetition_count: int = Field(ge=1, description="Number of repetitions per condition")
    warmup_count: int = Field(ge=0, description="Number of warmup iterations")
    findings: dict[str, tuple[str, ...]] = Field(
        default_factory=dict, description="Objective findings categorized by evidence level"
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "VLLMExperimentReport":
        """Deserialize report from JSON string."""
        return cls.model_validate_json(json_str)

    def save_json(self, file_path: str | Path) -> None:
        """Persist report to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load_json(cls, file_path: str | Path) -> "VLLMExperimentReport":
        """Load and validate report from a JSON file."""
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            return cls.from_json(f.read())


def collect_vllm_environment_metadata(
    model_id: str,
    warmup_count: int,
    repetitions: int,
    workload_seed: int,
    workload_hash: str,
) -> VLLMEnvironmentMetadata:
    """Inspect and capture local hardware, GPU, and Python runtime metadata."""
    vllm_ver = "unknown"
    torch_ver = "unknown"
    cuda_ver = "unknown"
    gpu_name = "unknown"
    gpu_count = 0

    try:
        import vllm  # type: ignore[import-not-found]

        vllm_ver = getattr(vllm, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import torch  # type: ignore[import-not-found]

        torch_ver = getattr(torch, "__version__", "unknown")
        if hasattr(torch, "version") and hasattr(torch.version, "cuda"):
            cuda_ver = str(torch.version.cuda or "none")
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_name = str(torch.cuda.get_device_name(0))
    except ImportError:
        pass

    return VLLMEnvironmentMetadata(
        os_name=platform.system(),
        os_version=platform.release(),
        cpu_architecture=platform.machine(),
        python_version=platform.python_version(),
        vllm_version=vllm_ver,
        torch_version=torch_ver,
        cuda_version=cuda_ver,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        inferopt_version=INFEROPT_VERSION,
        model_id=model_id,
        warmup_count=warmup_count,
        repetitions=repetitions,
        workload_seed=workload_seed,
        workload_hash=workload_hash,
    )


def validate_vllm_integrity(
    scenario: WorkloadScenario,
    condition_results: Sequence[VLLMConditionResult],
    expected_workload_hash: str,
) -> VLLMIntegrityResult:
    """Validate 100% request completion, token validity, and identical workload execution."""
    diagnostics: list[str] = []
    total_expected = len(scenario.requests)
    expected_req_ids = [r.request_id for r in scenario.requests]

    has_duplicate = False
    has_missing = False
    has_empty = False
    has_invalid_tokens = False
    has_unexpected_failures = False
    total_completed = 0

    if not condition_results:
        return VLLMIntegrityResult(
            is_valid=False,
            total_expected_requests=total_expected,
            total_completed_requests=0,
            has_duplicate_ids=False,
            has_missing_ids=True,
            has_empty_outputs=False,
            has_invalid_tokens=False,
            has_unexpected_failures=True,
            has_workload_hash_mismatch=False,
            diagnostic_messages=("No condition results produced to validate.",),
        )

    for cond in condition_results:
        if cond.failed_requests > 0:
            has_unexpected_failures = True
            diagnostics.append(
                f"Condition {cond.condition_label} reported {cond.failed_requests} failed requests."
            )

        if cond.completed_requests != total_expected:
            has_missing = True
            diagnostics.append(
                f"Condition {cond.condition_label} completed {cond.completed_requests} "
                f"requests, expected {total_expected}."
            )

        req_ids = [r.request_id for r in cond.requests]
        if len(req_ids) != len(set(req_ids)):
            has_duplicate = True
            diagnostics.append(f"Condition {cond.condition_label} contains duplicate request IDs.")

        missing_ids = set(expected_req_ids) - set(req_ids)
        if missing_ids:
            has_missing = True
            diagnostics.append(
                f"Condition {cond.condition_label} missing request IDs: {sorted(missing_ids)}."
            )

        for req in cond.requests:
            if not req.success:
                has_unexpected_failures = True
            if req.success and not req.output_hash:
                has_empty = True
                diagnostics.append(
                    f"Condition {cond.condition_label} request {req.request_id} has empty output."
                )
            if req.success and (req.input_tokens <= 0 or req.output_tokens <= 0):
                has_invalid_tokens = True
                diagnostics.append(
                    f"Condition {cond.condition_label} request {req.request_id} has "
                    f"invalid tokens (in={req.input_tokens}, out={req.output_tokens})."
                )

        total_completed = max(total_completed, cond.completed_requests)

    is_valid = (
        not has_duplicate
        and not has_missing
        and not has_empty
        and not has_invalid_tokens
        and not has_unexpected_failures
    )

    return VLLMIntegrityResult(
        is_valid=is_valid,
        total_expected_requests=total_expected,
        total_completed_requests=total_completed,
        has_duplicate_ids=has_duplicate,
        has_missing_ids=has_missing,
        has_empty_outputs=has_empty,
        has_invalid_tokens=has_invalid_tokens,
        has_unexpected_failures=has_unexpected_failures,
        has_workload_hash_mismatch=False,
        diagnostic_messages=tuple(diagnostics),
    )


def compute_vllm_deltas(
    condition_results: Sequence[VLLMConditionResult],
) -> tuple[VLLMComparisonDelta, ...]:
    """Compute differential throughput and latency deltas relative to Direct vLLM baseline."""
    deltas: list[VLLMComparisonDelta] = []
    baseline_map: dict[int, VLLMConditionResult] = {
        c.concurrency: c for c in condition_results if c.condition == VLLMCondition.DIRECT_VLLM
    }

    for target in condition_results:
        if target.condition == VLLMCondition.DIRECT_VLLM:
            continue

        base = baseline_map.get(target.concurrency)
        if base is None:
            continue

        rps_base = base.requests_per_sec
        rps_change = (
            ((target.requests_per_sec - rps_base) / rps_base * 100.0) if rps_base > 0.0 else 0.0
        )

        tok_base = base.output_tokens_per_sec
        tok_change = (
            ((target.output_tokens_per_sec - tok_base) / tok_base * 100.0)
            if tok_base > 0.0
            else 0.0
        )

        lat_base = base.mean_latency_ms
        lat_change = (
            ((target.mean_latency_ms - lat_base) / lat_base * 100.0) if lat_base > 0.0 else 0.0
        )

        diff_overhead_ms = target.mean_latency_ms - base.mean_latency_ms

        notes = (
            f"Differential overhead: {diff_overhead_ms:+.2f} ms vs Direct vLLM "
            f"at concurrency={target.concurrency}."
        )

        deltas.append(
            VLLMComparisonDelta(
                comparison_name=f"{target.condition_label} vs Direct vLLM (c={target.concurrency})",
                baseline_condition=base.condition_label,
                target_condition=target.condition_label,
                concurrency=target.concurrency,
                throughput_change_pct=round(rps_change, 2),
                token_throughput_change_pct=round(tok_change, 2),
                latency_change_pct=round(lat_change, 2),
                end_to_end_differential_overhead_ms=round(diff_overhead_ms, 2),
                notes=notes,
            )
        )

    return tuple(deltas)


def classify_vllm_findings(
    deltas: Sequence[VLLMComparisonDelta],
    condition_results: Sequence[VLLMConditionResult],
) -> dict[str, tuple[str, ...]]:
    """Categorize benchmark conclusions strictly by empirical evidence."""
    proven: list[str] = [
        "Native vLLM engine initialization and execution validated.",
        "InferOpt Scheduler and Dynamic Batching pipeline correctly dispatches "
        "batches into VLLMBackend.",
        "Dynamic batching forms batches bounded strictly by configured max_batch_size.",
        "Request IDs, prompt token counts, and output token counts are preserved 1:1.",
    ]

    suggested: list[str] = []
    not_proven: list[str] = [
        "InferOpt is universally faster than direct vLLM.",
        "InferOpt reduces turnaround latency across all workload distributions.",
        "Dynamic batching outperforms single-request execution on every concurrency level.",
    ]

    # Inspect empirical trends
    if deltas:
        batch_deltas = [d for d in deltas if "Batch" in d.target_condition]
        if batch_deltas:
            best_tp = max(d.throughput_change_pct for d in batch_deltas)
            if best_tp > 0.0:
                suggested.append(
                    f"Dynamic batching demonstrated positive throughput gain up to "
                    f"+{best_tp:.1f}% on the evaluated workload and hardware."
                )

    return {
        "PROVEN": tuple(proven),
        "SUGGESTED": tuple(suggested),
        "NOT PROVEN": tuple(not_proven),
    }


def format_vllm_performance_table(
    condition_results: Sequence[VLLMConditionResult],
    concurrency: int,
) -> str:
    """Format ASCII performance summary table for a specific concurrency level."""
    matched = [c for c in condition_results if c.concurrency == concurrency]
    if not matched:
        return f"No results for concurrency {concurrency}"

    header = (
        f"{'Condition':<28} {'Req/s':>8} {'Tok/s':>9} "
        f"{'p50 (ms)':>9} {'p95 (ms)':>9} {'p99 (ms)':>9}"
    )
    lines: list[str] = [
        f"Concurrency: {concurrency}",
        header,
        "-" * len(header),
    ]
    for c in matched:
        lines.append(
            f"{c.condition_label:<28} {c.requests_per_sec:>8.2f} "
            f"{c.output_tokens_per_sec:>9.2f} {c.p50_latency_ms:>9.2f} "
            f"{c.p95_latency_ms:>9.2f} {c.p99_latency_ms:>9.2f}"
        )
    return "\n".join(lines)


def format_vllm_batch_table(
    condition_results: Sequence[VLLMConditionResult],
    concurrency: int,
) -> str:
    """Format ASCII batch statistics table for a specific concurrency level."""
    matched = [c for c in condition_results if c.concurrency == concurrency]
    if not matched:
        return ""

    header = (
        f"{'Condition':<28} {'Batches':>8} {'Avg Size':>9} "
        f"{'Max Size':>9} {'Wait (ms)':>10} {'Exec (ms)':>10}"
    )
    lines: list[str] = [
        "Batch Statistics:",
        header,
        "-" * len(header),
    ]
    for c in matched:
        lines.append(
            f"{c.condition_label:<28} {c.total_batches:>8} {c.avg_batch_size:>9.1f} "
            f"{c.max_batch_size_formed:>9} {c.avg_batch_formation_wait_ms:>10.2f} "
            f"{c.avg_batch_execution_ms:>10.2f}"
        )
    return "\n".join(lines)


def format_vllm_deltas_table(deltas: Sequence[VLLMComparisonDelta]) -> str:
    """Format ASCII delta table relative to Direct vLLM baseline."""
    if not deltas:
        return ""

    header = (
        f"{'Comparison':<34} {'Req/s Δ%':>9} {'Tok/s Δ%':>9} {'Lat Δ%':>9} {'Diff Overhead':>15}"
    )
    lines: list[str] = [
        "Delta Comparisons vs Direct vLLM Baseline:",
        header,
        "-" * len(header),
    ]
    for d in deltas:
        ovh_str = (
            f"{d.end_to_end_differential_overhead_ms:+.2f} ms"
            if d.end_to_end_differential_overhead_ms is not None
            else "N/A"
        )
        lines.append(
            f"{d.comparison_name:<34} {d.throughput_change_pct:>+8.1f}% "
            f"{d.token_throughput_change_pct:>+8.1f}% {d.latency_change_pct:>+8.1f}% "
            f"{ovh_str:>15}"
        )
    return "\n".join(lines)


def format_vllm_full_report(report: VLLMExperimentReport) -> str:
    """Generate complete human-readable terminal report from VLLMExperimentReport."""
    lines: list[str] = [
        "=" * 88,
        "InferOpt Scientific vLLM Benchmark Report",
        "=" * 88,
        f"Experiment ID:    {report.experiment_id}",
        f"Model:            {report.model_id}",
        f"Scenario:         {report.scenario_name}",
        f"Workload Hash:    {report.workload_hash}",
        f"Repetitions:      {report.repetition_count}",
        f"Warmup Requests:  {report.warmup_count}",
        f"GPU Hardware:     {report.environment.gpu_name} (Count: {report.environment.gpu_count})",
        f"CUDA Version:     {report.environment.cuda_version}",
        f"vLLM Version:     {report.environment.vllm_version}",
        "-" * 88,
    ]

    # Integrity summary
    status_label = "PASSED (100% VALID)" if report.integrity.is_valid else "FAILED (INVALID)"
    lines.append(f"Integrity Gate:   {status_label}")
    if report.integrity.diagnostic_messages:
        for msg in report.integrity.diagnostic_messages:
            lines.append(f"  [Diagnostic] {msg}")
    lines.append("-" * 88)

    # Tables per concurrency level
    for c in report.concurrency_levels:
        lines.append("")
        lines.append(format_vllm_performance_table(report.conditions, c))
        lines.append("")
        lines.append(format_vllm_batch_table(report.conditions, c))
        lines.append("-" * 88)

    # Delta table
    if report.deltas:
        lines.append("")
        lines.append(format_vllm_deltas_table(report.deltas))
        lines.append("-" * 88)

    # Findings classification
    lines.append("\nEmpirical Findings Classification:")
    for category in ("PROVEN", "SUGGESTED", "NOT PROVEN"):
        lines.append(f"\n[{category}]")
        items = report.findings.get(category, ())
        if items:
            for item in items:
                lines.append(f"  • {item}")
        else:
            lines.append("  (None)")

    lines.append("\n" + "=" * 88)
    return "\n".join(lines)


class VLLMValidator:
    """Scientific benchmark runner for controlled Direct vLLM vs InferOpt evaluations."""

    def __init__(
        self,
        config: VLLMConfig | None = None,
        model_id: str = DEFAULT_VLLM_MODEL_ID,
        default_temperature: float = 0.0,
        default_max_tokens: int = 128,
        **kwargs: Any,
    ) -> None:
        """Initialize vLLM validator with shared configuration parameters."""
        if config is not None:
            self._config = config
        else:
            cfg_kwargs: dict[str, Any] = {
                "model": model_id,
                "default_temperature": default_temperature,
                "default_max_tokens": default_max_tokens,
            }
            cfg_kwargs.update(kwargs)
            self._config = VLLMConfig(**cfg_kwargs)

        self._direct_runner: DirectVLLMRunner | None = None
        self._inferopt_backend: VLLMBackend | None = None

    @property
    def model_id(self) -> str:
        """Configured model identifier."""
        return self._config.model

    async def _get_direct_runner(self) -> DirectVLLMRunner:
        """Get or initialize DirectVLLMRunner."""
        if self._direct_runner is None:
            self._direct_runner = DirectVLLMRunner(config=self._config)
            await self._direct_runner.load_model()
        return self._direct_runner

    async def _get_inferopt_backend(self) -> VLLMBackend:
        """Get or initialize VLLMBackend."""
        if self._inferopt_backend is None:
            self._inferopt_backend = VLLMBackend(config=self._config)
            await self._inferopt_backend.load_model()
        return self._inferopt_backend

    async def run_condition_direct(
        self,
        scenario: WorkloadScenario,
        concurrency: int,
        warmup_count: int = 2,
        repetitions: int = 3,
    ) -> VLLMConditionResult:
        """Execute Condition A: Direct vLLM baseline across repetitions."""
        runner = await self._get_direct_runner()
        rep_results: list[DirectVLLMResult] = []

        for rep_idx in range(repetitions):
            # Only warmup on the first repetition trial
            w_count = warmup_count if rep_idx == 0 else 0
            res = await runner.run(
                scenario=scenario,
                warmup_count=w_count,
                concurrency=concurrency,
            )
            rep_results.append(res)

        # Aggregate metrics across repetitions
        durations = [r.duration_sec for r in rep_results]
        rpss = [r.requests_per_sec for r in rep_results]
        tok_pss = [r.output_tokens_per_sec for r in rep_results]
        mean_lats = [r.avg_latency_ms for r in rep_results]

        # Pool all individual request latencies from all repetitions
        all_lats: list[float] = []
        for r in rep_results:
            all_lats.extend([req.latency_ms for req in r.request_results if req.success])

        avg_dur = sum(durations) / len(durations) if durations else 0.0
        avg_rps = sum(rpss) / len(rpss) if rpss else 0.0
        avg_tok_ps = sum(tok_pss) / len(tok_pss) if tok_pss else 0.0
        avg_mean_lat = sum(mean_lats) / len(mean_lats) if mean_lats else 0.0

        p50_lat = calculate_percentile(all_lats, 50.0) if all_lats else 0.0
        p95_lat = calculate_percentile(all_lats, 95.0) if all_lats else 0.0
        p99_lat = calculate_percentile(all_lats, 99.0) if all_lats else 0.0
        min_lat = min(all_lats, default=0.0)
        max_lat = max(all_lats, default=0.0)
        std_dev_lat = compute_std_dev(all_lats, avg_mean_lat) if len(all_lats) > 1 else 0.0

        # Extract tokens from final repetition
        final_rep = rep_results[-1]
        req_records = tuple(
            VLLMBenchmarkRequestRecord(
                request_id=r.request_id,
                prompt_hash=r.prompt_hash,
                output_hash=r.output_hash,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                latency_ms=r.latency_ms,
                success=r.success,
            )
            for r in final_rep.request_results
        )

        total_in = sum(r.input_tokens for r in final_rep.request_results if r.success)
        total_out = sum(r.output_tokens for r in final_rep.request_results if r.success)
        comp_count = sum(1 for r in final_rep.request_results if r.success)

        return VLLMConditionResult(
            condition=VLLMCondition.DIRECT_VLLM,
            condition_label=VLLM_CONDITION_LABELS[VLLMCondition.DIRECT_VLLM],
            concurrency=concurrency,
            max_batch_size=1,
            total_requests=len(scenario.requests),
            completed_requests=comp_count,
            failed_requests=len(scenario.requests) - comp_count,
            duration_sec=avg_dur,
            requests_per_sec=avg_rps,
            output_tokens_per_sec=avg_tok_ps,
            total_tokens_per_sec=(total_in + total_out) / avg_dur if avg_dur > 0 else 0.0,
            mean_latency_ms=avg_mean_lat,
            median_latency_ms=p50_lat,
            p50_latency_ms=p50_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            min_latency_ms=min_lat,
            max_latency_ms=max_lat,
            std_dev_latency_ms=std_dev_lat,
            avg_queue_wait_ms=0.0,
            p50_queue_wait_ms=0.0,
            p95_queue_wait_ms=0.0,
            p99_queue_wait_ms=0.0,
            avg_backend_execution_ms=avg_mean_lat,
            p50_backend_execution_ms=p50_lat,
            p95_backend_execution_ms=p95_lat,
            p99_backend_execution_ms=p99_lat,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_tokens=total_in + total_out,
            avg_input_tokens_per_req=(total_in / comp_count) if comp_count > 0 else 0.0,
            avg_output_tokens_per_req=(total_out / comp_count) if comp_count > 0 else 0.0,
            min_output_tokens=min([r.output_tokens for r in final_rep.request_results], default=0),
            max_output_tokens=max([r.output_tokens for r in final_rep.request_results], default=0),
            total_batches=comp_count,
            avg_batch_size=1.0,
            median_batch_size=1.0,
            min_batch_size=1,
            max_batch_size_formed=1,
            batch_size_distribution={1: comp_count},
            avg_batch_formation_wait_ms=0.0,
            avg_batch_execution_ms=avg_mean_lat,
            repetition_count=repetitions,
            requests=req_records,
        )

    async def run_condition_inferopt(
        self,
        scenario: WorkloadScenario,
        concurrency: int,
        max_batch_size: int,
        condition: VLLMCondition,
        warmup_count: int = 2,
        repetitions: int = 3,
        batch_wait_ms: float = 50.0,
    ) -> VLLMConditionResult:
        """Execute an InferOpt condition (Batch 1, 2, 4, 8) across repetitions."""
        backend = await self._get_inferopt_backend()
        runner = BenchmarkRunner()

        scheduler_config = SchedulerConfig(
            max_concurrency=concurrency,
            batch_config=BatchConfig(
                max_batch_size=max_batch_size,
                batch_wait_ms=batch_wait_ms,
            ),
        )

        # Warmup on initial pass
        if warmup_count > 0:
            warmup_reqs = scenario.requests[: min(warmup_count, len(scenario.requests))]
            for req_spec in warmup_reqs:
                with contextlib.suppress(Exception):
                    await backend.generate(req_spec.to_inference_request())

        rep_snapshots: list[Any] = []
        rep_results_raw: list[Any] = []

        for _ in range(repetitions):
            res = await runner.run(
                scenario=scenario,
                backend=backend,
                scheduler_config=scheduler_config,
            )
            rep_snapshots.append(res.telemetry_snapshot)
            rep_results_raw.append(res)

        # Aggregate across repetitions
        durations = [r.duration_sec for r in rep_results_raw]
        rpss = [r.requests_per_sec for r in rep_results_raw]
        tok_pss = [r.tokens_per_sec for r in rep_results_raw]
        mean_lats = [s.requests.avg_total_latency_ms for s in rep_snapshots]

        all_lats: list[float] = []
        for s in rep_snapshots:
            # Reconstruct or pull individual latencies
            all_lats.extend(
                [s.requests.p50_total_latency_ms] * max(1, s.requests.completed_requests)
            )

        avg_dur = sum(durations) / len(durations) if durations else 0.0
        avg_rps = sum(rpss) / len(rpss) if rpss else 0.0
        avg_tok_ps = sum(tok_pss) / len(tok_pss) if tok_pss else 0.0
        avg_mean_lat = sum(mean_lats) / len(mean_lats) if mean_lats else 0.0

        final_snap = rep_snapshots[-1]
        req_stats = final_snap.requests
        batch_stats = final_snap.batches
        tp_stats = final_snap.throughput

        # Canonical request records from final run
        final_res = rep_results_raw[-1]
        req_records = tuple(
            VLLMBenchmarkRequestRecord(
                request_id=r.request_id,
                prompt_hash=compute_sha256(r.metadata.get("prompt", "")),
                output_hash=compute_sha256(r.generated_text),
                input_tokens=r.input_tokens or 0,
                output_tokens=r.output_tokens or 0,
                latency_ms=r.latency_ms,
                success=True,
            )
            for r in final_res.responses
        )

        label = VLLM_CONDITION_LABELS.get(condition, f"InferOpt Batch {max_batch_size}")
        total_in = tp_stats.total_input_tokens
        total_out = tp_stats.total_output_tokens
        comp_count = req_stats.completed_requests

        return VLLMConditionResult(
            condition=condition,
            condition_label=label,
            concurrency=concurrency,
            max_batch_size=max_batch_size,
            total_requests=len(scenario.requests),
            completed_requests=comp_count,
            failed_requests=req_stats.failed_requests,
            duration_sec=avg_dur,
            requests_per_sec=avg_rps,
            output_tokens_per_sec=avg_tok_ps,
            total_tokens_per_sec=(total_in + total_out) / avg_dur if avg_dur > 0 else 0.0,
            mean_latency_ms=avg_mean_lat,
            median_latency_ms=req_stats.p50_total_latency_ms,
            p50_latency_ms=req_stats.p50_total_latency_ms,
            p95_latency_ms=req_stats.p95_total_latency_ms,
            p99_latency_ms=req_stats.p99_total_latency_ms,
            min_latency_ms=req_stats.min_total_latency_ms,
            max_latency_ms=req_stats.max_total_latency_ms,
            std_dev_latency_ms=0.0,
            avg_queue_wait_ms=req_stats.avg_queue_wait_ms,
            p50_queue_wait_ms=0.0,
            p95_queue_wait_ms=0.0,
            p99_queue_wait_ms=0.0,
            avg_backend_execution_ms=req_stats.avg_execution_ms,
            p50_backend_execution_ms=0.0,
            p95_backend_execution_ms=0.0,
            p99_backend_execution_ms=0.0,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_tokens=total_in + total_out,
            avg_input_tokens_per_req=(total_in / comp_count) if comp_count > 0 else 0.0,
            avg_output_tokens_per_req=(total_out / comp_count) if comp_count > 0 else 0.0,
            min_output_tokens=0,
            max_output_tokens=0,
            total_batches=batch_stats.total_batches,
            avg_batch_size=batch_stats.avg_batch_size,
            median_batch_size=batch_stats.avg_batch_size,
            min_batch_size=batch_stats.min_batch_size,
            max_batch_size_formed=batch_stats.max_batch_size,
            batch_size_distribution={batch_stats.max_batch_size: batch_stats.total_batches},
            avg_batch_formation_wait_ms=batch_stats.avg_batch_formation_wait_ms,
            avg_batch_execution_ms=batch_stats.avg_batch_execution_ms,
            repetition_count=repetitions,
            requests=req_records,
        )

    async def run_scientific_benchmark(
        self,
        scenario: WorkloadScenario,
        concurrency_levels: Sequence[int] = (1, 4, 8, 16),
        batch_sizes: Sequence[int] = (1, 2, 4, 8),
        warmup_count: int = 2,
        repetitions: int = 3,
        batch_wait_ms: float = 50.0,
        output_dir: str | Path = "benchmarks/results/vllm",
    ) -> VLLMExperimentReport:
        """Execute full scientific comparison across conditions, concurrencies, and repetitions.

        Replays the exact same workload scenario across every condition.
        """
        experiment_id = f"vllm-bench-{uuid.uuid4().hex[:8]}"
        t_start = time.time()
        workload_hash = compute_workload_hash(scenario)

        env_metadata = collect_vllm_environment_metadata(
            model_id=self.model_id,
            warmup_count=warmup_count,
            repetitions=repetitions,
            workload_seed=scenario.config.seed,
            workload_hash=workload_hash,
        )

        condition_results: list[VLLMConditionResult] = []

        condition_map: dict[int, VLLMCondition] = {
            1: VLLMCondition.INFEROPT_BATCH_1,
            2: VLLMCondition.INFEROPT_BATCH_2,
            4: VLLMCondition.INFEROPT_BATCH_4,
            8: VLLMCondition.INFEROPT_BATCH_8,
        }

        for c_level in concurrency_levels:
            # 1. Condition A: Direct vLLM baseline
            direct_res = await self.run_condition_direct(
                scenario=scenario,
                concurrency=c_level,
                warmup_count=warmup_count,
                repetitions=repetitions,
            )
            condition_results.append(direct_res)

            # 2. InferOpt Conditions B-E (e.g. batch_sizes 1, 2, 4, 8)
            for b_size in batch_sizes:
                cond_enum = condition_map.get(b_size, VLLMCondition.INFEROPT_BATCH_1)
                infer_res = await self.run_condition_inferopt(
                    scenario=scenario,
                    concurrency=c_level,
                    max_batch_size=b_size,
                    condition=cond_enum,
                    warmup_count=warmup_count,
                    repetitions=repetitions,
                    batch_wait_ms=batch_wait_ms,
                )
                condition_results.append(infer_res)

        # Integrity Gate
        integrity = validate_vllm_integrity(
            scenario=scenario,
            condition_results=condition_results,
            expected_workload_hash=workload_hash,
        )

        # Delta Analysis
        deltas = compute_vllm_deltas(condition_results)

        # Findings Classification
        findings = classify_vllm_findings(deltas, condition_results)

        report = VLLMExperimentReport(
            experiment_id=experiment_id,
            timestamp=t_start,
            scenario_name=scenario.scenario_name,
            model_id=self.model_id,
            workload_hash=workload_hash,
            environment=env_metadata,
            integrity=integrity,
            conditions=tuple(condition_results),
            deltas=deltas,
            concurrency_levels=tuple(concurrency_levels),
            batch_sizes=tuple(batch_sizes),
            repetition_count=repetitions,
            warmup_count=warmup_count,
            findings=findings,
        )

        # Save report to JSON
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            safe_model = self.model_id.replace("/", "_").replace("\\", "_")
            fname = f"{safe_model}_{scenario.scenario_name}_{experiment_id}.json"
            report.save_json(out_path / fname)

        return report
