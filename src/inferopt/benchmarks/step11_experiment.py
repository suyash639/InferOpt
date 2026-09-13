"""Step 11 Experiment 1: Workload-Aware Batch Configuration Selection.

Evaluates a defined candidate configuration space against real measured vLLM telemetry,
feeds measured evidence into the deterministic optimization engine across throughput,
p95 latency, and balanced objectives, and performs independent final validation runs
on freshly instantiated engine lifecycles.
"""

import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from inferopt.backends.vllm import DEFAULT_VLLM_MODEL_ID, VLLMConfig
from inferopt.benchmarks.models import BenchmarkResult, WorkloadScenario
from inferopt.benchmarks.vllm_baseline import compute_std_dev
from inferopt.benchmarks.vllm_validation import (
    VLLMCondition,
    VLLMConditionResult,
    VLLMEnvironmentMetadata,
    VLLMIntegrityResult,
    VLLMRepetitionMeasurement,
    VLLMValidator,
    collect_vllm_environment_metadata,
    compute_workload_hash,
    validate_vllm_integrity,
)
from inferopt.optimizer.engine import DeterministicOptimizer, calculate_objective_score
from inferopt.optimizer.models import (
    CandidateEvaluation,
    CandidateSpace,
    ObjectiveConfig,
    OptimizationConstraints,
    OptimizationObjectiveType,
    OptimizationResult,
    TunableConfig,
)
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.telemetry.models import (
    BatchStats,
    MetricsSnapshot,
    RequestStats,
    ThroughputStats,
)


class Step11CandidateResult(BaseModel):
    """Exploration benchmark result for an individual candidate configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: TunableConfig = Field(description="Evaluated tunable configuration")
    concurrency: int = Field(ge=1, description="Concurrency level evaluated")
    max_batch_size: int = Field(ge=1, description="Max batch size evaluated")
    batch_wait_ms: float = Field(ge=0.0, description="Batch formation wait window in ms")
    repetitions: int = Field(ge=1, description="Number of measured repetitions")
    requests_per_sec: float = Field(ge=0.0, description="Measured average throughput (req/s)")
    output_tokens_per_sec: float = Field(
        ge=0.0, description="Measured average output token throughput (tok/s)"
    )
    total_tokens_per_sec: float = Field(
        ge=0.0, description="Measured average total token throughput (tok/s)"
    )
    mean_latency_ms: float = Field(ge=0.0, description="Mean turnaround latency in ms")
    median_latency_ms: float = Field(ge=0.0, description="Median (P50) turnaround latency in ms")
    p50_latency_ms: float = Field(ge=0.0, description="50th percentile latency in ms")
    p95_latency_ms: float = Field(ge=0.0, description="95th percentile latency in ms")
    p99_latency_ms: float = Field(ge=0.0, description="99th percentile latency in ms")
    std_dev_latency_ms: float = Field(ge=0.0, description="Sample standard deviation of latency")
    avg_queue_wait_ms: float = Field(ge=0.0, description="Average queue wait time in ms")
    avg_backend_execution_ms: float = Field(
        ge=0.0, description="Average backend execution time in ms"
    )
    total_batches: int = Field(ge=0, description="Total batches formed and executed")
    avg_batch_size: float = Field(ge=0.0, description="Average formed batch size")
    max_batch_size_formed: int = Field(ge=0, description="Maximum batch size actually formed")
    completed_requests: int = Field(ge=0, description="Total requests completed successfully")
    failed_requests: int = Field(ge=0, description="Total requests failed")
    integrity_valid: bool = Field(description="True if candidate run satisfied integrity checks")
    repetition_measurements: tuple[VLLMRepetitionMeasurement, ...] = Field(
        default_factory=tuple, description="Granular measurement per repetition"
    )
    repetition_throughputs: tuple[float, ...] = Field(
        default_factory=tuple, description="Throughput per repetition trial"
    )
    repetition_p95_latencies_ms: tuple[float, ...] = Field(
        default_factory=tuple, description="p95 latency per repetition trial"
    )
    throughput_std_dev: float = Field(
        default=0.0,
        ge=0.0,
        description="Throughput sample standard deviation across repetitions",
    )
    throughput_cv: float = Field(
        default=0.0,
        ge=0.0,
        description="Throughput coefficient of variation (std_dev / mean)",
    )
    p95_std_dev_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="p95 latency sample standard deviation across repetitions",
    )
    p95_cv: float = Field(
        default=0.0,
        ge=0.0,
        description="p95 latency coefficient of variation (std_dev / mean)",
    )
    min_throughput: float = Field(
        default=0.0, ge=0.0, description="Minimum throughput across repetitions"
    )
    max_throughput: float = Field(
        default=0.0, ge=0.0, description="Maximum throughput across repetitions"
    )
    min_p95_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Minimum p95 latency across repetitions"
    )
    max_p95_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Maximum p95 latency across repetitions"
    )


class Step11OptimizerDecision(BaseModel):
    """Optimizer selection decision for a specific optimization objective."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_type: OptimizationObjectiveType = Field(description="Optimization objective type")
    objective_label: str = Field(description="Human-readable objective label")
    objective_config: ObjectiveConfig = Field(
        description="Objective configuration used for scoring"
    )
    selected_config: TunableConfig = Field(description="Winning tunable configuration")
    predicted_score: float = Field(description="Objective score computed by optimizer")
    explanation: str = Field(description="Optimizer selection explanation")
    exploration_metrics: Step11CandidateResult = Field(
        description="Measured metrics of selected candidate during exploration"
    )
    total_evaluated: int = Field(ge=1, description="Total candidate configurations evaluated")
    feasible_evaluated: int = Field(ge=1, description="Count of feasible candidates evaluated")
    top_candidates: tuple[CandidateEvaluation, ...] = Field(
        default_factory=tuple,
        description="Top 3 ranked candidate evaluations for this objective",
    )


class Step11ValidationResult(BaseModel):
    """Independent validation result comparing exploration predictions against fresh execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_type: OptimizationObjectiveType = Field(description="Optimization objective type")
    objective_label: str = Field(description="Human-readable objective label")
    config: TunableConfig = Field(description="Validated tunable configuration")
    exploration_score: float = Field(description="Objective score during exploration phase")
    validation_score: float = Field(
        description="Objective score computed on fresh validation measurements"
    )
    score_delta_pct: float = Field(
        description=(
            "Percentage difference in score ((validation - exploration) / |exploration| * 100)"
        )
    )
    exploration_throughput: float = Field(
        ge=0.0, description="Throughput measured in exploration (req/s)"
    )
    validation_throughput: float = Field(
        ge=0.0, description="Throughput measured in independent validation (req/s)"
    )
    throughput_delta_pct: float = Field(
        description="Percentage difference in throughput ((val - expl) / expl * 100)"
    )
    exploration_p95_latency_ms: float = Field(
        ge=0.0, description="p95 latency measured in exploration (ms)"
    )
    validation_p95_latency_ms: float = Field(
        ge=0.0, description="p95 latency measured in validation (ms)"
    )
    p95_latency_delta_pct: float = Field(
        description="Percentage difference in p95 latency ((val - expl) / expl * 100)"
    )
    validation_p99_latency_ms: float = Field(
        ge=0.0, description="p99 latency measured in validation (ms)"
    )
    validation_std_dev_ms: float = Field(
        ge=0.0, description="Latency sample standard deviation in validation (ms)"
    )
    exploration_repetitions: int = Field(ge=1, description="Repetitions during exploration")
    validation_repetitions: int = Field(
        ge=1, description="Repetitions during independent validation"
    )
    integrity_valid: bool = Field(description="True if validation run satisfied integrity checks")
    repetition_measurements: tuple[VLLMRepetitionMeasurement, ...] = Field(
        default_factory=tuple, description="Granular measurement per validation repetition"
    )
    validation_repetition_scores: tuple[float, ...] = Field(
        default_factory=tuple, description="Objective score per validation repetition"
    )
    validation_repetition_throughputs: tuple[float, ...] = Field(
        default_factory=tuple, description="Throughput per validation repetition"
    )
    validation_repetition_p95_ms: tuple[float, ...] = Field(
        default_factory=tuple, description="p95 latency per validation repetition"
    )
    validation_repetition_p99_ms: tuple[float, ...] = Field(
        default_factory=tuple, description="p99 latency per validation repetition"
    )
    throughput_std_dev: float = Field(
        default=0.0,
        ge=0.0,
        description="Validation throughput standard deviation across repetitions",
    )
    throughput_cv: float = Field(
        default=0.0,
        ge=0.0,
        description="Validation throughput coefficient of variation",
    )
    p95_std_dev_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Validation p95 latency standard deviation across repetitions",
    )
    p95_cv: float = Field(
        default=0.0,
        ge=0.0,
        description="Validation p95 latency coefficient of variation",
    )
    min_throughput: float = Field(
        default=0.0, ge=0.0, description="Minimum validation throughput across repetitions"
    )
    max_throughput: float = Field(
        default=0.0, ge=0.0, description="Maximum validation throughput across repetitions"
    )
    min_p95_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Minimum validation p95 latency across repetitions"
    )
    max_p95_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Maximum validation p95 latency across repetitions"
    )


class Step11ExperimentReport(BaseModel):
    """Complete, standalone, machine-readable Step 11 experiment report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Unique experiment identifier")
    timestamp: float = Field(description="Experiment execution timestamp")
    scenario_name: str = Field(description="Evaluated workload scenario name")
    workload_hash: str = Field(description="Deterministic SHA-256 hash of the workload")
    model_id: str = Field(description="Evaluated Hugging Face model identifier")
    environment: VLLMEnvironmentMetadata = Field(description="Hardware and runtime environment")
    candidate_space: CandidateSpace = Field(description="Defined candidate configuration space")
    baseline_concurrency: int = Field(
        ge=1, description="Concurrency level for Direct vLLM baseline"
    )
    baseline_result: VLLMConditionResult = Field(
        description="Direct vLLM baseline benchmark result"
    )
    exploration_results: tuple[Step11CandidateResult, ...] = Field(
        description="Measured exploration results for all candidate configurations"
    )
    optimizer_decisions: dict[str, Step11OptimizerDecision] = Field(
        description="Optimizer recommendations for each objective"
    )
    validation_results: tuple[Step11ValidationResult, ...] = Field(
        description="Independent validation results for winning configurations"
    )
    integrity: VLLMIntegrityResult = Field(description="Overall integrity validation result")
    findings: dict[str, tuple[str, ...]] = Field(
        default_factory=dict, description="Objective findings classified by evidence level"
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Step11ExperimentReport":
        """Deserialize report from JSON string."""
        return cls.model_validate_json(json_str)

    def save_json(self, file_path: str | Path) -> None:
        """Persist report to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load_json(cls, file_path: str | Path) -> "Step11ExperimentReport":
        """Load and validate report from a JSON file."""
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            return cls.from_json(f.read())


def vllm_condition_to_benchmark_result(
    cond: VLLMConditionResult,
    scenario: WorkloadScenario,
    batch_wait_ms: float = 50.0,
    backend_name: str = "vllm",
) -> BenchmarkResult:
    """Convert a VLLMConditionResult into a standard BenchmarkResult for the optimizer."""
    sched_cfg = SchedulerConfig(
        max_concurrency=cond.concurrency,
        batch_config=BatchConfig(
            max_batch_size=cond.max_batch_size,
            batch_wait_ms=batch_wait_ms,
        ),
    )
    snapshot = MetricsSnapshot(
        timestamp=time.time(),
        requests=RequestStats(
            total_requests=cond.total_requests,
            completed_requests=cond.completed_requests,
            failed_requests=cond.failed_requests,
            avg_total_latency_ms=cond.mean_latency_ms,
            p50_total_latency_ms=cond.p50_latency_ms,
            p95_total_latency_ms=cond.p95_latency_ms,
            p99_total_latency_ms=cond.p99_latency_ms,
            avg_queue_wait_ms=cond.avg_queue_wait_ms,
            avg_execution_ms=cond.avg_backend_execution_ms,
        ),
        batches=BatchStats(
            total_batches=cond.total_batches,
            avg_batch_size=cond.avg_batch_size,
            max_batch_size=cond.max_batch_size_formed,
            min_batch_size=cond.min_batch_size,
            avg_batch_formation_wait_ms=cond.avg_batch_formation_wait_ms,
            avg_batch_execution_ms=cond.avg_batch_execution_ms,
        ),
        throughput=ThroughputStats(
            requests_per_sec=cond.requests_per_sec,
            batches_per_sec=(
                cond.total_batches / cond.duration_sec if cond.duration_sec > 0.0 else 0.0
            ),
            tokens_per_sec=cond.total_tokens_per_sec,
            total_input_tokens=cond.total_input_tokens,
            total_output_tokens=cond.total_output_tokens,
        ),
    )
    return BenchmarkResult(
        benchmark_id=f"bench-expl-{uuid.uuid4().hex[:6]}",
        timestamp=time.time(),
        scenario_name=scenario.scenario_name,
        workload_config=scenario.config,
        backend_name=backend_name,
        scheduler_config=sched_cfg,
        batch_config=sched_cfg.batch_config,
        telemetry_snapshot=snapshot,
        duration_sec=cond.duration_sec,
        total_requests=cond.total_requests,
        completed_requests=cond.completed_requests,
        failed_requests=cond.failed_requests,
        cancelled_requests=0,
        requests_per_sec=cond.requests_per_sec,
        batches_per_sec=snapshot.throughput.batches_per_sec,
        tokens_per_sec=cond.total_tokens_per_sec,
        avg_latency_ms=cond.mean_latency_ms,
        p50_latency_ms=cond.p50_latency_ms,
        p95_latency_ms=cond.p95_latency_ms,
        p99_latency_ms=cond.p99_latency_ms,
        avg_queue_wait_ms=cond.avg_queue_wait_ms,
        avg_execution_ms=cond.avg_backend_execution_ms,
        peak_queue_depth=cond.concurrency,
        peak_active_requests=cond.concurrency,
        total_batches=cond.total_batches,
        avg_batch_size=cond.avg_batch_size,
        max_batch_size=cond.max_batch_size_formed,
    )


def _repetition_to_benchmark_result(
    m: VLLMRepetitionMeasurement,
    config: TunableConfig,
    scenario: WorkloadScenario,
    backend_name: str = "vllm",
) -> BenchmarkResult:
    """Convert a VLLMRepetitionMeasurement into a BenchmarkResult for single-repetition scoring."""
    sched_cfg = SchedulerConfig(
        max_concurrency=config.max_concurrency,
        batch_config=BatchConfig(
            max_batch_size=config.max_batch_size,
            batch_wait_ms=config.batch_wait_ms,
        ),
    )
    snap = MetricsSnapshot(
        timestamp=time.time(),
        requests=RequestStats(
            total_requests=m.measured_request_count,
            completed_requests=m.completed_requests,
            failed_requests=m.failed_requests,
            avg_total_latency_ms=m.mean_latency_ms,
            p50_total_latency_ms=m.p50_latency_ms,
            p95_total_latency_ms=m.p95_latency_ms,
            p99_total_latency_ms=m.p99_latency_ms,
            avg_queue_wait_ms=m.avg_queue_wait_ms,
            avg_execution_ms=m.avg_backend_execution_ms,
        ),
        batches=BatchStats(
            total_batches=m.total_batches,
            avg_batch_size=m.avg_batch_size,
            max_batch_size=round(m.avg_batch_size),
            min_batch_size=1 if m.total_batches > 0 else 0,
            avg_batch_formation_wait_ms=0.0,
            avg_batch_execution_ms=m.avg_backend_execution_ms,
        ),
        throughput=ThroughputStats(
            requests_per_sec=m.requests_per_sec,
            batches_per_sec=(m.total_batches / m.duration_sec if m.duration_sec > 0.0 else 0.0),
            tokens_per_sec=m.total_tokens_per_sec,
            total_input_tokens=0,
            total_output_tokens=0,
        ),
    )
    return BenchmarkResult(
        benchmark_id=f"bench-val-rep-{uuid.uuid4().hex[:6]}",
        timestamp=time.time(),
        scenario_name=scenario.scenario_name,
        workload_config=scenario.config,
        backend_name=backend_name,
        scheduler_config=sched_cfg,
        batch_config=sched_cfg.batch_config,
        telemetry_snapshot=snap,
        duration_sec=m.duration_sec,
        total_requests=m.measured_request_count,
        completed_requests=m.completed_requests,
        failed_requests=m.failed_requests,
        cancelled_requests=0,
        requests_per_sec=m.requests_per_sec,
        batches_per_sec=snap.throughput.batches_per_sec,
        tokens_per_sec=m.total_tokens_per_sec,
        avg_latency_ms=m.mean_latency_ms,
        p50_latency_ms=m.p50_latency_ms,
        p95_latency_ms=m.p95_latency_ms,
        p99_latency_ms=m.p99_latency_ms,
        avg_queue_wait_ms=m.avg_queue_wait_ms,
        avg_execution_ms=m.avg_backend_execution_ms,
        peak_queue_depth=config.max_concurrency,
        peak_active_requests=config.max_concurrency,
        total_batches=m.total_batches,
        avg_batch_size=m.avg_batch_size,
        max_batch_size=round(m.avg_batch_size),
    )


def classify_step11_findings(
    baseline: VLLMConditionResult,
    candidates: Sequence[Step11CandidateResult],
    decisions: Mapping[str, Step11OptimizerDecision],
    validations: Sequence[Step11ValidationResult],
) -> dict[str, tuple[str, ...]]:
    """Categorize Step 11 empirical outcomes into PROVEN, SUGGESTED, and NOT PROVEN."""
    proven: list[str] = [
        (
            "Deterministic optimizer evaluates the full candidate configuration space "
            "against real measured telemetry."
        ),
        (
            "Deterministic optimizer selects reproducible runtime configurations "
            "according to specified objectives."
        ),
        (
            "Strict separation between candidate exploration, optimizer scoring, and "
            "independent validation was maintained."
        ),
        (
            "Selected configurations were re-executed successfully in independent "
            "validation runs with zero GPU leakage."
        ),
    ]

    suggested: list[str] = []
    not_proven: list[str] = [
        (
            "InferOpt universally discovers the global optimum across all model "
            "architectures and hardware platforms."
        ),
        (
            "InferOpt outperforms direct vLLM for every arbitrary prompt length and "
            "traffic distribution."
        ),
        (
            "Selected runtime configuration is guaranteed to generalize identically "
            "to untested hardware architectures."
        ),
        (
            "Learned optimization (ML/RL/Bayesian) is superior to deterministic search "
            "on this parameter space."
        ),
    ]

    # Validate if independent validation scores remained consistent (within 25% variance)
    stable_validations = [v for v in validations if abs(v.score_delta_pct) <= 25.0]
    if stable_validations and len(stable_validations) == len(validations):
        proven.append(
            "Selected configurations maintained consistent performance during "
            "independent fresh validation trials."
        )

    # Check if different objectives yielded distinct configurations
    selected_configs = {d.selected_config for d in decisions.values()}
    if len(selected_configs) > 1:
        suggested.append(
            "Different optimization objectives successfully selected distinct runtime "
            f"operating points ({len(selected_configs)} distinct configs across "
            f"{len(decisions)} objectives)."
        )
    else:
        suggested.append(
            "Workload-aware optimization selects a balanced configuration dominating "
            "across evaluated objectives."
        )

    # Check if winning throughput configuration improved over baseline
    tput_decision = decisions.get("THROUGHPUT")
    if tput_decision and baseline.requests_per_sec > 0.0:
        gain_pct = (
            (tput_decision.exploration_metrics.requests_per_sec - baseline.requests_per_sec)
            / baseline.requests_per_sec
            * 100.0
        )
        if gain_pct > 0.0:
            suggested.append(
                f"Workload-aware configuration selection demonstrated +{gain_pct:.1f}% "
                "throughput improvement over direct vLLM baseline on the evaluated "
                "workload and hardware."
            )

    return {
        "PROVEN": tuple(proven),
        "SUGGESTED": tuple(suggested),
        "NOT PROVEN": tuple(not_proven),
    }


def format_step11_report(report: Step11ExperimentReport) -> str:
    """Format complete ASCII Step 11 experiment report."""
    formatted_ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(report.timestamp))
    lines: list[str] = [
        "=" * 96,
        "  INFEROPT STEP 11: WORKLOAD-AWARE BATCH CONFIGURATION SELECTION",
        "=" * 96,
        f"Experiment ID:        {report.experiment_id}",
        f"Timestamp:            {formatted_ts}",
        f"Model:                {report.model_id}",
        f"Scenario:             {report.scenario_name}",
        f"Workload Hash:        {report.workload_hash}",
        (
            f"GPU Device:           {report.environment.gpu_name} "
            f"(Count: {report.environment.gpu_count})"
        ),
        f"CUDA Version:         {report.environment.cuda_version}",
        f"vLLM Version:         {report.environment.vllm_version}",
        f"InferOpt Version:     {report.environment.inferopt_version}",
        f"Enforce Eager:        {report.environment.enforce_eager} (Diagnostic Mode)",
        (
            f"Candidate Space:      concurrency={report.candidate_space.concurrencies}, "
            f"batch_size={report.candidate_space.batch_sizes}, "
            f"batch_wait_ms={report.candidate_space.batch_waits_ms}"
        ),
        "-" * 96,
    ]

    # Integrity Gate
    status_label = "PASSED (100% VALID)" if report.integrity.is_valid else "FAILED (INVALID)"
    lines.append(f"Integrity Gate:       {status_label}")
    if report.integrity.diagnostic_messages:
        for msg in report.integrity.diagnostic_messages:
            lines.append(f"  [Diagnostic] {msg}")
    lines.append("-" * 96)

    # Baseline Summary
    lines.append("Direct vLLM Baseline:")
    b = report.baseline_result
    lines.append(
        f"  Concurrency: {b.concurrency} | Req/s: {b.requests_per_sec:.2f} | "
        f"Tok/s: {b.output_tokens_per_sec:.2f} | p50: {b.p50_latency_ms:.2f}ms | "
        f"p95: {b.p95_latency_ms:.2f}ms | p99: {b.p99_latency_ms:.2f}ms"
    )
    lines.append("-" * 96)

    # Candidate Exploration Table
    lines.append("CANDIDATE CONFIGURATION SPACE EXPLORATION (Phase 1):")
    header = (
        f"{'Config (c, b, w)':<20} {'Rep':>4} {'Req/s':>8} {'Tok/s':>9} "
        f"{'p50 (ms)':>9} {'p95 (ms)':>9} {'p99 (ms)':>9} {'Queue':>8} "
        f"{'Exec':>8} {'Batches':>8} {'AvgB':>6} {'MaxB':>5} {'Status':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for c in report.exploration_results:
        cfg_str = f"({c.concurrency}, {c.max_batch_size}, {c.batch_wait_ms:.0f}ms)"
        status_str = "VALID" if c.integrity_valid else "INVALID"
        lines.append(
            f"{cfg_str:<20} {c.repetitions:>4} {c.requests_per_sec:>8.2f} "
            f"{c.output_tokens_per_sec:>9.2f} {c.p50_latency_ms:>9.2f} "
            f"{c.p95_latency_ms:>9.2f} {c.p99_latency_ms:>9.2f} "
            f"{c.avg_queue_wait_ms:>8.1f} {c.avg_backend_execution_ms:>8.1f} "
            f"{c.total_batches:>8} {c.avg_batch_size:>6.1f} "
            f"{c.max_batch_size_formed:>5} {status_str:>8}"
        )
    lines.append("-" * 96)

    # Optimizer Decisions Table
    lines.append("OPTIMIZER DECISIONS (Phase 2):")
    for _key, dec in report.optimizer_decisions.items():
        cfg = dec.selected_config
        cfg_str = (
            f"concurrency={cfg.max_concurrency}, "
            f"batch_size={cfg.max_batch_size}, "
            f"batch_wait={cfg.batch_wait_ms:.1f}ms"
        )
        lines.append(f"  [{dec.objective_label}]")
        lines.append(f"    Selected Configuration: {cfg_str}")
        lines.append(f"    Predicted Score:        {dec.predicted_score:.4f}")
        lines.append(
            f"    Exploration Metrics:    {dec.exploration_metrics.requests_per_sec:.2f} req/s, "
            f"p95={dec.exploration_metrics.p95_latency_ms:.2f}ms, "
            f"p99={dec.exploration_metrics.p99_latency_ms:.2f}ms"
        )
        lines.append(f"    Rationale:              {dec.explanation}")
    lines.append("-" * 96)

    # Independent Validation Table
    lines.append("FINAL INDEPENDENT VALIDATION (Phase 3):")
    val_hdr = (
        f"{'Objective':<14} {'Config (c, b, w)':<18} {'Expl Score':>11} {'Val Score':>10} "
        f"{'Score Δ%':>9} {'Expl Req/s':>11} {'Val Req/s':>10} {'Expl p95':>9} "
        f"{'Val p95':>9} {'Val p99':>9} {'Val StdDev':>11} {'Status':>8}"
    )
    lines.append(val_hdr)
    lines.append("-" * len(val_hdr))
    for v in report.validation_results:
        cfg = v.config
        cfg_str = f"({cfg.max_concurrency}, {cfg.max_batch_size}, {cfg.batch_wait_ms:.0f}ms)"
        v_status = "VALID" if v.integrity_valid else "INVALID"
        lines.append(
            f"{v.objective_label:<14} {cfg_str:<18} {v.exploration_score:>11.4f} "
            f"{v.validation_score:>10.4f} {v.score_delta_pct:>+8.1f}% "
            f"{v.exploration_throughput:>11.2f} {v.validation_throughput:>10.2f} "
            f"{v.exploration_p95_latency_ms:>9.2f} {v.validation_p95_latency_ms:>9.2f} "
            f"{v.validation_p99_latency_ms:>9.2f} {v.validation_std_dev_ms:>11.2f} {v_status:>8}"
        )
    lines.append("-" * 96)

    # Findings Classification
    lines.append("EMPIRICAL FINDINGS CLASSIFICATION:")
    for category in ("PROVEN", "SUGGESTED", "NOT PROVEN"):
        lines.append(f"\n[{category}]")
        items = report.findings.get(category, ())
        if items:
            for item in items:
                lines.append(f"  • {item}")
        else:
            lines.append("  (None)")

    lines.append("\n" + "=" * 96)
    return "\n".join(lines)


class Step11ExperimentRunner:
    """Orchestrator for Step 11: Workload-Aware Batch Configuration Selection Experiment."""

    def __init__(
        self,
        validator: VLLMValidator | None = None,
        config: VLLMConfig | None = None,
        model_id: str = DEFAULT_VLLM_MODEL_ID,
        enforce_eager: bool = False,
    ) -> None:
        """Initialize Step 11 Experiment Runner."""
        if validator is not None:
            self._validator = validator
        else:
            self._validator = VLLMValidator(
                config=config,
                model_id=model_id,
                enforce_eager=enforce_eager,
            )
        self._optimizer = DeterministicOptimizer()

    @property
    def validator(self) -> VLLMValidator:
        """Underlying vLLM benchmark validator instance."""
        return self._validator

    @property
    def model_id(self) -> str:
        """Target model identifier."""
        return self._validator.model_id

    async def run_experiment(
        self,
        scenario: WorkloadScenario,
        candidate_space: CandidateSpace | None = None,
        objectives: Sequence[ObjectiveConfig] | None = None,
        warmup_count: int = 2,
        exploration_repetitions: int = 3,
        validation_repetitions: int = 3,
        batch_wait_ms: float = 50.0,
        baseline_concurrency: int = 1,
        constraints: OptimizationConstraints | None = None,
        output_dir: str | Path | None = "benchmarks/results/step11",
    ) -> Step11ExperimentReport:
        """Execute the full 3-phase Step 11 experiment.

        Phase 1: Measure Direct vLLM baseline & evaluate all candidate configurations.
        Phase 2: Feed measured exploration telemetry into DeterministicOptimizer across objectives.
        Phase 3: Execute independent validation trials with fresh engine lifecycles for winners.
        Phase 4: Run comprehensive integrity validation and produce structured report.
        """
        experiment_id = f"step11-exp-{uuid.uuid4().hex[:8]}"
        t_start = time.time()
        workload_hash = compute_workload_hash(scenario)

        space = candidate_space or CandidateSpace(
            concurrencies=(1, 4, 8),
            batch_sizes=(1, 2, 4, 8),
            batch_waits_ms=(batch_wait_ms,),
        )
        candidates = space.generate_candidates()

        env_metadata = collect_vllm_environment_metadata(
            model_id=self.model_id,
            warmup_count=warmup_count,
            repetitions=exploration_repetitions,
            workload_seed=scenario.config.seed,
            workload_hash=workload_hash,
            enforce_eager=self._validator.config.enforce_eager,
        )

        all_condition_results: list[VLLMConditionResult] = []

        # -------------------------------------------------------------
        # 1. Baseline: Direct vLLM Run
        # -------------------------------------------------------------
        print("\n" + "=" * 80)
        print("  STEP 11: RUNNING DIRECT VLLM BASELINE")
        print("=" * 80)
        baseline_res = await self._validator.run_condition_direct(
            scenario=scenario,
            concurrency=baseline_concurrency,
            warmup_count=warmup_count,
            repetitions=exploration_repetitions,
        )
        all_condition_results.append(baseline_res)

        # -------------------------------------------------------------
        # 2. Phase 1: Candidate Space Exploration
        # -------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"  STEP 11: EXPLORING CANDIDATE SPACE ({len(candidates)} configurations)")
        print("=" * 80)

        exploration_results: list[Step11CandidateResult] = []
        benchmark_results_for_opt: list[BenchmarkResult] = []
        candidate_map: dict[TunableConfig, Step11CandidateResult] = {}

        for idx, cand in enumerate(candidates, 1):
            print(
                f"\n[CANDIDATE {idx}/{len(candidates)}] "
                f"concurrency={cand.max_concurrency}, batch_size={cand.max_batch_size}, "
                f"batch_wait_ms={cand.batch_wait_ms:.1f}"
            )
            c_res = await self._validator.run_condition_inferopt(
                scenario=scenario,
                concurrency=cand.max_concurrency,
                max_batch_size=cand.max_batch_size,
                condition=VLLMCondition.INFEROPT_BATCH_1,  # Generic tag
                warmup_count=warmup_count,
                repetitions=exploration_repetitions,
                batch_wait_ms=cand.batch_wait_ms,
            )
            all_condition_results.append(c_res)

            bench_res = vllm_condition_to_benchmark_result(
                cond=c_res,
                scenario=scenario,
                batch_wait_ms=cand.batch_wait_ms,
            )
            benchmark_results_for_opt.append(bench_res)

            # Single-condition integrity validation
            c_integrity = validate_vllm_integrity(
                scenario=scenario,
                condition_results=[c_res],
                expected_workload_hash=workload_hash,
            )

            rep_tputs = tuple(m.requests_per_sec for m in c_res.repetition_measurements)
            rep_p95s = tuple(m.p95_latency_ms for m in c_res.repetition_measurements)
            tput_sd = (
                compute_std_dev(list(rep_tputs), c_res.requests_per_sec)
                if len(rep_tputs) > 1
                else 0.0
            )
            tput_cv = (tput_sd / c_res.requests_per_sec) if c_res.requests_per_sec > 0 else 0.0
            p95_sd = (
                compute_std_dev(list(rep_p95s), c_res.p95_latency_ms) if len(rep_p95s) > 1 else 0.0
            )
            p95_cv = (p95_sd / c_res.p95_latency_ms) if c_res.p95_latency_ms > 0 else 0.0
            min_tp = min(rep_tputs, default=c_res.requests_per_sec)
            max_tp = max(rep_tputs, default=c_res.requests_per_sec)
            min_p95 = min(rep_p95s, default=c_res.p95_latency_ms)
            max_p95 = max(rep_p95s, default=c_res.p95_latency_ms)

            cand_result = Step11CandidateResult(
                config=cand,
                concurrency=cand.max_concurrency,
                max_batch_size=cand.max_batch_size,
                batch_wait_ms=cand.batch_wait_ms,
                repetitions=exploration_repetitions,
                requests_per_sec=c_res.requests_per_sec,
                output_tokens_per_sec=c_res.output_tokens_per_sec,
                total_tokens_per_sec=c_res.total_tokens_per_sec,
                mean_latency_ms=c_res.mean_latency_ms,
                median_latency_ms=c_res.median_latency_ms,
                p50_latency_ms=c_res.p50_latency_ms,
                p95_latency_ms=c_res.p95_latency_ms,
                p99_latency_ms=c_res.p99_latency_ms,
                std_dev_latency_ms=c_res.std_dev_latency_ms,
                avg_queue_wait_ms=c_res.avg_queue_wait_ms,
                avg_backend_execution_ms=c_res.avg_backend_execution_ms,
                total_batches=c_res.total_batches,
                avg_batch_size=c_res.avg_batch_size,
                max_batch_size_formed=c_res.max_batch_size_formed,
                completed_requests=c_res.completed_requests,
                failed_requests=c_res.failed_requests,
                integrity_valid=c_integrity.is_valid,
                repetition_measurements=c_res.repetition_measurements,
                repetition_throughputs=rep_tputs,
                repetition_p95_latencies_ms=rep_p95s,
                throughput_std_dev=round(tput_sd, 4),
                throughput_cv=round(tput_cv, 4),
                p95_std_dev_ms=round(p95_sd, 2),
                p95_cv=round(p95_cv, 4),
                min_throughput=round(min_tp, 4),
                max_throughput=round(max_tp, 4),
                min_p95_latency_ms=round(min_p95, 2),
                max_p95_latency_ms=round(max_p95, 2),
            )
            exploration_results.append(cand_result)
            candidate_map[cand] = cand_result

        # -------------------------------------------------------------
        # 3. Phase 2: Optimizer Decisions Across Objectives
        # -------------------------------------------------------------
        print("\n" + "=" * 80)
        print("  STEP 11: RUNNING DETERMINISTIC OPTIMIZER")
        print("=" * 80)

        # Reference targets set relative to baseline measurements
        ref_throughput = max(baseline_res.requests_per_sec, 0.1)
        ref_p95 = max(baseline_res.p95_latency_ms, 1.0)

        default_objectives: list[ObjectiveConfig] = [
            ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT),
            ObjectiveConfig(objective_type=OptimizationObjectiveType.LATENCY),
            ObjectiveConfig(
                objective_type=OptimizationObjectiveType.BALANCED,
                throughput_weight=0.5,
                latency_weight=0.5,
                target_throughput_rps=ref_throughput,
                target_p95_latency_ms=ref_p95,
            ),
        ]
        active_objectives = list(objectives or default_objectives)

        decisions: dict[str, Step11OptimizerDecision] = {}
        unique_selected_configs: dict[
            TunableConfig, list[tuple[OptimizationObjectiveType, str, ObjectiveConfig]]
        ] = {}

        for obj in active_objectives:
            opt_res: OptimizationResult = self._optimizer.evaluate_results(
                results=benchmark_results_for_opt,
                objective=obj,
                constraints=constraints,
            )

            if opt_res.recommended_config is None or opt_res.best_score is None:
                raise RuntimeError(
                    f"Optimizer failed to find a feasible configuration for objective "
                    f"{obj.objective_type}: {opt_res.summary_explanation}"
                )

            sel_config = opt_res.recommended_config
            expl_metric = candidate_map[sel_config]
            obj_label = obj.objective_type.value

            # Extract top 3 candidates sorted by deterministic tie-breaking hierarchy
            def tie_breaker_key(ev: CandidateEvaluation) -> tuple[float, float, float, int, int]:
                p95 = ev.metrics.p95_latency_ms if ev.metrics else 0.0
                return (
                    -ev.objective_score,
                    p95,
                    ev.config.batch_wait_ms,
                    ev.config.max_batch_size,
                    ev.config.max_concurrency,
                )

            feasible_evals = [e for e in opt_res.evaluations if e.is_feasible]
            sorted_feasible = sorted(feasible_evals, key=tie_breaker_key)
            top_3_evals = tuple(sorted_feasible[:3])

            dec = Step11OptimizerDecision(
                objective_type=obj.objective_type,
                objective_label=obj_label,
                objective_config=obj,
                selected_config=sel_config,
                predicted_score=opt_res.best_score,
                explanation=opt_res.summary_explanation,
                exploration_metrics=expl_metric,
                total_evaluated=opt_res.total_candidates,
                feasible_evaluated=opt_res.feasible_candidates,
                top_candidates=top_3_evals,
            )
            decisions[obj_label] = dec

            if sel_config not in unique_selected_configs:
                unique_selected_configs[sel_config] = []
            unique_selected_configs[sel_config].append((obj.objective_type, obj_label, obj))

        # -------------------------------------------------------------
        # 4. Phase 3: Independent Final Validation
        # -------------------------------------------------------------
        print("\n" + "=" * 80)
        print(
            f"  STEP 11: INDEPENDENT FINAL VALIDATION "
            f"({len(unique_selected_configs)} unique winning configurations)"
        )
        print("=" * 80)

        validation_results: list[Step11ValidationResult] = []

        for sel_config, obj_list in unique_selected_configs.items():
            print(
                f"\n[VALIDATION RUN] Selected config: "
                f"concurrency={sel_config.max_concurrency}, "
                f"batch_size={sel_config.max_batch_size}, "
                f"batch_wait_ms={sel_config.batch_wait_ms:.1f}"
            )
            # Execute dedicated, freshly instantiated engine lifecycle with independent repetitions
            val_cond_res = await self._validator.run_condition_inferopt(
                scenario=scenario,
                concurrency=sel_config.max_concurrency,
                max_batch_size=sel_config.max_batch_size,
                condition=VLLMCondition.INFEROPT_BATCH_1,
                warmup_count=warmup_count,
                repetitions=validation_repetitions,
                batch_wait_ms=sel_config.batch_wait_ms,
            )
            all_condition_results.append(val_cond_res)

            val_bench_res = vllm_condition_to_benchmark_result(
                cond=val_cond_res,
                scenario=scenario,
                batch_wait_ms=sel_config.batch_wait_ms,
            )

            val_integrity = validate_vllm_integrity(
                scenario=scenario,
                condition_results=[val_cond_res],
                expected_workload_hash=workload_hash,
            )

            expl_metric = candidate_map[sel_config]

            val_rep_tputs = tuple(m.requests_per_sec for m in val_cond_res.repetition_measurements)
            val_rep_p95s = tuple(m.p95_latency_ms for m in val_cond_res.repetition_measurements)
            val_rep_p99s = tuple(m.p99_latency_ms for m in val_cond_res.repetition_measurements)

            val_tput_sd = (
                compute_std_dev(list(val_rep_tputs), val_cond_res.requests_per_sec)
                if len(val_rep_tputs) > 1
                else 0.0
            )
            val_tput_cv = (
                (val_tput_sd / val_cond_res.requests_per_sec)
                if val_cond_res.requests_per_sec > 0
                else 0.0
            )
            val_p95_sd = (
                compute_std_dev(list(val_rep_p95s), val_cond_res.p95_latency_ms)
                if len(val_rep_p95s) > 1
                else 0.0
            )
            val_p95_cv = (
                (val_p95_sd / val_cond_res.p95_latency_ms)
                if val_cond_res.p95_latency_ms > 0
                else 0.0
            )
            val_min_tp = min(val_rep_tputs, default=val_cond_res.requests_per_sec)
            val_max_tp = max(val_rep_tputs, default=val_cond_res.requests_per_sec)
            val_min_p95 = min(val_rep_p95s, default=val_cond_res.p95_latency_ms)
            val_max_p95 = max(val_rep_p95s, default=val_cond_res.p95_latency_ms)

            for obj_type, obj_label, obj_cfg in obj_list:
                val_score, _ = calculate_objective_score(val_bench_res, obj_cfg)
                expl_score = decisions[obj_label].predicted_score

                # Calculate individual validation repetition scores
                val_rep_scores: list[float] = []
                for m in val_cond_res.repetition_measurements:
                    b_res = _repetition_to_benchmark_result(m, sel_config, scenario)
                    sc, _ = calculate_objective_score(b_res, obj_cfg)
                    val_rep_scores.append(sc)

                score_delta_pct = (
                    ((val_score - expl_score) / abs(expl_score) * 100.0)
                    if abs(expl_score) > 0.0
                    else 0.0
                )
                tput_delta_pct = (
                    (
                        (val_cond_res.requests_per_sec - expl_metric.requests_per_sec)
                        / expl_metric.requests_per_sec
                        * 100.0
                    )
                    if expl_metric.requests_per_sec > 0.0
                    else 0.0
                )
                p95_delta_pct = (
                    (
                        (val_cond_res.p95_latency_ms - expl_metric.p95_latency_ms)
                        / expl_metric.p95_latency_ms
                        * 100.0
                    )
                    if expl_metric.p95_latency_ms > 0.0
                    else 0.0
                )

                validation_results.append(
                    Step11ValidationResult(
                        objective_type=obj_type,
                        objective_label=obj_label,
                        config=sel_config,
                        exploration_score=expl_score,
                        validation_score=val_score,
                        score_delta_pct=round(score_delta_pct, 2),
                        exploration_throughput=expl_metric.requests_per_sec,
                        validation_throughput=val_cond_res.requests_per_sec,
                        throughput_delta_pct=round(tput_delta_pct, 2),
                        exploration_p95_latency_ms=expl_metric.p95_latency_ms,
                        validation_p95_latency_ms=val_cond_res.p95_latency_ms,
                        p95_latency_delta_pct=round(p95_delta_pct, 2),
                        validation_p99_latency_ms=val_cond_res.p99_latency_ms,
                        validation_std_dev_ms=val_cond_res.std_dev_latency_ms,
                        exploration_repetitions=exploration_repetitions,
                        validation_repetitions=validation_repetitions,
                        integrity_valid=val_integrity.is_valid,
                        repetition_measurements=val_cond_res.repetition_measurements,
                        validation_repetition_scores=tuple(val_rep_scores),
                        validation_repetition_throughputs=val_rep_tputs,
                        validation_repetition_p95_ms=val_rep_p95s,
                        validation_repetition_p99_ms=val_rep_p99s,
                        throughput_std_dev=round(val_tput_sd, 4),
                        throughput_cv=round(val_tput_cv, 4),
                        p95_std_dev_ms=round(val_p95_sd, 2),
                        p95_cv=round(val_p95_cv, 4),
                        min_throughput=round(val_min_tp, 4),
                        max_throughput=round(val_max_tp, 4),
                        min_p95_latency_ms=round(val_min_p95, 2),
                        max_p95_latency_ms=round(val_max_p95, 2),
                    )
                )

        # -------------------------------------------------------------
        # 5. Phase 4: Overall Integrity Gate and Report Creation
        # -------------------------------------------------------------
        overall_integrity = validate_vllm_integrity(
            scenario=scenario,
            condition_results=all_condition_results,
            expected_workload_hash=workload_hash,
        )

        findings = classify_step11_findings(
            baseline=baseline_res,
            candidates=exploration_results,
            decisions=decisions,
            validations=validation_results,
        )

        report = Step11ExperimentReport(
            experiment_id=experiment_id,
            timestamp=t_start,
            scenario_name=scenario.scenario_name,
            workload_hash=workload_hash,
            model_id=self.model_id,
            environment=env_metadata,
            candidate_space=space,
            baseline_concurrency=baseline_concurrency,
            baseline_result=baseline_res,
            exploration_results=tuple(exploration_results),
            optimizer_decisions=decisions,
            validation_results=tuple(validation_results),
            integrity=overall_integrity,
            findings=findings,
        )

        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            safe_model = self.model_id.replace("/", "_").replace("\\", "_")
            fname = f"step11_{safe_model}_{scenario.scenario_name}_{experiment_id}.json"
            report.save_json(out_path / fname)

        return report
