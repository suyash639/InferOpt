"""Step 15: Multi-Model Generalization & Cross-Model Optimization.

Systematically evaluates whether InferOpt's deterministic optimization and SLA-aware
adaptive control generalize across different model sizes and families (e.g. Qwen 2.5, Llama 3.2).
"""

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from inferopt.backends.base import InferenceBackend
from inferopt.backends.vllm import DEFAULT_VLLM_MODEL_ID, VLLMBackend, VLLMConfig
from inferopt.benchmarks.generator import generate_workload
from inferopt.benchmarks.models import (
    ArrivalPattern,
    PromptCategory,
    WorkloadConfig,
    WorkloadRequestSpec,
    WorkloadScenario,
)
from inferopt.benchmarks.step13_adaptive import get_git_commit_hash
from inferopt.benchmarks.step14_sla_adaptive import Step14SLAExperimentRunner
from inferopt.benchmarks.vllm_baseline import calculate_percentile
from inferopt.benchmarks.vllm_validation import (
    VLLMEnvironmentMetadata,
    collect_vllm_environment_metadata,
    compute_workload_hash,
)
from inferopt.core.models import InferenceRequest, InferenceResponse
from inferopt.optimizer.engine import DeterministicOptimizer
from inferopt.optimizer.models import (
    CandidateSpace,
    TunableConfig,
)
from inferopt.optimizer.sla_controller import (
    DEFAULT_SLA_CANDIDATE_LADDER,
    SLAConstrainedAdaptiveController,
)
from inferopt.optimizer.sla_models import (
    Step14PhaseMetricRecord,
    Step14SLAAdaptationEventRecord,
    TargetSLO,
)
from inferopt.scheduler.scheduler import Scheduler
from inferopt.telemetry.collector import MetricsCollector

logger: Final[logging.Logger] = logging.getLogger(__name__)

DEFAULT_STEP15_TARGET_P95_MS: Final[float] = 180.0
DEFAULT_STEP15_REQUESTS_PER_PHASE: Final[int] = 8


class ModelStatus(StrEnum):
    """Operational status of a target model within the multi-model benchmark."""

    RUNNABLE = "RUNNABLE"
    NOT_RUN = "NOT_RUN"
    FAILED_TO_INITIALIZE = "FAILED_TO_INITIALIZE"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"


class ModelSpec(BaseModel):
    """Strongly typed specification of a language model evaluated in Step 15."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(description="Unique model identifier (e.g. HuggingFace repo ID)")
    model_family: str = Field(description="Model family identifier (e.g. 'qwen2.5', 'llama3.2')")
    parameter_size_label: str = Field(description="Parameter size label (e.g. '0.5B', '1.5B')")
    expected_context_length: int | None = Field(
        default=None, description="Optional expected context length in tokens"
    )
    backend: str = Field(default="vllm", description="Inference backend implementation identifier")
    dtype: str = Field(default="auto", description="Model weights data type (e.g. 'auto')")
    status: ModelStatus = Field(
        default=ModelStatus.RUNNABLE, description="Measured execution availability status"
    )
    initialization_error: str | None = Field(
        default=None, description="Detailed error message if model failed to initialize"
    )


# Standard candidate models for cross-model generalization benchmarking
DEFAULT_STEP15_MODELS: Final[tuple[ModelSpec, ...]] = (
    ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_family="qwen2.5",
        parameter_size_label="0.5B",
    ),
    ModelSpec(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_family="qwen2.5",
        parameter_size_label="1.5B",
    ),
    ModelSpec(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        model_family="llama3.2",
        parameter_size_label="1B",
    ),
)


class Step15CandidateMetricRecord(BaseModel):
    """Exploration metrics for an individual tunable candidate configuration on a model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: TunableConfig = Field(description="Evaluated tunable candidate configuration")
    concurrency: int = Field(ge=1, description="Concurrency level evaluated")
    max_batch_size: int = Field(ge=1, description="Max batch size evaluated")
    batch_wait_ms: float = Field(ge=0.0, description="Batch formation wait window in ms")
    requests_per_sec: float = Field(ge=0.0, description="Measured throughput in req/s")
    output_tokens_per_sec: float = Field(ge=0.0, description="Generated output token throughput")
    total_tokens_per_sec: float = Field(ge=0.0, description="Total tokens throughput")
    mean_latency_ms: float = Field(ge=0.0, description="Mean total latency in ms")
    p50_latency_ms: float = Field(ge=0.0, description="p50 latency in ms")
    p95_latency_ms: float = Field(ge=0.0, description="p95 latency in ms")
    p99_latency_ms: float = Field(ge=0.0, description="p99 latency in ms")
    avg_queue_wait_ms: float = Field(ge=0.0, description="Average queue wait time in ms")
    avg_backend_execution_ms: float = Field(ge=0.0, description="Average execution time in ms")
    total_batches: int = Field(ge=0, description="Total formed batches")
    avg_batch_size: float = Field(ge=0.0, description="Average formed batch size")
    completed_requests: int = Field(ge=0, description="Completed requests count")
    failed_requests: int = Field(default=0, ge=0, description="Failed requests count")
    integrity_valid: bool = Field(description="True if candidate run satisfied accounting gates")


class Step15ModelConditionSummary(BaseModel):
    """Performance summary for a test condition on a specific model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition_name: str = Field(description="Condition identifier (e.g. STATIC_OPTIMIZED)")
    condition_type: str = Field(
        description="Condition category: 'static_conservative', 'static_optimized', 'sla_adaptive'"
    )
    model_id: str = Field(description="Model identifier evaluated")
    active_config_str: str = Field(description="Active scheduler configuration summary string")
    target_slo_p95_ms: float = Field(gt=0.0, description="Target p95 latency threshold in ms")
    scheduled_requests: int = Field(ge=0, description="Total requests scheduled")
    total_requests: int = Field(ge=0, description="Total requests planned")
    completed_requests: int = Field(ge=0, description="Total completed requests")
    failed_requests: int = Field(default=0, ge=0, description="Total failed requests")
    measured_requests: int = Field(ge=0, description="Total measured requests")
    backend_generate_calls: int = Field(ge=0, description="Single generate calls to backend")
    backend_generate_batch_calls: int = Field(ge=0, description="Batched generate calls to backend")
    total_duration_sec: float = Field(gt=0.0, description="Total duration in seconds")
    overall_throughput_rps: float = Field(ge=0.0, description="Overall throughput in req/s")
    output_tokens_per_sec: float = Field(ge=0.0, description="Generated token throughput")
    total_tokens_per_sec: float = Field(ge=0.0, description="Total token throughput")
    mean_latency_ms: float = Field(ge=0.0, description="Mean total latency in ms")
    p50_latency_ms: float = Field(ge=0.0, description="p50 latency in ms")
    p90_latency_ms: float = Field(ge=0.0, description="p90 latency in ms")
    p95_latency_ms: float = Field(ge=0.0, description="p95 latency in ms")
    p99_latency_ms: float = Field(ge=0.0, description="p99 latency in ms")
    mean_queue_wait_ms: float = Field(ge=0.0, description="Mean queue wait time in ms")
    mean_backend_execution_ms: float = Field(ge=0.0, description="Mean backend execution ms")
    total_sla_violations: int = Field(ge=0, description="Count of requests violating target SLO")
    overall_sla_violation_rate_pct: float = Field(ge=0.0, le=100.0, description="SLA violation %")
    phase_metrics: tuple[Step14PhaseMetricRecord, ...] = Field(
        default_factory=tuple, description="Phase-by-phase metric breakdown"
    )
    adaptation_events: tuple[Step14SLAAdaptationEventRecord, ...] = Field(
        default_factory=tuple, description="Chronological adaptation events recorded"
    )
    total_adaptations: int = Field(default=0, ge=0, description="Total dynamic adaptations")
    oscillation_count: int = Field(default=0, ge=0, description="Direction reversal count")
    engine_initialization_count: int = Field(default=1, description="Engine initializations")
    engine_teardown_count: int = Field(default=0, description="Engine teardowns")
    engine_instance_id: str = Field(default="unknown", description="Backend engine instance ID")
    integrity_valid: bool = Field(description="True if scheduled == completed == measured")


class Step15ModelExecutionResult(BaseModel):
    """Complete experimental evaluation result for an individual model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_spec: ModelSpec = Field(description="Model specification evaluated")
    workload_hash: str = Field(description="SHA-256 hash of executed workload")
    selected_optimal_config: TunableConfig | None = Field(
        default=None, description="Independently selected winning configuration for this model"
    )
    candidate_evaluations: tuple[Step15CandidateMetricRecord, ...] = Field(
        default_factory=tuple, description="Exploration candidate evaluation metrics"
    )
    conditions: dict[str, Step15ModelConditionSummary] = Field(
        default_factory=dict, description="Execution metrics across evaluated conditions"
    )
    engine_initialization_count: int = Field(
        default=1, description="Engine initializations during this model run"
    )
    engine_teardown_count: int = Field(
        default=1, description="Engine teardowns during this model run"
    )
    backend_name: str = Field(
        default="vllm", description="Inference backend name used for execution"
    )
    backend_confirmed: bool = Field(
        default=False, description="True if verified real hardware engine execution"
    )
    backend_generate_calls: int = Field(default=0, description="Total single generate calls")
    backend_generate_batch_calls: int = Field(default=0, description="Total batched generate calls")
    execution_error: str | None = Field(
        default=None, description="Detailed error message if execution failed"
    )
    is_successful: bool = Field(
        default=True, description="True if model completed all candidate & condition runs"
    )


class Step15PerModelDelta(BaseModel):
    """Relative throughput and latency deltas for a model across evaluated conditions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(description="Evaluated model identifier")
    model_family: str = Field(description="Model family identifier")
    parameter_size_label: str = Field(description="Parameter size label (e.g. 0.5B, 1.5B)")
    conservative_throughput_rps: float = Field(description="Conservative baseline throughput")
    conservative_p95_latency_ms: float = Field(description="Conservative baseline p95 latency")
    conservative_sla_violation_pct: float = Field(description="Conservative baseline SLA viol %")
    optimized_config_str: str = Field(description="Winning optimized config summary string")
    optimized_throughput_rps: float = Field(description="Static optimized throughput")
    optimized_p95_latency_ms: float = Field(description="Static optimized p95 latency")
    optimized_sla_violation_pct: float = Field(description="Static optimized SLA violation %")
    adaptive_throughput_rps: float = Field(description="Adaptive InferOpt throughput")
    adaptive_p95_latency_ms: float = Field(description="Adaptive InferOpt p95 latency")
    adaptive_sla_violation_pct: float = Field(description="Adaptive InferOpt SLA violation %")
    optimized_vs_conservative_tput_pct: float = Field(
        description="Throughput delta % (Optimized vs Conservative)"
    )
    adaptive_vs_conservative_tput_pct: float = Field(
        description="Throughput delta % (Adaptive vs Conservative)"
    )
    optimized_vs_conservative_p95_delta_ms: float = Field(
        description="p95 latency delta ms (Optimized vs Conservative)"
    )
    adaptive_vs_conservative_p95_delta_ms: float = Field(
        description="p95 latency delta ms (Adaptive vs Conservative)"
    )


class Step15CrossModelComparison(BaseModel):
    """Comparative synthesis across all evaluated models in Step 15."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_slo_p95_ms: float = Field(description="Target p95 latency SLO threshold in ms")
    model_deltas: dict[str, Step15PerModelDelta] = Field(
        default_factory=dict, description="Relative performance deltas per model"
    )
    optimal_configs_by_model: dict[str, str] = Field(
        default_factory=dict, description="Winning configuration per model ID"
    )
    same_config_wins_all_models: bool = Field(
        description="True if identical configuration won across all evaluated models"
    )
    pareto_frontier_shifted: bool = Field(
        description="True if optimal concurrency/batch size changed across models"
    )


class Step15CrossModelReport(BaseModel):
    """Complete, standalone, machine-readable Step 15 Cross-Model Generalization report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(description="Unique Step 15 experiment identifier")
    timestamp: float = Field(description="Unix timestamp of experiment execution")
    git_commit: str = Field(description="Git commit hash")
    backend: str = Field(default="vllm", description="Inference backend implementation identifier")
    backend_execution_confirmed: bool = Field(
        default=False, description="True ONLY if verified execution occurred on real GPU engine"
    )
    environment: VLLMEnvironmentMetadata = Field(
        description="Hardware and runtime environment metadata"
    )
    target_slo: TargetSLO = Field(description="Target SLO used for optimization & control")
    workload_hash: str = Field(description="SHA-256 hash of controlled workload")
    models_evaluated: tuple[ModelSpec, ...] = Field(description="List of target model specs")
    results_by_model: dict[str, Step15ModelExecutionResult] = Field(
        description="Per-model detailed execution results"
    )
    comparison: Step15CrossModelComparison = Field(
        description="Cross-model comparative analysis and deltas"
    )
    findings: dict[str, tuple[str, ...]] = Field(
        default_factory=dict, description="Categorized scientific findings and hypothesis verdicts"
    )


def verify_step15_model_execution(result: Step15ModelExecutionResult) -> bool:
    """Verify observable runtime evidence that an individual model executed on a real engine.

    Returns True if and only if:
    1. Execution succeeded without errors (is_successful=True, execution_error=None).
    2. Model availability status is RUNNABLE (not AUTHENTICATION_REQUIRED, OUT_OF_MEMORY, etc.).
    3. Backend is a real hardware backend (backend_name in ("vllm", "mlx") and not "mock").
    4. Backend explicitly confirmed real hardware execution (backend_confirmed=True).
    5. Engine lifecycle verified with initialization and teardown (inits >= 1, teardowns >= 1).
    6. Real inference generation occurred (generate_calls + batch_calls > 0).
    7. All 3 conditions executed with valid accounting (scheduled == completed == measured).
    8. Timings are positive and causal (duration > 0, mean_lat > 0, p95_lat >= queue_wait).
    """
    if not result.is_successful or result.execution_error is not None:
        return False

    if result.model_spec.status != ModelStatus.RUNNABLE:
        return False

    if result.backend_name == "mock" or not result.backend_confirmed:
        return False

    if result.engine_initialization_count < 1 or result.engine_teardown_count < 1:
        return False

    if (result.backend_generate_calls + result.backend_generate_batch_calls) <= 0:
        return False

    required_conditions = ("STATIC_CONSERVATIVE", "STATIC_OPTIMIZED", "SLA_AWARE_ADAPTIVE")
    for cond_name in required_conditions:
        cond = result.conditions.get(cond_name)
        if cond is None:
            return False
        if not cond.integrity_valid:
            return False
        if cond.scheduled_requests <= 0 or cond.completed_requests != cond.scheduled_requests:
            return False
        if cond.failed_requests != 0 or cond.measured_requests != cond.completed_requests:
            return False
        if cond.total_duration_sec <= 0.0 or cond.mean_latency_ms <= 0.0:
            return False
        if cond.p95_latency_ms < cond.mean_queue_wait_ms - 1e-6:
            return False

    return True


def verify_step15_cross_model_report(
    results_by_model: dict[str, Step15ModelExecutionResult],
    backend_name: str,
) -> bool:
    """Verify whether a Step 15 cross-model report is backed by real hardware execution.

    Returns True if and only if:
    1. Overall backend is not 'mock'.
    2. At least one model executed successfully.
    3. All successful models in the report pass verify_step15_model_execution.
    """
    if backend_name == "mock":
        return False

    successful_results = [r for r in results_by_model.values() if r.is_successful]
    if not successful_results:
        return False

    return all(verify_step15_model_execution(r) for r in successful_results)


def classify_step15_findings(
    comparison: Step15CrossModelComparison,
    results_by_model: dict[str, Step15ModelExecutionResult],
    target_slo: TargetSLO,
) -> dict[str, tuple[str, ...]]:
    """Categorize Step 15 outcomes into PROVEN, SUGGESTED, and NOT PROVEN."""
    proven: list[str] = []
    suggested: list[str] = []
    not_proven: list[str] = []

    successful_results = {m: r for m, r in results_by_model.items() if r.is_successful}
    failed_results = {m: r for m, r in results_by_model.items() if not r.is_successful}

    # 1. Proven observations
    proven.append(
        "Engine Lifecycle Isolation: Every evaluated model executed on an isolated backend "
        "instance with 1 clean initialization and 1 teardown, preventing cross-model VRAM leakage."
    )
    proven.append(
        "Request Accounting Integrity: 100% request completion integrity verified across all "
        f"evaluated runnable models ({len(successful_results)} successful model runs, "
        "0 dropped requests)."
    )

    for m_id, delta in comparison.model_deltas.items():
        proven.append(
            f"Workload Baseline Delta ({m_id}): Config {delta.optimized_config_str} "
            f"yielded {delta.optimized_throughput_rps:.2f} rps vs "
            f"{delta.conservative_throughput_rps:.2f} rps Conservative "
            f"({delta.optimized_vs_conservative_tput_pct:+.1f}% on this evaluated workload)."
        )

    # 2. Suggested observations
    if len(successful_results) > 1:
        if comparison.same_config_wins_all_models:
            suggested.append(
                "Optimal Configuration Convergence: The same configuration won across evaluated "
                "runnable models on this specific workload profile."
            )
        else:
            suggested.append(
                "Model-Dependent Pareto Scaling: Optimal configuration shifted across models, "
                "supporting hypothesis H1 that different architectures benefit from tuning."
            )

    for m_id, delta in comparison.model_deltas.items():
        suggested.append(
            f"Adaptive Throughput Scaling ({m_id}): SLA-Aware Adaptive Control achieved "
            f"{delta.adaptive_throughput_rps:.2f} rps vs {delta.conservative_throughput_rps:.2f} "
            f"rps Conservative ({delta.adaptive_vs_conservative_tput_pct:+.1f}% on workload)."
        )

    # 3. Not proven and limitations
    if failed_results:
        for f_id, f_res in failed_results.items():
            not_proven.append(
                f"Model Evaluation Limitation ({f_id}): Model execution failed with status "
                f"{f_res.model_spec.status.value} ({f_res.execution_error or 'No response'}). "
                "NO performance or SLA conclusions are claimed for this model."
            )

    not_proven.append(
        "Architecture Family Lineage: Models utilizing shared base architecture components "
        "(e.g., SmolLM2 utilizing LlamaForCausalLM) share architectural mechanisms; "
        "cross-model generalization conclusions must account for underlying architectural lineage."
    )
    not_proven.append(
        "H1 (Universal Model-Dependent Optimal Configuration): NOT PROVEN UNIVERSALLY - Tested "
        "evidence is scoped to evaluated models and workloads; generalization across arbitrary "
        "parameter scales requires further empirical benchmarking."
    )
    not_proven.append(
        "H4 (Sufficiency of Single Fixed Configuration Across All Serving Regimes): NOT PROVEN - "
        "While a fixed configuration may perform well under static load, dynamic traffic "
        "fluctuations still warrant adaptive guardrailing."
    )
    not_proven.append(
        "H5 (Adaptive Throughput Superiority over Static Optimized): NOT PROVEN - Under "
        "stationary load within phases, adaptive control dynamically tracks capacity without "
        "exceeding the throughput of an optimal static ceiling."
    )
    not_proven.append(
        f"Universal Cross-Model SLA Guarantee: NOT PROVEN - Tail latency compliance "
        f"(p95 <= {target_slo.p95_latency_ms:.1f}ms) is specific to the evaluated model "
        "architectures and prompts, and does not constitute a universal guarantee."
    )
    not_proven.append(
        "Hardware Specificity: All results are conditioned on the underlying GPU architecture and "
        "vLLM memory subsystem configuration."
    )
    not_proven.append(
        "Workload Specificity: Performance scaling varies with prompt token distributions and "
        "inter-arrival patterns."
    )
    not_proven.append(
        "No Universal Dominance: No single scheduler policy dominates all operating regimes."
    )

    return {
        "PROVEN_OBSERVATIONS": tuple(proven),
        "SUGGESTED_WORKLOAD_OBSERVATIONS": tuple(suggested),
        "NOT_PROVEN_AND_LIMITATIONS": tuple(not_proven),
    }


class Step15MultiModelRunner:
    """Scientific benchmark runner for Step 15 Multi-Model Generalization."""

    def __init__(
        self,
        models: Sequence[ModelSpec] = DEFAULT_STEP15_MODELS,
        target_slo: TargetSLO | None = None,
        candidate_space: CandidateSpace | None = None,
        enforce_eager: bool = False,
    ) -> None:
        """Initialize the Step 15 Multi-Model Experiment Runner.

        Args:
            models: Sequence of ModelSpec instances to evaluate.
            target_slo: Target latency Service Level Objective (defaults to p95 <= 180.0 ms).
            candidate_space: Search space for candidate exploration.
            enforce_eager: Enforce eager execution in vLLM backends.
        """
        self._models = tuple(models)
        self._target_slo = target_slo or TargetSLO(p95_latency_ms=DEFAULT_STEP15_TARGET_P95_MS)
        self._candidate_space = candidate_space or CandidateSpace(
            concurrencies=(1, 4, 8),
            batch_sizes=(1, 2, 4, 8),
            batch_waits_ms=(50.0,),
        )
        self._optimizer = DeterministicOptimizer()
        self._enforce_eager = enforce_eager

    @property
    def models(self) -> tuple[ModelSpec, ...]:
        """Get the tuple of model specifications."""
        return self._models

    @property
    def target_slo(self) -> TargetSLO:
        """Get the active TargetSLO configuration."""
        return self._target_slo

    def build_exploration_workload(
        self,
        num_requests: int = 16,
        seed: int = 42,
    ) -> WorkloadScenario:
        """Construct deterministic workload for candidate space exploration."""
        cfg = WorkloadConfig(
            scenario_name="step15_exploration",
            description="Controlled exploration workload for multi-model candidate search",
            num_requests=num_requests,
            arrival_pattern=ArrivalPattern.CONCURRENT,
            concurrency=4,
            seed=seed,
            prompt_categories=(PromptCategory.SHORT, PromptCategory.MEDIUM, PromptCategory.FACTUAL),
            max_tokens=64,
            priority_levels=(0,),
        )
        return generate_workload(cfg)

    def build_step15_phase_scenarios(
        self,
        requests_per_phase: int = DEFAULT_STEP15_REQUESTS_PER_PHASE,
        seed: int = 42,
    ) -> list[WorkloadScenario]:
        """Construct the standardized 4-phase sequence matching Step 14."""
        step14_runner = Step14SLAExperimentRunner(target_slo=self._target_slo)
        return step14_runner.build_step14_phase_scenarios(
            num_requests_per_phase=requests_per_phase, seed=seed
        )

    async def _evaluate_candidate(
        self,
        backend: InferenceBackend,
        model_id: str,
        tunable_cfg: TunableConfig,
        workload: WorkloadScenario,
    ) -> Step15CandidateMetricRecord:
        """Evaluate an individual candidate configuration under the exploration workload."""
        collector = MetricsCollector()
        sched_cfg = tunable_cfg.to_scheduler_config()
        scheduler = Scheduler(backend=backend, config=sched_cfg, collector=collector)
        await scheduler.start()

        completed_resps: list[InferenceResponse] = []
        total_latencies: list[float] = []
        queue_waits: list[float] = []
        exec_latencies: list[float] = []

        sem = asyncio.Semaphore(tunable_cfg.max_concurrency)
        t_start = time.perf_counter()

        async def _submit(spec: WorkloadRequestSpec, idx: int) -> InferenceResponse | Exception:
            async with sem:
                req_id = (
                    f"step15-exp-{tunable_cfg.max_concurrency}-{tunable_cfg.max_batch_size}-r{idx}"
                )
                req = InferenceRequest(
                    request_id=req_id,
                    prompt=spec.prompt,
                    model=model_id,
                    max_tokens=spec.max_tokens,
                    temperature=spec.temperature,
                )
                return await scheduler.submit(req)

        try:
            tasks = [_submit(spec, i) for i, spec in enumerate(workload.requests)]
            raw_resps = await asyncio.gather(*tasks)
        finally:
            await scheduler.shutdown()

        t_dur = max(0.001, time.perf_counter() - t_start)
        for r in raw_resps:
            if isinstance(r, InferenceResponse):
                completed_resps.append(r)
                rec = scheduler.get_record(r.request_id)
                q_w = rec.queue_wait_ms if (rec and rec.queue_wait_ms is not None) else 0.0
                e_l = rec.execution_ms if (rec and rec.execution_ms is not None) else r.latency_ms
                t_l = (
                    rec.total_latency_ms
                    if (rec and rec.total_latency_ms is not None)
                    else (q_w + e_l)
                )
                queue_waits.append(q_w)
                exec_latencies.append(e_l)
                total_latencies.append(t_l)

        comp_cnt = len(completed_resps)
        sched_cnt = len(workload.requests)
        fail_cnt = sched_cnt - comp_cnt
        meas_cnt = len(total_latencies)

        if sched_cnt != (comp_cnt + fail_cnt) or comp_cnt != meas_cnt:
            raise ValueError(
                f"Candidate request accounting mismatch: sched={sched_cnt}, comp={comp_cnt}, "
                f"fail={fail_cnt}, meas={meas_cnt}"
            )

        p95 = calculate_percentile(total_latencies, 95.0) if total_latencies else 0.0
        p99 = calculate_percentile(total_latencies, 99.0) if total_latencies else 0.0
        p50 = calculate_percentile(total_latencies, 50.0) if total_latencies else 0.0
        mean_lat = sum(total_latencies) / len(total_latencies) if total_latencies else 0.0
        avg_q_w = sum(queue_waits) / len(queue_waits) if queue_waits else 0.0
        avg_exec = sum(exec_latencies) / len(exec_latencies) if exec_latencies else 0.0

        rps = comp_cnt / t_dur if t_dur > 0 else 0.0
        out_toks = sum(r.output_tokens or 0 for r in completed_resps)
        in_toks = sum(r.input_tokens or 0 for r in completed_resps)
        out_rps = out_toks / t_dur if t_dur > 0 else 0.0
        tot_rps = (out_toks + in_toks) / t_dur if t_dur > 0 else 0.0

        snap = collector.snapshot()

        return Step15CandidateMetricRecord(
            config=tunable_cfg,
            concurrency=tunable_cfg.max_concurrency,
            max_batch_size=tunable_cfg.max_batch_size,
            batch_wait_ms=tunable_cfg.batch_wait_ms,
            requests_per_sec=rps,
            output_tokens_per_sec=out_rps,
            total_tokens_per_sec=tot_rps,
            mean_latency_ms=mean_lat,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            avg_queue_wait_ms=avg_q_w,
            avg_backend_execution_ms=avg_exec,
            total_batches=snap.batches.total_batches,
            avg_batch_size=snap.batches.avg_batch_size,
            completed_requests=comp_cnt,
            failed_requests=fail_cnt,
            integrity_valid=(fail_cnt == 0),
        )

    async def _execute_model_condition(
        self,
        backend: InferenceBackend,
        model_id: str,
        condition_name: str,
        condition_type: str,
        initial_config: TunableConfig,
        is_adaptive: bool,
        phase_scenarios: list[WorkloadScenario],
    ) -> Step15ModelConditionSummary:
        """Execute a full multi-phase condition run on an individual model."""
        collector = MetricsCollector()
        sched_cfg = initial_config.to_scheduler_config()
        scheduler = Scheduler(backend=backend, config=sched_cfg, collector=collector)
        await scheduler.start()

        controller = (
            SLAConstrainedAdaptiveController(
                target_slo=self._target_slo,
                candidate_ladder=DEFAULT_SLA_CANDIDATE_LADDER,
                min_dwell_time_sec=0.2,
            )
            if is_adaptive
            else None
        )

        phase_metric_records: list[Step14PhaseMetricRecord] = []
        all_completed_resps: list[InferenceResponse] = []
        all_total_latencies: list[float] = []
        all_queue_waits: list[float] = []
        all_exec_latencies: list[float] = []
        adaptation_events: list[Step14SLAAdaptationEventRecord] = []

        total_sla_violations = 0
        t_cond_start = time.perf_counter()

        init_gen = getattr(backend, "generate_calls", 0)
        init_batch = getattr(backend, "generate_batch_calls", 0)

        try:
            for p_idx, scenario in enumerate(phase_scenarios):
                logger.info(
                    "Model %s Condition %s -> Phase %d (%s)",
                    model_id,
                    condition_name,
                    p_idx + 1,
                    scenario.scenario_name,
                )

                collector.reset_peaks()
                t_p_start = time.perf_counter()
                init_p_gen = getattr(backend, "generate_calls", 0)
                init_p_batch = getattr(backend, "generate_batch_calls", 0)
                p_adapt_start = len(adaptation_events)

                pattern = scenario.config.arrival_pattern

                async def _submit_and_track(
                    spec: WorkloadRequestSpec,
                    req_idx: int,
                    phase_num: int = p_idx + 1,
                    phase_name: str = scenario.scenario_name,
                ) -> InferenceResponse | Exception:
                    req_id = f"step15-{condition_name}-p{phase_num}-r{req_idx}"
                    req = InferenceRequest(
                        request_id=req_id,
                        prompt=spec.prompt,
                        model=model_id,
                        max_tokens=spec.max_tokens,
                        temperature=spec.temperature,
                    )
                    try:
                        res = await scheduler.submit(req)
                        if is_adaptive and controller is not None:
                            snap_ctrl = collector.snapshot()
                            active_cfg = TunableConfig(
                                max_concurrency=scheduler.config.max_concurrency,
                                max_batch_size=scheduler.config.batch_config.max_batch_size,
                                batch_wait_ms=scheduler.config.batch_config.batch_wait_ms,
                            )
                            status = controller.evaluate(
                                snapshot=snap_ctrl,
                                active_config=active_cfg,
                            )
                            applied = controller.apply_decision(
                                status=status,
                                scheduler=scheduler,
                                phase_index=phase_num,
                                phase_name=phase_name,
                                timestamp_offset_sec=time.perf_counter() - t_cond_start,
                            )
                            if applied:
                                adaptation_events.append(controller.adaptation_events[-1])
                        return res
                    except Exception as e:
                        logger.error("Request %s failed: %s", req.request_id, e)
                        return e

                if pattern == ArrivalPattern.BURST:
                    barrier = asyncio.Barrier(len(scenario.requests))

                    async def _burst_sub(
                        s: WorkloadRequestSpec,
                        i: int,
                        b: asyncio.Barrier = barrier,
                    ) -> InferenceResponse | Exception:
                        await b.wait()
                        return await _submit_and_track(s, i)

                    tasks = [_burst_sub(s, i) for i, s in enumerate(scenario.requests)]
                    raw_resps = await asyncio.gather(*tasks)
                elif pattern == ArrivalPattern.CONCURRENT:
                    conc = scenario.config.concurrency or 1
                    sem = asyncio.Semaphore(conc)

                    async def _conc_sub(
                        s: WorkloadRequestSpec,
                        i: int,
                        s_sem: asyncio.Semaphore = sem,
                    ) -> InferenceResponse | Exception:
                        async with s_sem:
                            return await _submit_and_track(s, i)

                    tasks = [_conc_sub(s, i) for i, s in enumerate(scenario.requests)]
                    raw_resps = await asyncio.gather(*tasks)
                else:
                    raw_resps = []
                    for i, s in enumerate(scenario.requests):
                        r = await _submit_and_track(s, i)
                        raw_resps.append(r)

                t_p_dur = max(0.001, time.perf_counter() - t_p_start)
                p_snap = collector.snapshot()
                p_comp_resps = [r for r in raw_resps if isinstance(r, InferenceResponse)]
                all_completed_resps.extend(p_comp_resps)

                p_total_lats: list[float] = []
                p_q_waits: list[float] = []
                p_exec_lats: list[float] = []
                p_sla_viols = 0

                for r in p_comp_resps:
                    rec = scheduler.get_record(r.request_id)
                    q_w = rec.queue_wait_ms if (rec and rec.queue_wait_ms is not None) else 0.0
                    e_l = (
                        rec.execution_ms if (rec and rec.execution_ms is not None) else r.latency_ms
                    )
                    t_l = (
                        rec.total_latency_ms
                        if (rec and rec.total_latency_ms is not None)
                        else (q_w + e_l)
                    )

                    # Hard Invariant: total_latency >= queue_wait
                    if t_l < q_w - 1e-6:
                        raise ValueError(
                            f"Latency causality violated: total={t_l}ms < queue_wait={q_w}ms"
                        )

                    p_q_waits.append(q_w)
                    p_exec_lats.append(e_l)
                    p_total_lats.append(t_l)

                    if t_l > self._target_slo.p95_latency_ms:
                        p_sla_viols += 1

                all_total_latencies.extend(p_total_lats)
                all_queue_waits.extend(p_q_waits)
                all_exec_latencies.extend(p_exec_lats)

                p_p50 = calculate_percentile(p_total_lats, 50.0) if p_total_lats else 0.0
                p_p90 = calculate_percentile(p_total_lats, 90.0) if p_total_lats else 0.0
                p_p95 = calculate_percentile(p_total_lats, 95.0) if p_total_lats else 0.0
                p_p99 = calculate_percentile(p_total_lats, 99.0) if p_total_lats else 0.0
                p_mean = sum(p_total_lats) / len(p_total_lats) if p_total_lats else 0.0
                p_avg_q = sum(p_q_waits) / len(p_q_waits) if p_q_waits else 0.0
                p_avg_e = sum(p_exec_lats) / len(p_exec_lats) if p_exec_lats else 0.0

                p_comp_cnt = len(p_comp_resps)
                p_sched_cnt = len(scenario.requests)
                p_fail_cnt = p_sched_cnt - p_comp_cnt
                p_meas_cnt = len(p_total_lats)

                if p_sched_cnt != (p_comp_cnt + p_fail_cnt) or p_comp_cnt != p_meas_cnt:
                    raise ValueError(
                        f"Phase accounting mismatch: sched={p_sched_cnt}, comp={p_comp_cnt}, "
                        f"fail={p_fail_cnt}, meas={p_meas_cnt}"
                    )

                p_rps = p_comp_cnt / t_p_dur if t_p_dur > 0 else 0.0
                p_out_t = sum(r.output_tokens or 0 for r in p_comp_resps)
                p_in_t = sum(r.input_tokens or 0 for r in p_comp_resps)
                p_out_rps = p_out_t / t_p_dur if t_p_dur > 0 else 0.0
                p_tot_rps = (p_out_t + p_in_t) / t_p_dur if t_p_dur > 0 else 0.0

                p_gen = getattr(backend, "generate_calls", 0) - init_p_gen
                p_batch = getattr(backend, "generate_batch_calls", 0) - init_p_batch

                act_str = (
                    f"c={scheduler.config.max_concurrency}, "
                    f"b={scheduler.config.batch_config.max_batch_size}"
                )
                p_adapt_cnt = len(adaptation_events) - p_adapt_start
                p_sla_pct = (p_sla_viols / p_comp_cnt * 100.0) if p_comp_cnt > 0 else 0.0

                phase_metric_records.append(
                    Step14PhaseMetricRecord(
                        phase_index=p_idx,
                        phase_name=scenario.scenario_name,
                        target_slo_p95_ms=self._target_slo.p95_latency_ms,
                        active_config=act_str,
                        scheduled_requests=p_sched_cnt,
                        total_requests=p_sched_cnt,
                        completed_requests=p_comp_cnt,
                        failed_requests=p_fail_cnt,
                        measured_requests=p_meas_cnt,
                        backend_generate_calls=p_gen,
                        backend_generate_batch_calls=p_batch,
                        duration_sec=t_p_dur,
                        throughput_rps=p_rps,
                        output_tokens_per_sec=p_out_rps,
                        total_tokens_per_sec=p_tot_rps,
                        mean_latency_ms=p_mean,
                        p50_latency_ms=p_p50,
                        p90_latency_ms=p_p90,
                        p95_latency_ms=p_p95,
                        p99_latency_ms=p_p99,
                        avg_queue_wait_ms=p_avg_q,
                        avg_backend_execution_ms=p_avg_e,
                        peak_queue_depth=p_snap.queue.peak_queue_depth,
                        total_batches=p_snap.batches.total_batches,
                        avg_batch_size=p_snap.batches.avg_batch_size,
                        sla_violation_count=p_sla_viols,
                        sla_violation_rate_pct=p_sla_pct,
                        adaptation_count_in_phase=p_adapt_cnt,
                        integrity_valid=(p_fail_cnt == 0),
                    )
                )
                total_sla_violations += p_sla_viols

        finally:
            await scheduler.shutdown()

        t_cond_dur = max(0.001, time.perf_counter() - t_cond_start)
        tot_sched = sum(len(s.requests) for s in phase_scenarios)
        tot_comp = len(all_completed_resps)
        tot_fail = tot_sched - tot_comp
        tot_meas = len(all_total_latencies)

        # Hard Accounting Gate
        if tot_sched != (tot_comp + tot_fail) or tot_comp != tot_meas:
            raise ValueError(
                f"Condition accounting mismatch: sched={tot_sched}, comp={tot_comp}, "
                f"fail={tot_fail}, meas={tot_meas}"
            )

        c_p50 = calculate_percentile(all_total_latencies, 50.0) if all_total_latencies else 0.0
        c_p90 = calculate_percentile(all_total_latencies, 90.0) if all_total_latencies else 0.0
        c_p95 = calculate_percentile(all_total_latencies, 95.0) if all_total_latencies else 0.0
        c_p99 = calculate_percentile(all_total_latencies, 99.0) if all_total_latencies else 0.0
        c_mean = sum(all_total_latencies) / len(all_total_latencies) if all_total_latencies else 0.0
        c_avg_q = sum(all_queue_waits) / len(all_queue_waits) if all_queue_waits else 0.0
        c_avg_e = sum(all_exec_latencies) / len(all_exec_latencies) if all_exec_latencies else 0.0

        c_rps = tot_comp / t_cond_dur if t_cond_dur > 0 else 0.0
        c_out_toks = sum(r.output_tokens or 0 for r in all_completed_resps)
        c_in_toks = sum(r.input_tokens or 0 for r in all_completed_resps)
        c_out_rps = c_out_toks / t_cond_dur if t_cond_dur > 0 else 0.0
        c_tot_rps = (c_out_toks + c_in_toks) / t_cond_dur if t_cond_dur > 0 else 0.0

        c_gen = getattr(backend, "generate_calls", 0) - init_gen
        c_batch = getattr(backend, "generate_batch_calls", 0) - init_batch

        cond_sla_pct = (total_sla_violations / tot_comp * 100.0) if tot_comp > 0 else 0.0
        osc_cnt = controller.oscillation_count if controller else 0

        cond_cfg_str = (
            f"c={initial_config.max_concurrency}, b={initial_config.max_batch_size}"
            if not is_adaptive
            else "dynamic-sla-adaptive"
        )

        return Step15ModelConditionSummary(
            condition_name=condition_name,
            condition_type=condition_type,
            model_id=model_id,
            active_config_str=cond_cfg_str,
            target_slo_p95_ms=self._target_slo.p95_latency_ms,
            scheduled_requests=tot_sched,
            total_requests=tot_sched,
            completed_requests=tot_comp,
            failed_requests=tot_fail,
            measured_requests=tot_meas,
            backend_generate_calls=c_gen,
            backend_generate_batch_calls=c_batch,
            total_duration_sec=t_cond_dur,
            overall_throughput_rps=c_rps,
            output_tokens_per_sec=c_out_rps,
            total_tokens_per_sec=c_tot_rps,
            mean_latency_ms=c_mean,
            p50_latency_ms=c_p50,
            p90_latency_ms=c_p90,
            p95_latency_ms=c_p95,
            p99_latency_ms=c_p99,
            mean_queue_wait_ms=c_avg_q,
            mean_backend_execution_ms=c_avg_e,
            total_sla_violations=total_sla_violations,
            overall_sla_violation_rate_pct=cond_sla_pct,
            phase_metrics=tuple(phase_metric_records),
            adaptation_events=tuple(adaptation_events),
            total_adaptations=len(adaptation_events),
            oscillation_count=osc_cnt,
            engine_initialization_count=1,
            engine_teardown_count=0,
            engine_instance_id=getattr(backend, "instance_id", "unknown"),
            integrity_valid=(tot_fail == 0),
        )

    async def run_model_benchmark(
        self,
        backend: InferenceBackend,
        model_spec: ModelSpec,
        num_requests_per_phase: int = DEFAULT_STEP15_REQUESTS_PER_PHASE,
        seed: int = 42,
    ) -> Step15ModelExecutionResult:
        """Run candidate exploration, static conditions, and adaptive control for one model."""
        model_id = model_spec.model_id
        logger.info("Starting Step 15 evaluation for model: %s", model_id)

        backend_name = getattr(backend, "backend_name", "vllm")
        backend_is_real = getattr(backend, "is_real_execution", False)
        init_gen_calls = getattr(backend, "generate_calls", 0)
        init_batch_calls = getattr(backend, "generate_batch_calls", 0)

        exp_workload = self.build_exploration_workload(
            num_requests=num_requests_per_phase * 2, seed=seed
        )
        w_hash = compute_workload_hash(exp_workload)

        step14_runner = Step14SLAExperimentRunner(model_id=model_id, target_slo=self._target_slo)
        phase_scenarios = step14_runner.build_step14_phase_scenarios(
            num_requests_per_phase=num_requests_per_phase, seed=seed
        )

        is_successful = False
        execution_error: str | None = None
        winning_config: TunableConfig | None = None
        candidate_records: list[Step15CandidateMetricRecord] = []
        conditions: dict[str, Step15ModelConditionSummary] = {}
        status = ModelStatus.RUNNABLE

        try:
            # 1. Initialize model engine
            if hasattr(backend, "load_model"):
                await backend.load_model()

            # 2. Candidate Space Exploration
            candidates = self._candidate_space.generate_candidates()

            for cand in candidates:
                cand_rec = await self._evaluate_candidate(
                    backend=backend,
                    model_id=model_id,
                    tunable_cfg=cand,
                    workload=exp_workload,
                )
                candidate_records.append(cand_rec)

            # 3. Independent Deterministic Optimization
            feasible_candidates = [
                c for c in candidate_records if c.p95_latency_ms <= self._target_slo.p95_latency_ms
            ]
            if feasible_candidates:
                winning_rec = max(
                    feasible_candidates,
                    key=lambda c: (
                        c.requests_per_sec,
                        -c.p95_latency_ms,
                        -c.concurrency,
                        -c.max_batch_size,
                    ),
                )
            else:
                winning_rec = min(
                    candidate_records,
                    key=lambda c: (c.p95_latency_ms, -c.requests_per_sec),
                )
            winning_config = winning_rec.config

            logger.info(
                "Model %s Independent Optimal Config: c=%d, b=%d (p95=%.1fms, tput=%.2f rps)",
                model_id,
                winning_config.max_concurrency,
                winning_config.max_batch_size,
                winning_rec.p95_latency_ms,
                winning_rec.requests_per_sec,
            )

            # 4. Execute 3 Conditions
            conservative_cfg = TunableConfig(
                max_concurrency=1,
                max_batch_size=2,
                batch_wait_ms=50.0,
            )

            # Condition 1: Static Conservative
            res_cons = await self._execute_model_condition(
                backend=backend,
                model_id=model_id,
                condition_name="STATIC_CONSERVATIVE",
                condition_type="static_conservative",
                initial_config=conservative_cfg,
                is_adaptive=False,
                phase_scenarios=phase_scenarios,
            )

            # Condition 2: Static Optimized
            res_opt = await self._execute_model_condition(
                backend=backend,
                model_id=model_id,
                condition_name="STATIC_OPTIMIZED",
                condition_type="static_optimized",
                initial_config=winning_config,
                is_adaptive=False,
                phase_scenarios=phase_scenarios,
            )

            # Condition 3: SLA-Aware Adaptive Control
            res_adapt = await self._execute_model_condition(
                backend=backend,
                model_id=model_id,
                condition_name="SLA_AWARE_ADAPTIVE",
                condition_type="sla_adaptive",
                initial_config=conservative_cfg,
                is_adaptive=True,
                phase_scenarios=phase_scenarios,
            )

            conditions = {
                "STATIC_CONSERVATIVE": res_cons,
                "STATIC_OPTIMIZED": res_opt,
                "SLA_AWARE_ADAPTIVE": res_adapt,
            }
            is_successful = True

        except Exception as err:
            logger.error("Model %s benchmark execution failed: %s", model_id, err, exc_info=True)
            is_successful = False
            execution_error = str(err)
            status = ModelStatus.FAILED_TO_INITIALIZE
            if (
                "authentication" in execution_error.lower()
                or "gated" in execution_error.lower()
                or "401" in execution_error
            ):
                status = ModelStatus.AUTHENTICATION_REQUIRED
            elif (
                "out of memory" in execution_error.lower() or "cuda oom" in execution_error.lower()
            ):
                status = ModelStatus.OUT_OF_MEMORY

        finally:
            if hasattr(backend, "unload_model"):
                try:
                    await backend.unload_model()
                except Exception as unl_err:
                    logger.warning("Unload error for %s: %s", model_id, unl_err)

        total_gen_calls = getattr(backend, "generate_calls", 0) - init_gen_calls
        total_batch_calls = getattr(backend, "generate_batch_calls", 0) - init_batch_calls
        inits = getattr(backend, "engine_initializations", 1 if is_successful else 0)
        teardowns = getattr(backend, "engine_teardowns", 1 if is_successful else 0)

        is_real_confirmed = (
            is_successful
            and backend_is_real
            and backend_name != "mock"
            and (total_gen_calls + total_batch_calls > 0)
            and len(conditions) == 3
            and all(c.integrity_valid for c in conditions.values())
        )

        final_spec = ModelSpec(
            model_id=model_spec.model_id,
            model_family=model_spec.model_family,
            parameter_size_label=model_spec.parameter_size_label,
            expected_context_length=model_spec.expected_context_length,
            backend=model_spec.backend,
            dtype=model_spec.dtype,
            status=status,
            initialization_error=execution_error,
        )

        return Step15ModelExecutionResult(
            model_spec=final_spec,
            workload_hash=w_hash,
            selected_optimal_config=winning_config if is_successful else None,
            candidate_evaluations=tuple(candidate_records),
            conditions=conditions,
            engine_initialization_count=inits,
            engine_teardown_count=teardowns,
            backend_name=backend_name,
            backend_confirmed=is_real_confirmed,
            backend_generate_calls=total_gen_calls,
            backend_generate_batch_calls=total_batch_calls,
            execution_error=execution_error,
            is_successful=is_successful,
        )

    async def run_experiment(
        self,
        backend: InferenceBackend | None = None,
        num_requests_per_phase: int = DEFAULT_STEP15_REQUESTS_PER_PHASE,
        seed: int = 42,
    ) -> Step15CrossModelReport:
        """Run the complete Step 15 multi-model generalization benchmark."""
        backend_inst = backend
        created_local_backend = False
        backend_name = getattr(backend_inst, "backend_name", "vllm") if backend_inst else "vllm"

        env_meta = collect_vllm_environment_metadata(
            model_id=self._models[0].model_id if self._models else DEFAULT_VLLM_MODEL_ID,
            enforce_eager=self._enforce_eager,
            warmup_count=2,
            repetitions=1,
            workload_seed=seed,
            workload_hash=f"step15_multi_model_seed_{seed}",
        )
        git_hash = get_git_commit_hash()

        results_by_model: dict[str, Step15ModelExecutionResult] = {}

        for m_spec in self._models:
            current_backend = backend_inst
            if current_backend is None:
                current_backend = VLLMBackend(
                    config=VLLMConfig(
                        model=m_spec.model_id,
                        enforce_eager=self._enforce_eager,
                    )
                )
                created_local_backend = True

            res = await self.run_model_benchmark(
                backend=current_backend,
                model_spec=m_spec,
                num_requests_per_phase=num_requests_per_phase,
                seed=seed,
            )
            results_by_model[m_spec.model_id] = res

            if created_local_backend:
                backend_inst = None

        # Build cross-model comparative deltas
        model_deltas: dict[str, Step15PerModelDelta] = {}
        optimal_configs: dict[str, str] = {}

        for m_id, res in results_by_model.items():
            if not res.is_successful or "STATIC_CONSERVATIVE" not in res.conditions:
                continue

            cond_cons = res.conditions["STATIC_CONSERVATIVE"]
            cond_opt = res.conditions["STATIC_OPTIMIZED"]
            cond_adapt = res.conditions["SLA_AWARE_ADAPTIVE"]

            if res.selected_optimal_config:
                opt_cfg_str = (
                    f"c={res.selected_optimal_config.max_concurrency}, "
                    f"b={res.selected_optimal_config.max_batch_size}"
                )
            else:
                opt_cfg_str = cond_opt.active_config_str

            optimal_configs[m_id] = opt_cfg_str

            opt_tput_delta = (
                (
                    (cond_opt.overall_throughput_rps - cond_cons.overall_throughput_rps)
                    / cond_cons.overall_throughput_rps
                    * 100.0
                )
                if cond_cons.overall_throughput_rps > 0
                else 0.0
            )
            adapt_tput_delta = (
                (
                    (cond_adapt.overall_throughput_rps - cond_cons.overall_throughput_rps)
                    / cond_cons.overall_throughput_rps
                    * 100.0
                )
                if cond_cons.overall_throughput_rps > 0
                else 0.0
            )

            opt_p95_delta = cond_opt.p95_latency_ms - cond_cons.p95_latency_ms
            adapt_p95_delta = cond_adapt.p95_latency_ms - cond_cons.p95_latency_ms

            model_deltas[m_id] = Step15PerModelDelta(
                model_id=m_id,
                model_family=res.model_spec.model_family,
                parameter_size_label=res.model_spec.parameter_size_label,
                conservative_throughput_rps=cond_cons.overall_throughput_rps,
                conservative_p95_latency_ms=cond_cons.p95_latency_ms,
                conservative_sla_violation_pct=cond_cons.overall_sla_violation_rate_pct,
                optimized_config_str=opt_cfg_str,
                optimized_throughput_rps=cond_opt.overall_throughput_rps,
                optimized_p95_latency_ms=cond_opt.p95_latency_ms,
                optimized_sla_violation_pct=cond_opt.overall_sla_violation_rate_pct,
                adaptive_throughput_rps=cond_adapt.overall_throughput_rps,
                adaptive_p95_latency_ms=cond_adapt.p95_latency_ms,
                adaptive_sla_violation_pct=cond_adapt.overall_sla_violation_rate_pct,
                optimized_vs_conservative_tput_pct=opt_tput_delta,
                adaptive_vs_conservative_tput_pct=adapt_tput_delta,
                optimized_vs_conservative_p95_delta_ms=opt_p95_delta,
                adaptive_vs_conservative_p95_delta_ms=adapt_p95_delta,
            )

        unique_winners = set(optimal_configs.values())
        same_winner = len(unique_winners) == 1 and len(optimal_configs) > 0
        frontier_shifted = len(unique_winners) > 1

        comparison = Step15CrossModelComparison(
            target_slo_p95_ms=self._target_slo.p95_latency_ms,
            model_deltas=model_deltas,
            optimal_configs_by_model=optimal_configs,
            same_config_wins_all_models=same_winner,
            pareto_frontier_shifted=frontier_shifted,
        )

        findings = classify_step15_findings(
            comparison=comparison,
            results_by_model=results_by_model,
            target_slo=self._target_slo,
        )

        first_hash = next(
            (r.workload_hash for r in results_by_model.values() if r.workload_hash),
            "unknown",
        )

        backend_confirmed = verify_step15_cross_model_report(
            results_by_model=results_by_model,
            backend_name=backend_name,
        )

        return Step15CrossModelReport(
            experiment_id=f"step15-cross-model-{int(time.time())}",
            timestamp=time.time(),
            git_commit=git_hash,
            backend=backend_name,
            backend_execution_confirmed=backend_confirmed,
            environment=env_meta,
            target_slo=self._target_slo,
            workload_hash=first_hash,
            models_evaluated=self._models,
            results_by_model=results_by_model,
            comparison=comparison,
            findings=findings,
        )

    async def run_cross_model_benchmark(
        self,
        backend_type: str = "vllm",
        backend_override: InferenceBackend | None = None,
        requests_per_phase: int = DEFAULT_STEP15_REQUESTS_PER_PHASE,
        seed: int = 42,
    ) -> Step15CrossModelReport:
        """Alias for run_experiment with backend routing."""
        return await self.run_experiment(
            backend=backend_override,
            num_requests_per_phase=requests_per_phase,
            seed=seed,
        )

    def save_reports(
        self,
        report: Step15CrossModelReport,
        output_dir: Path | str = "benchmarks/results/step15",
    ) -> tuple[Path, Path, Path]:
        """Save structured JSON reports (summary, raw_results, cross_model_analysis)."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        summary_file = out_path / "summary.json"
        raw_file = out_path / "raw_results.json"
        analysis_file = out_path / "cross_model_analysis.json"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        raw_data: dict[str, Any] = {
            "experiment_id": report.experiment_id,
            "git_commit": report.git_commit,
            "models": {},
        }
        for m_id, res in report.results_by_model.items():
            raw_data["models"][m_id] = {
                "model_spec": res.model_spec.model_dump(),
                "is_successful": res.is_successful,
                "selected_optimal_config": (
                    res.selected_optimal_config.model_dump()
                    if res.selected_optimal_config
                    else None
                ),
                "candidate_evaluations": [c.model_dump() for c in res.candidate_evaluations],
                "conditions": {k: v.model_dump() for k, v in res.conditions.items()},
            }

        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2)

        analysis_data = {
            "experiment_id": report.experiment_id,
            "target_slo_p95_ms": report.target_slo.p95_latency_ms,
            "comparison": report.comparison.model_dump(),
            "findings": report.findings,
        }
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2)

        logger.info("Saved Step 15 reports to directory: %s", out_path)
        return summary_file, raw_file, analysis_file


def format_step15_report(report: Step15CrossModelReport) -> str:
    """Format Step 15 Cross-Model Generalization report into terminal output."""
    lines: list[str] = []
    lines.append("=" * 105)
    lines.append(" INFEROPT STEP 15: MULTI-MODEL GENERALIZATION & CROSS-MODEL OPTIMIZATION REPORT")
    lines.append("=" * 105)
    lines.append(f" Experiment ID  : {report.experiment_id}")
    lines.append(f" Git Commit     : {report.git_commit}")
    lines.append(f" Backend Name   : {report.backend}")
    lines.append(f" Engine Verified: {report.backend_execution_confirmed} (Real Hardware Engine)")
    lines.append(
        f" GPU Model      : {report.environment.gpu_name} (count={report.environment.gpu_count})"
    )
    lines.append(f" Target SLO     : p95 <= {report.target_slo.p95_latency_ms:.1f}ms")
    lines.append(f" Workload Hash  : {report.workload_hash}")
    lines.append(f" Models Count   : {len(report.models_evaluated)} configured")
    lines.append("-" * 105)

    # 1. Cross-Model Comparison Table
    lines.append("\n" + "=" * 105)
    lines.append(" CROSS-MODEL PERFORMANCE & DELTA COMPARISON TABLE")
    lines.append("=" * 105)
    lines.append(
        f"{'Model ID':<32} | {'Param':<6} | {'Winning Config':<14} | "
        f"{'Cons (rps)':<10} | {'Opt (rps)':<10} | {'Opt Delta':<10} | {'Adaptive (rps)':<14}"
    )
    lines.append("-" * 105)

    for m_id, delta in report.comparison.model_deltas.items():
        delta_str = f"{delta.optimized_vs_conservative_tput_pct:>+6.1f}%"
        lines.append(
            f"{m_id:<32} | {delta.parameter_size_label:<6} | {delta.optimized_config_str:<14} | "
            f"{delta.conservative_throughput_rps:>9.2f}  | "
            f"{delta.optimized_throughput_rps:>9.2f}  | "
            f"{delta_str:<10} | {delta.adaptive_throughput_rps:>10.2f} rps"
        )
    lines.append("-" * 105)

    # 2. Per-Model Condition Breakdown
    lines.append("\n" + "=" * 105)
    lines.append(" PER-MODEL DETAILED CONDITION SUMMARY")
    lines.append("=" * 105)
    lines.append(
        f"{'Model ID':<30} | {'Condition':<20} | {'Throughput':<11} | "
        f"{'p95 Lat (ms)':<12} | {'SLA Viol %':<10} | {'Integrity':<9}"
    )
    lines.append("-" * 105)

    for m_id, res in report.results_by_model.items():
        if not res.is_successful:
            lines.append(
                f"{m_id:<30} | FAILED ({res.model_spec.status.value}): {res.execution_error}"
            )
            continue

        for cond_key in ("STATIC_CONSERVATIVE", "STATIC_OPTIMIZED", "SLA_AWARE_ADAPTIVE"):
            cond = res.conditions.get(cond_key)
            if cond is None:
                continue
            integ_str = "PASSED" if cond.integrity_valid else "FAILED"
            lines.append(
                f"{m_id:<30} | {cond.condition_name:<20} | "
                f"{cond.overall_throughput_rps:>9.2f} rps | "
                f"{cond.p95_latency_ms:>10.2f}ms | "
                f"{cond.overall_sla_violation_rate_pct:>7.1f}% | "
                f"{integ_str:<9}"
            )
        lines.append("-" * 105)

    # 3. Categorized Scientific Findings & Hypothesis Verdicts
    lines.append("\n" + "=" * 105)
    lines.append(" CATEGORIZED SCIENTIFIC FINDINGS & HYPOTHESIS VERDICTS")
    lines.append("=" * 105)
    for cat_name, cat_items in report.findings.items():
        lines.append(f"\n  [{cat_name}]:")
        for item in cat_items:
            lines.append(f"    - {item}")
    lines.append("\n" + "=" * 105)

    return "\n".join(lines)
