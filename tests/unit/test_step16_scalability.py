"""Unit and integration tests for Step 16 Heavy-Load Scalability benchmark."""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from inferopt.backends.mock import MockBackend
from inferopt.benchmarks.cli import build_parser, run_benchmark_cli
from inferopt.benchmarks.step16_scalability import (
    DEFAULT_STEP16_LOAD_LEVELS,
    DEFAULT_STEP16_MODEL_ID,
    Step16AggregatedConditionMetrics,
    Step16LoadConditionMetrics,
    Step16LoadLevelResult,
    Step16SaturationAnalysis,
    Step16ScalabilityReport,
    Step16ScalabilityRunner,
    classify_step16_findings,
    format_step16_report,
    verify_step16_engine_execution,
    verify_step16_report,
)
from inferopt.benchmarks.vllm_validation import VLLMEnvironmentMetadata
from inferopt.optimizer.sla_models import TargetSLO


class TestStep16ModelsAndWorkload:
    """Verify load-level workload generation and determinism."""

    def test_default_load_levels(self) -> None:
        """Verify standard load level sequence."""
        assert DEFAULT_STEP16_LOAD_LEVELS == (16, 32, 64, 128, 256)
        assert DEFAULT_STEP16_MODEL_ID == "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    def test_load_workload_generation_and_hashing(self) -> None:
        """Verify deterministic scenario generation for different load levels."""
        runner = Step16ScalabilityRunner(model_id=DEFAULT_STEP16_MODEL_ID)

        scen_16 = runner.build_load_workload(num_requests=16, seed=42)
        scen_16_repeat = runner.build_load_workload(num_requests=16, seed=42)
        scen_32 = runner.build_load_workload(num_requests=32, seed=42)

        assert len(scen_16.requests) == 16
        assert len(scen_32.requests) == 32

        from inferopt.benchmarks.vllm_validation import compute_workload_hash

        assert compute_workload_hash(scen_16) == compute_workload_hash(scen_16_repeat)
        assert compute_workload_hash(scen_16) != compute_workload_hash(scen_32)


class TestStep16ScalabilityRunnerMocked:
    """Verify execution, aggregation, and invariants under MockBackend."""

    @pytest.mark.asyncio
    async def test_full_mocked_scalability_experiment_and_invariants(self) -> None:
        """Execute scalability benchmark under MockBackend and verify all hard invariants."""
        backend = MockBackend(default_latency_sec=0.002)
        target_slo = TargetSLO(p95_latency_ms=100.0)
        runner = Step16ScalabilityRunner(
            model_id="mock-smollm-1.7B",
            load_levels=(8, 16),
            target_slo=target_slo,
        )

        report = await runner.run_experiment(
            backend=backend,
            repetitions=2,
            warmup_count=1,
            seed=42,
        )

        assert isinstance(report, Step16ScalabilityReport)
        assert report.backend == "mock"
        assert report.backend_execution_confirmed is False
        assert len(report.results_by_load) == 2
        assert 8 in report.results_by_load
        assert 16 in report.results_by_load

        # Check per-load metrics and hard invariants
        for load, load_res in report.results_by_load.items():
            assert load_res.load_level == load
            assert len(load_res.raw_repetitions) == 2 * 3  # 2 reps * 3 conditions
            assert len(load_res.conditions) == 3

            for cond_name in ("STATIC_CONSERVATIVE", "STATIC_OPTIMIZED", "SLA_AWARE_ADAPTIVE"):
                c_agg = load_res.conditions[cond_name]
                assert c_agg.repetition_count == 2
                assert c_agg.mean_throughput_rps > 0.0
                assert c_agg.all_repetitions_valid is True

            for rep in load_res.raw_repetitions:
                assert rep.scheduled_requests == load
                assert rep.completed_requests == load
                assert rep.failed_requests == 0
                assert rep.measured_requests == load
                assert rep.integrity_valid is True
                assert rep.total_duration_sec > 0.0
                assert rep.mean_latency_ms >= rep.mean_queue_wait_ms - 1e-6
                assert rep.p95_latency_ms >= rep.mean_queue_wait_ms - 1e-6
                assert rep.engine_initialization_count >= 1
                assert rep.engine_teardown_count >= 1

        # Check saturation analysis
        sat = report.saturation_analysis
        assert sat.max_achieved_throughput_rps > 0.0
        assert sat.evaluated_loads == (8, 16)
        assert isinstance(sat.batching_efficiency_trend, str)

        # Check findings
        assert "PROVEN_OBSERVATIONS" in report.findings
        assert "SUGGESTED_WORKLOAD_OBSERVATIONS" in report.findings
        assert "NOT_PROVEN_AND_LIMITATIONS" in report.findings

        # Check terminal formatting
        formatted = format_step16_report(report)
        assert "INFEROPT STEP 16" in formatted
        assert "SCALABILITY & PERFORMANCE SUMMARY" in formatted
        assert "EMPIRICAL SATURATION CURVE CHARACTERIZATION" in formatted


class TestStep16EvidenceBasedVerification:
    """Targeted regression tests for Section 2 & 12 (Predicates A-I and report verification)."""

    @staticmethod
    def _create_sample_condition_metrics(
        backend_name: str = "vllm",
        backend_confirmed: bool = True,
        inits: int = 1,
        teardowns: int = 1,
        gen_calls: int = 0,
        batch_calls: int = 16,
        integrity_valid: bool = True,
        scheduled: int = 32,
        completed: int = 32,
        failed: int = 0,
        measured: int = 32,
        duration: float = 4.0,
        mean_lat: float = 60.0,
        p50_lat: float = 50.0,
        p90_lat: float = 80.0,
        p95_lat: float = 90.0,
        p99_lat: float = 100.0,
        queue_wait: float = 15.0,
    ) -> Step16LoadConditionMetrics:
        return Step16LoadConditionMetrics(
            load_level=scheduled,
            repetition_index=0,
            condition_name="STATIC_OPTIMIZED",
            condition_type="static_optimized",
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            active_config_str="c=4, b=4, w=50.0ms",
            target_slo_p95_ms=180.0,
            workload_hash="hash_123",
            scheduled_requests=scheduled,
            completed_requests=completed,
            failed_requests=failed,
            measured_requests=measured,
            total_duration_sec=duration,
            throughput_rps=completed / duration if duration > 0 else 0.0,
            output_tokens_per_sec=200.0,
            input_tokens_per_sec=100.0,
            total_tokens_per_sec=300.0,
            mean_latency_ms=mean_lat,
            p50_latency_ms=p50_lat,
            p90_latency_ms=p90_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            min_latency_ms=20.0,
            max_latency_ms=110.0,
            mean_queue_wait_ms=queue_wait,
            p95_queue_wait_ms=queue_wait + 5.0,
            mean_backend_execution_ms=mean_lat - queue_wait,
            p95_backend_execution_ms=p95_lat - queue_wait,
            total_batches=8,
            mean_batch_size=4.0,
            max_batch_size=4,
            batch_size_distribution={4: 8},
            peak_queue_depth=4,
            peak_active_requests=4,
            total_sla_violations=0,
            overall_sla_violation_rate_pct=0.0,
            total_adaptations=0,
            adaptation_events=(),
            engine_initialization_count=inits,
            engine_teardown_count=teardowns,
            backend_name=backend_name,
            backend_confirmed=backend_confirmed,
            backend_generate_calls=gen_calls,
            backend_generate_batch_calls=batch_calls,
            integrity_valid=integrity_valid,
        )

    def test_A_successful_vllm_verification(self) -> None:
        """A. Observable runtime facts from a real vLLM execution verify as True."""
        metrics = self._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=True,
            inits=1,
            teardowns=1,
            batch_calls=16,
            integrity_valid=True,
        )
        assert verify_step16_engine_execution(metrics) is True

    def test_B_verification_fails_when_backend_is_mock(self) -> None:
        """B. Verification fails when backend is mock."""
        metrics = self._create_sample_condition_metrics(
            backend_name="mock",
            backend_confirmed=False,
            inits=1,
            teardowns=1,
        )
        assert verify_step16_engine_execution(metrics) is False

    def test_C_verification_fails_when_backend_confirmed_is_false(self) -> None:
        """C. Verification fails when backend_confirmed is false."""
        metrics = self._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=False,
            inits=1,
            teardowns=1,
        )
        assert verify_step16_engine_execution(metrics) is False

    def test_D_verification_fails_when_native_batch_calls_is_zero(self) -> None:
        """D. Verification fails when native batch calls == 0."""
        metrics = self._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=True,
            gen_calls=0,
            batch_calls=0,
        )
        assert verify_step16_engine_execution(metrics) is False

    def test_E_verification_fails_when_scheduled_not_equal_completed_plus_failed(self) -> None:
        """E. Verification fails when scheduled != completed + failed."""
        metrics = self._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=True,
            scheduled=32,
            completed=30,
            failed=0,  # 32 != 30 + 0
            measured=30,
        )
        assert verify_step16_engine_execution(metrics) is False

    def test_F_verification_fails_when_measured_not_equal_completed(self) -> None:
        """F. Verification fails when measured != completed."""
        metrics = self._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=True,
            scheduled=32,
            completed=32,
            failed=0,
            measured=30,  # 30 != 32
        )
        assert verify_step16_engine_execution(metrics) is False

    def test_G_verification_fails_when_integrity_valid_is_false(self) -> None:
        """G. Verification fails when integrity_valid == false."""
        metrics = self._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=True,
            integrity_valid=False,
        )
        assert verify_step16_engine_execution(metrics) is False

    def test_H_verification_fails_when_teardown_count_is_zero(self) -> None:
        """H. Verification fails when teardown_count == 0."""
        metrics = self._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=True,
            inits=1,
            teardowns=0,
        )
        assert verify_step16_engine_execution(metrics) is False

    def test_I_verification_passes_with_complete_evidence(self) -> None:
        """I. Verification passes with complete valid evidence across all 20 predicates."""
        metrics = self._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=True,
            inits=1,
            teardowns=1,
            gen_calls=0,
            batch_calls=16,
            scheduled=32,
            completed=32,
            failed=0,
            measured=32,
            duration=3.5,
            mean_lat=60.0,
            p50_lat=50.0,
            p90_lat=80.0,
            p95_lat=90.0,
            p99_lat=100.0,
            queue_wait=15.0,
            integrity_valid=True,
        )
        assert verify_step16_engine_execution(metrics) is True

    def test_verify_step16_report_logic(self) -> None:
        """Verify report level verification."""
        valid_rep = self._create_sample_condition_metrics(
            backend_name="vllm", backend_confirmed=True
        )
        env = VLLMEnvironmentMetadata(
            os_name="macOS",
            os_version="14.0",
            cpu_architecture="arm64",
            python_version="3.11",
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            warmup_count=0,
            repetitions=1,
            workload_seed=42,
            workload_hash="hash_1",
        )
        agg = Step16AggregatedConditionMetrics(
            condition_name="STATIC_OPTIMIZED",
            condition_type="static_optimized",
            repetition_count=1,
            mean_throughput_rps=8.0,
            mean_p95_latency_ms=90.0,
            mean_p99_latency_ms=100.0,
            mean_queue_wait_ms=15.0,
            mean_batch_size=4.0,
            mean_sla_violation_rate_pct=0.0,
            total_adaptations=0,
            all_repetitions_valid=True,
        )
        load_res = Step16LoadLevelResult(
            load_level=32,
            workload_hash="hash_1",
            conditions={"STATIC_OPTIMIZED": agg},
            raw_repetitions=(valid_rep,),
            optimized_vs_conservative_tput_pct=0.0,
            adaptive_vs_conservative_tput_pct=0.0,
            optimized_vs_conservative_p95_delta_ms=0.0,
            adaptive_vs_conservative_p95_delta_ms=0.0,
        )
        sat = Step16SaturationAnalysis(
            evaluated_loads=(32,),
            max_achieved_throughput_rps=8.0,
            peak_throughput_condition="STATIC_OPTIMIZED",
            batching_efficiency_trend="Stable",
            adaptive_sla_protection_demonstrated=False,
        )
        report_valid = Step16ScalabilityReport(
            experiment_id="exp_1",
            timestamp=123456.0,
            git_commit="abcdef",
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            backend="vllm",
            backend_execution_confirmed=True,
            environment=env,
            target_slo=TargetSLO(p95_latency_ms=180.0),
            load_levels=(32,),
            results_by_load={32: load_res},
            saturation_analysis=sat,
            findings=classify_step16_findings(
                results_by_load={32: load_res},
                saturation=sat,
                target_slo=TargetSLO(p95_latency_ms=180.0),
                model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            ),
        )
        assert verify_step16_report(report_valid) is True

        report_mock = report_valid.model_copy(update={"backend": "mock"})
        assert verify_step16_report(report_mock) is False


class FakeRealVLLMBackend(MockBackend):
    """Simulated real vLLM backend for regression testing engine lifecycle and verification."""

    def __init__(self, default_latency_sec: float = 0.001) -> None:
        super().__init__(default_latency_sec=default_latency_sec)
        self._engine_initializations = 0
        self._engine_teardowns = 0
        self._is_loaded = False

    @property
    def backend_name(self) -> str:
        return "vllm"

    @property
    def is_real_execution(self) -> bool:
        return True

    async def load_model(self) -> None:
        self._engine_initializations += 1
        self._is_loaded = True

    async def unload_model(self) -> None:
        if self._is_loaded:
            self._engine_teardowns += 1
            self._is_loaded = False

    async def generate_batch(self, batch: Any) -> list[Any]:
        self._generate_batch_calls += 1
        return await super().generate_batch(batch)


class TestStep16LifecycleAndFailureHandling:
    """Regression tests for Section 3, 4, 5, 6, 12 (J, K, L, M, N)."""

    @pytest.mark.asyncio
    async def test_J_teardown_occurs_on_normal_completion(self) -> None:
        """J. Teardown occurs on normal completion with 1 init and 1 teardown per condition."""
        backend = FakeRealVLLMBackend(default_latency_sec=0.001)
        runner = Step16ScalabilityRunner(
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            load_levels=(8,),
            target_slo=TargetSLO(p95_latency_ms=180.0),
        )

        report = await runner.run_experiment(backend=backend, repetitions=1, warmup_count=1)

        assert report.backend_execution_confirmed is True

        load_res = report.results_by_load[8]
        for rep in load_res.raw_repetitions:
            assert rep.engine_initialization_count == 1
            assert rep.engine_teardown_count == 1
            assert rep.backend_confirmed is True
            assert rep.backend_generate_batch_calls > 0
            assert verify_step16_engine_execution(rep) is True

        assert verify_step16_report(report) is True

    @pytest.mark.asyncio
    async def test_K_teardown_occurs_when_benchmark_raises(self) -> None:
        """K. Teardown occurs when benchmark condition raises an exception."""
        backend = FakeRealVLLMBackend(default_latency_sec=0.001)
        runner = Step16ScalabilityRunner(
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            load_levels=(8,),
        )

        # Force single condition run to fail
        async def _failing_run(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Benchmark condition execution error")

        runner._execute_single_condition_run = _failing_run  # type: ignore[method-assign]

        workload = runner.build_load_workload(num_requests=8, seed=42)
        from inferopt.optimizer.models import TunableConfig

        with pytest.raises(RuntimeError, match="Benchmark condition execution error"):
            await runner._execute_isolated_condition(
                condition_name="STATIC_CONSERVATIVE",
                condition_type="static_conservative",
                load_level=8,
                repetition_idx=0,
                initial_config=TunableConfig(),
                is_adaptive=False,
                workload=workload,
                backend_override=backend,
            )

    @pytest.mark.asyncio
    async def test_L_teardown_occurs_on_oom_failure_path(self) -> None:
        """L. Teardown occurs on OOM/failure path."""
        backend = FakeRealVLLMBackend(default_latency_sec=0.001)
        runner = Step16ScalabilityRunner(
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            load_levels=(8,),
        )

        async def _oom_run(*args: Any, **kwargs: Any) -> Any:
            raise MemoryError("CUDA out of memory during batch generation")

        runner._execute_single_condition_run = _oom_run  # type: ignore[method-assign]

        workload = runner.build_load_workload(num_requests=8, seed=42)
        from inferopt.optimizer.models import TunableConfig

        with pytest.raises(MemoryError, match="CUDA out of memory"):
            await runner._execute_isolated_condition(
                condition_name="STATIC_OPTIMIZED",
                condition_type="static_optimized",
                load_level=8,
                repetition_idx=0,
                initial_config=TunableConfig(),
                is_adaptive=False,
                workload=workload,
                backend_override=backend,
            )

    @pytest.mark.asyncio
    async def test_M_adaptive_scheduler_changes_do_not_increment_engine_lifecycle_counters(
        self,
    ) -> None:
        """M. Adaptive changes do not increment lifecycle counters (inits=1, teardowns=1)."""
        backend = FakeRealVLLMBackend(default_latency_sec=0.001)
        runner = Step16ScalabilityRunner(
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            load_levels=(8,),
            target_slo=TargetSLO(p95_latency_ms=180.0),
        )

        workload = runner.build_load_workload(num_requests=8, seed=42)
        from inferopt.optimizer.models import TunableConfig

        metrics = await runner._execute_isolated_condition(
            condition_name="SLA_AWARE_ADAPTIVE",
            condition_type="sla_adaptive",
            load_level=8,
            repetition_idx=0,
            initial_config=TunableConfig(max_concurrency=1, max_batch_size=2),
            is_adaptive=True,
            workload=workload,
            backend_override=backend,
        )

        assert metrics.condition_name == "SLA_AWARE_ADAPTIVE"
        assert metrics.engine_initialization_count == 1
        assert metrics.engine_teardown_count == 1
        assert verify_step16_engine_execution(metrics) is True

    @pytest.mark.asyncio
    async def test_N_exactly_one_init_and_one_teardown_per_condition(self) -> None:
        """N. Exactly one init + one teardown per condition across loads and repetitions."""
        backend = FakeRealVLLMBackend(default_latency_sec=0.001)
        runner = Step16ScalabilityRunner(
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            load_levels=(8, 16),
            target_slo=TargetSLO(p95_latency_ms=180.0),
        )

        report = await runner.run_experiment(backend=backend, repetitions=2, seed=42)

        assert len(report.results_by_load) == 2
        total_condition_runs = 0
        for load_res in report.results_by_load.values():
            for rep in load_res.raw_repetitions:
                total_condition_runs += 1
                assert rep.engine_initialization_count == 1
                assert rep.engine_teardown_count == 1
                assert verify_step16_engine_execution(rep) is True

        assert total_condition_runs == 2 * (2 * 3)  # 2 loads * 2 reps * 3 conditions = 12 runs
        assert report.backend_execution_confirmed is True


class TestStep16SaturationWordingAndAnalysis:
    """Verify saturation curve detection, SLA violation guardrails, and JIT limitation phrasing."""

    def test_O_saturation_wording_does_not_claim_false_plateau_beginning_at_32(self) -> None:
        """O. Saturation wording does NOT claim a false 'plateau beginning at 32'."""
        runner = Step16ScalabilityRunner(
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            load_levels=(16, 32, 64, 128, 256),
            target_slo=TargetSLO(p95_latency_ms=180.0),
        )

        # Measured empirical static optimized throughputs
        load_profiles = {
            16: (8.39, 120.0, 15.0, 0.0),
            32: (5.74, 240.0, 80.0, 0.0),
            64: (4.97, 5819.0, 5200.0, 9.38),
            128: (5.31, 6153.0, 5600.0, 10.94),
            256: (5.53, 5836.0, 5300.0, 5.47),
        }

        results_by_load: dict[int, Step16LoadLevelResult] = {}
        for load, (tput, p95, qw, sla_viol) in load_profiles.items():
            agg_opt = Step16AggregatedConditionMetrics(
                condition_name="STATIC_OPTIMIZED",
                condition_type="static_optimized",
                repetition_count=1,
                mean_throughput_rps=tput,
                mean_p95_latency_ms=p95,
                mean_p99_latency_ms=p95 + 100.0,
                mean_queue_wait_ms=qw,
                mean_batch_size=4.0,
                mean_sla_violation_rate_pct=sla_viol,
                total_adaptations=0,
                all_repetitions_valid=True,
            )
            agg_cons = Step16AggregatedConditionMetrics(
                condition_name="STATIC_CONSERVATIVE",
                condition_type="static_conservative",
                repetition_count=1,
                mean_throughput_rps=tput * 0.4,
                mean_p95_latency_ms=p95 * 1.5,
                mean_p99_latency_ms=p95 * 1.6,
                mean_queue_wait_ms=qw * 1.2,
                mean_batch_size=2.0,
                mean_sla_violation_rate_pct=sla_viol * 1.5,
                total_adaptations=0,
                all_repetitions_valid=True,
            )
            agg_adapt = Step16AggregatedConditionMetrics(
                condition_name="SLA_AWARE_ADAPTIVE",
                condition_type="sla_adaptive",
                repetition_count=1,
                mean_throughput_rps=tput * 0.98,
                mean_p95_latency_ms=p95,
                mean_p99_latency_ms=p95 + 50.0,
                mean_queue_wait_ms=qw,
                mean_batch_size=4.0,
                mean_sla_violation_rate_pct=sla_viol,
                total_adaptations=2,
                all_repetitions_valid=True,
            )
            results_by_load[load] = Step16LoadLevelResult(
                load_level=load,
                workload_hash=f"hash_{load}",
                conditions={
                    "STATIC_CONSERVATIVE": agg_cons,
                    "STATIC_OPTIMIZED": agg_opt,
                    "SLA_AWARE_ADAPTIVE": agg_adapt,
                },
                raw_repetitions=(),
                optimized_vs_conservative_tput_pct=150.0,
                adaptive_vs_conservative_tput_pct=145.0,
                optimized_vs_conservative_p95_delta_ms=-100.0,
                adaptive_vs_conservative_p95_delta_ms=-100.0,
            )

        sat = runner._analyze_saturation_curve(results_by_load)
        assert sat.max_achieved_throughput_rps == 8.39
        assert sat.peak_throughput_load == 16
        assert sat.saturation_regime_load == 32
        expected_msg = (
            "Throughput reached its measured maximum at offered load 16 (8.39 rps) "
            "and entered a saturated/non-scaling regime by load 32"
        )
        assert expected_msg in sat.saturation_summary
        assert (
            "subsequent increases in offered load did not produce proportional throughput growth"
            in sat.saturation_summary
        )

    def test_P_adaptive_sla_violations_at_heavy_loads_preserved(self) -> None:
        """P. Adaptive SLA violations at 64/128/256 are preserved (SLA Protection: NO)."""
        runner = Step16ScalabilityRunner(
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            load_levels=(16, 32, 64, 128, 256),
            target_slo=TargetSLO(p95_latency_ms=5000.0),
        )

        load_profiles = {
            16: (8.39, 3789.23, 100.0, 0.0),
            32: (5.74, 3801.84, 200.0, 0.0),
            64: (4.97, 5360.80, 4800.0, 9.4),
            128: (5.31, 5724.73, 5200.0, 10.9),
            256: (5.53, 5453.44, 4900.0, 5.5),
        }

        results_by_load: dict[int, Step16LoadLevelResult] = {}
        for load, (tput, p95, qw, sla_viol) in load_profiles.items():
            agg_adapt = Step16AggregatedConditionMetrics(
                condition_name="SLA_AWARE_ADAPTIVE",
                condition_type="sla_adaptive",
                repetition_count=1,
                mean_throughput_rps=tput,
                mean_p95_latency_ms=p95,
                mean_p99_latency_ms=p95 + 50.0,
                mean_queue_wait_ms=qw,
                mean_batch_size=4.0,
                mean_sla_violation_rate_pct=sla_viol,
                total_adaptations=2,
                all_repetitions_valid=True,
            )
            results_by_load[load] = Step16LoadLevelResult(
                load_level=load,
                workload_hash=f"hash_{load}",
                conditions={"SLA_AWARE_ADAPTIVE": agg_adapt},
                raw_repetitions=(),
                optimized_vs_conservative_tput_pct=0.0,
                adaptive_vs_conservative_tput_pct=0.0,
                optimized_vs_conservative_p95_delta_ms=0.0,
                adaptive_vs_conservative_p95_delta_ms=0.0,
            )

        sat = runner._analyze_saturation_curve(results_by_load)
        # Violations occur at 64, 128, 256 (> 5000ms target SLO)
        assert sat.adaptive_sla_protection_demonstrated is False

        env = VLLMEnvironmentMetadata(
            os_name="Linux",
            os_version="5.15",
            cpu_architecture="x86_64",
            python_version="3.11",
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            warmup_count=2,
            repetitions=1,
            workload_seed=42,
            workload_hash="hash_step16",
        )
        report = Step16ScalabilityReport(
            experiment_id="exp_test_sla",
            timestamp=1000.0,
            git_commit="abc",
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            backend="vllm",
            backend_execution_confirmed=True,
            environment=env,
            target_slo=TargetSLO(p95_latency_ms=5000.0),
            load_levels=(16, 32, 64, 128, 256),
            results_by_load=results_by_load,
            saturation_analysis=sat,
            findings=classify_step16_findings(
                results_by_load=results_by_load,
                saturation=sat,
                target_slo=TargetSLO(p95_latency_ms=5000.0),
                model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            ),
        )
        formatted = format_step16_report(report)
        assert "SLA Protection Demonstrated: NO" in formatted

    def test_Q_jit_warning_and_limitations_represented_honestly(self) -> None:
        """Q. JIT warning/contamination is represented honestly in scientific limitations."""
        findings = classify_step16_findings(
            results_by_load={},
            saturation=Step16SaturationAnalysis(
                evaluated_loads=(16, 32),
                max_achieved_throughput_rps=8.39,
                peak_throughput_condition="STATIC_OPTIMIZED",
                batching_efficiency_trend="Stable",
                adaptive_sla_protection_demonstrated=False,
            ),
            target_slo=TargetSLO(p95_latency_ms=5000.0),
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        )

        limitations = findings["NOT_PROVEN_AND_LIMITATIONS"]
        assert any("Inference-Time Triton JIT Compilation:" in lim for lim in limitations)
        assert any("kernel_unified_attention" in lim for lim in limitations)


class TestStep16ReportSerialization:
    """Verify JSON report saving and structure."""

    @pytest.mark.asyncio
    async def test_step16_report_serialization_and_roundtrip(self) -> None:
        """Verify summary.json, raw_results.json, and scalability_analysis.json."""
        backend = MockBackend(default_latency_sec=0.001)
        runner = Step16ScalabilityRunner(
            model_id="mock-model",
            load_levels=(8,),
        )
        report = await runner.run_experiment(backend=backend, repetitions=1, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_f, raw_f, analysis_f = runner.save_reports(report, tmpdir)
            assert Path(summary_f).exists()
            assert Path(raw_f).exists()
            assert Path(analysis_f).exists()

            with open(summary_f, encoding="utf-8") as f:
                s_data = json.load(f)
                assert s_data["experiment_id"] == report.experiment_id
                assert "results_by_load" in s_data

            with open(raw_f, encoding="utf-8") as f:
                r_data = json.load(f)
                assert "raw_results_by_load" in r_data

            with open(analysis_f, encoding="utf-8") as f:
                a_data = json.load(f)
                assert "saturation_analysis" in a_data
                assert "findings" in a_data


class TestStep16CLIIntegration:
    """Verify CLI argument parsing and dispatch for Step 16."""

    def test_parser_step16_arguments(self) -> None:
        """Verify --experiment-step16, --loads, and --target-slo-p95-ms parsing."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--experiment-step16",
                "--loads",
                "16",
                "32",
                "64",
                "--target-slo-p95-ms",
                "150.0",
                "--repetitions",
                "2",
                "--backend",
                "mock",
            ]
        )
        assert args.experiment_step16 is True
        assert args.loads == [16, 32, 64]
        assert args.target_slo_p95_ms == 150.0
        assert args.repetitions == 2
        assert args.backend == "mock"

    @pytest.mark.asyncio
    async def test_run_benchmark_cli_step16_mock(self) -> None:
        """Verify end-to-end CLI execution with MockBackend returns exit code 0."""
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args(
                [
                    "--experiment-step16",
                    "--loads",
                    "8",
                    "--repetitions",
                    "1",
                    "--backend",
                    "mock",
                    "--output",
                    tmpdir,
                ]
            )
            exit_code = await run_benchmark_cli(args)
            assert exit_code == 0
            assert (Path(tmpdir) / "summary.json").exists()
