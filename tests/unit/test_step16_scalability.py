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
    Step16LoadConditionMetrics,
    Step16ScalabilityReport,
    Step16ScalabilityRunner,
    classify_step16_findings,
    format_step16_report,
    verify_step16_engine_execution,
    verify_step16_report,
)
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


class TestStep16EngineVerification:
    """Targeted regression tests for Step 16 Engine Verification."""

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
        p95_lat: float = 90.0,
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
            p50_latency_ms=50.0,
            p90_latency_ms=80.0,
            p95_latency_ms=p95_lat,
            p99_latency_ms=100.0,
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

    def test_valid_real_vllm_evidence_verified_true(self) -> None:
        """Observable runtime facts from a real vLLM execution verify as True."""
        metrics = self._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=True,
            inits=1,
            teardowns=1,
            batch_calls=16,
            integrity_valid=True,
        )
        assert verify_step16_engine_execution(metrics) is True

    def test_mock_backend_verified_false(self) -> None:
        """MockBackend execution must always verify as False."""
        metrics = self._create_sample_condition_metrics(
            backend_name="mock",
            backend_confirmed=False,
            inits=1,
            teardowns=1,
        )
        assert verify_step16_engine_execution(metrics) is False

    def test_zero_backend_calls_verified_false(self) -> None:
        """Zero backend generate calls must verify as False."""
        metrics = self._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=True,
            gen_calls=0,
            batch_calls=0,
        )
        assert verify_step16_engine_execution(metrics) is False

    def test_incomplete_accounting_verified_false(self) -> None:
        """Incomplete or dropped request accounting must verify as False."""
        metrics = self._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=True,
            scheduled=32,
            completed=30,
            failed=2,
            measured=30,
            integrity_valid=False,
        )
        assert verify_step16_engine_execution(metrics) is False

    def test_verify_step16_report_logic(self) -> None:
        """Verify report level verification."""
        valid_rep = self._create_sample_condition_metrics(
            backend_name="vllm", backend_confirmed=True
        )
        from inferopt.benchmarks.step16_scalability import (
            Step16AggregatedConditionMetrics,
            Step16LoadLevelResult,
            Step16SaturationAnalysis,
        )
        from inferopt.benchmarks.vllm_validation import VLLMEnvironmentMetadata

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


class TestStep16TeardownReconciliationAndFailureCleanup:
    """Regression tests for engine teardown reconciliation, OOM cleanup, and native batching."""

    @pytest.mark.asyncio
    async def test_teardown_reconciliation_post_unload(self) -> None:
        """Verify that post-teardown reconciliation updates engine_teardown_count to 1."""
        backend = FakeRealVLLMBackend(default_latency_sec=0.001)
        runner = Step16ScalabilityRunner(
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            load_levels=(8,),
            target_slo=TargetSLO(p95_latency_ms=180.0),
        )

        report = await runner.run_experiment(backend=backend, repetitions=1, warmup_count=1)

        assert report.backend_execution_confirmed is True
        assert backend.engine_initializations == 1
        assert backend.engine_teardowns == 1

        load_res = report.results_by_load[8]
        for rep in load_res.raw_repetitions:
            assert rep.engine_initialization_count == 1
            assert rep.engine_teardown_count == 1
            assert rep.backend_confirmed is True
            assert rep.backend_generate_batch_calls > 0
            assert rep.backend_generate_calls == 0
            assert verify_step16_engine_execution(rep) is True

        assert verify_step16_report(report) is True

    @pytest.mark.asyncio
    async def test_failure_path_cleanup_in_finally(self) -> None:
        """Verify unload_model() is deterministically called when an execution error occurs."""
        backend = FakeRealVLLMBackend(default_latency_sec=0.001)
        runner = Step16ScalabilityRunner(
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            load_levels=(8,),
        )

        # Force candidate evaluation to raise an error
        async def _failing_eval(*args: Any, **kwargs: Any) -> float:
            raise RuntimeError("Simulated request failure")

        runner._evaluate_candidate = _failing_eval  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="Simulated request failure"):
            await runner.run_experiment(backend=backend)

        assert backend.engine_initializations == 1
        assert backend.engine_teardowns == 1

    @pytest.mark.asyncio
    async def test_cuda_oom_cleanup_in_finally(self) -> None:
        """Verify unload_model() is invoked even if a simulated CUDA OOM occurs."""
        backend = FakeRealVLLMBackend(default_latency_sec=0.001)
        runner = Step16ScalabilityRunner(
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            load_levels=(8,),
        )

        async def _oom_eval(*args: Any, **kwargs: Any) -> float:
            raise MemoryError("CUDA out of memory during inference")

        runner._evaluate_candidate = _oom_eval  # type: ignore[method-assign]

        with pytest.raises(MemoryError, match="CUDA out of memory"):
            await runner.run_experiment(backend=backend)

        assert backend.engine_initializations == 1
        assert backend.engine_teardowns == 1

    def test_native_batch_execution_verification(self) -> None:
        """Verify native batch (batch_calls > 0, generate_calls == 0) satisfies verification."""
        metrics_native_batch = TestStep16EngineVerification._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=True,
            inits=1,
            teardowns=1,
            gen_calls=0,
            batch_calls=12,
            integrity_valid=True,
        )
        assert verify_step16_engine_execution(metrics_native_batch) is True

        metrics_no_calls = TestStep16EngineVerification._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=True,
            inits=1,
            teardowns=1,
            gen_calls=0,
            batch_calls=0,
        )
        assert verify_step16_engine_execution(metrics_no_calls) is False

        metrics_teardown_zero = TestStep16EngineVerification._create_sample_condition_metrics(
            backend_name="vllm",
            backend_confirmed=True,
            inits=1,
            teardowns=0,
            batch_calls=12,
        )
        assert verify_step16_engine_execution(metrics_teardown_zero) is False


class TestStep16SaturationWordingAndAnalysis:
    """Verify saturation curve detection and scientifically defensible phrasing."""

    def test_saturation_wording_logic_for_empirical_run(self) -> None:
        """Verify saturation analysis with measured T4 empirical progression."""
        runner = Step16ScalabilityRunner(
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            load_levels=(16, 32, 64, 128, 256),
            target_slo=TargetSLO(p95_latency_ms=180.0),
        )

        # Measured empirical static optimized throughputs & latencies
        load_profiles = {
            16: (7.57, 120.0, 15.0, 0.0),
            32: (5.25, 240.0, 80.0, 0.0),
            64: (4.48, 5819.0, 5200.0, 9.38),
            128: (4.87, 6153.0, 5600.0, 10.94),
            256: (5.13, 5836.0, 5300.0, 5.47),
        }

        from inferopt.benchmarks.step16_scalability import (
            Step16AggregatedConditionMetrics,
            Step16LoadLevelResult,
        )

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
        assert sat.max_achieved_throughput_rps == 7.57
        assert sat.peak_throughput_load == 16
        assert sat.throughput_plateau_load == 32
        assert "Throughput reached its measured maximum at load 16" in sat.saturation_summary
        assert "entered a saturated/non-scaling regime by load 32" in sat.saturation_summary

        findings = classify_step16_findings(
            results_by_load=results_by_load,
            saturation=sat,
            target_slo=TargetSLO(p95_latency_ms=180.0),
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        )

        assert any(
            "Throughput Saturation Regime:" in s
            for s in findings["SUGGESTED_WORKLOAD_OBSERVATIONS"]
        )
        assert any(
            "queue wait times exceeded physical compute capacity" in s
            for s in findings["NOT_PROVEN_AND_LIMITATIONS"]
        )
