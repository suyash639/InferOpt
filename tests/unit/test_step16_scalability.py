"""Unit and integration tests for Step 16 Heavy-Load Scalability benchmark."""

import json
import tempfile
from pathlib import Path

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
