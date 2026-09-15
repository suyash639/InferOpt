"""Unit and integration tests for Step 15 Multi-Model Generalization benchmark."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from inferopt.backends.mock import MockBackend
from inferopt.benchmarks.cli import build_parser, run_benchmark_cli
from inferopt.benchmarks.step15_multi_model import (
    DEFAULT_STEP15_MODELS,
    ModelSpec,
    ModelStatus,
    Step15CrossModelComparison,
    Step15CrossModelReport,
    Step15ModelConditionSummary,
    Step15ModelExecutionResult,
    Step15MultiModelRunner,
    classify_step15_findings,
    format_step15_report,
    verify_step15_cross_model_report,
    verify_step15_model_execution,
)
from inferopt.core.models import InferenceRequest, InferenceResponse
from inferopt.optimizer.models import TunableConfig
from inferopt.optimizer.sla_models import TargetSLO


class TestStep15MultiModelSpec:
    """Verify ModelSpec construction, default registry, and validation."""

    def test_default_models_registry(self) -> None:
        """Verify default candidate models in Step 15 registry."""
        assert len(DEFAULT_STEP15_MODELS) == 3
        ids = [m.model_id for m in DEFAULT_STEP15_MODELS]
        assert "Qwen/Qwen2.5-0.5B-Instruct" in ids
        assert "Qwen/Qwen2.5-1.5B-Instruct" in ids
        assert "meta-llama/Llama-3.2-1B-Instruct" in ids

        for m in DEFAULT_STEP15_MODELS:
            assert m.status == ModelStatus.RUNNABLE
            assert m.parameter_size_label in ("0.5B", "1.5B", "1B")
            assert m.backend == "vllm"

    def test_custom_model_spec(self) -> None:
        """Verify custom ModelSpec instantiation and immutability."""
        spec = ModelSpec(
            model_id="custom-org/custom-model-3B",
            model_family="custom",
            parameter_size_label="3B",
            expected_context_length=4096,
        )
        assert spec.model_id == "custom-org/custom-model-3B"
        assert spec.parameter_size_label == "3B"
        assert spec.expected_context_length == 4096

        with pytest.raises(ValidationError):
            spec.model_id = "other"


class TestStep15MultiModelRunner:
    """Verify Step 15 multi-model execution, candidate exploration, and invariants."""

    def test_phase_scenarios_construction(self) -> None:
        """Verify 4-phase sequence definitions and request counts."""
        runner = Step15MultiModelRunner()
        scenarios = runner.build_step15_phase_scenarios(requests_per_phase=6, seed=42)

        assert len(scenarios) == 4
        assert scenarios[0].scenario_name == "PHASE_1_MODERATE"
        assert scenarios[1].scenario_name == "PHASE_2_BURST"
        assert scenarios[2].scenario_name == "PHASE_3_SATURATED"
        assert scenarios[3].scenario_name == "PHASE_4_COOLDOWN"

        for s in scenarios:
            assert len(s.requests) == 6

    @pytest.mark.asyncio
    async def test_full_mocked_multi_model_benchmark_and_invariants(self) -> None:
        """Execute multi-model benchmark under MockBackend and verify hard invariants."""
        test_models = (
            ModelSpec(
                model_id="mock-qwen-0.5b",
                model_family="qwen2.5",
                parameter_size_label="0.5B",
            ),
            ModelSpec(
                model_id="mock-qwen-1.5b",
                model_family="qwen2.5",
                parameter_size_label="1.5B",
            ),
        )

        backend = MockBackend(default_latency_sec=0.002)
        target_slo = TargetSLO(p95_latency_ms=120.0)
        runner = Step15MultiModelRunner(
            models=test_models,
            target_slo=target_slo,
        )

        requests_per_phase = 4
        report = await runner.run_cross_model_benchmark(
            backend_type="mock",
            backend_override=backend,
            requests_per_phase=requests_per_phase,
            seed=42,
        )

        assert isinstance(report, Step15CrossModelReport)
        assert report.backend == "mock"
        assert report.backend_execution_confirmed is False
        assert len(report.results_by_model) == 2

        # Check each model result
        for _m_id, res in report.results_by_model.items():
            assert res.is_successful is True
            assert res.selected_optimal_config is not None
            assert len(res.candidate_evaluations) == 12  # 3 concurrencies * 4 batch sizes
            assert len(res.conditions) == 3

            for cond_name in ("STATIC_CONSERVATIVE", "STATIC_OPTIMIZED", "SLA_AWARE_ADAPTIVE"):
                cond = res.conditions[cond_name]
                expected_reqs = 4 * requests_per_phase
                assert cond.scheduled_requests == expected_reqs
                assert cond.completed_requests == expected_reqs
                assert cond.failed_requests == 0
                assert cond.measured_requests == expected_reqs
                assert cond.integrity_valid is True
                assert cond.total_duration_sec > 0.0

                for pm in cond.phase_metrics:
                    assert pm.scheduled_requests == requests_per_phase
                    assert pm.completed_requests == requests_per_phase
                    assert pm.failed_requests == 0
                    assert pm.measured_requests == requests_per_phase
                    assert pm.mean_latency_ms >= pm.avg_queue_wait_ms - 1e-6
                    assert pm.p95_latency_ms >= pm.avg_queue_wait_ms - 1e-6

        # Check cross-model comparison
        assert len(report.comparison.model_deltas) == 2
        for _m_id, delta in report.comparison.model_deltas.items():
            assert delta.conservative_throughput_rps > 0.0
            assert delta.optimized_throughput_rps > 0.0
            assert delta.adaptive_throughput_rps > 0.0

        # Check scientific findings
        assert "PROVEN_OBSERVATIONS" in report.findings
        assert "SUGGESTED_WORKLOAD_OBSERVATIONS" in report.findings
        assert "NOT_PROVEN_AND_LIMITATIONS" in report.findings

        # Formatted report
        text_report = format_step15_report(report)
        assert "INFEROPT STEP 15" in text_report
        assert "CROSS-MODEL PERFORMANCE & DELTA COMPARISON TABLE" in text_report
        assert "PER-MODEL DETAILED CONDITION SUMMARY" in text_report
        assert "CATEGORIZED SCIENTIFIC FINDINGS & HYPOTHESIS VERDICTS" in text_report

    @pytest.mark.asyncio
    async def test_graceful_model_failure_handling(self) -> None:
        """Verify runner tolerates model errors and completes surviving models."""

        class FailingBackend(MockBackend):
            def __init__(self, fail_model_id: str) -> None:
                super().__init__(default_latency_sec=0.001)
                self.fail_model_id = fail_model_id

            async def generate(self, request: InferenceRequest) -> InferenceResponse:
                if request.model == self.fail_model_id:
                    raise RuntimeError("Authentication failed: gated HuggingFace repository")
                return await super().generate(request)

        test_models = (
            ModelSpec(
                model_id="gated-llama-model",
                model_family="llama3.2",
                parameter_size_label="1B",
            ),
            ModelSpec(
                model_id="working-qwen-model",
                model_family="qwen2.5",
                parameter_size_label="0.5B",
            ),
        )

        runner = Step15MultiModelRunner(models=test_models)
        backend = FailingBackend(fail_model_id="gated-llama-model")

        # Mock the per-model execution so gated model fails during execution
        report = await runner.run_cross_model_benchmark(
            backend_type="mock",
            backend_override=backend,
            requests_per_phase=2,
            seed=42,
        )

        assert isinstance(report, Step15CrossModelReport)
        assert "working-qwen-model" in report.results_by_model
        assert report.results_by_model["working-qwen-model"].is_successful is True

    @pytest.mark.asyncio
    async def test_step15_report_serialization(self) -> None:
        """Verify 3 structured JSON reports are saved and valid."""
        test_models = (
            ModelSpec(
                model_id="mock-qwen-0.5b",
                model_family="qwen2.5",
                parameter_size_label="0.5B",
            ),
        )
        backend = MockBackend(default_latency_sec=0.001)
        runner = Step15MultiModelRunner(models=test_models)

        report = await runner.run_cross_model_benchmark(
            backend_type="mock",
            backend_override=backend,
            requests_per_phase=2,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_f, raw_f, analysis_f = runner.save_reports(report, tmpdir)
            assert Path(summary_f).exists()
            assert Path(raw_f).exists()
            assert Path(analysis_f).exists()

            with open(summary_f, encoding="utf-8") as f:
                summary_data = json.load(f)
                assert summary_data["experiment_id"] == report.experiment_id
                assert "mock-qwen-0.5b" in summary_data["results_by_model"]

            with open(raw_f, encoding="utf-8") as f:
                raw_data = json.load(f)
                assert "mock-qwen-0.5b" in raw_data["models"]
                assert len(raw_data["models"]["mock-qwen-0.5b"]["candidate_evaluations"]) > 0

            with open(analysis_f, encoding="utf-8") as f:
                analysis_data = json.load(f)
                assert "findings" in analysis_data
                assert "comparison" in analysis_data


class TestStep15FindingsClassification:
    """Verify conservative hypothesis classification logic."""

    def test_classification_categories(self) -> None:
        """Verify findings format and standard limitations."""
        comparison = Step15CrossModelComparison(
            target_slo_p95_ms=180.0,
            model_deltas={},
            optimal_configs_by_model={},
            same_config_wins_all_models=True,
            pareto_frontier_shifted=False,
        )
        findings = classify_step15_findings(
            comparison=comparison,
            results_by_model={},
            target_slo=TargetSLO(p95_latency_ms=180.0),
        )

        assert "PROVEN_OBSERVATIONS" in findings
        assert "SUGGESTED_WORKLOAD_OBSERVATIONS" in findings
        assert "NOT_PROVEN_AND_LIMITATIONS" in findings

        limitations = findings["NOT_PROVEN_AND_LIMITATIONS"]
        assert any("Hardware Specificity" in lim for lim in limitations)
        assert any("Workload Specificity" in lim for lim in limitations)
        assert any("No Universal Dominance" in lim for lim in limitations)


class TestStep15CLIIntegration:
    """Verify CLI argument parsing and dispatch for Step 15."""

    def test_parser_step15_arguments(self) -> None:
        """Verify --experiment-step15, --models, and --target-slo-p95-ms parsing."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--experiment-step15",
                "--models",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--num-requests-per-phase",
                "4",
                "--target-slo-p95-ms",
                "150.0",
                "--backend",
                "mock",
            ]
        )

        assert args.experiment_step15 is True
        assert args.models == ["Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]
        assert args.num_requests_per_phase == 4
        assert args.target_slo_p95_ms == 150.0
        assert args.backend == "mock"

    @pytest.mark.asyncio
    async def test_run_benchmark_cli_step15_mock(self) -> None:
        """Verify end-to-end CLI execution with MockBackend returns exit code 0."""
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args(
                [
                    "--experiment-step15",
                    "--models",
                    "mock-model-1",
                    "--num-requests-per-phase",
                    "2",
                    "--backend",
                    "mock",
                    "--output",
                    tmpdir,
                ]
            )

            exit_code = await run_benchmark_cli(args)
            assert exit_code == 0
            assert (Path(tmpdir) / "summary.json").exists()


class TestStep15EngineVerification:
    """Targeted regression tests for Step 15 Real Hardware Engine Verification."""

    @staticmethod
    def _create_sample_condition(
        condition_name: str,
        integrity_valid: bool = True,
        scheduled: int = 16,
        completed: int = 16,
        failed: int = 0,
        measured: int = 16,
        duration: float = 2.5,
        mean_lat: float = 50.0,
        p95_lat: float = 80.0,
        queue_wait: float = 10.0,
    ) -> Step15ModelConditionSummary:
        return Step15ModelConditionSummary(
            condition_name=condition_name,
            condition_type=condition_name.lower(),
            model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            active_config_str="c=4,b=4,w=50.0ms",
            target_slo_p95_ms=180.0,
            scheduled_requests=scheduled,
            total_requests=scheduled,
            completed_requests=completed,
            failed_requests=failed,
            measured_requests=measured,
            backend_generate_calls=0,
            backend_generate_batch_calls=4,
            total_duration_sec=duration,
            overall_throughput_rps=scheduled / duration if duration > 0 else 0.0,
            output_tokens_per_sec=100.0,
            total_tokens_per_sec=200.0,
            mean_latency_ms=mean_lat,
            p50_latency_ms=45.0,
            p90_latency_ms=70.0,
            p95_latency_ms=p95_lat,
            p99_latency_ms=90.0,
            mean_queue_wait_ms=queue_wait,
            mean_backend_execution_ms=mean_lat - queue_wait,
            total_sla_violations=0,
            overall_sla_violation_rate_pct=0.0,
            phase_metrics=(),
            adaptation_events=(),
            total_adaptations=0,
            oscillation_count=0,
            engine_initialization_count=1,
            engine_teardown_count=1,
            engine_instance_id="vllm_engine_test",
            integrity_valid=integrity_valid,
        )

    @classmethod
    def _create_sample_conditions(
        cls, integrity_valid: bool = True
    ) -> dict[str, Step15ModelConditionSummary]:
        return {
            "STATIC_CONSERVATIVE": cls._create_sample_condition(
                "STATIC_CONSERVATIVE", integrity_valid=integrity_valid
            ),
            "STATIC_OPTIMIZED": cls._create_sample_condition(
                "STATIC_OPTIMIZED", integrity_valid=integrity_valid
            ),
            "SLA_AWARE_ADAPTIVE": cls._create_sample_condition(
                "SLA_AWARE_ADAPTIVE", integrity_valid=integrity_valid
            ),
        }

    def test_valid_real_vllm_evidence_verified_true(self) -> None:
        """Observable runtime facts from a real vLLM execution must verify as True."""
        result = Step15ModelExecutionResult(
            model_spec=ModelSpec(
                model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                model_family="smollm2",
                parameter_size_label="1.7B",
                status=ModelStatus.RUNNABLE,
            ),
            workload_hash="abc123hash",
            selected_optimal_config=TunableConfig(max_concurrency=4, max_batch_size=4),
            candidate_evaluations=(),
            conditions=self._create_sample_conditions(integrity_valid=True),
            engine_initialization_count=1,
            engine_teardown_count=1,
            backend_name="vllm",
            backend_confirmed=True,
            backend_generate_calls=0,
            backend_generate_batch_calls=48,
            execution_error=None,
            is_successful=True,
        )
        assert verify_step15_model_execution(result) is True

    def test_mock_backend_verified_false(self) -> None:
        """MockBackend execution must always verify as False."""
        result = Step15ModelExecutionResult(
            model_spec=ModelSpec(
                model_id="mock-model",
                model_family="custom",
                parameter_size_label="0.5B",
                status=ModelStatus.RUNNABLE,
            ),
            workload_hash="abc123hash",
            selected_optimal_config=TunableConfig(max_concurrency=2, max_batch_size=2),
            candidate_evaluations=(),
            conditions=self._create_sample_conditions(integrity_valid=True),
            engine_initialization_count=1,
            engine_teardown_count=1,
            backend_name="mock",
            backend_confirmed=False,
            backend_generate_calls=10,
            backend_generate_batch_calls=0,
            execution_error=None,
            is_successful=True,
        )
        assert verify_step15_model_execution(result) is False

    def test_vllm_initialization_failure_verified_false(self) -> None:
        """Failed engine initialization must verify as False."""
        result = Step15ModelExecutionResult(
            model_spec=ModelSpec(
                model_id="Qwen/Qwen2.5-1.5B-Instruct",
                model_family="qwen2.5",
                parameter_size_label="1.5B",
                status=ModelStatus.FAILED_TO_INITIALIZE,
                initialization_error="CUDA out of memory",
            ),
            workload_hash="abc123hash",
            selected_optimal_config=None,
            candidate_evaluations=(),
            conditions={},
            engine_initialization_count=0,
            engine_teardown_count=0,
            backend_name="vllm",
            backend_confirmed=False,
            backend_generate_calls=0,
            backend_generate_batch_calls=0,
            execution_error="CUDA out of memory",
            is_successful=False,
        )
        assert verify_step15_model_execution(result) is False

    def test_zero_backend_calls_verified_false(self) -> None:
        """Zero completed backend generate calls must verify as False."""
        result = Step15ModelExecutionResult(
            model_spec=ModelSpec(
                model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                model_family="smollm2",
                parameter_size_label="1.7B",
                status=ModelStatus.RUNNABLE,
            ),
            workload_hash="abc123hash",
            selected_optimal_config=TunableConfig(max_concurrency=4, max_batch_size=4),
            candidate_evaluations=(),
            conditions=self._create_sample_conditions(integrity_valid=True),
            engine_initialization_count=1,
            engine_teardown_count=1,
            backend_name="vllm",
            backend_confirmed=True,
            backend_generate_calls=0,
            backend_generate_batch_calls=0,
            execution_error=None,
            is_successful=True,
        )
        assert verify_step15_model_execution(result) is False

    def test_incomplete_request_accounting_verified_false(self) -> None:
        """Incomplete or dropped request accounting must verify as False."""
        invalid_conditions = self._create_sample_conditions(integrity_valid=False)
        invalid_conditions["STATIC_CONSERVATIVE"] = self._create_sample_condition(
            "STATIC_CONSERVATIVE",
            integrity_valid=False,
            scheduled=16,
            completed=14,
            failed=2,
            measured=14,
        )

        result = Step15ModelExecutionResult(
            model_spec=ModelSpec(
                model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                model_family="smollm2",
                parameter_size_label="1.7B",
                status=ModelStatus.RUNNABLE,
            ),
            workload_hash="abc123hash",
            selected_optimal_config=TunableConfig(max_concurrency=4, max_batch_size=4),
            candidate_evaluations=(),
            conditions=invalid_conditions,
            engine_initialization_count=1,
            engine_teardown_count=1,
            backend_name="vllm",
            backend_confirmed=True,
            backend_generate_calls=0,
            backend_generate_batch_calls=48,
            execution_error=None,
            is_successful=True,
        )
        assert verify_step15_model_execution(result) is False

    def test_authentication_failure_verified_false(self) -> None:
        """Gated HuggingFace model authentication failure must verify as False."""
        result = Step15ModelExecutionResult(
            model_spec=ModelSpec(
                model_id="meta-llama/Llama-3.2-1B-Instruct",
                model_family="llama3.2",
                parameter_size_label="1B",
                status=ModelStatus.AUTHENTICATION_REQUIRED,
                initialization_error="401 Client Error: Repository is gated",
            ),
            workload_hash="abc123hash",
            selected_optimal_config=None,
            candidate_evaluations=(),
            conditions={},
            engine_initialization_count=0,
            engine_teardown_count=0,
            backend_name="vllm",
            backend_confirmed=False,
            backend_generate_calls=0,
            backend_generate_batch_calls=0,
            execution_error="401 Client Error: Repository is gated",
            is_successful=False,
        )
        assert verify_step15_model_execution(result) is False

    def test_successful_real_batch_inference_verified_true(self) -> None:
        """Batch-only real vLLM inference satisfies verification."""
        result = Step15ModelExecutionResult(
            model_spec=ModelSpec(
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                model_family="qwen2.5",
                parameter_size_label="0.5B",
                status=ModelStatus.RUNNABLE,
            ),
            workload_hash="qwen_hash_1",
            selected_optimal_config=TunableConfig(max_concurrency=8, max_batch_size=8),
            candidate_evaluations=(),
            conditions=self._create_sample_conditions(integrity_valid=True),
            engine_initialization_count=1,
            engine_teardown_count=1,
            backend_name="vllm",
            backend_confirmed=True,
            backend_generate_calls=0,
            backend_generate_batch_calls=64,
            execution_error=None,
            is_successful=True,
        )
        assert verify_step15_model_execution(result) is True

    def test_cross_model_report_verification(self) -> None:
        """Cross-model report verification with runnable real vLLM and gated failure."""
        real_smollm_res = Step15ModelExecutionResult(
            model_spec=ModelSpec(
                model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                model_family="smollm2",
                parameter_size_label="1.7B",
                status=ModelStatus.RUNNABLE,
            ),
            workload_hash="hash_smollm",
            selected_optimal_config=TunableConfig(max_concurrency=4, max_batch_size=4),
            candidate_evaluations=(),
            conditions=self._create_sample_conditions(integrity_valid=True),
            engine_initialization_count=1,
            engine_teardown_count=1,
            backend_name="vllm",
            backend_confirmed=True,
            backend_generate_calls=0,
            backend_generate_batch_calls=48,
            execution_error=None,
            is_successful=True,
        )

        gated_llama_res = Step15ModelExecutionResult(
            model_spec=ModelSpec(
                model_id="meta-llama/Llama-3.2-1B-Instruct",
                model_family="llama3.2",
                parameter_size_label="1B",
                status=ModelStatus.AUTHENTICATION_REQUIRED,
                initialization_error="401 Client Error: Repository is gated",
            ),
            workload_hash="hash_llama",
            selected_optimal_config=None,
            candidate_evaluations=(),
            conditions={},
            engine_initialization_count=0,
            engine_teardown_count=0,
            backend_name="vllm",
            backend_confirmed=False,
            backend_generate_calls=0,
            backend_generate_batch_calls=0,
            execution_error="401 Client Error: Repository is gated",
            is_successful=False,
        )

        results_mixed = {
            "HuggingFaceTB/SmolLM2-1.7B-Instruct": real_smollm_res,
            "meta-llama/Llama-3.2-1B-Instruct": gated_llama_res,
        }
        # Mixed report has real verified execution for runnable model
        assert verify_step15_cross_model_report(results_mixed, backend_name="vllm") is True

        # Mock backend report returns False
        assert verify_step15_cross_model_report(results_mixed, backend_name="mock") is False

        # All failed models report returns False
        results_only_failed = {"meta-llama/Llama-3.2-1B-Instruct": gated_llama_res}
        assert verify_step15_cross_model_report(results_only_failed, backend_name="vllm") is False
