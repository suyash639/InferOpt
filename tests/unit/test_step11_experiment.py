"""Unit and mock integration tests for Step 11: Workload-Aware Batch Configuration Selection."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from inferopt import __version__ as INFEROPT_VERSION
from inferopt.benchmarks.cli import build_parser, run_benchmark_cli
from inferopt.benchmarks.generator import get_concurrent_4_workload
from inferopt.benchmarks.step11_experiment import (
    Step11CandidateResult,
    Step11ExperimentReport,
    Step11ExperimentRunner,
    Step11OptimizerDecision,
    Step11ValidationResult,
    classify_step11_findings,
    format_step11_report,
    vllm_condition_to_benchmark_result,
)
from inferopt.benchmarks.vllm_validation import (
    VLLMBenchmarkRequestRecord,
    VLLMCondition,
    VLLMConditionResult,
    VLLMEnvironmentMetadata,
    VLLMIntegrityResult,
    VLLMValidator,
    compute_workload_hash,
)
from inferopt.optimizer.models import (
    CandidateSpace,
    ObjectiveConfig,
    OptimizationObjectiveType,
    TunableConfig,
)


def _create_mock_condition_result(
    concurrency: int,
    max_batch_size: int,
    requests_per_sec: float,
    p95_latency_ms: float,
    condition: VLLMCondition = VLLMCondition.INFEROPT_BATCH_1,
    failed_requests: int = 0,
    repetitions: int = 3,
) -> VLLMConditionResult:
    """Create a realistic VLLMConditionResult for unit testing."""
    scenario = get_concurrent_4_workload(seed=42)
    req_records = tuple(
        VLLMBenchmarkRequestRecord(
            request_id=r.request_id,
            prompt_hash="mock_prompt_hash",
            output_hash="mock_output_hash",
            input_tokens=10,
            output_tokens=20,
            latency_ms=p95_latency_ms * 0.8,
            success=(failed_requests == 0),
        )
        for r in scenario.requests
    )
    total_reqs = len(scenario.requests)
    comp_reqs = total_reqs - failed_requests

    return VLLMConditionResult(
        condition=condition,
        condition_label=f"Mock (c={concurrency}, b={max_batch_size})",
        concurrency=concurrency,
        max_batch_size=max_batch_size,
        total_requests=total_reqs,
        completed_requests=comp_reqs,
        failed_requests=failed_requests,
        duration_sec=total_reqs / requests_per_sec if requests_per_sec > 0 else 1.0,
        requests_per_sec=requests_per_sec,
        output_tokens_per_sec=requests_per_sec * 20.0,
        total_tokens_per_sec=requests_per_sec * 30.0,
        mean_latency_ms=p95_latency_ms * 0.75,
        median_latency_ms=p95_latency_ms * 0.7,
        p50_latency_ms=p95_latency_ms * 0.7,
        p95_latency_ms=p95_latency_ms,
        p99_latency_ms=p95_latency_ms * 1.1,
        min_latency_ms=p95_latency_ms * 0.5,
        max_latency_ms=p95_latency_ms * 1.2,
        std_dev_latency_ms=p95_latency_ms * 0.1,
        avg_queue_wait_ms=5.0,
        avg_backend_execution_ms=p95_latency_ms * 0.7,
        total_input_tokens=comp_reqs * 10,
        total_output_tokens=comp_reqs * 20,
        total_tokens=comp_reqs * 30,
        avg_input_tokens_per_req=10.0,
        avg_output_tokens_per_req=20.0,
        min_output_tokens=20,
        max_output_tokens=20,
        total_batches=max(1, total_reqs // max_batch_size),
        avg_batch_size=float(min(max_batch_size, total_reqs)),
        median_batch_size=float(min(max_batch_size, total_reqs)),
        min_batch_size=1,
        max_batch_size_formed=min(max_batch_size, total_reqs),
        batch_size_distribution={
            min(max_batch_size, total_reqs): max(1, total_reqs // max_batch_size)
        },
        avg_batch_formation_wait_ms=2.0,
        avg_batch_execution_ms=p95_latency_ms * 0.7,
        repetition_count=repetitions,
        requests=req_records,
    )


class TestStep11ModelsAndSerialization:
    """Unit tests for Step 11 domain models and JSON roundtrips."""

    def test_candidate_result_creation_and_immutability(self) -> None:
        cfg = TunableConfig(max_concurrency=4, max_batch_size=2, batch_wait_ms=50.0)
        res = Step11CandidateResult(
            config=cfg,
            concurrency=4,
            max_batch_size=2,
            batch_wait_ms=50.0,
            repetitions=3,
            requests_per_sec=12.5,
            output_tokens_per_sec=250.0,
            total_tokens_per_sec=350.0,
            mean_latency_ms=80.0,
            median_latency_ms=75.0,
            p50_latency_ms=75.0,
            p95_latency_ms=95.0,
            p99_latency_ms=110.0,
            std_dev_latency_ms=8.5,
            avg_queue_wait_ms=10.0,
            avg_backend_execution_ms=70.0,
            total_batches=8,
            avg_batch_size=2.0,
            max_batch_size_formed=2,
            completed_requests=16,
            failed_requests=0,
            integrity_valid=True,
        )
        assert res.concurrency == 4
        assert res.max_batch_size == 2
        assert res.requests_per_sec == 12.5
        assert res.integrity_valid is True

        with pytest.raises(ValidationError):
            res.requests_per_sec = 20.0

    def test_vllm_condition_to_benchmark_result(self) -> None:
        scenario = get_concurrent_4_workload(seed=42)
        cond = _create_mock_condition_result(
            concurrency=8,
            max_batch_size=4,
            requests_per_sec=15.0,
            p95_latency_ms=120.0,
        )
        bench = vllm_condition_to_benchmark_result(
            cond=cond,
            scenario=scenario,
            batch_wait_ms=50.0,
        )
        assert bench.requests_per_sec == 15.0
        assert bench.p95_latency_ms == 120.0
        assert bench.scheduler_config.max_concurrency == 8
        assert bench.scheduler_config.batch_config.max_batch_size == 4
        assert bench.scheduler_config.batch_config.batch_wait_ms == 50.0
        assert bench.completed_requests == len(scenario.requests)

    def test_experiment_report_json_roundtrip(self) -> None:
        scenario = get_concurrent_4_workload(seed=42)
        workload_hash = compute_workload_hash(scenario)

        env = VLLMEnvironmentMetadata(
            os_name="Linux",
            os_version="5.15",
            cpu_architecture="x86_64",
            python_version="3.12.0",
            vllm_version="0.29.0",
            torch_version="2.4.0",
            cuda_version="12.4",
            gpu_name="Tesla T4",
            gpu_count=1,
            inferopt_version=INFEROPT_VERSION,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            enforce_eager=True,
            warmup_count=2,
            repetitions=3,
            workload_seed=42,
            workload_hash=workload_hash,
        )

        base_res = _create_mock_condition_result(
            concurrency=1,
            max_batch_size=1,
            requests_per_sec=2.0,
            p95_latency_ms=500.0,
            condition=VLLMCondition.DIRECT_VLLM,
        )

        cfg_cand = TunableConfig(max_concurrency=4, max_batch_size=4, batch_wait_ms=50.0)
        cand_res = Step11CandidateResult(
            config=cfg_cand,
            concurrency=4,
            max_batch_size=4,
            batch_wait_ms=50.0,
            repetitions=3,
            requests_per_sec=8.0,
            output_tokens_per_sec=160.0,
            total_tokens_per_sec=240.0,
            mean_latency_ms=150.0,
            median_latency_ms=140.0,
            p50_latency_ms=140.0,
            p95_latency_ms=180.0,
            p99_latency_ms=200.0,
            std_dev_latency_ms=12.0,
            avg_queue_wait_ms=10.0,
            avg_backend_execution_ms=140.0,
            total_batches=4,
            avg_batch_size=4.0,
            max_batch_size_formed=4,
            completed_requests=16,
            failed_requests=0,
            integrity_valid=True,
        )

        obj_cfg = ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT)
        decision = Step11OptimizerDecision(
            objective_type=OptimizationObjectiveType.THROUGHPUT,
            objective_label="THROUGHPUT",
            objective_config=obj_cfg,
            selected_config=cfg_cand,
            predicted_score=8.0,
            explanation="Maximum throughput achieved",
            exploration_metrics=cand_res,
            total_evaluated=1,
            feasible_evaluated=1,
        )

        validation = Step11ValidationResult(
            objective_type=OptimizationObjectiveType.THROUGHPUT,
            objective_label="THROUGHPUT",
            config=cfg_cand,
            exploration_score=8.0,
            validation_score=7.95,
            score_delta_pct=-0.62,
            exploration_throughput=8.0,
            validation_throughput=7.95,
            throughput_delta_pct=-0.62,
            exploration_p95_latency_ms=180.0,
            validation_p95_latency_ms=182.0,
            p95_latency_delta_pct=1.11,
            validation_p99_latency_ms=205.0,
            validation_std_dev_ms=13.0,
            exploration_repetitions=3,
            validation_repetitions=3,
            integrity_valid=True,
        )

        integrity = VLLMIntegrityResult(
            is_valid=True,
            total_expected_requests=16,
            total_completed_requests=16,
        )

        report = Step11ExperimentReport(
            experiment_id="test-exp-001",
            timestamp=1700000000.0,
            scenario_name=scenario.scenario_name,
            workload_hash=workload_hash,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            environment=env,
            candidate_space=CandidateSpace(
                concurrencies=(4,), batch_sizes=(4,), batch_waits_ms=(50.0,)
            ),
            baseline_concurrency=1,
            baseline_result=base_res,
            exploration_results=(cand_res,),
            optimizer_decisions={"THROUGHPUT": decision},
            validation_results=(validation,),
            integrity=integrity,
            findings={"PROVEN": ("Test proven claim",)},
        )

        json_str = report.to_json()
        loaded = Step11ExperimentReport.from_json(json_str)
        assert loaded.experiment_id == report.experiment_id
        assert loaded.workload_hash == report.workload_hash
        assert loaded.baseline_result.requests_per_sec == base_res.requests_per_sec
        assert len(loaded.exploration_results) == 1
        assert loaded.optimizer_decisions["THROUGHPUT"].predicted_score == 8.0
        assert len(loaded.validation_results) == 1
        assert loaded.validation_results[0].validation_throughput == 7.95

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "report.json"
            report.save_json(file_path)
            loaded_from_disk = Step11ExperimentReport.load_json(file_path)
            assert loaded_from_disk.experiment_id == report.experiment_id


class TestStep11FindingsAndFormatting:
    """Unit tests for empirical findings classification and report formatting."""

    def test_classify_step11_findings(self) -> None:
        base = _create_mock_condition_result(1, 1, 1.5, 500.0)
        cand1 = Step11CandidateResult(
            config=TunableConfig(max_concurrency=4, max_batch_size=4, batch_wait_ms=50.0),
            concurrency=4,
            max_batch_size=4,
            batch_wait_ms=50.0,
            repetitions=3,
            requests_per_sec=8.5,
            output_tokens_per_sec=170.0,
            total_tokens_per_sec=250.0,
            mean_latency_ms=120.0,
            median_latency_ms=110.0,
            p50_latency_ms=110.0,
            p95_latency_ms=140.0,
            p99_latency_ms=160.0,
            std_dev_latency_ms=10.0,
            avg_queue_wait_ms=5.0,
            avg_backend_execution_ms=115.0,
            total_batches=4,
            avg_batch_size=4.0,
            max_batch_size_formed=4,
            completed_requests=16,
            failed_requests=0,
            integrity_valid=True,
        )
        cand2 = Step11CandidateResult(
            config=TunableConfig(max_concurrency=1, max_batch_size=1, batch_wait_ms=50.0),
            concurrency=1,
            max_batch_size=1,
            batch_wait_ms=50.0,
            repetitions=3,
            requests_per_sec=2.0,
            output_tokens_per_sec=40.0,
            total_tokens_per_sec=60.0,
            mean_latency_ms=50.0,
            median_latency_ms=45.0,
            p50_latency_ms=45.0,
            p95_latency_ms=55.0,
            p99_latency_ms=60.0,
            std_dev_latency_ms=4.0,
            avg_queue_wait_ms=0.0,
            avg_backend_execution_ms=50.0,
            total_batches=16,
            avg_batch_size=1.0,
            max_batch_size_formed=1,
            completed_requests=16,
            failed_requests=0,
            integrity_valid=True,
        )

        decisions = {
            "THROUGHPUT": Step11OptimizerDecision(
                objective_type=OptimizationObjectiveType.THROUGHPUT,
                objective_label="THROUGHPUT",
                objective_config=ObjectiveConfig(
                    objective_type=OptimizationObjectiveType.THROUGHPUT
                ),
                selected_config=cand1.config,
                predicted_score=8.5,
                explanation="Highest throughput",
                exploration_metrics=cand1,
                total_evaluated=2,
                feasible_evaluated=2,
            ),
            "LATENCY": Step11OptimizerDecision(
                objective_type=OptimizationObjectiveType.LATENCY,
                objective_label="LATENCY",
                objective_config=ObjectiveConfig(objective_type=OptimizationObjectiveType.LATENCY),
                selected_config=cand2.config,
                predicted_score=-55.0,
                explanation="Lowest p95 latency",
                exploration_metrics=cand2,
                total_evaluated=2,
                feasible_evaluated=2,
            ),
        }

        validations = [
            Step11ValidationResult(
                objective_type=OptimizationObjectiveType.THROUGHPUT,
                objective_label="THROUGHPUT",
                config=cand1.config,
                exploration_score=8.5,
                validation_score=8.4,
                score_delta_pct=-1.18,
                exploration_throughput=8.5,
                validation_throughput=8.4,
                throughput_delta_pct=-1.18,
                exploration_p95_latency_ms=140.0,
                validation_p95_latency_ms=142.0,
                p95_latency_delta_pct=1.43,
                validation_p99_latency_ms=165.0,
                validation_std_dev_ms=11.0,
                exploration_repetitions=3,
                validation_repetitions=3,
                integrity_valid=True,
            ),
            Step11ValidationResult(
                objective_type=OptimizationObjectiveType.LATENCY,
                objective_label="LATENCY",
                config=cand2.config,
                exploration_score=-55.0,
                validation_score=-54.0,
                score_delta_pct=-1.82,
                exploration_throughput=2.0,
                validation_throughput=2.05,
                throughput_delta_pct=2.5,
                exploration_p95_latency_ms=55.0,
                validation_p95_latency_ms=54.0,
                p95_latency_delta_pct=-1.82,
                validation_p99_latency_ms=59.0,
                validation_std_dev_ms=3.8,
                exploration_repetitions=3,
                validation_repetitions=3,
                integrity_valid=True,
            ),
        ]

        findings = classify_step11_findings(
            baseline=base,
            candidates=[cand1, cand2],
            decisions=decisions,
            validations=validations,
        )

        assert "PROVEN" in findings
        assert "SUGGESTED" in findings
        assert "NOT PROVEN" in findings
        assert len(findings["PROVEN"]) >= 4
        assert any("distinct runtime operating points" in s for s in findings["SUGGESTED"])
        assert any("throughput improvement" in s for s in findings["SUGGESTED"])

    def test_format_step11_report(self) -> None:
        scenario = get_concurrent_4_workload(seed=42)
        workload_hash = compute_workload_hash(scenario)

        env = VLLMEnvironmentMetadata(
            os_name="Linux",
            os_version="5.15",
            cpu_architecture="x86_64",
            python_version="3.12.0",
            vllm_version="0.29.0",
            torch_version="2.4.0",
            cuda_version="12.4",
            gpu_name="Tesla T4",
            gpu_count=1,
            inferopt_version=INFEROPT_VERSION,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            enforce_eager=True,
            warmup_count=2,
            repetitions=3,
            workload_seed=42,
            workload_hash=workload_hash,
        )

        base_res = _create_mock_condition_result(
            1, 1, 1.2, 800.0, condition=VLLMCondition.DIRECT_VLLM
        )
        cfg_cand = TunableConfig(max_concurrency=4, max_batch_size=4, batch_wait_ms=50.0)
        cand_res = Step11CandidateResult(
            config=cfg_cand,
            concurrency=4,
            max_batch_size=4,
            batch_wait_ms=50.0,
            repetitions=3,
            requests_per_sec=8.0,
            output_tokens_per_sec=160.0,
            total_tokens_per_sec=240.0,
            mean_latency_ms=150.0,
            median_latency_ms=140.0,
            p50_latency_ms=140.0,
            p95_latency_ms=180.0,
            p99_latency_ms=200.0,
            std_dev_latency_ms=12.0,
            avg_queue_wait_ms=10.0,
            avg_backend_execution_ms=140.0,
            total_batches=4,
            avg_batch_size=4.0,
            max_batch_size_formed=4,
            completed_requests=16,
            failed_requests=0,
            integrity_valid=True,
        )

        obj_cfg = ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT)
        decision = Step11OptimizerDecision(
            objective_type=OptimizationObjectiveType.THROUGHPUT,
            objective_label="THROUGHPUT",
            objective_config=obj_cfg,
            selected_config=cfg_cand,
            predicted_score=8.0,
            explanation="Maximum throughput achieved",
            exploration_metrics=cand_res,
            total_evaluated=1,
            feasible_evaluated=1,
        )

        validation = Step11ValidationResult(
            objective_type=OptimizationObjectiveType.THROUGHPUT,
            objective_label="THROUGHPUT",
            config=cfg_cand,
            exploration_score=8.0,
            validation_score=7.95,
            score_delta_pct=-0.62,
            exploration_throughput=8.0,
            validation_throughput=7.95,
            throughput_delta_pct=-0.62,
            exploration_p95_latency_ms=180.0,
            validation_p95_latency_ms=182.0,
            p95_latency_delta_pct=1.11,
            validation_p99_latency_ms=205.0,
            validation_std_dev_ms=13.0,
            exploration_repetitions=3,
            validation_repetitions=3,
            integrity_valid=True,
        )

        report = Step11ExperimentReport(
            experiment_id="test-report-formatting",
            timestamp=1700000000.0,
            scenario_name=scenario.scenario_name,
            workload_hash=workload_hash,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            environment=env,
            candidate_space=CandidateSpace(
                concurrencies=(4,), batch_sizes=(4,), batch_waits_ms=(50.0,)
            ),
            baseline_concurrency=1,
            baseline_result=base_res,
            exploration_results=(cand_res,),
            optimizer_decisions={"THROUGHPUT": decision},
            validation_results=(validation,),
            integrity=VLLMIntegrityResult(
                is_valid=True, total_expected_requests=16, total_completed_requests=16
            ),
            findings={
                "PROVEN": ("Test proven fact",),
                "SUGGESTED": ("Test suggested hypothesis",),
                "NOT PROVEN": ("Test unproven claim",),
            },
        )

        text = format_step11_report(report)
        assert "INFEROPT STEP 11: WORKLOAD-AWARE BATCH CONFIGURATION SELECTION" in text
        assert "Experiment ID:        test-report-formatting" in text
        assert "Integrity Gate:       PASSED (100% VALID)" in text
        assert "CANDIDATE CONFIGURATION SPACE EXPLORATION (Phase 1):" in text
        assert "OPTIMIZER DECISIONS (Phase 2):" in text
        assert "FINAL INDEPENDENT VALIDATION (Phase 3):" in text
        assert "EMPIRICAL FINDINGS CLASSIFICATION:" in text
        assert "[PROVEN]" in text
        assert "[SUGGESTED]" in text
        assert "[NOT PROVEN]" in text


class TestStep11ExperimentRunnerMocked:
    """Mock integration tests for Step11ExperimentRunner orchestration."""

    @pytest.mark.asyncio
    async def test_full_step11_experiment_run(self) -> None:
        scenario = get_concurrent_4_workload(seed=42)
        mock_validator = MagicMock(spec=VLLMValidator)
        mock_validator.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        mock_validator.config = MagicMock()
        mock_validator.config.enforce_eager = True

        # Map candidate configs to synthetic realistic metrics
        async def mock_run_direct(
            scenario: Any, concurrency: int, warmup_count: int, repetitions: int
        ) -> VLLMConditionResult:
            return _create_mock_condition_result(
                concurrency=concurrency,
                max_batch_size=1,
                requests_per_sec=1.2,
                p95_latency_ms=800.0,
                condition=VLLMCondition.DIRECT_VLLM,
                repetitions=repetitions,
            )

        async def mock_run_inferopt(
            scenario: Any,
            concurrency: int,
            max_batch_size: int,
            condition: Any,
            warmup_count: int,
            repetitions: int,
            batch_wait_ms: float,
        ) -> VLLMConditionResult:
            # Batching improves throughput with batch size
            # Concurrency 1, max_batch 1 -> lowest latency (50ms), throughput 2.0
            # Concurrency 8, max_batch 8 -> highest throughput (10.0), latency 200ms
            base_rps = 1.0 + (max_batch_size * 1.0) + (concurrency * 0.2)
            p95_lat = 50.0 + (max_batch_size * 15.0) + (concurrency * 5.0)
            return _create_mock_condition_result(
                concurrency=concurrency,
                max_batch_size=max_batch_size,
                requests_per_sec=base_rps,
                p95_latency_ms=p95_lat,
                repetitions=repetitions,
            )

        mock_validator.run_condition_direct = AsyncMock(side_effect=mock_run_direct)
        mock_validator.run_condition_inferopt = AsyncMock(side_effect=mock_run_inferopt)

        runner = Step11ExperimentRunner(validator=mock_validator)

        # Candidate space: concurrencies=(1, 4), batch_sizes=(1, 4), batch_waits_ms=(50.0,)
        # Produces 4 candidates
        space = CandidateSpace(
            concurrencies=(1, 4),
            batch_sizes=(1, 4),
            batch_waits_ms=(50.0,),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report = await runner.run_experiment(
                scenario=scenario,
                candidate_space=space,
                warmup_count=1,
                exploration_repetitions=2,
                validation_repetitions=2,
                output_dir=tmpdir,
            )

            # Baseline + 4 exploration runs + independent validation runs
            assert mock_validator.run_condition_direct.await_count == 1
            # 4 exploration candidates + at least 1 validation run
            assert mock_validator.run_condition_inferopt.await_count >= 5

            assert len(report.exploration_results) == 4
            assert len(report.optimizer_decisions) == 3
            assert "THROUGHPUT" in report.optimizer_decisions
            assert "LATENCY" in report.optimizer_decisions
            assert "BALANCED" in report.optimizer_decisions

            # Verify that THROUGHPUT selected highest batch size candidate
            tput_winner = report.optimizer_decisions["THROUGHPUT"].selected_config
            assert tput_winner.max_batch_size == 4

            # Verify that LATENCY selected lowest latency candidate (c=1, b=1)
            lat_winner = report.optimizer_decisions["LATENCY"].selected_config
            assert lat_winner.max_concurrency == 1
            assert lat_winner.max_batch_size == 1

            # Verify validation results exist and are populated
            assert len(report.validation_results) == 3
            assert report.integrity.is_valid is True

            # Verify report was saved to output directory
            saved_files = list(Path(tmpdir).glob("step11_*.json"))
            assert len(saved_files) == 1

    @pytest.mark.asyncio
    async def test_validation_deduplication_when_same_config_wins_all_objectives(self) -> None:
        scenario = get_concurrent_4_workload(seed=42)
        mock_validator = MagicMock(spec=VLLMValidator)
        mock_validator.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        mock_validator.config = MagicMock()
        mock_validator.config.enforce_eager = True

        async def mock_run_direct(
            scenario: Any, concurrency: int, warmup_count: int, repetitions: int
        ) -> VLLMConditionResult:
            return _create_mock_condition_result(
                concurrency=concurrency,
                max_batch_size=1,
                requests_per_sec=1.0,
                p95_latency_ms=100.0,
                condition=VLLMCondition.DIRECT_VLLM,
                repetitions=repetitions,
            )

        # In this mock, (c=2, b=2) has highest throughput AND lowest latency
        async def mock_run_inferopt(
            scenario: Any,
            concurrency: int,
            max_batch_size: int,
            condition: Any,
            warmup_count: int,
            repetitions: int,
            batch_wait_ms: float,
        ) -> VLLMConditionResult:
            if concurrency == 2 and max_batch_size == 2:
                rps = 20.0
                lat = 10.0
            else:
                rps = 5.0
                lat = 50.0
            return _create_mock_condition_result(
                concurrency=concurrency,
                max_batch_size=max_batch_size,
                requests_per_sec=rps,
                p95_latency_ms=lat,
                repetitions=repetitions,
            )

        mock_validator.run_condition_direct = AsyncMock(side_effect=mock_run_direct)
        mock_validator.run_condition_inferopt = AsyncMock(side_effect=mock_run_inferopt)

        runner = Step11ExperimentRunner(validator=mock_validator)
        space = CandidateSpace(
            concurrencies=(1, 2),
            batch_sizes=(1, 2),
            batch_waits_ms=(50.0,),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report = await runner.run_experiment(
                scenario=scenario,
                candidate_space=space,
                warmup_count=1,
                exploration_repetitions=2,
                validation_repetitions=2,
                output_dir=tmpdir,
            )
            # 4 exploration runs + exactly 1 validation run (since same config won all 3 objectives)
            assert mock_validator.run_condition_inferopt.await_count == 5
            # But 3 validation result entries (one per objective)
            assert len(report.validation_results) == 3
            assert all(
                v.config.max_concurrency == 2 and v.config.max_batch_size == 2
                for v in report.validation_results
            )

    @pytest.mark.asyncio
    async def test_optimization_with_constraints_and_infeasibility(self) -> None:
        scenario = get_concurrent_4_workload(seed=42)
        mock_validator = MagicMock(spec=VLLMValidator)
        mock_validator.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        mock_validator.config = MagicMock()
        mock_validator.config.enforce_eager = True

        async def mock_run_direct(
            scenario: Any, concurrency: int, warmup_count: int, repetitions: int
        ) -> VLLMConditionResult:
            return _create_mock_condition_result(
                concurrency=concurrency,
                max_batch_size=1,
                requests_per_sec=1.0,
                p95_latency_ms=100.0,
                condition=VLLMCondition.DIRECT_VLLM,
                repetitions=repetitions,
            )

        async def mock_run_inferopt(
            scenario: Any,
            concurrency: int,
            max_batch_size: int,
            condition: Any,
            warmup_count: int,
            repetitions: int,
            batch_wait_ms: float,
        ) -> VLLMConditionResult:
            return _create_mock_condition_result(
                concurrency=concurrency,
                max_batch_size=max_batch_size,
                requests_per_sec=5.0,
                p95_latency_ms=200.0,
                repetitions=repetitions,
            )

        mock_validator.run_condition_direct = AsyncMock(side_effect=mock_run_direct)
        mock_validator.run_condition_inferopt = AsyncMock(side_effect=mock_run_inferopt)

        runner = Step11ExperimentRunner(validator=mock_validator)
        space = CandidateSpace(
            concurrencies=(1,),
            batch_sizes=(1,),
            batch_waits_ms=(50.0,),
        )

        from inferopt.optimizer.models import OptimizationConstraints

        # Require p95 latency <= 50ms (impossible for 200ms candidate)
        strict_constraints = OptimizationConstraints(max_p95_latency_ms=50.0)

        with pytest.raises(RuntimeError, match="Optimizer failed to find a feasible configuration"):
            await runner.run_experiment(
                scenario=scenario,
                candidate_space=space,
                constraints=strict_constraints,
                output_dir=None,
            )


class TestStep11CLIExtension:
    """Unit tests for Step 11 CLI argument parsing and dispatch."""

    def test_cli_parser_step11_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--experiment-step11",
                "--concurrency",
                "1",
                "4",
                "8",
                "--batch-sizes",
                "1",
                "2",
                "4",
                "8",
                "--batch-wait-ms",
                "50.0",
                "--warmup",
                "2",
                "--repetitions",
                "3",
                "--val-repetitions",
                "4",
                "--enforce-eager",
            ]
        )
        assert args.experiment_step11 is True
        assert args.concurrency == [1, 4, 8]
        assert args.batch_sizes == [1, 2, 4, 8]
        assert args.batch_wait_ms == 50.0
        assert args.warmup == 2
        assert args.repetitions == 3
        assert args.val_repetitions == 4
        assert args.enforce_eager is True

    @pytest.mark.asyncio
    async def test_run_benchmark_cli_step11_dispatch(self) -> None:
        mock_report = MagicMock()
        mock_report.integrity.is_valid = True

        mock_runner = MagicMock()
        mock_runner.run_experiment = AsyncMock(return_value=mock_report)

        with (
            patch(
                "inferopt.benchmarks.step11_experiment.Step11ExperimentRunner",
                return_value=mock_runner,
            ),
            patch(
                "inferopt.benchmarks.step11_experiment.format_step11_report",
                return_value="Step 11 Mock Report",
            ),
        ):
            parser = build_parser()
            args = parser.parse_args(
                [
                    "--experiment-step11",
                    "--scenario",
                    "concurrent_4",
                    "--num-requests",
                    "8",
                    "--concurrency",
                    "1",
                    "4",
                    "--batch-sizes",
                    "1",
                    "2",
                    "--warmup",
                    "1",
                    "--repetitions",
                    "2",
                    "--val-repetitions",
                    "2",
                ]
            )
            rc = await run_benchmark_cli(args)
            assert rc == 0
            assert mock_runner.run_experiment.await_count == 1
