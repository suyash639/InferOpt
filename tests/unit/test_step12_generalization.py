"""Unit and mock integration tests for Step 12: Multi-Workload Generalization Experiment."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inferopt import __version__ as INFEROPT_VERSION
from inferopt.benchmarks.cli import build_parser, run_benchmark_cli
from inferopt.benchmarks.generator import get_concurrent_4_workload
from inferopt.benchmarks.models import ArrivalPattern
from inferopt.benchmarks.step11_experiment import (
    Step11CandidateResult,
    Step11ExperimentReport,
    Step11OptimizerDecision,
    Step11ValidationResult,
)
from inferopt.benchmarks.step12_generalization import (
    CrossWorkloadSummaryRow,
    Step12GeneralizationReport,
    Step12GeneralizationRunner,
    Step12WorkloadReport,
    classify_step12_findings,
    compute_cross_workload_summary,
    format_step12_report,
    get_step12_workload_matrix,
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


def _create_mock_vllm_condition_result(
    concurrency: int,
    max_batch_size: int,
    requests_per_sec: float,
    p95_latency_ms: float,
    condition: VLLMCondition = VLLMCondition.INFEROPT_BATCH_1,
    failed_requests: int = 0,
    repetitions: int = 3,
    scenario: Any = None,
) -> VLLMConditionResult:
    """Construct realistic VLLMConditionResult for unit testing."""
    sc = scenario or get_concurrent_4_workload(seed=42)
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
        for r in sc.requests
    )
    total_reqs = len(sc.requests)
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


def _create_mock_step11_report(
    scenario_name: str,
    base_rps: float,
    base_p95: float,
    tput_cfg: TunableConfig,
    tput_rps: float,
    lat_cfg: TunableConfig,
    lat_p95: float,
    bal_cfg: TunableConfig,
    bal_rps: float,
    bal_p95: float,
    score_delta_pct: float = 1.5,
) -> Step11ExperimentReport:
    """Create a mock Step11ExperimentReport for generalization testing."""
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
        workload_hash="mock_hash_" + scenario_name,
    )
    base_res = _create_mock_vllm_condition_result(
        concurrency=1,
        max_batch_size=1,
        requests_per_sec=base_rps,
        p95_latency_ms=base_p95,
        condition=VLLMCondition.DIRECT_VLLM,
    )
    cand_tput = Step11CandidateResult(
        config=tput_cfg,
        concurrency=tput_cfg.max_concurrency,
        max_batch_size=tput_cfg.max_batch_size,
        batch_wait_ms=tput_cfg.batch_wait_ms,
        repetitions=3,
        requests_per_sec=tput_rps,
        output_tokens_per_sec=tput_rps * 20.0,
        total_tokens_per_sec=tput_rps * 30.0,
        mean_latency_ms=100.0,
        median_latency_ms=90.0,
        p50_latency_ms=90.0,
        p95_latency_ms=120.0,
        p99_latency_ms=140.0,
        std_dev_latency_ms=10.0,
        avg_queue_wait_ms=5.0,
        avg_backend_execution_ms=95.0,
        total_batches=4,
        avg_batch_size=float(tput_cfg.max_batch_size),
        max_batch_size_formed=tput_cfg.max_batch_size,
        completed_requests=16,
        failed_requests=0,
        integrity_valid=True,
    )
    cand_lat = Step11CandidateResult(
        config=lat_cfg,
        concurrency=lat_cfg.max_concurrency,
        max_batch_size=lat_cfg.max_batch_size,
        batch_wait_ms=lat_cfg.batch_wait_ms,
        repetitions=3,
        requests_per_sec=2.0,
        output_tokens_per_sec=40.0,
        total_tokens_per_sec=60.0,
        mean_latency_ms=lat_p95 * 0.8,
        median_latency_ms=lat_p95 * 0.75,
        p50_latency_ms=lat_p95 * 0.75,
        p95_latency_ms=lat_p95,
        p99_latency_ms=lat_p95 * 1.1,
        std_dev_latency_ms=5.0,
        avg_queue_wait_ms=1.0,
        avg_backend_execution_ms=lat_p95 * 0.8,
        total_batches=16,
        avg_batch_size=1.0,
        max_batch_size_formed=1,
        completed_requests=16,
        failed_requests=0,
        integrity_valid=True,
    )
    cand_bal = Step11CandidateResult(
        config=bal_cfg,
        concurrency=bal_cfg.max_concurrency,
        max_batch_size=bal_cfg.max_batch_size,
        batch_wait_ms=bal_cfg.batch_wait_ms,
        repetitions=3,
        requests_per_sec=bal_rps,
        output_tokens_per_sec=bal_rps * 20.0,
        total_tokens_per_sec=bal_rps * 30.0,
        mean_latency_ms=bal_p95 * 0.8,
        median_latency_ms=bal_p95 * 0.75,
        p50_latency_ms=bal_p95 * 0.75,
        p95_latency_ms=bal_p95,
        p99_latency_ms=bal_p95 * 1.1,
        std_dev_latency_ms=8.0,
        avg_queue_wait_ms=4.0,
        avg_backend_execution_ms=bal_p95 * 0.75,
        total_batches=8,
        avg_batch_size=float(bal_cfg.max_batch_size),
        max_batch_size_formed=bal_cfg.max_batch_size,
        completed_requests=16,
        failed_requests=0,
        integrity_valid=True,
    )

    decisions = {
        "THROUGHPUT": Step11OptimizerDecision(
            objective_type=OptimizationObjectiveType.THROUGHPUT,
            objective_label="THROUGHPUT",
            objective_config=ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT),
            selected_config=tput_cfg,
            predicted_score=tput_rps,
            explanation="Throughput winner",
            exploration_metrics=cand_tput,
            total_evaluated=3,
            feasible_evaluated=3,
        ),
        "LATENCY": Step11OptimizerDecision(
            objective_type=OptimizationObjectiveType.LATENCY,
            objective_label="LATENCY",
            objective_config=ObjectiveConfig(objective_type=OptimizationObjectiveType.LATENCY),
            selected_config=lat_cfg,
            predicted_score=-lat_p95,
            explanation="Latency winner",
            exploration_metrics=cand_lat,
            total_evaluated=3,
            feasible_evaluated=3,
        ),
        "BALANCED": Step11OptimizerDecision(
            objective_type=OptimizationObjectiveType.BALANCED,
            objective_label="BALANCED",
            objective_config=ObjectiveConfig(objective_type=OptimizationObjectiveType.BALANCED),
            selected_config=bal_cfg,
            predicted_score=0.75,
            explanation="Balanced winner",
            exploration_metrics=cand_bal,
            total_evaluated=3,
            feasible_evaluated=3,
        ),
    }

    val_bal = Step11ValidationResult(
        objective_type=OptimizationObjectiveType.BALANCED,
        objective_label="BALANCED",
        config=bal_cfg,
        exploration_score=0.75,
        validation_score=0.75 * (1.0 + (score_delta_pct / 100.0)),
        score_delta_pct=score_delta_pct,
        exploration_throughput=bal_rps,
        validation_throughput=bal_rps * 0.98,
        throughput_delta_pct=-2.0,
        exploration_p95_latency_ms=bal_p95,
        validation_p95_latency_ms=bal_p95 * 1.02,
        p95_latency_delta_pct=2.0,
        validation_p99_latency_ms=bal_p95 * 1.15,
        validation_std_dev_ms=6.0,
        exploration_repetitions=3,
        validation_repetitions=3,
        integrity_valid=True,
    )

    return Step11ExperimentReport(
        experiment_id="mock-exp-" + scenario_name,
        timestamp=1700000000.0,
        scenario_name=scenario_name,
        workload_hash="mock_hash_" + scenario_name,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        environment=env,
        candidate_space=CandidateSpace(
            concurrencies=(1, 4, 8), batch_sizes=(1, 2, 4, 8), batch_waits_ms=(50.0,)
        ),
        baseline_concurrency=1,
        baseline_result=base_res,
        exploration_results=(cand_tput, cand_lat, cand_bal),
        optimizer_decisions=decisions,
        validation_results=(val_bal,),
        integrity=VLLMIntegrityResult(
            is_valid=True, total_expected_requests=16, total_completed_requests=16
        ),
        findings={"PROVEN": ("Test proven fact",)},
    )


class TestStep12WorkloadMatrix:
    """Unit tests for the 4-class workload matrix construction and hashing."""

    def test_get_step12_workload_matrix_structure_and_classes(self) -> None:
        matrix = get_step12_workload_matrix(seed=42, num_requests=16)
        assert set(matrix.keys()) == {"A_LIGHT", "B_BURSTY", "C_SATURATED", "D_MIXED"}

        # Class A: Light / Low-concurrency
        sc_a = matrix["A_LIGHT"]
        assert sc_a.config.concurrency == 1
        assert len(sc_a.requests) == 16

        # Class B: Moderate / Bursty
        sc_b = matrix["B_BURSTY"]
        assert sc_b.config.arrival_pattern == ArrivalPattern.BURST
        assert len(sc_b.requests) == 16

        # Class C: High-concurrency / Saturated
        sc_c = matrix["C_SATURATED"]
        assert sc_c.config.concurrency == 8
        assert len(sc_c.requests) == 16

        # Class D: Mixed / Variable
        sc_d = matrix["D_MIXED"]
        assert len(sc_d.config.priority_levels) > 1
        assert len(sc_d.requests) == 16

        # Distinct workload hashes
        hashes = {compute_workload_hash(sc) for sc in matrix.values()}
        assert len(hashes) == 4

    def test_workload_matrix_custom_seed_and_request_count(self) -> None:
        matrix_10 = get_step12_workload_matrix(seed=99, num_requests=10)
        assert all(len(sc.requests) == 10 for sc in matrix_10.values())
        assert matrix_10["A_LIGHT"].config.seed == 99


class TestStep12ModelsAndSerialization:
    """Unit tests for Step 12 domain models, summary logic, and JSON roundtrip."""

    def test_cross_workload_summary_row_and_summary_creation(self) -> None:
        row = CrossWorkloadSummaryRow(
            workload_key="A_LIGHT",
            workload_name="light_low_concurrency",
            arrival_pattern="CONCURRENT",
            baseline_rps=1.5,
            baseline_p95_ms=500.0,
            tput_winner_config="c=4, b=4",
            tput_winner_rps=6.0,
            lat_winner_config="c=1, b=1",
            lat_winner_p95_ms=60.0,
            balanced_winner_config="c=4, b=2",
            balanced_winner_rps=5.0,
            balanced_winner_p95_ms=80.0,
            balanced_val_delta_pct=1.2,
            is_stable=True,
            integrity_valid=True,
        )
        assert row.workload_key == "A_LIGHT"
        assert row.is_stable is True

    def test_compute_cross_workload_summary_and_question_answering(self) -> None:
        cfg_1_1 = TunableConfig(max_concurrency=1, max_batch_size=1, batch_wait_ms=50.0)
        cfg_4_2 = TunableConfig(max_concurrency=4, max_batch_size=2, batch_wait_ms=50.0)
        cfg_4_4 = TunableConfig(max_concurrency=4, max_batch_size=4, batch_wait_ms=50.0)
        cfg_8_8 = TunableConfig(max_concurrency=8, max_batch_size=8, batch_wait_ms=50.0)

        r_a = _create_mock_step11_report(
            "light_low_concurrency", 1.5, 400.0, cfg_4_2, 6.0, cfg_1_1, 50.0, cfg_4_2, 5.5, 65.0
        )
        r_b = _create_mock_step11_report(
            "moderate_bursty", 1.8, 500.0, cfg_4_4, 8.0, cfg_1_1, 60.0, cfg_4_2, 7.0, 80.0
        )
        r_c = _create_mock_step11_report(
            "high_concurrency_saturated",
            2.0,
            600.0,
            cfg_8_8,
            12.0,
            cfg_1_1,
            80.0,
            cfg_4_4,
            10.0,
            110.0,
        )
        r_d = _create_mock_step11_report(
            "mixed_variable", 1.7, 550.0, cfg_8_8, 10.0, cfg_1_1, 70.0, cfg_4_4, 9.0, 95.0
        )

        reports = {
            "A_LIGHT": Step12WorkloadReport(
                workload_key="A_LIGHT",
                workload_name="light_low_concurrency",
                workload_description="Class A",
                workload_hash="h_a",
                request_count=16,
                arrival_pattern="CONCURRENT",
                experiment_report=r_a,
            ),
            "B_BURSTY": Step12WorkloadReport(
                workload_key="B_BURSTY",
                workload_name="moderate_bursty",
                workload_description="Class B",
                workload_hash="h_b",
                request_count=16,
                arrival_pattern="BURST",
                experiment_report=r_b,
            ),
            "C_SATURATED": Step12WorkloadReport(
                workload_key="C_SATURATED",
                workload_name="high_concurrency_saturated",
                workload_description="Class C",
                workload_hash="h_c",
                request_count=16,
                arrival_pattern="CONCURRENT",
                experiment_report=r_c,
            ),
            "D_MIXED": Step12WorkloadReport(
                workload_key="D_MIXED",
                workload_name="mixed_variable",
                workload_description="Class D",
                workload_hash="h_d",
                request_count=16,
                arrival_pattern="CONCURRENT",
                experiment_report=r_d,
            ),
        }

        summary = compute_cross_workload_summary(reports)
        assert summary.total_workloads == 4
        assert len(summary.rows) == 4
        assert summary.distinct_latency_winners == 1  # c=1, b=1 for all latency
        assert summary.distinct_throughput_winners == 3  # c=4,b=2; c=4,b=4; c=8,b=8
        assert summary.total_integrity_failures == 0
        assert summary.high_variance_count == 0

        # Check answers to all 8 generalization questions
        assert len(summary.answers_to_generalization_questions) == 8
        for i in range(1, 9):
            prefix = f"{i}_"
            matched_keys = [
                k for k in summary.answers_to_generalization_questions if k.startswith(prefix)
            ]
            assert len(matched_keys) == 1, f"Missing question answer for #{i}"

    def test_step12_generalization_report_json_roundtrip(self) -> None:
        cfg_1_1 = TunableConfig(max_concurrency=1, max_batch_size=1, batch_wait_ms=50.0)
        cfg_4_4 = TunableConfig(max_concurrency=4, max_batch_size=4, batch_wait_ms=50.0)
        r_a = _create_mock_step11_report(
            "light_low_concurrency", 1.5, 400.0, cfg_4_4, 6.0, cfg_1_1, 50.0, cfg_4_4, 5.5, 65.0
        )
        reports = {
            "A_LIGHT": Step12WorkloadReport(
                workload_key="A_LIGHT",
                workload_name="light_low_concurrency",
                workload_description="Class A",
                workload_hash="h_a",
                request_count=16,
                arrival_pattern="CONCURRENT",
                experiment_report=r_a,
            ),
        }
        summary = compute_cross_workload_summary(reports)
        findings = classify_step12_findings(summary)

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
            workload_hash="h_a",
        )

        report = Step12GeneralizationReport(
            experiment_id="step12-test-001",
            timestamp=1700000000.0,
            git_commit="abcdef123456",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            environment=env,
            candidate_space=CandidateSpace(
                concurrencies=(1, 4), batch_sizes=(1, 4), batch_waits_ms=(50.0,)
            ),
            workload_reports=reports,
            summary=summary,
            findings=findings,
        )

        json_str = report.to_json()
        loaded = Step12GeneralizationReport.from_json(json_str)
        assert loaded.experiment_id == report.experiment_id
        assert loaded.git_commit == "abcdef123456"
        assert len(loaded.workload_reports) == 1
        assert loaded.summary.total_workloads == 1

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "step12_report.json"
            report.save_json(fpath)
            loaded_disk = Step12GeneralizationReport.load_json(fpath)
            assert loaded_disk.experiment_id == report.experiment_id


class TestStep12FindingsAndFormatting:
    """Unit tests for Step 12 findings classification and report formatting."""

    def test_classify_step12_findings(self) -> None:
        cfg_1_1 = TunableConfig(max_concurrency=1, max_batch_size=1, batch_wait_ms=50.0)
        cfg_4_4 = TunableConfig(max_concurrency=4, max_batch_size=4, batch_wait_ms=50.0)
        r_a = _create_mock_step11_report(
            "light_low_concurrency", 1.5, 400.0, cfg_4_4, 6.0, cfg_1_1, 50.0, cfg_4_4, 5.5, 65.0
        )
        reports = {
            "A_LIGHT": Step12WorkloadReport(
                workload_key="A_LIGHT",
                workload_name="light_low_concurrency",
                workload_description="Class A",
                workload_hash="h_a",
                request_count=16,
                arrival_pattern="CONCURRENT",
                experiment_report=r_a,
            ),
        }
        summary = compute_cross_workload_summary(reports)
        findings = classify_step12_findings(summary)

        assert "PROVEN" in findings
        assert "SUGGESTED" in findings
        assert "NOT PROVEN" in findings
        assert len(findings["PROVEN"]) >= 3
        assert len(findings["NOT PROVEN"]) >= 4

    def test_format_step12_report(self) -> None:
        cfg_1_1 = TunableConfig(max_concurrency=1, max_batch_size=1, batch_wait_ms=50.0)
        cfg_4_4 = TunableConfig(max_concurrency=4, max_batch_size=4, batch_wait_ms=50.0)
        r_a = _create_mock_step11_report(
            "light_low_concurrency", 1.5, 400.0, cfg_4_4, 6.0, cfg_1_1, 50.0, cfg_4_4, 5.5, 65.0
        )
        reports = {
            "A_LIGHT": Step12WorkloadReport(
                workload_key="A_LIGHT",
                workload_name="light_low_concurrency",
                workload_description="Class A",
                workload_hash="h_a",
                request_count=16,
                arrival_pattern="CONCURRENT",
                experiment_report=r_a,
            ),
        }
        summary = compute_cross_workload_summary(reports)
        findings = classify_step12_findings(summary)

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
            workload_hash="h_a",
        )

        report = Step12GeneralizationReport(
            experiment_id="step12-format-test",
            timestamp=1700000000.0,
            git_commit="abcdef123456",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            environment=env,
            candidate_space=CandidateSpace(
                concurrencies=(1, 4), batch_sizes=(1, 4), batch_waits_ms=(50.0,)
            ),
            workload_reports=reports,
            summary=summary,
            findings=findings,
        )

        text = format_step12_report(report)
        assert "INFEROPT STEP 12: MULTI-WORKLOAD GENERALIZATION EXPERIMENT" in text
        assert "EVALUATED WORKLOAD MATRIX (4 Workload Classes):" in text
        assert "CROSS-WORKLOAD GENERALIZATION SUMMARY MATRIX:" in text
        assert "ANSWERS TO GENERALIZATION QUESTIONS:" in text
        assert "EMPIRICAL FINDINGS CLASSIFICATION:" in text
        assert "[PROVEN]" in text
        assert "[SUGGESTED]" in text
        assert "[NOT PROVEN]" in text


class TestStep12GeneralizationRunnerMocked:
    """Mock integration tests for Step12GeneralizationRunner orchestration."""

    @pytest.mark.asyncio
    async def test_full_step12_generalization_run(self) -> None:
        mock_validator = MagicMock(spec=VLLMValidator)
        mock_validator.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        mock_validator.config = MagicMock()
        mock_validator.config.enforce_eager = True

        async def mock_run_direct(
            scenario: Any, concurrency: int, warmup_count: int, repetitions: int
        ) -> VLLMConditionResult:
            return _create_mock_vllm_condition_result(
                concurrency=concurrency,
                max_batch_size=1,
                requests_per_sec=1.5,
                p95_latency_ms=500.0,
                condition=VLLMCondition.DIRECT_VLLM,
                repetitions=repetitions,
                scenario=scenario,
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
            rps = 1.0 + (max_batch_size * 1.5) + (concurrency * 0.5)
            p95_lat = 40.0 + (max_batch_size * 10.0) + (concurrency * 5.0)
            return _create_mock_vllm_condition_result(
                concurrency=concurrency,
                max_batch_size=max_batch_size,
                requests_per_sec=rps,
                p95_latency_ms=p95_lat,
                repetitions=repetitions,
                scenario=scenario,
            )

        mock_validator.run_condition_direct = AsyncMock(side_effect=mock_run_direct)
        mock_validator.run_condition_inferopt = AsyncMock(side_effect=mock_run_inferopt)

        runner = Step12GeneralizationRunner(validator=mock_validator)

        # 2 workloads x (2 concurrency x 2 batch sizes = 4 candidates)
        matrix = {
            "A_LIGHT": get_step12_workload_matrix()["A_LIGHT"],
            "B_BURSTY": get_step12_workload_matrix()["B_BURSTY"],
        }
        space = CandidateSpace(
            concurrencies=(1, 4),
            batch_sizes=(1, 4),
            batch_waits_ms=(50.0,),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report = await runner.run_experiment(
                workloads=matrix,
                candidate_space=space,
                warmup_count=1,
                exploration_repetitions=2,
                validation_repetitions=2,
                output_dir=tmpdir,
            )

            assert len(report.workload_reports) == 2
            assert "A_LIGHT" in report.workload_reports
            assert "B_BURSTY" in report.workload_reports
            assert report.summary.total_workloads == 2
            assert report.summary.total_integrity_failures == 0

            # Verify saved report
            saved_files = list(Path(tmpdir).glob("step12_generalization_*.json"))
            assert len(saved_files) == 1


class TestStep12CLIExtension:
    """Unit tests for Step 12 CLI argument parsing and dispatch."""

    def test_cli_parser_step12_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--experiment-step12",
                "--workloads",
                "A_LIGHT",
                "C_SATURATED",
                "--num-requests",
                "12",
                "--concurrency",
                "1",
                "4",
                "8",
                "--batch-sizes",
                "1",
                "2",
                "4",
                "8",
                "--val-repetitions",
                "4",
            ]
        )
        assert args.experiment_step12 is True
        assert args.workloads == ["A_LIGHT", "C_SATURATED"]
        assert args.num_requests == 12
        assert args.concurrency == [1, 4, 8]
        assert args.batch_sizes == [1, 2, 4, 8]
        assert args.val_repetitions == 4

    @pytest.mark.asyncio
    async def test_run_benchmark_cli_step12_dispatch(self) -> None:
        mock_summary = MagicMock()
        mock_summary.total_integrity_failures = 0

        mock_report = MagicMock()
        mock_report.summary = mock_summary

        mock_runner = MagicMock()
        mock_runner.run_experiment = AsyncMock(return_value=mock_report)

        with (
            patch(
                "inferopt.benchmarks.step12_generalization.Step12GeneralizationRunner",
                return_value=mock_runner,
            ),
            patch(
                "inferopt.benchmarks.step12_generalization.format_step12_report",
                return_value="Step 12 Mock Report",
            ),
        ):
            parser = build_parser()
            args = parser.parse_args(
                [
                    "--experiment-step12",
                    "--workloads",
                    "A_LIGHT",
                    "B_BURSTY",
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
