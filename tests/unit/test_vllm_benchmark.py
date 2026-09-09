"""Unit and mock integration tests for the Scientific vLLM Benchmark Harness (Step 10)."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from inferopt.backends.vllm import DEFAULT_VLLM_MODEL_ID, VLLMConfig
from inferopt.benchmarks.generator import get_concurrent_4_workload, get_single_workload
from inferopt.benchmarks.vllm_baseline import (
    DirectVLLMResult,
    DirectVLLMRunner,
    calculate_percentile,
    compute_sha256,
    compute_std_dev,
)
from inferopt.benchmarks.vllm_validation import (
    VLLM_CONDITION_LABELS,
    VLLMBenchmarkRequestRecord,
    VLLMComparisonDelta,
    VLLMCondition,
    VLLMConditionResult,
    VLLMExperimentReport,
    VLLMIntegrityResult,
    VLLMValidator,
    classify_vllm_findings,
    collect_vllm_environment_metadata,
    compute_vllm_deltas,
    compute_workload_hash,
    format_vllm_batch_table,
    format_vllm_deltas_table,
    format_vllm_full_report,
    format_vllm_performance_table,
    validate_vllm_integrity,
)

HAS_VLLM: bool = "vllm" in sys.modules
RUN_VLLM_EXPLICIT: bool = os.environ.get("INFEROPT_RUN_VLLM_TESTS") == "1"


def _create_mock_vllm_output(
    prompt: str = "Test prompt",
    text: str = "Test response from vLLM engine",
    prompt_tokens: list[int] | None = None,
    output_tokens: list[int] | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    """Construct mock RequestOutput matching vLLM internal data structure."""
    req_out = MagicMock()
    req_out.prompt = prompt
    req_out.prompt_token_ids = prompt_tokens if prompt_tokens is not None else [101, 102, 103, 104]

    comp_out = MagicMock()
    comp_out.text = text
    comp_out.token_ids = (
        output_tokens if output_tokens is not None else [201, 202, 203, 204, 205, 206]
    )
    comp_out.finish_reason = finish_reason

    req_out.outputs = [comp_out]
    return req_out


def _setup_mock_vllm_module() -> MagicMock:
    """Set up a mock vllm module that produces realistic outputs."""
    mock_llm_instance = MagicMock()

    def _mock_generate(
        prompts: list[str],
        sampling_params: Any = None,
        use_tqdm: bool = False,
    ) -> list[MagicMock]:
        return [
            _create_mock_vllm_output(
                prompt=p,
                text=f"Completed response for: {p}",
                prompt_tokens=[1, 2, 3, 4],
                output_tokens=[10, 11, 12, 13, 14, 15, 16, 17],
            )
            for p in prompts
        ]

    mock_llm_instance.generate.side_effect = _mock_generate
    mock_llm_cls = MagicMock(return_value=mock_llm_instance)
    mock_sampling_cls = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))

    mock_vllm = MagicMock()
    mock_vllm.LLM = mock_llm_cls
    mock_vllm.SamplingParams = mock_sampling_cls
    return mock_vllm


class TestStatisticalHelpers:
    """Unit tests for statistical calculation functions."""

    def test_compute_sha256(self) -> None:
        h1 = compute_sha256("test prompt")
        h2 = compute_sha256("test prompt")
        h3 = compute_sha256("different prompt")
        assert len(h1) == 64
        assert h1 == h2
        assert h1 != h3

    def test_calculate_percentile_empty(self) -> None:
        assert calculate_percentile([], 50.0) == 0.0

    def test_calculate_percentile_single(self) -> None:
        assert calculate_percentile([42.0], 50.0) == 42.0
        assert calculate_percentile([42.0], 95.0) == 42.0

    def test_calculate_percentile_distribution(self) -> None:
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert calculate_percentile(data, 50.0) == 30.0
        assert calculate_percentile(data, 0.0) == 10.0
        assert calculate_percentile(data, 100.0) == 50.0

    def test_compute_std_dev(self) -> None:
        assert compute_std_dev([]) == 0.0
        assert compute_std_dev([5.0]) == 0.0
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        std = compute_std_dev(data)
        assert round(std, 2) == 2.14


class TestWorkloadHashing:
    """Unit tests for deterministic workload hashing."""

    def test_workload_hash_determinism(self) -> None:
        sc1 = get_concurrent_4_workload(seed=42)
        sc2 = get_concurrent_4_workload(seed=42)
        sc_diff_seed = get_concurrent_4_workload(seed=99)
        sc_single = get_single_workload(seed=42)

        h1 = compute_workload_hash(sc1)
        h2 = compute_workload_hash(sc2)
        h_diff = compute_workload_hash(sc_diff_seed)
        h_single = compute_workload_hash(sc_single)

        assert h1 == h2
        assert len(h1) == 64
        assert h1 != h_diff
        assert h1 != h_single


class TestVLLMBenchmarkModelsAndIntegrity:
    """Unit tests for result models, serialization, and integrity gate validation."""

    def test_environment_metadata_collection(self) -> None:
        meta = collect_vllm_environment_metadata(
            model_id="mock-model",
            warmup_count=2,
            repetitions=3,
            workload_seed=42,
            workload_hash="abc123hash",
        )
        assert meta.model_id == "mock-model"
        assert meta.warmup_count == 2
        assert meta.repetitions == 3
        assert meta.workload_seed == 42
        assert meta.workload_hash == "abc123hash"
        assert meta.python_version != ""

    def test_integrity_validation_success(self) -> None:
        scenario = get_concurrent_4_workload(seed=42)
        w_hash = compute_workload_hash(scenario)

        req_records = tuple(
            VLLMBenchmarkRequestRecord(
                request_id=f"req-{i}",
                prompt_hash=f"phash-{i}",
                output_hash=f"ohash-{i}",
                input_tokens=10,
                output_tokens=20,
                latency_ms=50.0,
                success=True,
            )
            for i in range(4)
        )

        cond_res = VLLMConditionResult(
            condition=VLLMCondition.DIRECT_VLLM,
            condition_label=VLLM_CONDITION_LABELS[VLLMCondition.DIRECT_VLLM],
            concurrency=4,
            max_batch_size=1,
            total_requests=4,
            completed_requests=4,
            failed_requests=0,
            duration_sec=0.5,
            requests_per_sec=8.0,
            output_tokens_per_sec=160.0,
            total_tokens_per_sec=240.0,
            mean_latency_ms=50.0,
            median_latency_ms=50.0,
            p50_latency_ms=50.0,
            p95_latency_ms=50.0,
            p99_latency_ms=50.0,
            min_latency_ms=50.0,
            max_latency_ms=50.0,
            std_dev_latency_ms=0.0,
            avg_queue_wait_ms=0.0,
            avg_backend_execution_ms=50.0,
            total_input_tokens=40,
            total_output_tokens=80,
            total_tokens=120,
            avg_input_tokens_per_req=10.0,
            avg_output_tokens_per_req=20.0,
            min_output_tokens=20,
            max_output_tokens=20,
            total_batches=4,
            avg_batch_size=1.0,
            median_batch_size=1.0,
            min_batch_size=1,
            max_batch_size_formed=1,
            avg_batch_formation_wait_ms=0.0,
            avg_batch_execution_ms=50.0,
            repetition_count=1,
            requests=req_records,
        )

        # Match exact request IDs from scenario
        scenario_requests = [
            VLLMBenchmarkRequestRecord(
                request_id=r.request_id,
                prompt_hash="phash",
                output_hash="ohash",
                input_tokens=10,
                output_tokens=20,
                latency_ms=50.0,
                success=True,
            )
            for r in scenario.requests
        ]
        cond_res_matched = cond_res.model_copy(update={"requests": tuple(scenario_requests)})

        integrity = validate_vllm_integrity(
            scenario=scenario,
            condition_results=[cond_res_matched],
            expected_workload_hash=w_hash,
        )

        assert integrity.is_valid is True
        assert integrity.total_expected_requests == 4
        assert integrity.total_completed_requests == 4
        assert integrity.has_duplicate_ids is False
        assert integrity.has_missing_ids is False
        assert integrity.has_empty_outputs is False
        assert integrity.has_invalid_tokens is False
        assert integrity.has_unexpected_failures is False

    def test_integrity_validation_missing_ids(self) -> None:
        scenario = get_concurrent_4_workload(seed=42)
        w_hash = compute_workload_hash(scenario)

        # Incomplete request list
        req_records = (
            VLLMBenchmarkRequestRecord(
                request_id=scenario.requests[0].request_id,
                prompt_hash="p1",
                output_hash="o1",
                input_tokens=10,
                output_tokens=20,
                latency_ms=50.0,
                success=True,
            ),
        )

        cond_res = VLLMConditionResult(
            condition=VLLMCondition.DIRECT_VLLM,
            condition_label="A. Direct vLLM",
            concurrency=1,
            max_batch_size=1,
            total_requests=4,
            completed_requests=1,
            failed_requests=3,
            duration_sec=0.5,
            requests_per_sec=2.0,
            output_tokens_per_sec=40.0,
            total_tokens_per_sec=60.0,
            mean_latency_ms=50.0,
            median_latency_ms=50.0,
            p50_latency_ms=50.0,
            p95_latency_ms=50.0,
            p99_latency_ms=50.0,
            min_latency_ms=50.0,
            max_latency_ms=50.0,
            std_dev_latency_ms=0.0,
            avg_queue_wait_ms=0.0,
            avg_backend_execution_ms=50.0,
            total_input_tokens=10,
            total_output_tokens=20,
            total_tokens=30,
            avg_input_tokens_per_req=10.0,
            avg_output_tokens_per_req=20.0,
            min_output_tokens=20,
            max_output_tokens=20,
            total_batches=1,
            avg_batch_size=1.0,
            median_batch_size=1.0,
            max_batch_size_formed=1,
            avg_batch_formation_wait_ms=0.0,
            avg_batch_execution_ms=50.0,
            repetition_count=1,
            requests=req_records,
        )

        integrity = validate_vllm_integrity(
            scenario=scenario,
            condition_results=[cond_res],
            expected_workload_hash=w_hash,
        )

        assert integrity.is_valid is False
        assert integrity.has_missing_ids is True
        assert integrity.has_unexpected_failures is True

    def test_experiment_report_json_roundtrip(self) -> None:
        scenario = get_single_workload(seed=42)
        w_hash = compute_workload_hash(scenario)
        meta = collect_vllm_environment_metadata("mock", 1, 1, 42, w_hash)
        integrity = VLLMIntegrityResult(
            is_valid=True,
            total_expected_requests=1,
            total_completed_requests=1,
            has_duplicate_ids=False,
            has_missing_ids=False,
            has_empty_outputs=False,
            has_invalid_tokens=False,
            has_unexpected_failures=False,
        )

        report = VLLMExperimentReport(
            experiment_id="test-exp-1",
            timestamp=1700000000.0,
            scenario_name="single",
            model_id="mock-model",
            workload_hash=w_hash,
            environment=meta,
            integrity=integrity,
            conditions=(),
            deltas=(),
            concurrency_levels=(1,),
            batch_sizes=(1,),
            repetition_count=1,
            warmup_count=1,
            findings={"PROVEN": ("Test proven finding",)},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = Path(tmp_dir) / "test_report.json"
            report.save_json(fpath)
            loaded = VLLMExperimentReport.load_json(fpath)

            assert loaded.experiment_id == report.experiment_id
            assert loaded.workload_hash == w_hash
            assert loaded.findings["PROVEN"] == ("Test proven finding",)


class TestDeltasAndReportFormatting:
    """Unit tests for delta comparisons and terminal report formatting."""

    def test_compute_vllm_deltas(self) -> None:
        direct = VLLMConditionResult(
            condition=VLLMCondition.DIRECT_VLLM,
            condition_label="A. Direct vLLM",
            concurrency=4,
            max_batch_size=1,
            total_requests=4,
            completed_requests=4,
            failed_requests=0,
            duration_sec=1.0,
            requests_per_sec=4.0,
            output_tokens_per_sec=100.0,
            total_tokens_per_sec=150.0,
            mean_latency_ms=100.0,
            median_latency_ms=100.0,
            p50_latency_ms=100.0,
            p95_latency_ms=100.0,
            p99_latency_ms=100.0,
            min_latency_ms=100.0,
            max_latency_ms=100.0,
            std_dev_latency_ms=0.0,
            avg_queue_wait_ms=0.0,
            avg_backend_execution_ms=100.0,
            total_input_tokens=50,
            total_output_tokens=100,
            total_tokens=150,
            avg_input_tokens_per_req=12.5,
            avg_output_tokens_per_req=25.0,
            min_output_tokens=25,
            max_output_tokens=25,
            total_batches=4,
            avg_batch_size=1.0,
            median_batch_size=1.0,
            max_batch_size_formed=1,
            avg_batch_formation_wait_ms=0.0,
            avg_batch_execution_ms=100.0,
            repetition_count=1,
            requests=(),
        )

        batch4 = direct.model_copy(
            update={
                "condition": VLLMCondition.INFEROPT_BATCH_4,
                "condition_label": "D. InferOpt Batch 4",
                "max_batch_size": 4,
                "requests_per_sec": 8.0,  # 100% throughput gain
                "output_tokens_per_sec": 200.0,
                "mean_latency_ms": 110.0,  # 10ms differential overhead
            }
        )

        deltas = compute_vllm_deltas([direct, batch4])
        assert len(deltas) == 1
        d = deltas[0]
        assert d.concurrency == 4
        assert d.throughput_change_pct == 100.0
        assert d.token_throughput_change_pct == 100.0
        assert d.latency_change_pct == 10.0
        assert d.end_to_end_differential_overhead_ms == 10.0

    def test_classify_vllm_findings(self) -> None:
        delta = VLLMComparisonDelta(
            comparison_name="Batch 4 vs Direct",
            baseline_condition="Direct",
            target_condition="Batch 4",
            concurrency=4,
            throughput_change_pct=25.0,
            token_throughput_change_pct=25.0,
            latency_change_pct=5.0,
            end_to_end_differential_overhead_ms=5.0,
        )
        findings = classify_vllm_findings([delta], [])
        assert "PROVEN" in findings
        assert "SUGGESTED" in findings
        assert "NOT PROVEN" in findings
        assert any("dynamic batching" in s.lower() for s in findings["PROVEN"])

    def test_format_vllm_tables(self) -> None:
        cond = VLLMConditionResult(
            condition=VLLMCondition.DIRECT_VLLM,
            condition_label="A. Direct vLLM",
            concurrency=4,
            max_batch_size=1,
            total_requests=4,
            completed_requests=4,
            failed_requests=0,
            duration_sec=1.0,
            requests_per_sec=4.0,
            output_tokens_per_sec=100.0,
            total_tokens_per_sec=150.0,
            mean_latency_ms=100.0,
            median_latency_ms=100.0,
            p50_latency_ms=100.0,
            p95_latency_ms=100.0,
            p99_latency_ms=100.0,
            min_latency_ms=100.0,
            max_latency_ms=100.0,
            std_dev_latency_ms=0.0,
            avg_queue_wait_ms=0.0,
            avg_backend_execution_ms=100.0,
            total_input_tokens=50,
            total_output_tokens=100,
            total_tokens=150,
            avg_input_tokens_per_req=12.5,
            avg_output_tokens_per_req=25.0,
            min_output_tokens=25,
            max_output_tokens=25,
            total_batches=4,
            avg_batch_size=1.0,
            median_batch_size=1.0,
            max_batch_size_formed=1,
            avg_batch_formation_wait_ms=0.0,
            avg_batch_execution_ms=100.0,
            repetition_count=1,
            requests=(),
        )

        perf_table = format_vllm_performance_table([cond], concurrency=4)
        assert "A. Direct vLLM" in perf_table
        assert "4.00" in perf_table

        batch_table = format_vllm_batch_table([cond], concurrency=4)
        assert "A. Direct vLLM" in batch_table

        delta = VLLMComparisonDelta(
            comparison_name="Test Delta",
            baseline_condition="Direct",
            target_condition="Batch 4",
            concurrency=4,
            throughput_change_pct=15.0,
            token_throughput_change_pct=15.0,
            latency_change_pct=-2.0,
            end_to_end_differential_overhead_ms=-2.0,
        )
        delta_table = format_vllm_deltas_table([delta])
        assert "Test Delta" in delta_table


class TestDirectVLLMRunnerMocked:
    """Unit tests for DirectVLLMRunner using mock vLLM."""

    @pytest.mark.asyncio
    async def test_direct_runner_execution_and_tokens(self) -> None:
        mock_vllm = _setup_mock_vllm_module()
        runner = DirectVLLMRunner(model_id=DEFAULT_VLLM_MODEL_ID)

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            scenario = get_concurrent_4_workload(seed=42)
            result = await runner.run(
                scenario=scenario,
                warmup_count=1,
                concurrency=2,
            )

            assert isinstance(result, DirectVLLMResult)
            assert result.total_requests == 4
            assert result.completed_requests == 4
            assert result.failed_requests == 0
            assert result.requests_per_sec > 0.0
            assert result.total_input_tokens == 16  # 4 tokens * 4
            assert result.total_output_tokens == 32  # 8 tokens * 4
            assert len(result.request_results) == 4
            assert all(r.success for r in result.request_results)
            assert all(r.output_tokens == 8 for r in result.request_results)


class TestVLLMValidatorMocked:
    """Unit tests for full scientific VLLMValidator with mock vLLM engine."""

    @pytest.mark.asyncio
    async def test_full_scientific_benchmark_run(self) -> None:
        mock_vllm = _setup_mock_vllm_module()
        validator = VLLMValidator(model_id=DEFAULT_VLLM_MODEL_ID)

        with (
            patch.dict(sys.modules, {"vllm": mock_vllm}),
            tempfile.TemporaryDirectory() as tmp_dir,
        ):
            scenario = get_concurrent_4_workload(seed=42)
            report = await validator.run_scientific_benchmark(
                scenario=scenario,
                concurrency_levels=(1, 4),
                batch_sizes=(1, 2, 4),
                warmup_count=1,
                repetitions=1,
                output_dir=tmp_dir,
            )

            assert isinstance(report, VLLMExperimentReport)
            assert report.scenario_name == "concurrent_4"
            assert report.workload_hash == compute_workload_hash(scenario)
            assert len(report.concurrency_levels) == 2
            assert len(report.batch_sizes) == 3

            # 2 concurrency levels * (1 Direct + 3 Batch sizes) = 8 condition results
            assert len(report.conditions) == 8

            # Verify all conditions completed
            for cond in report.conditions:
                assert cond.completed_requests == 4
                assert cond.failed_requests == 0
                assert cond.requests_per_sec > 0.0

            # Full terminal report format string
            full_rep_text = format_vllm_full_report(report)
            assert "InferOpt Scientific vLLM Benchmark Report" in full_rep_text
            assert "PROVEN" in full_rep_text


class TestLiveVLLMBenchmark:
    """Live hardware tests executed only on compatible NVIDIA GPU systems."""

    @pytest.mark.skipif(
        not (HAS_VLLM or RUN_VLLM_EXPLICIT),
        reason="Live vLLM benchmark requires an NVIDIA GPU with vLLM installed.",
    )
    @pytest.mark.asyncio
    async def test_live_vllm_benchmark_smoke(self) -> None:
        validator = VLLMValidator(
            config=VLLMConfig(
                model=DEFAULT_VLLM_MODEL_ID,
                default_max_tokens=16,
                default_temperature=0.0,
            )
        )
        scenario = get_concurrent_4_workload(seed=42)

        with tempfile.TemporaryDirectory() as tmp_dir:
            report = await validator.run_scientific_benchmark(
                scenario=scenario,
                concurrency_levels=(1,),
                batch_sizes=(1, 4),
                warmup_count=1,
                repetitions=1,
                output_dir=tmp_dir,
            )
            assert report.integrity.is_valid is True
            assert len(report.conditions) == 2
