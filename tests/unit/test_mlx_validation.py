"""Unit and integration tests for MLX validation and baseline comparison subsystem."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from inferopt.benchmarks.cli import build_parser
from inferopt.benchmarks.generator import (
    get_concurrent_4_workload,
    get_single_workload,
)
from inferopt.benchmarks.mlx_baseline import (
    DirectMLXRequestResult,
    DirectMLXResult,
    DirectMLXRunner,
    calculate_percentile,
    compute_sha256,
    compute_std_dev,
)
from inferopt.benchmarks.mlx_validation import (
    AuditCondition,
    AuditConditionResult,
    AuditExperimentReport,
    AuditRequestRecord,
    MLXValidator,
    format_audit_batch_table,
    format_audit_deltas_table,
    format_audit_full_report,
    format_audit_performance_table,
    format_audit_token_table,
    format_comparison_table,
)
from inferopt.core.models import InferenceBatch, InferenceRequest

try:
    import mlx.core as mx  # noqa: F401
    import mlx_lm  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

RUN_MLX_EXPLICIT = os.getenv("INFEROPT_RUN_MLX_TESTS") == "1"


class TestDirectMLXRunnerMocked:
    """Unit tests for DirectMLXRunner using mocks without requiring real MLX."""

    @pytest.mark.asyncio
    async def test_mocked_direct_run_sequential_with_hashes(self) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text: [4, 5] if "Direct" in text else [1, 2, 3]

        runner = DirectMLXRunner(model_id="mock-qwen")
        runner._model = mock_model
        runner._tokenizer = mock_tokenizer

        with patch("mlx_lm.generate", return_value="Direct output text"):
            scenario = get_single_workload(seed=42)
            result = await runner.run(scenario=scenario, warmup_count=0)

            assert isinstance(result, DirectMLXResult)
            assert result.scenario_name == "single"
            assert result.model_id == "mock-qwen"
            assert result.total_requests == 1
            assert result.completed_requests == 1
            assert result.failed_requests == 0
            assert result.requests_per_sec > 0.0
            assert len(result.request_results) == 1
            req_res = result.request_results[0]
            assert req_res.generated_text == "Direct output text"
            assert req_res.input_tokens == 3
            assert req_res.output_tokens == 2
            assert req_res.prompt_hash == compute_sha256(scenario.requests[0].prompt)
            assert req_res.output_hash == compute_sha256("Direct output text")
            assert result.min_output_tokens == 2
            assert result.max_output_tokens == 2
            assert result.avg_input_tokens_per_req == 3.0
            assert result.avg_output_tokens_per_req == 2.0

    @pytest.mark.asyncio
    async def test_mocked_direct_run_concurrent(self) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2]

        runner = DirectMLXRunner(model_id="mock-qwen")
        runner._model = mock_model
        runner._tokenizer = mock_tokenizer

        with patch("mlx_lm.generate", return_value="Concurrent output"):
            scenario = get_concurrent_4_workload(seed=42)
            result = await runner.run(scenario=scenario, warmup_count=0)

            assert result.total_requests == 4
            assert result.completed_requests == 4
            assert len(result.request_results) == 4
            assert all(r.success for r in result.request_results)
            assert all(
                r.output_hash == compute_sha256("Concurrent output") for r in result.request_results
            )

    @pytest.mark.asyncio
    async def test_mocked_direct_run_native_batch(self) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        runner = DirectMLXRunner(model_id="mock-qwen")
        runner._model = mock_model
        runner._tokenizer = mock_tokenizer

        mock_batch_resp = MagicMock()
        mock_batch_resp.texts = ["Out 1", "Out 2", "Out 3", "Out 4"]

        with patch("mlx_lm.batch_generate", return_value=mock_batch_resp):
            scenario = get_concurrent_4_workload(seed=42)
            result = await runner.run_native_batch(scenario=scenario, warmup_count=0)

            assert result.total_requests == 4
            assert result.completed_requests == 4
            assert result.failed_requests == 0
            assert len(result.request_results) == 4
            assert result.request_results[0].generated_text == "Out 1"
            assert result.request_results[3].generated_text == "Out 4"
            assert result.request_results[0].output_hash == compute_sha256("Out 1")

    @pytest.mark.asyncio
    async def test_mocked_direct_run_handles_failure(self) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2]

        runner = DirectMLXRunner(model_id="mock-qwen")
        runner._model = mock_model
        runner._tokenizer = mock_tokenizer

        with patch("mlx_lm.generate", side_effect=RuntimeError("Metal driver error")):
            scenario = get_single_workload(seed=42)
            result = await runner.run(scenario=scenario, warmup_count=0)

            assert result.total_requests == 1
            assert result.completed_requests == 0
            assert result.failed_requests == 1
            assert result.request_results[0].success is False
            assert "Metal driver error" in (result.request_results[0].error_message or "")


class TestStatisticalUtils:
    """Unit tests for statistical distribution math."""

    def test_percentile_and_std_dev(self) -> None:
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert calculate_percentile(vals, 50.0) == 30.0
        assert calculate_percentile(vals, 0.0) == 10.0
        assert calculate_percentile(vals, 100.0) == 50.0

        std = compute_std_dev(vals)
        assert std == pytest.approx(15.811388, rel=1e-4)

        # Single element std dev is 0.0
        assert compute_std_dev([42.0]) == 0.0
        assert compute_std_dev([]) == 0.0


class TestAuditIntegrityGate:
    """Unit tests for audit integrity and exact output equivalence checks."""

    def test_audit_integrity_valid(self) -> None:
        validator = MLXValidator()
        scenario = get_concurrent_4_workload(seed=42)

        req_records = tuple(
            AuditRequestRecord(
                request_id=req.request_id,
                prompt_hash=compute_sha256(req.prompt),
                output_hash=compute_sha256("Same output"),
                input_tokens=10,
                output_tokens=20,
                success=True,
            )
            for req in scenario.requests
        )

        res_a = AuditConditionResult(
            condition=AuditCondition.DIRECT_SINGLE,
            condition_label="A. Direct MLX Single",
            concurrency=1,
            max_batch_size=1,
            total_requests=4,
            completed_requests=4,
            failed_requests=0,
            duration_sec=0.2,
            requests_per_sec=20.0,
            output_tokens_per_sec=400.0,
            total_tokens_per_sec=600.0,
            mean_latency_ms=50.0,
            median_latency_ms=50.0,
            p50_latency_ms=50.0,
            p95_latency_ms=50.0,
            p99_latency_ms=50.0,
            min_latency_ms=50.0,
            max_latency_ms=50.0,
            std_dev_latency_ms=0.0,
            total_input_tokens=40,
            total_output_tokens=80,
            avg_input_tokens_per_req=10.0,
            avg_output_tokens_per_req=20.0,
            min_output_tokens=20,
            max_output_tokens=20,
            total_batches=4,
            avg_batch_size=1.0,
            median_batch_size=1.0,
            max_batch_size_formed=1,
            batch_size_distribution={1: 4},
            avg_batch_wait_ms=0.0,
            avg_backend_execution_ms=50.0,
            repetition_count=1,
            requests=req_records,
        )

        res_c = res_a.model_copy(
            update={
                "condition": AuditCondition.INFEROPT_BATCH_1,
                "condition_label": "C. InferOpt Batch 1",
            }
        )

        integrity = validator._verify_audit_integrity(scenario, [res_a, res_c])
        assert integrity.is_valid is True
        assert integrity.status == "COMPARISON_VALID"
        assert integrity.exact_output_match is True
        assert len(integrity.diagnostics) == 0

    def test_audit_integrity_detects_output_mismatch(self) -> None:
        validator = MLXValidator()
        scenario = get_single_workload(seed=42)

        rec_a = (
            AuditRequestRecord(
                request_id=scenario.requests[0].request_id,
                prompt_hash=compute_sha256(scenario.requests[0].prompt),
                output_hash=compute_sha256("Output A"),
                input_tokens=10,
                output_tokens=20,
                success=True,
            ),
        )
        rec_c = (
            AuditRequestRecord(
                request_id=scenario.requests[0].request_id,
                prompt_hash=compute_sha256(scenario.requests[0].prompt),
                output_hash=compute_sha256("Output B (Divergent)"),
                input_tokens=10,
                output_tokens=20,
                success=True,
            ),
        )

        res_a = AuditConditionResult(
            condition=AuditCondition.DIRECT_SINGLE,
            condition_label="A. Direct MLX Single",
            concurrency=1,
            max_batch_size=1,
            total_requests=1,
            completed_requests=1,
            failed_requests=0,
            duration_sec=0.1,
            requests_per_sec=10.0,
            output_tokens_per_sec=200.0,
            total_tokens_per_sec=300.0,
            mean_latency_ms=100.0,
            median_latency_ms=100.0,
            p50_latency_ms=100.0,
            p95_latency_ms=100.0,
            p99_latency_ms=100.0,
            min_latency_ms=100.0,
            max_latency_ms=100.0,
            std_dev_latency_ms=0.0,
            total_input_tokens=10,
            total_output_tokens=20,
            avg_input_tokens_per_req=10.0,
            avg_output_tokens_per_req=20.0,
            min_output_tokens=20,
            max_output_tokens=20,
            total_batches=1,
            avg_batch_size=1.0,
            median_batch_size=1.0,
            max_batch_size_formed=1,
            batch_size_distribution={1: 1},
            avg_batch_wait_ms=0.0,
            avg_backend_execution_ms=100.0,
            repetition_count=1,
            requests=rec_a,
        )
        res_c = res_a.model_copy(
            update={
                "condition": AuditCondition.INFEROPT_BATCH_1,
                "condition_label": "C. InferOpt Batch 1",
                "requests": rec_c,
            }
        )

        integrity = validator._verify_audit_integrity(scenario, [res_a, res_c])
        assert integrity.is_valid is True
        assert integrity.status == "OUTPUT_MISMATCH"
        assert integrity.exact_output_match is False
        assert len(integrity.output_mismatches) == 1
        assert scenario.requests[0].request_id in integrity.output_mismatches


class TestAuditDeltaCalculations:
    """Unit tests for relative percentage deltas and differential overhead calculations."""

    def test_calculate_audit_deltas(self) -> None:
        validator = MLXValidator()

        res_a = AuditConditionResult(
            condition=AuditCondition.DIRECT_SINGLE,
            condition_label="A. Direct MLX Single",
            concurrency=1,
            max_batch_size=1,
            total_requests=4,
            completed_requests=4,
            failed_requests=0,
            duration_sec=0.2,
            requests_per_sec=20.0,
            output_tokens_per_sec=400.0,
            total_tokens_per_sec=600.0,
            mean_latency_ms=50.0,
            median_latency_ms=50.0,
            p50_latency_ms=50.0,
            p95_latency_ms=50.0,
            p99_latency_ms=50.0,
            min_latency_ms=50.0,
            max_latency_ms=50.0,
            std_dev_latency_ms=0.0,
            total_input_tokens=40,
            total_output_tokens=80,
            avg_input_tokens_per_req=10.0,
            avg_output_tokens_per_req=20.0,
            min_output_tokens=20,
            max_output_tokens=20,
            total_batches=4,
            avg_batch_size=1.0,
            median_batch_size=1.0,
            max_batch_size_formed=1,
            batch_size_distribution={1: 4},
            avg_batch_wait_ms=0.0,
            avg_backend_execution_ms=50.0,
            repetition_count=1,
            requests=(),
        )
        res_b = res_a.model_copy(
            update={
                "condition": AuditCondition.DIRECT_NATIVE_BATCH,
                "condition_label": "B. Direct MLX Native Batch",
                "requests_per_sec": 40.0,
                "output_tokens_per_sec": 800.0,
                "mean_latency_ms": 25.0,
            }
        )
        res_c = res_a.model_copy(
            update={
                "condition": AuditCondition.INFEROPT_BATCH_1,
                "condition_label": "C. InferOpt Batch 1",
                "requests_per_sec": 19.0,
                "output_tokens_per_sec": 380.0,
                "mean_latency_ms": 52.5,
            }
        )
        res_e = res_a.model_copy(
            update={
                "condition": AuditCondition.INFEROPT_BATCH_4,
                "condition_label": "E. InferOpt Batch 4",
                "requests_per_sec": 38.0,
                "output_tokens_per_sec": 760.0,
                "mean_latency_ms": 26.0,
            }
        )

        deltas = validator._calculate_audit_deltas([res_a, res_b, res_c, res_e])
        assert len(deltas) >= 3

        # Differential Overhead
        ovh_delta = next(d for d in deltas if "Overhead" in d.comparison_name)
        assert ovh_delta.end_to_end_differential_overhead_ms == pytest.approx(2.5)
        assert ovh_delta.throughput_change_pct == pytest.approx(-5.0)
        assert ovh_delta.latency_change_pct == pytest.approx(5.0)

        # Native Batch
        nb_delta = next(d for d in deltas if "Native" in d.comparison_name)
        assert nb_delta.throughput_change_pct == pytest.approx(100.0)
        assert nb_delta.latency_change_pct == pytest.approx(-50.0)


class TestAuditMockedExecutionAndFormatting:
    """Unit tests executing a complete mocked audit run and verifying table formatters."""

    @pytest.mark.asyncio
    async def test_mocked_audit_experiment_run(self) -> None:
        from inferopt.core.models import InferenceResponse

        validator = MLXValidator(model_id="mock-qwen")
        scenario = get_concurrent_4_workload(seed=42)

        mock_direct_res = DirectMLXResult(
            baseline_id="b1",
            timestamp=1.0,
            scenario_name="concurrent_4",
            model_id="mock-qwen",
            duration_sec=0.2,
            total_requests=4,
            completed_requests=4,
            failed_requests=0,
            requests_per_sec=20.0,
            output_tokens_per_sec=400.0,
            total_input_tokens=40,
            total_output_tokens=80,
            avg_input_tokens_per_req=10.0,
            avg_output_tokens_per_req=20.0,
            min_output_tokens=20,
            max_output_tokens=20,
            avg_latency_ms=50.0,
            p50_latency_ms=50.0,
            p95_latency_ms=50.0,
            p99_latency_ms=50.0,
            request_results=tuple(
                DirectMLXRequestResult(
                    request_id=req.request_id,
                    prompt=req.prompt,
                    prompt_hash=compute_sha256(req.prompt),
                    generated_text="Mock completion",
                    output_hash=compute_sha256("Mock completion"),
                    input_tokens=10,
                    output_tokens=20,
                    latency_ms=50.0,
                    success=True,
                )
                for req in scenario.requests
            ),
        )

        async def _mock_gen(req: InferenceRequest) -> InferenceResponse:
            return InferenceResponse(
                request_id=req.request_id,
                generated_text="Mock completion",
                input_tokens=10,
                output_tokens=20,
                latency_ms=50.0,
                backend_name="mlx",
            )

        async def _mock_batch_gen(batch: InferenceBatch) -> list[InferenceResponse]:
            return [
                InferenceResponse(
                    request_id=req.request_id,
                    generated_text="Mock completion",
                    input_tokens=10,
                    output_tokens=20,
                    latency_ms=50.0,
                    backend_name="mlx",
                )
                for req in batch.requests
            ]

        with (
            patch.object(validator._direct_runner, "load_model", return_value=None),
            patch.object(validator._backend, "load_model", return_value=None),
            patch.object(validator._direct_runner, "warmup", return_value=None),
            patch.object(validator._backend, "generate", side_effect=_mock_gen),
            patch.object(validator._backend, "generate_batch", side_effect=_mock_batch_gen),
            patch.object(validator._direct_runner, "run", return_value=mock_direct_res),
            patch.object(
                validator._direct_runner, "run_native_batch", return_value=mock_direct_res
            ),
        ):
            report = await validator.run_audit_experiment(
                scenario=scenario,
                concurrencies=(1, 4),
                batch_sizes=(1, 2, 4),
                warmup_count=1,
                repetitions=2,
            )

            assert isinstance(report, AuditExperimentReport)
            assert report.integrity.is_valid is True
            assert len(report.condition_results) == 10  # 2 concurrencies * (2 direct + 3 inferopt)
            assert len(report.deltas) > 0

            # Verify ASCII tables format without error
            p_table = format_audit_performance_table(report)
            assert "PERFORMANCE SUMMARY MATRIX" in p_table
            assert "A. Direct MLX Single" in p_table

            t_table = format_audit_token_table(report)
            assert "TOKEN WORK EQUIVALENCE" in t_table

            b_table = format_audit_batch_table(report)
            assert "BATCH FORMATION & DISPATCH DYNAMICS" in b_table

            d_table = format_audit_deltas_table(report)
            assert "RELATIVE DELTAS & DIFFERENTIAL OVERHEAD ANALYSIS" in d_table

            full_report = format_audit_full_report(report)
            assert "INFEROPT STEP 8.5 SCIENTIFIC MLX AUDIT REPORT" in full_report
            assert "No performance superiority claim is accepted" in full_report

            # JSON round-trip serialization
            json_str = report.model_dump_json()
            data = json.loads(json_str)
            assert data["model_id"] == "mock-qwen"
            assert data["integrity"]["status"] in ("COMPARISON_VALID", "OUTPUT_MISMATCH")


class TestCLIArgumentParsing:
    """Unit tests verifying CLI parsing for the enhanced audit mode."""

    def test_audit_cli_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--validate-mlx",
                "--audit",
                "--concurrency",
                "1",
                "4",
                "8",
                "--batch-sizes",
                "1",
                "2",
                "4",
                "8",
                "--warmup",
                "2",
                "--repetitions",
                "5",
                "--seed",
                "42",
            ]
        )
        assert args.validate_mlx is True
        assert args.audit is True
        assert args.concurrency == [1, 4, 8]
        assert args.batch_sizes == [1, 2, 4, 8]
        assert args.warmup == 2
        assert args.repetitions == 5
        assert args.seed == 42


@pytest.mark.skipif(
    not (HAS_MLX or RUN_MLX_EXPLICIT),
    reason="Requires Apple Silicon and MLX (set INFEROPT_RUN_MLX_TESTS=1 to enforce)",
)
class TestLiveMLXValidationIntegration:
    """Live validation tests executing real inference on Apple Silicon."""

    @pytest.mark.asyncio
    async def test_live_direct_runner_single(self) -> None:
        runner = DirectMLXRunner()
        scenario = get_single_workload(seed=42)
        res = await runner.run(scenario=scenario, warmup_count=1)

        assert res.total_requests == 1
        assert res.completed_requests == 1
        assert res.failed_requests == 0
        assert res.total_output_tokens > 0
        assert res.requests_per_sec > 0.0

    @pytest.mark.asyncio
    async def test_live_validation_experiment_single(self) -> None:
        validator = MLXValidator()
        scenario = get_single_workload(seed=42)
        report = await validator.run_validation_experiment(
            scenario=scenario,
            warmup_count=1,
            repetitions=2,
        )

        assert report.correctness_gate.is_valid is True
        assert len(report.direct_mlx_runs) == 2
        assert len(report.inferopt_runs) == 2
        assert len(report.comparisons) > 0

        table = format_comparison_table(report)
        assert "InferOpt Step 8.5 MLX Validation Report" in table
        assert "Average Latency" in table

    @pytest.mark.asyncio
    async def test_live_variable_length_batch_generation(self) -> None:
        from inferopt.backends.mlx import MLXBackend

        backend = MLXBackend()
        await backend.load_model()

        batch = InferenceBatch(
            requests=(
                InferenceRequest(
                    request_id="v-1", model="mlx", prompt="What is 1+1?", max_tokens=8
                ),
                InferenceRequest(
                    request_id="v-2",
                    model="mlx",
                    prompt="Explain gravity in two sentences.",
                    max_tokens=24,
                ),
            )
        )
        responses = await backend.generate_batch(batch)
        assert len(responses) == 2
        assert responses[0].request_id == "v-1"
        assert responses[1].request_id == "v-2"
        assert responses[0].generated_text.strip() != ""
        assert responses[1].generated_text.strip() != ""

    @pytest.mark.asyncio
    async def test_live_mlx_detokenizer_whitespace_regression(self) -> None:
        """Regression test demonstrating detokenizer whitespace semantics across MLX APIs.

        Proves that:
        1. mlx_lm.generate uses streaming detokenizer which trims initial token whitespace.
        2. mlx_lm.batch_generate uses tokenizer.decode which preserves initial token whitespace.
        3. Both execute identical autoregressive token steps and produce equivalent content.
        """
        import mlx_lm
        import mlx_lm.sample_utils

        loaded = mlx_lm.load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        model, tokenizer = loaded[0], loaded[1]
        prompt = "Summarize the benefits of dynamic batching in three bullet points."
        sampler = mlx_lm.sample_utils.make_sampler(temp=0.0)

        # 1. Single generation via mlx_lm.generate
        text_single = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=32, sampler=sampler, verbose=False
        )

        # 2. Batch generation via mlx_lm.batch_generate
        prompt_tokens = tokenizer.encode(prompt)
        batch_res = mlx_lm.batch_generate(
            model, tokenizer, prompts=[prompt_tokens], max_tokens=[32], verbose=False
        )
        text_batch = batch_res.texts[0]

        # Inherent API difference: batch_generate retains leading space, generate strips it
        assert text_batch.startswith(" ")
        assert text_batch.lstrip() == text_single.lstrip()

        # Both generate identical content after leading whitespace normalization
        toks_single = tokenizer.encode(text_single)
        toks_batch = tokenizer.encode(text_batch)
        # Differ at most by the isolated initial whitespace token
        assert abs(len(toks_single) - len(toks_batch)) <= 1
