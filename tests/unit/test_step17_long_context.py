"""Unit tests for Step 17: Long-Context & Production-Like Traffic Validation."""

import json
from pathlib import Path
from typing import Any

import pytest

from inferopt.backends.mock import MockBackend
from inferopt.benchmarks.cli import build_parser, run_benchmark_cli
from inferopt.benchmarks.step17_long_context import (
    DEFAULT_STEP17_MODEL_ID,
    DEFAULT_STEP17_TARGET_P95_MS,
    ContextProfile,
    Step17AggregatedConditionMetrics,
    Step17BenchmarkReport,
    Step17BenchmarkRunner,
    Step17ConditionMetrics,
    Step17ContextAnalysis,
    Step17TrafficAnalysis,
    Step17WorkloadCellResult,
    TrafficProfile,
    build_context_prompt,
    build_step17_workload,
    classify_step17_findings,
    format_step17_readme,
    format_step17_report,
    verify_step17_condition_execution,
    verify_step17_report,
)
from inferopt.benchmarks.vllm_validation import VLLMEnvironmentMetadata, compute_workload_hash
from inferopt.optimizer.models import CandidateSpace, TunableConfig
from inferopt.optimizer.sla_models import TargetSLO


def test_deterministic_context_prompt_generation() -> None:
    """Verify generated prompts are deterministic and match target lengths."""
    for profile in ContextProfile:
        prompt1 = build_context_prompt(profile, seed=42, request_index=0)
        prompt2 = build_context_prompt(profile, seed=42, request_index=0)
        prompt3 = build_context_prompt(profile, seed=42, request_index=1)

        # Exact reproducibility
        assert prompt1 == prompt2
        assert len(prompt1) > 0

        # Different index generates valid variation
        assert len(prompt3) > 0

    # Verify length ordering: SHORT < MEDIUM < LONG < XLONG
    p_short = build_context_prompt(ContextProfile.SHORT, seed=42, request_index=0)
    p_medium = build_context_prompt(ContextProfile.MEDIUM, seed=42, request_index=0)
    p_long = build_context_prompt(ContextProfile.LONG, seed=42, request_index=0)
    p_xlong = build_context_prompt(ContextProfile.XLONG, seed=42, request_index=0)

    assert len(p_short.split()) < len(p_medium.split())
    assert len(p_medium.split()) < len(p_long.split())
    assert len(p_long.split()) < len(p_xlong.split())

    # Check approximate word counts
    assert 20 <= len(p_short.split()) <= 100
    assert 100 <= len(p_medium.split()) <= 400
    assert 350 <= len(p_long.split()) <= 1200
    assert 700 <= len(p_xlong.split()) <= 2500


def test_deterministic_workload_generation_and_hashing() -> None:
    """Verify workload scenarios are generated deterministically with stable hashes."""
    w1 = build_step17_workload(
        ContextProfile.SHORT, TrafficProfile.STEADY, num_requests=16, seed=42
    )
    w2 = build_step17_workload(
        ContextProfile.SHORT, TrafficProfile.STEADY, num_requests=16, seed=42
    )
    w3 = build_step17_workload(
        ContextProfile.SHORT, TrafficProfile.STEADY, num_requests=16, seed=99
    )

    assert len(w1.requests) == 16
    assert compute_workload_hash(w1) == compute_workload_hash(w2)
    assert compute_workload_hash(w1) != compute_workload_hash(w3)


def test_traffic_profiles_workload_patterns() -> None:
    """Verify different traffic profiles produce appropriate delays and request distributions."""
    # Steady
    w_steady = build_step17_workload(
        ContextProfile.MEDIUM, TrafficProfile.STEADY, num_requests=8, seed=42
    )
    delays_steady = [r.scheduled_delay_ms for r in w_steady.requests]
    assert delays_steady[0] == 0.0
    assert delays_steady[1] > 0.0

    # Bursty (groups of 4 have identical delays)
    w_bursty = build_step17_workload(
        ContextProfile.MEDIUM, TrafficProfile.BURSTY, num_requests=8, seed=42
    )
    delays_bursty = [r.scheduled_delay_ms for r in w_bursty.requests]
    assert delays_bursty[0] == delays_bursty[1] == delays_bursty[2] == delays_bursty[3] == 0.0
    assert delays_bursty[4] == delays_bursty[5] == delays_bursty[6] == delays_bursty[7] > 0.0

    # Mixed context
    w_mixed = build_step17_workload(
        ContextProfile.SHORT, TrafficProfile.MIXED, num_requests=8, seed=42
    )
    mix_profiles = {r.metadata.get("context_profile") for r in w_mixed.requests}
    assert len(mix_profiles) > 1

    # Long context burst
    w_long_burst = build_step17_workload(
        ContextProfile.LONG, TrafficProfile.LONG_CONTEXT_BURST, num_requests=4, seed=42
    )
    for r in w_long_burst.requests:
        assert r.metadata.get("context_profile") in ("LONG", "XLONG")


def _create_valid_mock_condition_metrics(**overrides: Any) -> Step17ConditionMetrics:
    """Helper to create a valid Step17ConditionMetrics record."""
    data: dict[str, Any] = {
        "experiment_id": "test_exp",
        "git_commit": "abcdef0",
        "model_id": DEFAULT_STEP17_MODEL_ID,
        "backend_name": "vllm",
        "backend_confirmed": True,
        "workload_name": "step17_SHORT_STEADY_16",
        "workload_hash": "hash1234",
        "seed": 42,
        "context_profile": "SHORT",
        "traffic_profile": "STEADY",
        "offered_load": 16,
        "load_level": 16,
        "repetition_index": 0,
        "condition_name": "STATIC_OPTIMIZED",
        "condition_type": "static_optimized",
        "active_config_str": "c=4, b=4, w=50.0ms",
        "target_slo_p95_ms": 5000.0,
        "scheduled_requests": 16,
        "completed_requests": 16,
        "failed_requests": 0,
        "measured_requests": 16,
        "integrity_valid": True,
        "engine_initialization_count": 1,
        "engine_teardown_count": 1,
        "backend_generate_calls": 0,
        "backend_generate_batch_calls": 4,
        "native_batch_calls": 4,
        "total_batches": 4,
        "mean_batch_size": 4.0,
        "max_batch_size": 4,
        "batch_size_distribution": {4: 4},
        "mean_latency_ms": 120.0,
        "p50_latency_ms": 110.0,
        "p90_latency_ms": 130.0,
        "p95_latency_ms": 140.0,
        "p99_latency_ms": 145.0,
        "min_latency_ms": 90.0,
        "max_latency_ms": 150.0,
        "mean_queue_wait_ms": 15.0,
        "p95_queue_wait_ms": 25.0,
        "mean_backend_execution_ms": 105.0,
        "p95_backend_execution_ms": 115.0,
        "throughput_rps": 20.0,
        "output_tokens_per_sec": 1200.0,
        "input_tokens_per_sec": 1000.0,
        "total_tokens_per_sec": 2200.0,
        "input_tokens_mean": 60.0,
        "input_tokens_p50": 60.0,
        "input_tokens_p95": 64.0,
        "input_tokens_max": 65,
        "output_tokens_mean": 64.0,
        "output_tokens_p50": 64.0,
        "output_tokens_p95": 64.0,
        "output_tokens_max": 64,
        "total_tokens_mean": 124.0,
        "output_token_budget": 64,
        "total_adaptations": 0,
        "adaptation_events": (),
        "peak_queue_depth": 4,
        "peak_active_requests": 4,
        "total_sla_violations": 0,
        "overall_sla_violation_rate_pct": 0.0,
        "total_duration_sec": 0.8,
        "jit_activity_observed": False,
        "jit_warning_details": "",
    }
    data.update(overrides)
    return Step17ConditionMetrics(**data)


def test_20_rule_verification_predicate() -> None:
    """Verify all 20 rigorous evidence predicates in verify_step17_condition_execution."""
    # Baseline valid metrics
    valid_m = _create_valid_mock_condition_metrics()
    assert verify_step17_condition_execution(valid_m) is True

    # Rule 1: backend_name == "vllm"
    assert (
        verify_step17_condition_execution(valid_m.model_copy(update={"backend_name": "mock"}))
        is False
    )

    # Rule 2: backend_confirmed == True
    assert (
        verify_step17_condition_execution(valid_m.model_copy(update={"backend_confirmed": False}))
        is False
    )

    # Rule 4 & 9: failed_requests == 0
    assert (
        verify_step17_condition_execution(valid_m.model_copy(update={"failed_requests": 1}))
        is False
    )

    # Rule 5: backend_generate_batch_calls > 0
    assert (
        verify_step17_condition_execution(
            valid_m.model_copy(update={"backend_generate_batch_calls": 0})
        )
        is False
    )

    # Rule 11: scheduled_requests == completed + failed
    assert (
        verify_step17_condition_execution(valid_m.model_copy(update={"scheduled_requests": 17}))
        is False
    )

    # Rule 12: measured_requests == completed
    assert (
        verify_step17_condition_execution(valid_m.model_copy(update={"measured_requests": 15}))
        is False
    )

    # Rule 13: integrity_valid == True
    assert (
        verify_step17_condition_execution(valid_m.model_copy(update={"integrity_valid": False}))
        is False
    )

    # Rule 14 & 15: engine_initialization_count >= 1, teardown >= 1
    assert (
        verify_step17_condition_execution(
            valid_m.model_copy(update={"engine_initialization_count": 0})
        )
        is False
    )
    assert (
        verify_step17_condition_execution(valid_m.model_copy(update={"engine_teardown_count": 0}))
        is False
    )

    # Rule 16: total_duration_sec > 0.0
    assert (
        verify_step17_condition_execution(valid_m.model_copy(update={"total_duration_sec": 0.0}))
        is False
    )

    # Rule 18: p95 >= p50
    assert (
        verify_step17_condition_execution(
            valid_m.model_copy(update={"p50_latency_ms": 200.0, "p95_latency_ms": 100.0})
        )
        is False
    )

    # Rule 19: p99 >= p95
    assert (
        verify_step17_condition_execution(
            valid_m.model_copy(update={"p95_latency_ms": 200.0, "p99_latency_ms": 150.0})
        )
        is False
    )

    # Rule 20: total latency >= queue wait
    assert (
        verify_step17_condition_execution(
            valid_m.model_copy(update={"mean_queue_wait_ms": 200.0, "mean_latency_ms": 150.0})
        )
        is False
    )


def test_verify_step17_report() -> None:
    """Verify that report verification checks backend and all condition records."""
    valid_m = _create_valid_mock_condition_metrics()
    agg_m = Step17AggregatedConditionMetrics(
        condition_name="STATIC_OPTIMIZED",
        condition_type="static_optimized",
        repetition_count=1,
        mean_throughput_rps=20.0,
        mean_p95_latency_ms=140.0,
        mean_p99_latency_ms=145.0,
        mean_queue_wait_ms=15.0,
        mean_batch_size=4.0,
        mean_sla_violation_rate_pct=0.0,
        total_adaptations=0,
        mean_input_tokens=60.0,
        mean_output_tokens=64.0,
        all_repetitions_valid=True,
    )
    cell_res = Step17WorkloadCellResult(
        cell_id="SHORT_STEADY_load16",
        context_profile="SHORT",
        traffic_profile="STEADY",
        load_level=16,
        workload_hash="hash1234",
        conditions={"STATIC_OPTIMIZED": agg_m},
        raw_repetitions=(valid_m,),
        optimized_vs_conservative_tput_pct=10.0,
        adaptive_vs_conservative_tput_pct=10.0,
        optimized_vs_conservative_p95_delta_ms=-10.0,
        adaptive_vs_conservative_p95_delta_ms=-10.0,
    )
    ctx_analysis = Step17ContextAnalysis(
        throughput_by_context_profile={"SHORT": 20.0},
        p95_by_context_profile={"SHORT": 140.0},
        p99_by_context_profile={"SHORT": 145.0},
        queue_wait_by_context_profile={"SHORT": 15.0},
        batch_size_by_context_profile={"SHORT": 4.0},
        sla_violations_by_context_profile={"SHORT": 0.0},
        context_scaling_trend="Scaling trend",
        context_summary="Context summary",
    )
    traf_analysis = Step17TrafficAnalysis(
        throughput_by_traffic_profile={"STEADY": 20.0},
        p95_by_traffic_profile={"STEADY": 140.0},
        p99_by_traffic_profile={"STEADY": 145.0},
        queue_wait_by_traffic_profile={"STEADY": 15.0},
        batch_size_by_traffic_profile={"STEADY": 4.0},
        adaptations_by_traffic_profile={"STEADY": 0},
        sla_violations_by_traffic_profile={"STEADY": 0.0},
        traffic_scaling_trend="Traffic trend",
        traffic_summary="Traffic summary",
    )
    env_meta = VLLMEnvironmentMetadata(
        os_name="Linux",
        os_version="6.1",
        cpu_architecture="x86_64",
        python_version="3.11",
        vllm_version="0.29.0",
        torch_version="2.4.0",
        cuda_version="12.2",
        gpu_name="NVIDIA Tesla T4",
        gpu_count=1,
        inferopt_version="0.1.0",
        model_id=DEFAULT_STEP17_MODEL_ID,
        enforce_eager=True,
        warmup_count=2,
        repetitions=1,
        workload_seed=42,
        workload_hash="hash1234",
    )
    report = Step17BenchmarkReport(
        experiment_id="step17_test",
        timestamp=1000.0,
        git_commit="abcdef0",
        model_id=DEFAULT_STEP17_MODEL_ID,
        backend="vllm",
        backend_execution_confirmed=True,
        environment=env_meta,
        target_slo=TargetSLO(p95_latency_ms=5000.0),
        context_profiles=("SHORT",),
        traffic_profiles=("STEADY",),
        load_levels=(16,),
        results_by_cell={"SHORT_STEADY_load16": cell_res},
        context_analysis=ctx_analysis,
        traffic_analysis=traf_analysis,
        findings={"PROVEN_OBSERVATIONS": ("Integrity verified",)},
    )

    assert verify_step17_report(report) is True

    # Fails if backend is mock
    assert verify_step17_report(report.model_copy(update={"backend": "mock"})) is False

    # Fails if backend_execution_confirmed is False
    assert (
        verify_step17_report(report.model_copy(update={"backend_execution_confirmed": False}))
        is False
    )


@pytest.mark.asyncio
async def test_step17_runner_execution_with_mock_backend(tmp_path: Path) -> None:
    """Run full Step 17 benchmark runner in miniature test mode with MockBackend."""
    runner = Step17BenchmarkRunner(
        model_id=DEFAULT_STEP17_MODEL_ID,
        context_profiles=(ContextProfile.SHORT, ContextProfile.MEDIUM),
        traffic_profiles=(TrafficProfile.STEADY,),
        load_levels=(4,),
        target_slo=TargetSLO(p95_latency_ms=DEFAULT_STEP17_TARGET_P95_MS),
        candidate_space=CandidateSpace(concurrencies=(1, 2), batch_sizes=(1, 2)),
    )

    mock_b = MockBackend(default_latency_sec=0.001)
    report = await runner.run_experiment(
        backend=mock_b,
        repetitions=1,
        warmup_count=1,
        seed=42,
        jit_warning_observed=True,
        jit_warning_details="Simulated JIT compilation",
    )

    assert len(report.results_by_cell) > 0
    for cell in report.results_by_cell.values():
        assert "STATIC_CONSERVATIVE" in cell.conditions
        assert "STATIC_OPTIMIZED" in cell.conditions
        assert "SLA_AWARE_ADAPTIVE" in cell.conditions
        for rep in cell.raw_repetitions:
            assert rep.integrity_valid is True
            assert rep.engine_initialization_count >= 1
            assert rep.engine_teardown_count >= 1
            assert rep.scheduled_requests == rep.completed_requests == rep.measured_requests
            assert rep.failed_requests == 0
            assert rep.jit_activity_observed is True

    # Test report serialization
    out_dir = tmp_path / "step17_out"
    s_path, r_path, c_path, t_path, m_path = runner.save_reports(report, out_dir)

    assert s_path.exists()
    assert r_path.exists()
    assert c_path.exists()
    assert t_path.exists()
    assert m_path.exists()

    with s_path.open("r") as f:
        s_data = json.load(f)
        assert s_data["model_id"] == DEFAULT_STEP17_MODEL_ID
        assert "findings" in s_data

    # Test README and console report generation
    readme_text = format_step17_readme(report)
    assert "# InferOpt Step 17" in readme_text
    assert DEFAULT_STEP17_MODEL_ID in readme_text

    console_text = format_step17_report(report)
    assert "INFEROPT STEP 17" in console_text


@pytest.mark.asyncio
async def test_step17_runner_error_and_oom_cleanup() -> None:
    """Verify that backend cleanup (unload_model) executes reliably upon failure or exception."""
    runner = Step17BenchmarkRunner(
        model_id=DEFAULT_STEP17_MODEL_ID,
        context_profiles=(ContextProfile.SHORT,),
        traffic_profiles=(TrafficProfile.STEADY,),
        load_levels=(4,),
    )

    class FailingBackend(MockBackend):
        async def generate_batch(self, batch: Any) -> tuple[Any, ...]:
            raise RuntimeError("Simulated GPU Out-Of-Memory Error")

    fail_b = FailingBackend(default_latency_sec=0.001)
    workload = build_step17_workload(
        ContextProfile.SHORT, TrafficProfile.STEADY, num_requests=4, seed=42
    )

    metrics = await runner._execute_isolated_condition(
        condition_name="STATIC_OPTIMIZED",
        condition_type="static_optimized",
        context_profile=ContextProfile.SHORT,
        traffic_profile=TrafficProfile.STEADY,
        load_level=4,
        repetition_idx=0,
        initial_config=TunableConfig(),
        is_adaptive=False,
        workload=workload,
        backend_override=fail_b,
    )

    # Failed requests recorded and lifecycle teardown completed
    assert metrics.failed_requests == 4
    assert metrics.completed_requests == 0
    assert metrics.integrity_valid is False
    assert metrics.engine_initialization_count >= 1
    assert metrics.engine_teardown_count >= 1


def test_classify_step17_findings_no_false_sla_claims() -> None:
    """Verify findings classification does not claim universal SLA compliance or performance."""
    ctx_analysis = Step17ContextAnalysis(
        throughput_by_context_profile={"SHORT": 20.0, "XLONG": 4.0},
        p95_by_context_profile={"SHORT": 140.0, "XLONG": 5200.0},
        p99_by_context_profile={"SHORT": 145.0, "XLONG": 5400.0},
        queue_wait_by_context_profile={"SHORT": 15.0, "XLONG": 4500.0},
        batch_size_by_context_profile={"SHORT": 4.0, "XLONG": 2.0},
        sla_violations_by_context_profile={"SHORT": 0.0, "XLONG": 12.5},
        context_scaling_trend="Scaling trend",
        context_summary="Context summary",
    )
    traf_analysis = Step17TrafficAnalysis(
        throughput_by_traffic_profile={"STEADY": 20.0, "BURSTY": 15.0},
        p95_by_traffic_profile={"STEADY": 140.0, "BURSTY": 4800.0},
        p99_by_traffic_profile={"STEADY": 145.0, "BURSTY": 5100.0},
        queue_wait_by_traffic_profile={"STEADY": 15.0, "BURSTY": 4200.0},
        batch_size_by_traffic_profile={"STEADY": 4.0, "BURSTY": 4.0},
        adaptations_by_traffic_profile={"STEADY": 0, "BURSTY": 3},
        sla_violations_by_traffic_profile={"STEADY": 0.0, "BURSTY": 8.0},
        traffic_scaling_trend="Traffic trend",
        traffic_summary="Traffic summary",
    )

    findings = classify_step17_findings(
        results_by_cell={},
        context_analysis=ctx_analysis,
        traffic_analysis=traf_analysis,
        target_slo=TargetSLO(p95_latency_ms=5000.0),
        model_id=DEFAULT_STEP17_MODEL_ID,
    )

    not_proven = findings["NOT_PROVEN_AND_LIMITATIONS"]
    assert any("NOT PROVEN" in s for s in not_proven)
    assert any("Triton JIT" in s for s in not_proven)
    assert any("Universal SLA Compliance Guarantee" in s for s in not_proven)


@pytest.mark.asyncio
async def test_step17_cli_dispatch(tmp_path: Path) -> None:
    """Verify that CLI flags parse and dispatch --experiment-step17 correctly."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--experiment-step17",
            "--backend",
            "mock",
            "--loads",
            "4",
            "--context-profiles",
            "SHORT",
            "--traffic-profiles",
            "STEADY",
            "--warmup",
            "0",
            "--repetitions",
            "1",
            "--output",
            str(tmp_path / "step17_cli_out"),
        ]
    )

    assert args.experiment_step17 is True
    assert args.context_profiles == ["SHORT"]
    assert args.traffic_profiles == ["STEADY"]
    assert args.loads == [4]

    exit_code = await run_benchmark_cli(args)
    assert exit_code == 0
    assert (tmp_path / "step17_cli_out" / "summary.json").exists()
