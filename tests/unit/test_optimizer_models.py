"""Unit tests for optimizer domain models, candidate spaces, objectives, and constraints."""

import pytest
from pydantic import ValidationError

from inferopt.benchmarks.models import BenchmarkResult, WorkloadConfig
from inferopt.optimizer.models import (
    CandidateSpace,
    ObjectiveConfig,
    OptimizationConstraints,
    OptimizationObjectiveType,
    TunableConfig,
)
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.telemetry.models import MetricsSnapshot


def make_dummy_benchmark_result(
    concurrency: int = 4,
    batch_size: int = 4,
    batch_wait_ms: float = 0.0,
    requests_per_sec: float = 50.0,
    p95_latency_ms: float = 20.0,
    failed_requests: int = 0,
) -> BenchmarkResult:
    """Helper to construct synthetic BenchmarkResult without running inference."""
    sched_cfg = SchedulerConfig(
        max_concurrency=concurrency,
        batch_config=BatchConfig(max_batch_size=batch_size, batch_wait_ms=batch_wait_ms),
    )
    return BenchmarkResult(
        benchmark_id="b-test",
        scenario_name="test-scen",
        workload_config=WorkloadConfig(scenario_name="test-scen"),
        backend_name="mock",
        scheduler_config=sched_cfg,
        batch_config=sched_cfg.batch_config,
        telemetry_snapshot=MetricsSnapshot(),
        duration_sec=1.0,
        total_requests=50,
        completed_requests=50 - failed_requests,
        failed_requests=failed_requests,
        cancelled_requests=0,
        requests_per_sec=requests_per_sec,
        batches_per_sec=10.0,
        tokens_per_sec=500.0,
        avg_latency_ms=p95_latency_ms * 0.8,
        p50_latency_ms=p95_latency_ms * 0.7,
        p95_latency_ms=p95_latency_ms,
        p99_latency_ms=p95_latency_ms * 1.2,
        avg_queue_wait_ms=2.0,
        avg_execution_ms=p95_latency_ms * 0.7,
        peak_queue_depth=10,
        peak_active_requests=concurrency,
        total_batches=10,
        avg_batch_size=float(batch_size),
        max_batch_size=batch_size,
    )


class TestTunableConfig:
    """Tests for TunableConfig validation, immutability, and conversions."""

    def test_valid_creation_and_conversion(self) -> None:
        cfg = TunableConfig(max_concurrency=2, max_batch_size=8, batch_wait_ms=10.0)
        assert cfg.max_concurrency == 2
        assert cfg.max_batch_size == 8
        assert cfg.batch_wait_ms == 10.0

        sched_cfg = cfg.to_scheduler_config(max_queue_size=250, max_history_size=500)
        assert sched_cfg.max_concurrency == 2
        assert sched_cfg.batch_config.max_batch_size == 8
        assert sched_cfg.batch_config.batch_wait_ms == 10.0
        assert sched_cfg.max_queue_size == 250
        assert sched_cfg.max_history_size == 500

        restored = TunableConfig.from_scheduler_config(sched_cfg)
        assert restored == cfg

    def test_immutability(self) -> None:
        cfg = TunableConfig()
        with pytest.raises(ValidationError):
            cfg.max_concurrency = 8

    def test_invalid_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TunableConfig(max_concurrency=0)
        with pytest.raises(ValidationError):
            TunableConfig(max_batch_size=0)
        with pytest.raises(ValidationError):
            TunableConfig(batch_wait_ms=-1.0)


class TestCandidateSpace:
    """Tests for CandidateSpace Cartesian product generation."""

    def test_default_candidate_space(self) -> None:
        space = CandidateSpace()
        assert space.concurrencies == (1, 2, 4)
        assert space.batch_sizes == (1, 2, 4, 8)
        assert space.batch_waits_ms == (0.0, 5.0, 10.0, 25.0)
        assert space.total_candidates == 3 * 4 * 4  # 48

        candidates = space.generate_candidates()
        assert len(candidates) == 48
        assert candidates[0] == TunableConfig(
            max_concurrency=1, max_batch_size=1, batch_wait_ms=0.0
        )
        assert candidates[-1] == TunableConfig(
            max_concurrency=4, max_batch_size=8, batch_wait_ms=25.0
        )

    def test_deduplication_and_sorting(self) -> None:
        space = CandidateSpace(
            concurrencies=(4, 1, 2, 2),
            batch_sizes=(8, 2, 4, 4),
            batch_waits_ms=(10.0, 0.0, 0.0),
        )
        assert space.concurrencies == (1, 2, 4)
        assert space.batch_sizes == (2, 4, 8)
        assert space.batch_waits_ms == (0.0, 10.0)
        assert space.total_candidates == 3 * 3 * 2  # 18

    def test_invalid_candidate_space(self) -> None:
        with pytest.raises(ValidationError):
            CandidateSpace(concurrencies=())
        with pytest.raises(ValidationError):
            CandidateSpace(concurrencies=(0, 2))
        with pytest.raises(ValidationError):
            CandidateSpace(batch_waits_ms=(-5.0,))


class TestObjectiveConfig:
    """Tests for ObjectiveConfig validation."""

    def test_throughput_objective(self) -> None:
        obj = ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT)
        assert obj.objective_type == OptimizationObjectiveType.THROUGHPUT

    def test_balanced_objective_weights(self) -> None:
        obj = ObjectiveConfig(
            objective_type=OptimizationObjectiveType.BALANCED,
            throughput_weight=0.7,
            latency_weight=0.3,
        )
        assert obj.throughput_weight == 0.7
        assert obj.latency_weight == 0.3

    def test_invalid_weights_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ObjectiveConfig(throughput_weight=0.0, latency_weight=0.0)


class TestOptimizationConstraints:
    """Tests for SLA constraint evaluation and violation reporting."""

    def test_feasible_candidate(self) -> None:
        constraints = OptimizationConstraints(
            max_p95_latency_ms=50.0,
            min_throughput_rps=20.0,
            max_batch_size=8,
            max_concurrency=4,
        )
        cfg = TunableConfig(max_concurrency=2, max_batch_size=4, batch_wait_ms=0.0)
        res = make_dummy_benchmark_result(
            concurrency=2,
            batch_size=4,
            requests_per_sec=40.0,
            p95_latency_ms=30.0,
        )
        violations = constraints.evaluate(cfg, res)
        assert len(violations) == 0

    def test_p95_latency_violation(self) -> None:
        constraints = OptimizationConstraints(max_p95_latency_ms=25.0)
        cfg = TunableConfig(max_concurrency=4, max_batch_size=8)
        res = make_dummy_benchmark_result(p95_latency_ms=35.0)
        violations = constraints.evaluate(cfg, res)
        assert len(violations) == 1
        assert "exceeds max" in violations[0]

    def test_throughput_violation(self) -> None:
        constraints = OptimizationConstraints(min_throughput_rps=100.0)
        cfg = TunableConfig(max_concurrency=1, max_batch_size=1)
        res = make_dummy_benchmark_result(requests_per_sec=45.0)
        violations = constraints.evaluate(cfg, res)
        assert len(violations) == 1
        assert "is below min" in violations[0]

    def test_multiple_violations(self) -> None:
        constraints = OptimizationConstraints(
            max_p95_latency_ms=10.0,
            min_throughput_rps=100.0,
            max_batch_size=2,
            max_failed_requests=0,
        )
        cfg = TunableConfig(max_concurrency=4, max_batch_size=8)
        res = make_dummy_benchmark_result(
            p95_latency_ms=25.0,
            requests_per_sec=30.0,
            failed_requests=2,
        )
        violations = constraints.evaluate(cfg, res)
        assert len(violations) == 4  # p95, throughput, batch_size, failed_requests
