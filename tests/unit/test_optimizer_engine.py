"""Unit tests for DeterministicOptimizer ranking, tie-breaking, and experiment planning."""

from inferopt.benchmarks.models import BenchmarkResult, WorkloadConfig
from inferopt.optimizer.engine import DeterministicOptimizer
from inferopt.optimizer.models import (
    CandidateSpace,
    ObjectiveConfig,
    OptimizationConstraints,
    OptimizationObjectiveType,
    TunableConfig,
)
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.telemetry.models import MetricsSnapshot


def make_result(
    concurrency: int,
    batch_size: int,
    batch_wait_ms: float,
    requests_per_sec: float,
    p95_latency_ms: float,
    failed_requests: int = 0,
) -> BenchmarkResult:
    """Helper to construct synthetic BenchmarkResult."""
    sched_cfg = SchedulerConfig(
        max_concurrency=concurrency,
        batch_config=BatchConfig(max_batch_size=batch_size, batch_wait_ms=batch_wait_ms),
    )
    return BenchmarkResult(
        benchmark_id=f"b-{concurrency}-{batch_size}-{batch_wait_ms}",
        scenario_name="eval-scen",
        workload_config=WorkloadConfig(scenario_name="eval-scen"),
        backend_name="mock",
        scheduler_config=sched_cfg,
        batch_config=sched_cfg.batch_config,
        telemetry_snapshot=MetricsSnapshot(),
        duration_sec=1.0,
        total_requests=100,
        completed_requests=100 - failed_requests,
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


class TestDeterministicOptimizer:
    """Tests for deterministic scoring, ranking, constraint filtering, and tie-breaking."""

    def test_throughput_maximization(self) -> None:
        optimizer = DeterministicOptimizer()
        results = [
            make_result(
                concurrency=1,
                batch_size=1,
                batch_wait_ms=0.0,
                requests_per_sec=20.0,
                p95_latency_ms=10.0,
            ),
            make_result(
                concurrency=2,
                batch_size=4,
                batch_wait_ms=5.0,
                requests_per_sec=80.0,
                p95_latency_ms=35.0,
            ),
            make_result(
                concurrency=4,
                batch_size=8,
                batch_wait_ms=10.0,
                requests_per_sec=60.0,
                p95_latency_ms=45.0,
            ),
        ]
        objective = ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT)
        opt_res = optimizer.evaluate_results(results, objective)

        assert opt_res.is_feasible is True
        assert opt_res.recommended_config == TunableConfig(
            max_concurrency=2, max_batch_size=4, batch_wait_ms=5.0
        )
        assert opt_res.best_score == 80.0
        assert opt_res.total_candidates == 3
        assert opt_res.feasible_candidates == 3

    def test_latency_minimization(self) -> None:
        optimizer = DeterministicOptimizer()
        results = [
            make_result(
                concurrency=1,
                batch_size=1,
                batch_wait_ms=0.0,
                requests_per_sec=20.0,
                p95_latency_ms=12.0,
            ),
            make_result(
                concurrency=2,
                batch_size=4,
                batch_wait_ms=5.0,
                requests_per_sec=80.0,
                p95_latency_ms=35.0,
            ),
            make_result(
                concurrency=4,
                batch_size=8,
                batch_wait_ms=10.0,
                requests_per_sec=60.0,
                p95_latency_ms=45.0,
            ),
        ]
        objective = ObjectiveConfig(objective_type=OptimizationObjectiveType.LATENCY)
        opt_res = optimizer.evaluate_results(results, objective)

        assert opt_res.is_feasible is True
        assert opt_res.recommended_config == TunableConfig(
            max_concurrency=1, max_batch_size=1, batch_wait_ms=0.0
        )
        assert opt_res.best_score == -12.0

    def test_balanced_optimization(self) -> None:
        optimizer = DeterministicOptimizer()
        results = [
            make_result(
                concurrency=1,
                batch_size=1,
                batch_wait_ms=0.0,
                requests_per_sec=20.0,
                p95_latency_ms=10.0,
            ),
            make_result(
                concurrency=4,
                batch_size=8,
                batch_wait_ms=10.0,
                requests_per_sec=100.0,
                p95_latency_ms=100.0,
            ),
        ]
        # Equal weights 0.5, target_throughput=100.0, target_p95_latency_ms=100.0
        # Candidate 1: 0.5 * (20/100) - 0.5 * (10/100) = 0.10 - 0.05 = 0.05
        # Candidate 2: 0.5 * (100/100) - 0.5 * (100/100) = 0.50 - 0.50 = 0.00
        # Candidate 1 wins under balanced!
        objective = ObjectiveConfig(
            objective_type=OptimizationObjectiveType.BALANCED,
            throughput_weight=0.5,
            latency_weight=0.5,
            target_throughput_rps=100.0,
            target_p95_latency_ms=100.0,
        )
        opt_res = optimizer.evaluate_results(results, objective)
        assert opt_res.is_feasible is True
        assert opt_res.recommended_config == TunableConfig(
            max_concurrency=1, max_batch_size=1, batch_wait_ms=0.0
        )
        assert round(opt_res.best_score or 0.0, 4) == 0.05

    def test_constraint_filtering(self) -> None:
        optimizer = DeterministicOptimizer()
        # Candidate 2 has highest throughput (80 rps), but violates max_p95 constraint (35ms > 30ms)
        # Candidate 3 has 60 rps and 25ms -> should win!
        results = [
            make_result(
                concurrency=1,
                batch_size=1,
                batch_wait_ms=0.0,
                requests_per_sec=20.0,
                p95_latency_ms=10.0,
            ),
            make_result(
                concurrency=2,
                batch_size=4,
                batch_wait_ms=5.0,
                requests_per_sec=80.0,
                p95_latency_ms=35.0,
            ),
            make_result(
                concurrency=4,
                batch_size=8,
                batch_wait_ms=10.0,
                requests_per_sec=60.0,
                p95_latency_ms=25.0,
            ),
        ]
        objective = ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT)
        constraints = OptimizationConstraints(max_p95_latency_ms=30.0)
        opt_res = optimizer.evaluate_results(results, objective, constraints=constraints)

        assert opt_res.is_feasible is True
        assert opt_res.recommended_config == TunableConfig(
            max_concurrency=4, max_batch_size=8, batch_wait_ms=10.0
        )
        assert opt_res.best_score == 60.0
        assert opt_res.total_candidates == 3
        assert opt_res.feasible_candidates == 2

    def test_all_candidates_infeasible(self) -> None:
        optimizer = DeterministicOptimizer()
        results = [
            make_result(
                concurrency=1,
                batch_size=1,
                batch_wait_ms=0.0,
                requests_per_sec=20.0,
                p95_latency_ms=50.0,
            ),
            make_result(
                concurrency=2,
                batch_size=4,
                batch_wait_ms=5.0,
                requests_per_sec=80.0,
                p95_latency_ms=75.0,
            ),
        ]
        objective = ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT)
        constraints = OptimizationConstraints(max_p95_latency_ms=30.0)  # strict SLA
        opt_res = optimizer.evaluate_results(results, objective, constraints=constraints)

        assert opt_res.is_feasible is False
        assert opt_res.recommended_config is None
        assert opt_res.best_score is None
        assert opt_res.feasible_candidates == 0
        assert "Optimization failed" in opt_res.summary_explanation

    def test_deterministic_tie_breaking(self) -> None:
        optimizer = DeterministicOptimizer()
        # Two candidates have identical throughput (50 rps). Candidate 1 has lower p95 -> wins!
        results = [
            make_result(
                concurrency=2,
                batch_size=4,
                batch_wait_ms=0.0,
                requests_per_sec=50.0,
                p95_latency_ms=20.0,
            ),
            make_result(
                concurrency=2,
                batch_size=2,
                batch_wait_ms=0.0,
                requests_per_sec=50.0,
                p95_latency_ms=15.0,
            ),
        ]
        objective = ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT)
        opt_res = optimizer.evaluate_results(results, objective)
        assert opt_res.recommended_config == TunableConfig(
            max_concurrency=2, max_batch_size=2, batch_wait_ms=0.0
        )

        # Identical throughput and identical p95, but different batch_wait_ms -> smaller wait wins
        results_wait = [
            make_result(
                concurrency=2,
                batch_size=4,
                batch_wait_ms=10.0,
                requests_per_sec=50.0,
                p95_latency_ms=20.0,
            ),
            make_result(
                concurrency=2,
                batch_size=4,
                batch_wait_ms=5.0,
                requests_per_sec=50.0,
                p95_latency_ms=20.0,
            ),
        ]
        opt_res_wait = optimizer.evaluate_results(results_wait, objective)
        assert opt_res_wait.recommended_config == TunableConfig(
            max_concurrency=2, max_batch_size=4, batch_wait_ms=5.0
        )

    def test_create_experiment_plan(self) -> None:
        optimizer = DeterministicOptimizer()
        space = CandidateSpace(
            concurrencies=(1, 2),
            batch_sizes=(2, 4),
            batch_waits_ms=(0.0, 5.0),
        )
        plan = optimizer.create_experiment_plan(space)
        assert len(plan) == 2 * 2 * 2  # 8
        assert all(isinstance(cfg, SchedulerConfig) for cfg in plan)
        assert plan[0].max_concurrency == 1
        assert plan[0].batch_config.max_batch_size == 2
        assert plan[0].batch_config.batch_wait_ms == 0.0

    def test_empty_results_handling(self) -> None:
        optimizer = DeterministicOptimizer()
        objective = ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT)
        opt_res = optimizer.evaluate_results([], objective)
        assert opt_res.is_feasible is False
        assert opt_res.total_candidates == 0
        assert opt_res.recommended_config is None
