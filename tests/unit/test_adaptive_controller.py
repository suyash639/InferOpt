"""Unit tests for closed-loop AdaptiveController policies, gates, and rollbacks."""

import pytest

from inferopt.benchmarks.models import BenchmarkResult, WorkloadConfig
from inferopt.optimizer.adaptation_models import (
    AdaptationDecisionType,
    AdaptationPolicy,
)
from inferopt.optimizer.controller import AdaptiveController
from inferopt.optimizer.models import (
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
    completed_requests: int = 50,
    total_batches: int = 10,
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
        total_requests=completed_requests + failed_requests,
        completed_requests=completed_requests,
        failed_requests=failed_requests,
        cancelled_requests=0,
        requests_per_sec=requests_per_sec,
        batches_per_sec=float(total_batches),
        tokens_per_sec=500.0,
        avg_latency_ms=p95_latency_ms * 0.8,
        p50_latency_ms=p95_latency_ms * 0.7,
        p95_latency_ms=p95_latency_ms,
        p99_latency_ms=p95_latency_ms * 1.2,
        avg_queue_wait_ms=2.0,
        avg_execution_ms=p95_latency_ms * 0.7,
        peak_queue_depth=10,
        peak_active_requests=concurrency,
        total_batches=total_batches,
        avg_batch_size=float(batch_size),
        max_batch_size=batch_size,
    )


class TestAdaptationPolicy:
    """Tests for policy validation and bounds checks."""

    def test_default_policy(self) -> None:
        policy = AdaptationPolicy()
        assert policy.min_improvement_pct == 5.0
        assert policy.cooldown_windows == 2
        assert policy.min_completed_requests == 20
        assert policy.min_completed_batches == 5

    def test_invalid_bounds_raise_error(self) -> None:
        with pytest.raises(ValueError, match="min_concurrency"):
            AdaptationPolicy(min_concurrency=8, max_concurrency=4)

        with pytest.raises(ValueError, match="min_batch_size"):
            AdaptationPolicy(min_batch_size=16, max_batch_size=4)

        with pytest.raises(ValueError, match="min_batch_wait_ms"):
            AdaptationPolicy(min_batch_wait_ms=50.0, max_batch_wait_ms=10.0)

    def test_is_within_bounds(self) -> None:
        policy = AdaptationPolicy(
            min_concurrency=2,
            max_concurrency=8,
            min_batch_size=2,
            max_batch_size=8,
            min_batch_wait_ms=0.0,
            max_batch_wait_ms=20.0,
        )
        ok, _ = policy.is_within_bounds(
            TunableConfig(max_concurrency=4, max_batch_size=4, batch_wait_ms=5.0)
        )
        assert ok is True

        low_conc, _ = policy.is_within_bounds(
            TunableConfig(max_concurrency=1, max_batch_size=4, batch_wait_ms=5.0)
        )
        assert low_conc is False


class TestAdaptiveControllerEvaluationWindows:
    """Tests for evaluation window sufficiency thresholds."""

    def test_insufficient_requests(self) -> None:
        policy = AdaptationPolicy(min_completed_requests=30, min_completed_batches=5)
        controller = AdaptiveController(policy=policy)
        evidence = make_result(
            concurrency=2,
            batch_size=4,
            batch_wait_ms=0.0,
            requests_per_sec=50.0,
            p95_latency_ms=20.0,
            completed_requests=15,  # below 30
            total_batches=8,
        )
        decision = controller.evaluate(evidence)
        assert decision.decision_type == AdaptationDecisionType.INSUFFICIENT_DATA
        assert "15/30 completed requests" in decision.reason

    def test_insufficient_batches(self) -> None:
        policy = AdaptationPolicy(min_completed_requests=10, min_completed_batches=10)
        controller = AdaptiveController(policy=policy)
        evidence = make_result(
            concurrency=2,
            batch_size=4,
            batch_wait_ms=0.0,
            requests_per_sec=50.0,
            p95_latency_ms=20.0,
            completed_requests=25,
            total_batches=3,  # below 10
        )
        decision = controller.evaluate(evidence)
        assert decision.decision_type == AdaptationDecisionType.INSUFFICIENT_DATA
        assert "3/10 batches" in decision.reason


class TestAdaptiveControllerGating:
    """Tests for policy gating: improvement thresholds, cooldowns, constraints, and bounds."""

    def test_improvement_above_threshold_applies(self) -> None:
        policy = AdaptationPolicy(
            objective=ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT),
            min_improvement_pct=10.0,
            min_completed_requests=10,
            min_completed_batches=2,
        )
        controller = AdaptiveController(policy=policy)
        current = make_result(
            concurrency=2,
            batch_size=2,
            batch_wait_ms=0.0,
            requests_per_sec=40.0,
            p95_latency_ms=15.0,
        )
        candidates = [
            make_result(
                concurrency=4,
                batch_size=8,
                batch_wait_ms=5.0,
                requests_per_sec=60.0,  # 50% improvement
                p95_latency_ms=25.0,
            )
        ]
        decision = controller.evaluate(current, candidate_evidence=candidates)
        assert decision.decision_type == AdaptationDecisionType.APPLY
        assert decision.proposed_config == TunableConfig(
            max_concurrency=4, max_batch_size=8, batch_wait_ms=5.0
        )
        assert decision.improvement_pct == 50.0
        assert "improves THROUGHPUT by 50.00%" in decision.reason

    def test_improvement_below_threshold_rejects(self) -> None:
        policy = AdaptationPolicy(
            objective=ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT),
            min_improvement_pct=20.0,
            min_completed_requests=10,
            min_completed_batches=2,
        )
        controller = AdaptiveController(policy=policy)
        current = make_result(
            concurrency=2,
            batch_size=2,
            batch_wait_ms=0.0,
            requests_per_sec=50.0,
            p95_latency_ms=15.0,
        )
        candidates = [
            make_result(
                concurrency=4,
                batch_size=4,
                batch_wait_ms=0.0,
                requests_per_sec=52.0,  # only 4% improvement
                p95_latency_ms=16.0,
            )
        ]
        decision = controller.evaluate(current, candidate_evidence=candidates)
        assert decision.decision_type == AdaptationDecisionType.REJECT
        assert "below required threshold" in decision.reason

    def test_cooldown_suppression(self) -> None:
        policy = AdaptationPolicy(
            objective=ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT),
            min_improvement_pct=5.0,
            cooldown_windows=2,
            min_completed_requests=10,
            min_completed_batches=2,
        )
        controller = AdaptiveController(policy=policy)
        current = make_result(
            concurrency=1,
            batch_size=1,
            batch_wait_ms=0.0,
            requests_per_sec=20.0,
            p95_latency_ms=10.0,
        )
        candidates = [
            make_result(
                concurrency=4,
                batch_size=4,
                batch_wait_ms=5.0,
                requests_per_sec=50.0,
                p95_latency_ms=20.0,
            )
        ]

        # 1. First evaluation accepted and applied
        dec1 = controller.evaluate(current, candidate_evidence=candidates)
        assert dec1.decision_type == AdaptationDecisionType.APPLY
        applied = controller.apply_decision(dec1)
        assert applied is True
        assert controller.cooldown_remaining == 2

        # 2. Window 2: Suppressed by cooldown (cooldown decrements 2 -> 1)
        dec2 = controller.evaluate(current, candidate_evidence=candidates)
        assert dec2.decision_type == AdaptationDecisionType.COOLDOWN
        assert controller.cooldown_remaining == 1

        # 3. Window 3: Suppressed by cooldown (cooldown decrements 1 -> 0)
        dec3 = controller.evaluate(current, candidate_evidence=candidates)
        assert dec3.decision_type == AdaptationDecisionType.COOLDOWN
        assert controller.cooldown_remaining == 0

        # 4. Window 4: Cooldown expired, can evaluate again!
        dec4 = controller.evaluate(current, candidate_evidence=candidates)
        assert dec4.decision_type == AdaptationDecisionType.APPLY

    def test_latency_minimization_objective(self) -> None:
        policy = AdaptationPolicy(
            objective=ObjectiveConfig(objective_type=OptimizationObjectiveType.LATENCY),
            min_improvement_pct=10.0,
            min_completed_requests=10,
            min_completed_batches=2,
        )
        controller = AdaptiveController(policy=policy)
        current = make_result(
            concurrency=4,
            batch_size=8,
            batch_wait_ms=10.0,
            requests_per_sec=60.0,
            p95_latency_ms=40.0,
        )
        candidates = [
            make_result(
                concurrency=2,
                batch_size=2,
                batch_wait_ms=0.0,
                requests_per_sec=30.0,
                p95_latency_ms=20.0,  # 50% latency reduction
            )
        ]
        decision = controller.evaluate(current, candidate_evidence=candidates)
        assert decision.decision_type == AdaptationDecisionType.APPLY
        assert decision.proposed_config == TunableConfig(
            max_concurrency=2, max_batch_size=2, batch_wait_ms=0.0
        )
        assert decision.improvement_pct == 50.0

    def test_infeasible_candidate_rejected(self) -> None:
        policy = AdaptationPolicy(
            objective=ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT),
            constraints=OptimizationConstraints(max_p95_latency_ms=25.0),
            min_completed_requests=10,
            min_completed_batches=2,
        )
        controller = AdaptiveController(policy=policy)
        current = make_result(
            concurrency=2,
            batch_size=2,
            batch_wait_ms=0.0,
            requests_per_sec=30.0,
            p95_latency_ms=20.0,
        )
        # Candidate has high throughput but violates max_p95_latency_ms SLA (40ms > 25ms)
        candidates = [
            make_result(
                concurrency=8,
                batch_size=16,
                batch_wait_ms=10.0,
                requests_per_sec=120.0,
                p95_latency_ms=40.0,
            )
        ]
        decision = controller.evaluate(current, candidate_evidence=candidates)
        assert decision.decision_type == AdaptationDecisionType.INFEASIBLE


class TestAntiOscillationAndStability:
    """Tests proving that noisy metrics and rapid switching are suppressed."""

    def test_anti_oscillation_under_noisy_metrics(self) -> None:
        policy = AdaptationPolicy(
            objective=ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT),
            min_improvement_pct=10.0,
            cooldown_windows=3,
            min_completed_requests=10,
            min_completed_batches=2,
        )
        controller = AdaptiveController(policy=policy)

        cfg_a = make_result(2, 2, 0.0, requests_per_sec=40.0, p95_latency_ms=20.0)
        cfg_b = make_result(4, 4, 5.0, requests_per_sec=60.0, p95_latency_ms=25.0)

        # Window 1: Switch A -> B
        dec1 = controller.step(cfg_a, candidate_evidence=[cfg_b])
        assert dec1.decision_type == AdaptationDecisionType.APPLY
        assert dec1.proposed_config == TunableConfig(
            max_concurrency=4, max_batch_size=4, batch_wait_ms=5.0
        )

        # Windows 2, 3, 4: Fluctuating noise attempts to switch back B -> A.
        # All are suppressed by cooldown!
        cfg_b_noise = make_result(4, 4, 5.0, requests_per_sec=45.0, p95_latency_ms=25.0)
        cfg_a_noise = make_result(2, 2, 0.0, requests_per_sec=50.0, p95_latency_ms=20.0)

        for _ in range(3):
            dec_noise = controller.step(cfg_b_noise, candidate_evidence=[cfg_a_noise])
            assert dec_noise.decision_type == AdaptationDecisionType.COOLDOWN


class TestRollbackMechanisms:
    """Tests for automated rollback to known-good configurations on SLA degradation."""

    def test_rollback_on_sla_constraint_violation(self) -> None:
        policy = AdaptationPolicy(
            objective=ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT),
            constraints=OptimizationConstraints(max_p95_latency_ms=30.0),
            enable_rollback=True,
            min_completed_requests=10,
            min_completed_batches=2,
        )
        controller = AdaptiveController(policy=policy)

        # 1. Baseline known-good config A (20 RPS, 15ms)
        cfg_a = make_result(1, 1, 0.0, requests_per_sec=20.0, p95_latency_ms=15.0)
        cfg_b = make_result(4, 8, 10.0, requests_per_sec=80.0, p95_latency_ms=25.0)

        dec1 = controller.step(cfg_a, candidate_evidence=[cfg_b])
        assert dec1.decision_type == AdaptationDecisionType.APPLY

        # 2. In live deployment, config B encounters overload (p95 jumps to 55ms > 30ms SLA)
        cfg_b_degraded = make_result(4, 8, 10.0, requests_per_sec=70.0, p95_latency_ms=55.0)
        dec_rollback = controller.evaluate(cfg_b_degraded)
        assert dec_rollback.decision_type == AdaptationDecisionType.ROLLBACK
        assert dec_rollback.proposed_config == TunableConfig(
            max_concurrency=1, max_batch_size=1, batch_wait_ms=0.0
        )
        assert "violated constraints" in dec_rollback.reason

    def test_rollback_on_severe_relative_degradation(self) -> None:
        policy = AdaptationPolicy(
            objective=ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT),
            enable_rollback=True,
            rollback_degradation_pct=25.0,
            min_completed_requests=10,
            min_completed_batches=2,
        )
        controller = AdaptiveController(policy=policy)

        cfg_a = make_result(2, 2, 0.0, requests_per_sec=100.0, p95_latency_ms=15.0)
        cfg_b = make_result(4, 4, 5.0, requests_per_sec=120.0, p95_latency_ms=20.0)

        # Apply config B
        dec1 = controller.step(cfg_a, candidate_evidence=[cfg_b])
        assert dec1.decision_type == AdaptationDecisionType.APPLY

        # Config B collapses to 50 RPS (50% drop below baseline 100 RPS)
        cfg_b_collapsed = make_result(4, 4, 5.0, requests_per_sec=50.0, p95_latency_ms=20.0)
        dec_rollback = controller.evaluate(cfg_b_collapsed)
        assert dec_rollback.decision_type == AdaptationDecisionType.ROLLBACK
        assert dec_rollback.proposed_config == TunableConfig(
            max_concurrency=2, max_batch_size=2, batch_wait_ms=0.0
        )
        assert "degraded score by 50.0%" in dec_rollback.reason
