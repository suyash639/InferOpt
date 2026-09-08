"""End-to-end integration tests for closed-loop AdaptiveController and Scheduler."""

import asyncio

import pytest

from inferopt.backends.mock import MockBackend
from inferopt.benchmarks.models import BenchmarkResult, WorkloadConfig
from inferopt.core.models import InferenceRequest
from inferopt.optimizer.adaptation_models import (
    AdaptationDecisionType,
    AdaptationPolicy,
)
from inferopt.optimizer.controller import AdaptiveController
from inferopt.optimizer.models import (
    ObjectiveConfig,
    OptimizationObjectiveType,
    TunableConfig,
)
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.scheduler.scheduler import Scheduler
from inferopt.telemetry.models import MetricsSnapshot


def make_candidate_result(
    concurrency: int,
    batch_size: int,
    batch_wait_ms: float,
    requests_per_sec: float,
    p95_latency_ms: float,
) -> BenchmarkResult:
    """Helper to construct candidate BenchmarkResult."""
    sched_cfg = SchedulerConfig(
        max_concurrency=concurrency,
        batch_config=BatchConfig(max_batch_size=batch_size, batch_wait_ms=batch_wait_ms),
    )
    return BenchmarkResult(
        benchmark_id=f"cand-{concurrency}-{batch_size}",
        scenario_name="integration",
        workload_config=WorkloadConfig(scenario_name="integration"),
        backend_name="mock",
        scheduler_config=sched_cfg,
        batch_config=sched_cfg.batch_config,
        telemetry_snapshot=MetricsSnapshot(),
        duration_sec=1.0,
        total_requests=100,
        completed_requests=100,
        failed_requests=0,
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
        peak_queue_depth=5,
        peak_active_requests=concurrency,
        total_batches=20,
        avg_batch_size=float(batch_size),
        max_batch_size=batch_size,
    )


@pytest.mark.asyncio
class TestAdaptiveIntegration:
    """Integration test suite for closed-loop adaptation on active schedulers."""

    async def test_closed_loop_adaptation_with_live_scheduler(self) -> None:
        backend = MockBackend(default_latency_sec=0.010)
        initial_cfg = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=1, batch_wait_ms=0.0),
        )
        scheduler = Scheduler(backend=backend, config=initial_cfg)
        await scheduler.start()

        policy = AdaptationPolicy(
            objective=ObjectiveConfig(objective_type=OptimizationObjectiveType.THROUGHPUT),
            min_improvement_pct=10.0,
            cooldown_windows=1,
            min_completed_requests=5,
            min_completed_batches=2,
        )
        controller = AdaptiveController(policy=policy, scheduler=scheduler)

        try:
            # 1. Run baseline phase
            tasks1 = [
                asyncio.create_task(
                    scheduler.submit(
                        InferenceRequest(
                            request_id=f"base-{i}",
                            model="mock-model",
                            prompt="Test",
                            max_tokens=10,
                        )
                    )
                )
                for i in range(6)
            ]
            responses1 = await asyncio.gather(*tasks1)
            assert len(responses1) == 6

            # 2. Capture live snapshot and step controller with candidates
            snapshot1 = scheduler.collector.snapshot()
            candidates = [
                make_candidate_result(
                    concurrency=1,
                    batch_size=1,
                    batch_wait_ms=0.0,
                    requests_per_sec=15.0,
                    p95_latency_ms=12.0,
                ),
                make_candidate_result(
                    concurrency=2,
                    batch_size=4,
                    batch_wait_ms=5.0,
                    requests_per_sec=200.0,  # candidate improves throughput significantly
                    p95_latency_ms=22.0,
                ),
            ]

            decision = controller.step(snapshot1, candidate_evidence=candidates)
            assert decision.decision_type == AdaptationDecisionType.APPLY
            assert decision.is_applied is True
            assert decision.proposed_config == TunableConfig(
                max_concurrency=2, max_batch_size=4, batch_wait_ms=5.0
            )

            # Verify scheduler configuration was updated live
            assert scheduler.config.max_concurrency == 2
            assert scheduler.batch_config.max_batch_size == 4
            assert scheduler.batch_config.batch_wait_ms == 5.0

            # 3. Submit adapted request stream
            tasks2 = [
                asyncio.create_task(
                    scheduler.submit(
                        InferenceRequest(
                            request_id=f"adapt-{i}",
                            model="mock-model",
                            prompt="Test",
                            max_tokens=10,
                        )
                    )
                )
                for i in range(8)
            ]
            responses2 = await asyncio.gather(*tasks2)
            assert len(responses2) == 8
            assert all(r.generated_text != "" for r in responses2)

            # 4. Verify telemetry recorded the adaptation event
            adapt_events = scheduler.collector.recent_adaptations()
            assert len(adapt_events) >= 1
            assert adapt_events[-1].decision == "APPLY"
            assert adapt_events[-1].is_applied is True
            assert adapt_events[-1].new_config is not None
            assert adapt_events[-1].new_config["max_concurrency"] == 2
            assert adapt_events[-1].new_config["max_batch_size"] == 4
        finally:
            await scheduler.shutdown()
