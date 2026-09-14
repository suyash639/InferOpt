"""Unit tests for SLA-aware online adaptive controller and deadband hysteresis."""

import pytest

from inferopt.backends.mock import MockBackend
from inferopt.optimizer.models import TunableConfig
from inferopt.optimizer.sla_controller import (
    SLAConstrainedAdaptiveController,
)
from inferopt.optimizer.sla_models import SLAMode, TargetSLO
from inferopt.scheduler.scheduler import Scheduler
from inferopt.telemetry.collector import MetricsCollector
from inferopt.telemetry.models import (
    BatchStats,
    MetricsSnapshot,
    QueueStats,
    RequestStats,
    ThroughputStats,
)


def _make_mock_snapshot(
    completed_requests: int = 16,
    arrival_rate: float = 2.0,
    p95_total_latency: float = 120.0,
    avg_queue_wait: float = 10.0,
    peak_queue: int = 0,
    timestamp: float = 100.0,
) -> MetricsSnapshot:
    """Helper to construct deterministic MetricsSnapshot instances for testing."""
    return MetricsSnapshot(
        timestamp=timestamp,
        requests=RequestStats(
            total_requests=completed_requests,
            completed_requests=completed_requests,
            failed_requests=0,
            cancelled_requests=0,
            avg_total_latency_ms=p95_total_latency * 0.8,
            min_total_latency_ms=p95_total_latency * 0.5,
            max_total_latency_ms=p95_total_latency * 1.2,
            p50_total_latency_ms=p95_total_latency * 0.7,
            p95_total_latency_ms=p95_total_latency,
            p99_total_latency_ms=p95_total_latency * 1.1,
            avg_queue_wait_ms=avg_queue_wait,
            avg_execution_ms=p95_total_latency - avg_queue_wait,
        ),
        batches=BatchStats(
            total_batches=max(1, completed_requests // 2),
            completed_batches=max(1, completed_requests // 2),
            failed_batches=0,
            avg_batch_size=2.0,
            min_batch_size=1,
            max_batch_size=4,
            avg_batch_formation_wait_ms=5.0,
            avg_batch_execution_ms=20.0,
        ),
        queue=QueueStats(
            current_queue_depth=0,
            peak_queue_depth=peak_queue,
            current_active_requests=1,
            peak_active_requests=2,
        ),
        throughput=ThroughputStats(
            requests_per_sec=arrival_rate,
            batches_per_sec=arrival_rate / 2.0,
            tokens_per_sec=arrival_rate * 20.0,
            elapsed_sec=10.0,
        ),
    )


class TestTargetSLODomainModel:
    """Verify validation and boundary checking on TargetSLO."""

    def test_valid_target_slo(self) -> None:
        slo = TargetSLO(p95_latency_ms=180.0, p99_latency_ms=250.0, max_queue_wait_ms=50.0)
        assert slo.p95_latency_ms == 180.0
        assert slo.p99_latency_ms == 250.0
        assert slo.max_queue_wait_ms == 50.0
        assert slo.headroom_ratio == 0.75

    def test_p99_less_than_p95_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= p95_latency_ms"):
            TargetSLO(p95_latency_ms=200.0, p99_latency_ms=150.0)

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValueError):
            TargetSLO(p95_latency_ms=-10.0)


class TestSLAConstrainedAdaptiveController:
    """Verify control decisions, deadband hysteresis, and mitigation down-scaling."""

    def test_deadband_hold_mode(self) -> None:
        """When p95 is between 0.75*SLO and SLO, controller must maintain configuration (HOLD)."""
        slo = TargetSLO(p95_latency_ms=200.0, headroom_ratio=0.75)
        controller = SLAConstrainedAdaptiveController(target_slo=slo)

        # Observed p95 is 170ms (between 150ms and 200ms)
        snap = _make_mock_snapshot(p95_total_latency=170.0, peak_queue=3, arrival_rate=4.0)
        active_cfg = TunableConfig(max_concurrency=4, max_batch_size=4, batch_wait_ms=50.0)

        status = controller.evaluate(snapshot=snap, active_config=active_cfg, current_time=100.0)
        assert status.mode == SLAMode.HOLD
        assert status.slo_violated is False
        assert status.recommended_config == active_cfg

    def test_mitigate_mode_on_slo_violation(self) -> None:
        """When p95 exceeds target SLO, controller must downscale capacity (MITIGATE)."""
        slo = TargetSLO(p95_latency_ms=150.0, headroom_ratio=0.75)
        controller = SLAConstrainedAdaptiveController(target_slo=slo, min_dwell_time_sec=0.1)

        # Observed p95 is 220ms (> 150ms)
        snap = _make_mock_snapshot(p95_total_latency=220.0, avg_queue_wait=45.0, peak_queue=5)
        active_cfg = TunableConfig(
            max_concurrency=8, max_batch_size=8, batch_wait_ms=50.0
        )  # Level 3

        status = controller.evaluate(snapshot=snap, active_config=active_cfg, current_time=100.0)
        assert status.mode == SLAMode.MITIGATE
        assert status.slo_violated is True
        # Must recommend Level 2 (c=4, b=4)
        assert status.recommended_config.max_concurrency == 4
        assert status.recommended_config.max_batch_size == 4

    def test_expand_mode_with_latency_headroom(self) -> None:
        """When p95 has headroom and queue demands capacity, upscale (EXPAND)."""
        slo = TargetSLO(p95_latency_ms=200.0, headroom_ratio=0.75)  # threshold = 150ms
        controller = SLAConstrainedAdaptiveController(target_slo=slo, min_dwell_time_sec=0.1)

        # Observed p95 is 80ms (< 150ms) and peak_queue is 3 with arrival rate 4.0
        snap = _make_mock_snapshot(p95_total_latency=80.0, peak_queue=3, arrival_rate=4.0)
        active_cfg = TunableConfig(
            max_concurrency=1, max_batch_size=2, batch_wait_ms=50.0
        )  # Level 1

        status = controller.evaluate(snapshot=snap, active_config=active_cfg, current_time=100.0)
        assert status.mode == SLAMode.EXPAND
        assert status.slo_violated is False
        # Must recommend Level 2 (c=4, b=4)
        assert status.recommended_config.max_concurrency == 4
        assert status.recommended_config.max_batch_size == 4

    def test_dwell_time_cooldown_gating(self) -> None:
        """Reconfigurations must be held if minimum dwell time has not elapsed."""
        slo = TargetSLO(p95_latency_ms=150.0)
        controller = SLAConstrainedAdaptiveController(target_slo=slo, min_dwell_time_sec=5.0)

        # Trigger initial adaptation at t=10.0
        snap1 = _make_mock_snapshot(p95_total_latency=200.0)
        c8 = TunableConfig(max_concurrency=8, max_batch_size=8, batch_wait_ms=50.0)
        backend = MockBackend()
        sched = Scheduler(config=c8.to_scheduler_config(), backend=backend)

        status1 = controller.evaluate(snap1, c8, current_time=10.0)
        applied1 = controller.apply_decision(status1, sched, current_time=10.0)
        assert applied1 is True

        # Next evaluation at t=12.0 (< 5.0s cooldown)
        c4 = TunableConfig(max_concurrency=4, max_batch_size=4, batch_wait_ms=50.0)
        status2 = controller.evaluate(snap1, c4, current_time=12.0)
        assert status2.recommended_config == c4  # Dwell gate holds config
        applied2 = controller.apply_decision(status2, sched, current_time=12.0)
        assert applied2 is False

    @pytest.mark.asyncio
    async def test_apply_decision_dynamic_scheduler_reconfiguration(self) -> None:
        """Verify dynamic reconfiguration updates running Scheduler without engine restart."""
        slo = TargetSLO(p95_latency_ms=150.0)
        controller = SLAConstrainedAdaptiveController(target_slo=slo, min_dwell_time_sec=0.1)

        c8 = TunableConfig(max_concurrency=8, max_batch_size=8, batch_wait_ms=50.0)
        backend = MockBackend()
        collector = MetricsCollector()
        sched = Scheduler(config=c8.to_scheduler_config(), backend=backend, collector=collector)
        await sched.start()

        try:
            assert sched.config.max_concurrency == 8
            assert sched.config.batch_config.max_batch_size == 8

            snap = _make_mock_snapshot(p95_total_latency=220.0)
            status = controller.evaluate(snap, c8, current_time=100.0)
            applied = controller.apply_decision(status, sched, current_time=100.0)

            assert applied is True
            assert sched.config.max_concurrency == 4
            assert sched.config.batch_config.max_batch_size == 4
            assert len(controller.adaptation_events) == 1
        finally:
            await sched.shutdown()
