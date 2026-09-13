"""Unit and mock integration tests for Step 13: Online Adaptive Control."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inferopt.backends.mock import MockBackend
from inferopt.benchmarks.cli import build_parser, run_benchmark_cli
from inferopt.benchmarks.step13_adaptive import (
    Step13AdaptiveExperimentRunner,
    Step13AdaptiveReport,
    classify_step13_findings,
    format_step13_report,
)
from inferopt.core.models import InferenceRequest
from inferopt.optimizer.adaptation_models import (
    AdaptationDecisionType,
    AdaptationPolicy,
)
from inferopt.optimizer.controller import AdaptiveController
from inferopt.optimizer.models import TunableConfig
from inferopt.optimizer.regime_detector import (
    DeterministicRegimeDetector,
    RegimeDetectionConfig,
    RegimeDetectionResult,
    WorkloadRegime,
)
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.scheduler.scheduler import Scheduler
from inferopt.telemetry.models import (
    BatchStats,
    MetricsSnapshot,
    QueueStats,
    RequestStats,
    ThroughputStats,
)


def _make_mock_snapshot(
    completed_requests: int = 16,
    arrival_rate: float = 1.5,
    current_queue: int = 0,
    peak_queue: int = 0,
    current_active: int = 1,
    peak_active: int = 1,
    p95_latency_ms: float = 100.0,
    avg_batch_size: float = 1.0,
    total_batches: int = 16,
    timestamp: float = 100.0,
) -> MetricsSnapshot:
    """Helper to synthesize a MetricsSnapshot for regime detector testing."""
    return MetricsSnapshot(
        timestamp=timestamp,
        requests=RequestStats(
            total_requests=completed_requests,
            completed_requests=completed_requests,
            failed_requests=0,
            avg_total_latency_ms=p95_latency_ms * 0.8,
            p50_total_latency_ms=p95_latency_ms * 0.75,
            p95_total_latency_ms=p95_latency_ms,
            p99_total_latency_ms=p95_latency_ms * 1.1,
            avg_queue_wait_ms=5.0,
            avg_execution_ms=p95_latency_ms * 0.75,
        ),
        batches=BatchStats(
            total_batches=total_batches,
            completed_batches=total_batches,
            failed_batches=0,
            avg_batch_size=avg_batch_size,
            min_batch_size=1,
            max_batch_size=int(avg_batch_size),
        ),
        throughput=ThroughputStats(
            elapsed_sec=10.0,
            requests_per_sec=arrival_rate,
            batches_per_sec=total_batches / 10.0,
            tokens_per_sec=arrival_rate * 30.0,
            total_input_tokens=completed_requests * 10,
            total_output_tokens=completed_requests * 20,
        ),
        queue=QueueStats(
            current_queue_depth=current_queue,
            peak_queue_depth=peak_queue,
            current_active_requests=current_active,
            peak_active_requests=peak_active,
        ),
    )


class TestDeterministicRegimeDetector:
    """Unit tests for DeterministicRegimeDetector rules and thresholds."""

    def test_regime_detection_insufficient_data(self) -> None:
        """Verify UNKNOWN classification when request count is below threshold."""
        detector = DeterministicRegimeDetector(
            config=RegimeDetectionConfig(min_requests_for_detection=5)
        )
        snap = _make_mock_snapshot(completed_requests=2, current_active=0)
        res = detector.detect(snap)
        assert res.regime == WorkloadRegime.UNKNOWN
        assert res.confidence == 0.0
        assert "Insufficient requests" in res.reason

    def test_regime_detection_light_workload(self) -> None:
        """Verify LIGHT classification under low arrival rate and minimal queue depth."""
        detector = DeterministicRegimeDetector()
        snap = _make_mock_snapshot(
            completed_requests=16,
            arrival_rate=1.2,
            peak_queue=1,
            peak_active=1,
        )
        res = detector.detect(snap)
        assert res.regime == WorkloadRegime.LIGHT
        assert res.confidence == 1.0
        assert "Light traffic detected" in res.reason

    def test_regime_detection_bursty_workload(self) -> None:
        """Verify BURSTY classification when peak queue depth spikes >= 3."""
        detector = DeterministicRegimeDetector()
        snap = _make_mock_snapshot(
            completed_requests=16,
            arrival_rate=4.0,
            peak_queue=6,
            peak_active=3,
        )
        res = detector.detect(snap)
        assert res.regime == WorkloadRegime.BURSTY
        assert res.confidence == 1.0
        assert "Bursty traffic spike detected" in res.reason

    def test_regime_detection_saturated_workload(self) -> None:
        """Verify SATURATED classification when sustained active concurrency >= 5."""
        detector = DeterministicRegimeDetector()
        snap = _make_mock_snapshot(
            completed_requests=16,
            arrival_rate=8.0,
            peak_queue=8,
            peak_active=8,
        )
        res = detector.detect(snap)
        assert res.regime == WorkloadRegime.SATURATED
        assert res.confidence == 1.0
        assert "Saturated traffic detected" in res.reason

    def test_regime_detection_light_to_bursty(self) -> None:
        """Verify window delta detection transitioning from LIGHT to BURSTY regime."""
        detector = DeterministicRegimeDetector()
        snap1 = _make_mock_snapshot(
            completed_requests=16,
            arrival_rate=1.0,
            peak_queue=0,
            peak_active=1,
            timestamp=100.0,
        )
        snap2 = _make_mock_snapshot(
            completed_requests=24,
            arrival_rate=4.0,
            current_queue=4,
            peak_queue=5,
            current_active=3,
            peak_active=3,
            timestamp=102.0,
        )
        res1 = detector.detect(snap1)
        assert res1.regime == WorkloadRegime.LIGHT

        res2 = detector.detect(snap2, previous_snapshot=snap1)
        assert res2.regime == WorkloadRegime.BURSTY

    def test_regime_detection_bursty_to_saturated(self) -> None:
        """Verify window delta detection transitioning from BURSTY to SATURATED regime."""
        detector = DeterministicRegimeDetector()
        snap1 = _make_mock_snapshot(
            completed_requests=20,
            arrival_rate=4.0,
            peak_queue=4,
            peak_active=3,
            timestamp=100.0,
        )
        snap2 = _make_mock_snapshot(
            completed_requests=36,
            arrival_rate=8.0,
            peak_queue=8,
            peak_active=8,
            current_active=8,
            timestamp=102.0,
        )
        res1 = detector.detect(snap1)
        assert res1.regime == WorkloadRegime.BURSTY

        res2 = detector.detect(snap2, previous_snapshot=snap1)
        assert res2.regime == WorkloadRegime.SATURATED

    def test_regime_detection_saturated_to_light(self) -> None:
        """Verify window delta detection transitioning from SATURATED back to LIGHT."""
        detector = DeterministicRegimeDetector()
        snap1 = _make_mock_snapshot(
            completed_requests=40,
            arrival_rate=8.0,
            peak_queue=8,
            peak_active=8,
            timestamp=100.0,
        )
        snap2 = _make_mock_snapshot(
            completed_requests=44,
            arrival_rate=1.0,
            current_queue=0,
            peak_queue=1,
            current_active=1,
            peak_active=1,
            timestamp=104.0,
        )
        res1 = detector.detect(snap1)
        assert res1.regime == WorkloadRegime.SATURATED

        res2 = detector.detect(snap2, previous_snapshot=snap1)
        assert res2.regime == WorkloadRegime.LIGHT


class TestAdaptiveControllerRegimeIntegration:
    """Unit tests for AdaptiveController regime-aware closed-loop adaptation."""

    def test_evaluate_regime_triggers_apply(self) -> None:
        """Verify controller triggers APPLY when regime indicates reconfiguration."""
        controller = AdaptiveController(
            policy=AdaptationPolicy(
                min_dwell_time_sec=0.0,
                cooldown_windows=0,
                min_regime_evidence_count=1,
            )
        )
        snap = _make_mock_snapshot(current_active=1, current_queue=0)
        reg_res = RegimeDetectionResult(
            regime=WorkloadRegime.SATURATED,
            confidence=1.0,
            arrival_rate_rps=8.0,
            current_queue_depth=0,
            peak_queue_depth=8,
            active_concurrency=8,
            avg_batch_size=4.0,
            p95_latency_ms=120.0,
            reason="High concurrency detected",
        )
        curr_cfg = TunableConfig(max_concurrency=1, max_batch_size=2, batch_wait_ms=50.0)
        dec = controller.evaluate_regime(snap, reg_res, current_config=curr_cfg)
        assert dec.decision_type == AdaptationDecisionType.APPLY
        assert dec.proposed_config == TunableConfig(
            max_concurrency=8, max_batch_size=8, batch_wait_ms=50.0
        )
        assert dec.detected_regime == WorkloadRegime.SATURATED

    def test_evaluate_regime_no_change_when_already_matching(self) -> None:
        """Verify NO_CHANGE when active configuration already matches detected regime."""
        controller = AdaptiveController()
        snap = _make_mock_snapshot()
        reg_res = RegimeDetectionResult(
            regime=WorkloadRegime.LIGHT,
            confidence=1.0,
            arrival_rate_rps=1.0,
            current_queue_depth=0,
            peak_queue_depth=1,
            active_concurrency=1,
            avg_batch_size=1.0,
            p95_latency_ms=80.0,
            reason="Light regime",
        )
        curr_cfg = TunableConfig(max_concurrency=1, max_batch_size=2, batch_wait_ms=50.0)
        dec = controller.evaluate_regime(snap, reg_res, current_config=curr_cfg)
        assert dec.decision_type == AdaptationDecisionType.NO_CHANGE
        assert "already matches" in dec.reason

    def test_no_adaptation_during_stable_workload(self) -> None:
        """Verify continuous stable workload causes no unwanted reconfigurations."""
        controller = AdaptiveController(
            policy=AdaptationPolicy(min_dwell_time_sec=0.0, cooldown_windows=0)
        )
        snap = _make_mock_snapshot()
        reg_res = RegimeDetectionResult(
            regime=WorkloadRegime.LIGHT,
            confidence=1.0,
            arrival_rate_rps=1.0,
            current_queue_depth=0,
            peak_queue_depth=0,
            active_concurrency=1,
            avg_batch_size=1.0,
            p95_latency_ms=75.0,
            reason="Stable light traffic",
        )
        curr_cfg = TunableConfig(max_concurrency=1, max_batch_size=2, batch_wait_ms=50.0)

        for _ in range(5):
            dec = controller.step_regime(snap, reg_res, current_config=curr_cfg)
            assert dec.decision_type == AdaptationDecisionType.NO_CHANGE
            assert dec.is_applied is False

    def test_hysteresis_and_cooldown_prevents_thrashing(self) -> None:
        """Verify cooldown and dwell time prevent immediate re-adaptation."""
        controller = AdaptiveController(
            policy=AdaptationPolicy(
                min_dwell_time_sec=10.0,
                cooldown_windows=2,
            )
        )
        snap = _make_mock_snapshot()
        reg_res = RegimeDetectionResult(
            regime=WorkloadRegime.BURSTY,
            confidence=1.0,
            arrival_rate_rps=4.0,
            current_queue_depth=0,
            peak_queue_depth=4,
            active_concurrency=4,
            avg_batch_size=2.0,
            p95_latency_ms=100.0,
            reason="Burst detected",
        )
        curr_cfg = TunableConfig(max_concurrency=1, max_batch_size=2, batch_wait_ms=50.0)
        dec1 = controller.step_regime(snap, reg_res, current_config=curr_cfg)
        assert dec1.decision_type == AdaptationDecisionType.APPLY
        assert dec1.is_applied is True

        # Second step immediately after: cooldown/dwell time should suppress adaptation
        reg_res_light = RegimeDetectionResult(
            regime=WorkloadRegime.LIGHT,
            confidence=1.0,
            arrival_rate_rps=1.0,
            current_queue_depth=0,
            peak_queue_depth=1,
            active_concurrency=1,
            avg_batch_size=1.0,
            p95_latency_ms=80.0,
            reason="Light traffic",
        )
        dec2 = controller.evaluate_regime(
            snap, reg_res_light, current_config=dec1.proposed_config
        )
        assert dec2.decision_type == AdaptationDecisionType.COOLDOWN
        assert "Dwell time active" in dec2.reason or "Cooldown active" in dec2.reason

    def test_hysteresis_prevents_oscillation(self) -> None:
        """Verify min_regime_evidence_count dampens single-window noise/flapping."""
        controller = AdaptiveController(
            policy=AdaptationPolicy(
                min_dwell_time_sec=0.0,
                cooldown_windows=0,
                min_regime_evidence_count=2,
            )
        )
        snap = _make_mock_snapshot()
        reg_res_burst = RegimeDetectionResult(
            regime=WorkloadRegime.BURSTY,
            confidence=1.0,
            arrival_rate_rps=4.0,
            current_queue_depth=0,
            peak_queue_depth=4,
            active_concurrency=3,
            avg_batch_size=2.0,
            p95_latency_ms=90.0,
            reason="Bursty spike",
        )
        curr_cfg = TunableConfig(max_concurrency=1, max_batch_size=2, batch_wait_ms=50.0)

        # Window 1: Observed once -> NO_CHANGE because 1 < 2 evidence windows required
        dec1 = controller.evaluate_regime(snap, reg_res_burst, current_config=curr_cfg)
        assert dec1.decision_type == AdaptationDecisionType.NO_CHANGE
        assert "1/2 required consecutive windows" in dec1.reason

        # Window 2: Observed second time -> APPLY
        dec2 = controller.evaluate_regime(snap, reg_res_burst, current_config=curr_cfg)
        assert dec2.decision_type == AdaptationDecisionType.APPLY
        assert dec2.proposed_config == TunableConfig(
            max_concurrency=8, max_batch_size=8, batch_wait_ms=50.0
        )

    def test_deterministic_adaptation_decisions(self) -> None:
        """Verify that identical inputs deterministically yield identical decisions."""
        policy = AdaptationPolicy(min_dwell_time_sec=0.0, cooldown_windows=0)
        ctrl1 = AdaptiveController(policy=policy)
        ctrl2 = AdaptiveController(policy=policy)

        snap = _make_mock_snapshot()
        reg_res = RegimeDetectionResult(
            regime=WorkloadRegime.SATURATED,
            confidence=1.0,
            arrival_rate_rps=8.0,
            current_queue_depth=0,
            peak_queue_depth=8,
            active_concurrency=8,
            avg_batch_size=4.0,
            p95_latency_ms=130.0,
            reason="Saturated",
        )
        curr_cfg = TunableConfig(max_concurrency=1, max_batch_size=2, batch_wait_ms=50.0)

        dec1 = ctrl1.evaluate_regime(snap, reg_res, current_config=curr_cfg)
        dec2 = ctrl2.evaluate_regime(snap, reg_res, current_config=curr_cfg)

        assert dec1.decision_type == dec2.decision_type
        assert dec1.proposed_config == dec2.proposed_config
        assert dec1.reason == dec2.reason

    def test_invalid_config_rejection(self) -> None:
        """Verify that out-of-bounds target configuration is rejected as INFEASIBLE."""
        custom_policy_map = {
            WorkloadRegime.SATURATED: TunableConfig(
                max_concurrency=32, max_batch_size=64, batch_wait_ms=50.0
            )
        }
        policy = AdaptationPolicy(
            regime_policy=custom_policy_map,
            max_concurrency=16,
            max_batch_size=32,
            min_dwell_time_sec=0.0,
            cooldown_windows=0,
        )
        controller = AdaptiveController(policy=policy)
        snap = _make_mock_snapshot()
        reg_res = RegimeDetectionResult(
            regime=WorkloadRegime.SATURATED,
            confidence=1.0,
            arrival_rate_rps=10.0,
            current_queue_depth=0,
            peak_queue_depth=10,
            active_concurrency=10,
            avg_batch_size=8.0,
            p95_latency_ms=200.0,
            reason="High concurrency",
        )
        curr_cfg = TunableConfig(max_concurrency=1, max_batch_size=2, batch_wait_ms=50.0)

        dec = controller.evaluate_regime(snap, reg_res, current_config=curr_cfg)
        assert dec.decision_type == AdaptationDecisionType.INFEASIBLE
        assert "violates policy bounds" in dec.reason


class TestOnlineDynamicReconfigurationScheduler:
    """Unit tests verifying live dynamic reconfiguration on running Scheduler."""

    @pytest.mark.asyncio
    async def test_apply_config_preserves_scheduler_correctness(self) -> None:
        """Verify Scheduler.apply_config updates concurrency and batch params seamlessly."""
        backend = MockBackend(default_latency_sec=0.005)
        init_cfg = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=1, batch_wait_ms=20.0),
        )

        async with Scheduler(backend=backend, config=init_cfg) as scheduler:
            # Submit initial request
            req1 = InferenceRequest(model="mock", prompt="Hello initial", max_tokens=16)
            res1 = await scheduler.submit(req1)
            assert res1.generated_text is not None

            # Dynamically reconfigure to concurrency=4, batch_size=4
            new_cfg = SchedulerConfig(
                max_concurrency=4,
                batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=50.0),
            )
            scheduler.apply_config(new_cfg)
            assert scheduler.config.max_concurrency == 4
            assert scheduler.config.batch_config.max_batch_size == 4

            # Submit concurrent requests under new config
            reqs = [
                InferenceRequest(model="mock", prompt=f"Req {i}", max_tokens=16)
                for i in range(8)
            ]
            tasks = [scheduler.submit(r) for r in reqs]
            resps = await asyncio.gather(*tasks)
            assert len(resps) == 8
            assert all(r.generated_text is not None for r in resps)

    @pytest.mark.asyncio
    async def test_adaptation_with_inflight_requests(self) -> None:
        """Verify reconfiguring scheduler while requests are actively executing causes no drops."""
        backend = MockBackend(default_latency_sec=0.02)
        init_cfg = SchedulerConfig(
            max_concurrency=2,
            batch_config=BatchConfig(max_batch_size=2, batch_wait_ms=10.0),
        )

        async with Scheduler(backend=backend, config=init_cfg) as scheduler:
            reqs = [
                InferenceRequest(model="mock", prompt=f"Inflight {i}", max_tokens=16)
                for i in range(6)
            ]
            tasks = [asyncio.create_task(scheduler.submit(r)) for r in reqs]

            # Allow tasks to begin execution
            await asyncio.sleep(0.005)

            # Dynamically change config while in flight
            scheduler.apply_config(
                SchedulerConfig(
                    max_concurrency=8,
                    batch_config=BatchConfig(max_batch_size=8, batch_wait_ms=50.0),
                )
            )

            resps = await asyncio.gather(*tasks)
            assert len(resps) == 6
            assert all(r.generated_text is not None for r in resps)

    @pytest.mark.asyncio
    async def test_adaptation_with_non_empty_queue(self) -> None:
        """Verify queued backlog is processed correctly under new batch config after update."""
        backend = MockBackend(default_latency_sec=0.01)
        init_cfg = SchedulerConfig(
            max_concurrency=1,
            batch_config=BatchConfig(max_batch_size=1, batch_wait_ms=10.0),
        )

        async with Scheduler(backend=backend, config=init_cfg) as scheduler:
            reqs = [
                InferenceRequest(model="mock", prompt=f"Backlog {i}", max_tokens=16)
                for i in range(8)
            ]
            tasks = [asyncio.create_task(scheduler.submit(r)) for r in reqs]

            # Apply larger batch size and concurrency
            scheduler.apply_config(
                SchedulerConfig(
                    max_concurrency=4,
                    batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=50.0),
                )
            )

            resps = await asyncio.gather(*tasks)
            assert len(resps) == 8
            assert all(r.generated_text is not None for r in resps)

    @pytest.mark.asyncio
    async def test_engine_lifecycle_identity_unchanged(self) -> None:
        """Proves exactly 1 engine initialization and 1 teardown occurs for the entire run."""
        backend = MockBackend(default_latency_sec=0.001)
        runner = Step13AdaptiveExperimentRunner()
        report = await runner.run_experiment(backend=backend, num_requests_per_phase=4, seed=42)

        assert report.engine_initialization_count == 1
        assert report.engine_teardown_count == 1
        for _cond_name, cond in report.conditions.items():
            assert cond.engine_initialization_count == 1
            assert cond.engine_teardown_count == 1

    @pytest.mark.asyncio
    async def test_request_ids_remain_1_to_1_integrity(self) -> None:
        """Verify 100% completion, 0 duplicates, and 0 dropped request IDs."""
        backend = MockBackend(default_latency_sec=0.001)
        runner = Step13AdaptiveExperimentRunner()
        num_reqs = 6
        report = await runner.run_experiment(
            backend=backend, num_requests_per_phase=num_reqs, seed=42
        )

        total_expected = 4 * num_reqs  # 4 phases x 6 requests
        adaptive_cond = report.conditions["ADAPTIVE_INFEROPT"]
        assert adaptive_cond.total_requests == total_expected
        assert adaptive_cond.completed_requests == total_expected
        assert adaptive_cond.failed_requests == 0
        assert adaptive_cond.integrity_valid is True


class TestStep13EndToEndAdaptiveRunner:
    """Mock integration tests verifying multi-phase adaptation against MockBackend."""

    @pytest.mark.asyncio
    async def test_step13_full_phase_sequence_mocked(self) -> None:
        """Verify complete 4-phase adaptation sequence against single live MockBackend."""
        backend = MockBackend(default_latency_sec=0.001)
        runner = Step13AdaptiveExperimentRunner(
            model_id="mock-qwen-test",
            enforce_eager=True,
            policy=AdaptationPolicy(min_dwell_time_sec=0.0, cooldown_windows=0),
            detection_config=RegimeDetectionConfig(min_requests_for_detection=2),
        )

        report = await runner.run_experiment(
            backend=backend,
            num_requests_per_phase=8,
            seed=42,
        )

        assert report.engine_initialization_count == 1
        assert report.engine_teardown_count == 1
        assert "STATIC_CONSERVATIVE" in report.conditions
        assert "STATIC_AGGRESSIVE" in report.conditions
        assert "ADAPTIVE_INFEROPT" in report.conditions

        adaptive_cond = report.conditions["ADAPTIVE_INFEROPT"]
        assert adaptive_cond.integrity_valid is True
        assert adaptive_cond.completed_requests == 32  # 4 phases x 8 requests
        assert adaptive_cond.failed_requests == 0
        assert len(adaptive_cond.phase_metrics) == 4
        assert adaptive_cond.total_adaptations >= 1

        # Check baseline comparison
        comp = report.comparison
        assert comp.adaptive_tput > 0.0
        assert comp.static_conservative_tput > 0.0

        # Check evidence classifications
        findings = classify_step13_findings(comp, adaptive_cond)
        assert len(findings["PROVEN"]) >= 4
        assert len(findings["SUGGESTED"]) >= 2
        assert len(findings["NOT PROVEN"]) >= 4

        # Test ASCII report formatting
        ascii_rep = format_step13_report(report)
        assert "INFEROPT STEP 13: ONLINE ADAPTIVE CONTROL" in ascii_rep
        assert "OVERALL CONDITION COMPARISON TABLE" in ascii_rep
        assert "ADAPTIVE INFEROPT PHASE-BY-PHASE EXECUTION TRACE" in ascii_rep
        assert "SCIENTIFIC EVIDENCE CLASSIFICATION" in ascii_rep

    @pytest.mark.asyncio
    async def test_step13_report_json_roundtrip(self) -> None:
        """Verify Step13AdaptiveReport JSON serialization and file persistence."""
        backend = MockBackend(default_latency_sec=0.001)
        runner = Step13AdaptiveExperimentRunner()
        report = await runner.run_experiment(backend=backend, num_requests_per_phase=4, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "step13_report.json"
            report.save_json(out_file)
            assert out_file.exists()

            loaded = Step13AdaptiveReport.from_json(out_file.read_text(encoding="utf-8"))
            assert loaded.experiment_id == report.experiment_id
            assert loaded.num_requests_per_phase == report.num_requests_per_phase
            assert len(loaded.conditions) == len(report.conditions)


class TestStep13CLIParsingAndDispatch:
    """Unit tests for Step 13 CLI argument parsing and async dispatch."""

    def test_cli_argument_parsing(self) -> None:
        """Verify --experiment-step13 and related argument parsing."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--experiment-step13",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--num-requests-per-phase",
                "16",
                "--phase-sequence",
                "light",
                "bursty",
                "saturated",
                "light",
                "--enforce-eager",
            ]
        )
        assert args.experiment_step13 is True
        assert args.model == "Qwen/Qwen2.5-0.5B-Instruct"
        assert args.num_requests_per_phase == 16
        assert args.phase_sequence == ["light", "bursty", "saturated", "light"]
        assert args.enforce_eager is True

    @pytest.mark.asyncio
    async def test_run_benchmark_cli_step13_dispatch(self) -> None:
        """Verify CLI dispatch to Step13AdaptiveExperimentRunner."""
        mock_cond = MagicMock()
        mock_cond.integrity_valid = True

        mock_report = MagicMock()
        mock_report.conditions = {"ADAPTIVE_INFEROPT": mock_cond}

        mock_runner = MagicMock()
        mock_runner.run_experiment = AsyncMock(return_value=mock_report)

        with (
            patch(
                "inferopt.benchmarks.step13_adaptive.Step13AdaptiveExperimentRunner",
                return_value=mock_runner,
            ),
            patch(
                "inferopt.benchmarks.step13_adaptive.format_step13_report",
                return_value="Step 13 Mock Report",
            ),
        ):
            parser = build_parser()
            args = parser.parse_args(
                [
                    "--experiment-step13",
                    "--num-requests-per-phase",
                    "8",
                ]
            )
            rc = await run_benchmark_cli(args)
            assert rc == 0
            assert mock_runner.run_experiment.await_count == 1
