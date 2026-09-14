"""Unit and integration tests for Step 14 SLA-Aware Adaptive Control benchmark runner."""

import tempfile
from pathlib import Path

import pytest

from inferopt.backends.mock import MockBackend
from inferopt.benchmarks.cli import build_parser, run_benchmark_cli
from inferopt.benchmarks.step14_sla_adaptive import (
    Step14SLAExperimentRunner,
    format_step14_report,
)
from inferopt.optimizer.sla_models import Step14SLAReport, TargetSLO


class TestStep14SLAExperimentRunner:
    """Verify Step 14 end-to-end benchmark execution and hard runtime invariants."""

    def test_step14_phase_scenarios_construction(self) -> None:
        """Verify 4-phase sequence definitions and request counts."""
        runner = Step14SLAExperimentRunner()
        scenarios = runner.build_step14_phase_scenarios(num_requests_per_phase=8, seed=42)

        assert len(scenarios) == 4
        assert scenarios[0].scenario_name == "PHASE_1_MODERATE"
        assert scenarios[1].scenario_name == "PHASE_2_BURST"
        assert scenarios[2].scenario_name == "PHASE_3_SATURATED"
        assert scenarios[3].scenario_name == "PHASE_4_COOLDOWN"

        for s in scenarios:
            assert len(s.requests) == 8

    @pytest.mark.asyncio
    async def test_step14_full_mocked_experiment_and_invariants(self) -> None:
        """Execute full 3-condition experiment under MockBackend and verify 100% reconciliation."""
        backend = MockBackend(default_latency_sec=0.001)
        target_slo = TargetSLO(p95_latency_ms=150.0)
        runner = Step14SLAExperimentRunner(target_slo=target_slo)

        num_reqs = 4
        report = await runner.run_experiment(
            backend=backend,
            num_requests_per_phase=num_reqs,
            seed=42,
        )

        assert isinstance(report, Step14SLAReport)
        assert report.backend == "mock"
        assert report.backend_execution_confirmed is False
        assert report.target_slo.p95_latency_ms == 150.0

        # Hard invariant: 3 conditions * 4 phases * 4 reqs = 48 total requests
        expected_total_reqs = 3 * 4 * num_reqs
        assert report.scheduled_requests == expected_total_reqs
        assert report.completed_requests == expected_total_reqs
        assert report.failed_requests == 0
        assert report.measured_requests == expected_total_reqs
        assert backend.total_requests_executed == expected_total_reqs

        # Verify each condition
        for cond_name in ("STATIC_CONSERVATIVE", "STATIC_AGGRESSIVE", "SLA_AWARE_ADAPTIVE"):
            cond = report.conditions[cond_name]
            assert cond.scheduled_requests == 4 * num_reqs
            assert cond.completed_requests == 4 * num_reqs
            assert cond.failed_requests == 0
            assert cond.measured_requests == 4 * num_reqs
            assert cond.integrity_valid is True
            assert cond.total_duration_sec > 0.0

            expected_tput = cond.completed_requests / cond.total_duration_sec
            assert abs(cond.overall_throughput_rps - expected_tput) < 1e-4

            for pm in cond.phase_metrics:
                assert pm.scheduled_requests == num_reqs
                assert pm.completed_requests == num_reqs
                assert pm.failed_requests == 0
                assert pm.measured_requests == num_reqs
                assert pm.mean_latency_ms >= pm.avg_queue_wait_ms - 1e-6
                assert pm.p95_latency_ms >= pm.avg_queue_wait_ms - 1e-6

        # Formatted report validation
        text_report = format_step14_report(report)
        assert "INFEROPT STEP 14" in text_report
        assert "OVERALL CONDITION COMPARISON TABLE" in text_report
        assert "SLA-AWARE ADAPTIVE INFEROPT" in text_report

    @pytest.mark.asyncio
    async def test_step14_report_json_roundtrip(self) -> None:
        """Verify report serialization and round-trip deserialization."""
        backend = MockBackend(default_latency_sec=0.001)
        runner = Step14SLAExperimentRunner()

        report = await runner.run_experiment(backend=backend, num_requests_per_phase=2, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_file = runner.save_report(report, tmpdir)
            assert Path(saved_file).exists()

            with open(saved_file, encoding="utf-8") as f:
                data = f.read()

            loaded_report = Step14SLAReport.model_validate_json(data)
            assert loaded_report.experiment_id == report.experiment_id
            assert loaded_report.scheduled_requests == report.scheduled_requests
            assert loaded_report.completed_requests == report.completed_requests

    @pytest.mark.asyncio
    async def test_step14_cli_parsing_and_dispatch(self) -> None:
        """Verify CLI argument parsing and execution dispatch for Step 14."""
        parser = build_parser()
        args = parser.parse_args(
            [
                "--experiment-step14",
                "--target-slo-p95-ms",
                "175.0",
                "--num-requests-per-phase",
                "2",
                "--backend",
                "mock",
                "--output",
                "/tmp/test_step14_cli.json",
            ]
        )

        assert args.experiment_step14 is True
        assert args.target_slo_p95_ms == 175.0
        assert args.num_requests_per_phase == 2
        assert args.backend == "mock"

        exit_code = await run_benchmark_cli(args)
        assert exit_code == 0
