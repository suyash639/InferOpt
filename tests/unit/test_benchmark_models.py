"""Unit tests for workload specifications, scenario containers, and benchmark result models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from inferopt.benchmarks.models import (
    ArrivalPattern,
    BenchmarkResult,
    PromptCategory,
    WorkloadConfig,
    WorkloadRequestSpec,
    WorkloadScenario,
)
from inferopt.core.models import InferenceRequest
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.telemetry.models import MetricsSnapshot


class TestWorkloadRequestSpec:
    """Tests for WorkloadRequestSpec domain model."""

    def test_valid_request_spec(self) -> None:
        spec = WorkloadRequestSpec(
            request_id="r-001",
            model="test-model",
            prompt="Hello world",
            priority=2,
            max_tokens=64,
            temperature=0.5,
            metadata={"tag": "eval"},
            scheduled_delay_ms=10.0,
        )
        assert spec.request_id == "r-001"
        assert spec.model == "test-model"
        assert spec.prompt == "Hello world"
        assert spec.priority == 2
        assert spec.max_tokens == 64
        assert spec.temperature == 0.5
        assert spec.metadata == {"tag": "eval"}
        assert spec.scheduled_delay_ms == 10.0

    def test_conversion_to_inference_request(self) -> None:
        spec = WorkloadRequestSpec(
            request_id="r-001",
            model="test-model",
            prompt="Hello world",
            priority=2,
            max_tokens=64,
            temperature=0.5,
            metadata={"tag": "eval"},
        )
        req = spec.to_inference_request()
        assert isinstance(req, InferenceRequest)
        assert req.request_id == spec.request_id
        assert req.model == spec.model
        assert req.prompt == spec.prompt
        assert req.priority == spec.priority
        assert req.max_tokens == spec.max_tokens
        assert req.temperature == spec.temperature
        assert req.metadata == spec.metadata

    def test_immutability(self) -> None:
        spec = WorkloadRequestSpec(request_id="r-001", prompt="Prompt text")
        with pytest.raises(ValidationError):
            spec.prompt = "New prompt"

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            WorkloadRequestSpec.model_validate(
                {"request_id": "r-001", "prompt": "Prompt text", "unknown": 123}
            )


class TestWorkloadConfig:
    """Tests for WorkloadConfig model."""

    def test_default_config(self) -> None:
        cfg = WorkloadConfig(scenario_name="test-run")
        assert cfg.scenario_name == "test-run"
        assert cfg.num_requests == 10
        assert cfg.arrival_pattern == ArrivalPattern.CONCURRENT
        assert cfg.seed == 42
        assert cfg.prompt_categories == (PromptCategory.SHORT,)

    def test_invalid_num_requests(self) -> None:
        with pytest.raises(ValidationError):
            WorkloadConfig(scenario_name="test", num_requests=0)

    def test_immutability(self) -> None:
        cfg = WorkloadConfig(scenario_name="test")
        with pytest.raises(ValidationError):
            cfg.num_requests = 20


class TestWorkloadScenario:
    """Tests for WorkloadScenario serialization and persistence."""

    def test_scenario_creation_and_json_roundtrip(self, tmp_path: Path) -> None:
        cfg = WorkloadConfig(scenario_name="test-scenario", num_requests=2)
        reqs = (
            WorkloadRequestSpec(request_id="r-1", prompt="p1"),
            WorkloadRequestSpec(request_id="r-2", prompt="p2"),
        )
        scenario = WorkloadScenario(
            scenario_name="test-scenario",
            description="A test scenario",
            config=cfg,
            requests=reqs,
        )
        assert scenario.total_requests == 2

        json_str = scenario.to_json()
        restored = WorkloadScenario.from_json(json_str)
        assert restored.scenario_name == scenario.scenario_name
        assert restored.total_requests == 2
        assert restored.requests[0].request_id == "r-1"

        file_path = tmp_path / "scenario.json"
        scenario.save_json(file_path)
        assert file_path.exists()

        loaded = WorkloadScenario.load_json(file_path)
        assert loaded.scenario_name == scenario.scenario_name
        assert loaded.total_requests == 2


class TestBenchmarkResult:
    """Tests for BenchmarkResult serialization and persistence."""

    def test_result_creation_and_json_roundtrip(self, tmp_path: Path) -> None:
        cfg = WorkloadConfig(scenario_name="test-bench")
        sched_cfg = SchedulerConfig()
        batch_cfg = BatchConfig()
        snapshot = MetricsSnapshot()

        result = BenchmarkResult(
            benchmark_id="bench-1234",
            scenario_name="test-bench",
            workload_config=cfg,
            backend_name="mock",
            scheduler_config=sched_cfg,
            batch_config=batch_cfg,
            telemetry_snapshot=snapshot,
            duration_sec=1.5,
            total_requests=10,
            completed_requests=10,
            failed_requests=0,
            cancelled_requests=0,
            requests_per_sec=6.67,
            batches_per_sec=2.0,
            tokens_per_sec=100.0,
            avg_latency_ms=15.0,
            p50_latency_ms=14.0,
            p95_latency_ms=19.0,
            p99_latency_ms=20.0,
            avg_queue_wait_ms=1.0,
            avg_execution_ms=14.0,
            peak_queue_depth=5,
            peak_active_requests=4,
            total_batches=3,
            avg_batch_size=3.33,
            max_batch_size=4,
        )

        json_str = result.to_json()
        restored = BenchmarkResult.from_json(json_str)
        assert restored.benchmark_id == "bench-1234"
        assert restored.requests_per_sec == 6.67
        assert restored.completed_requests == 10

        file_path = tmp_path / "result.json"
        result.save_json(file_path)
        assert file_path.exists()

        loaded = BenchmarkResult.load_json(file_path)
        assert loaded.benchmark_id == result.benchmark_id
