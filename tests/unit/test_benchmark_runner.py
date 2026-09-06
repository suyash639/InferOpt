"""Unit tests for BenchmarkRunner arrival patterns, execution, and telemetry collection."""

import pytest

from inferopt.backends.mock import MockBackend
from inferopt.benchmarks.generator import (
    get_burst_workload,
    get_light_workload,
)
from inferopt.benchmarks.models import (
    ArrivalPattern,
    BenchmarkResult,
    WorkloadConfig,
    WorkloadRequestSpec,
    WorkloadScenario,
)
from inferopt.benchmarks.runner import BenchmarkRunner
from inferopt.core.exceptions import InferenceError
from inferopt.core.models import InferenceRequest, InferenceResponse
from inferopt.scheduler.config import BatchConfig, SchedulerConfig


class FailingBackend:
    """Mock backend that raises errors for testing failure handling in benchmarks."""

    def __init__(self) -> None:
        self.backend_name = "failing-backend"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        raise InferenceError("Synthetic failure")


class NonBatchMockBackend:
    """Mock backend that implements only standard InferenceBackend protocol."""

    def __init__(self) -> None:
        self.backend_name = "non-batch-mock"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            generated_text="test response",
            input_tokens=10,
            output_tokens=10,
            latency_ms=1.0,
            backend_name=self.backend_name,
        )


@pytest.mark.asyncio
class TestBenchmarkRunner:
    """Tests for BenchmarkRunner execution across diverse arrival patterns and configurations."""

    async def test_sequential_arrival_execution(self) -> None:
        backend = MockBackend(default_latency_sec=0.001)
        runner = BenchmarkRunner()

        cfg = WorkloadConfig(
            scenario_name="seq-test",
            num_requests=5,
            arrival_pattern=ArrivalPattern.SEQUENTIAL,
            max_tokens=16,
        )
        reqs = tuple(
            WorkloadRequestSpec(request_id=f"r-{i}", prompt=f"Prompt {i}") for i in range(5)
        )
        scenario = WorkloadScenario(scenario_name="seq-test", config=cfg, requests=reqs)

        result = await runner.run(scenario=scenario, backend=backend)
        assert isinstance(result, BenchmarkResult)
        assert result.total_requests == 5
        assert result.completed_requests == 5
        assert result.failed_requests == 0
        assert result.duration_sec > 0
        assert result.requests_per_sec > 0

    async def test_concurrent_arrival_execution(self) -> None:
        backend = MockBackend(default_latency_sec=0.001)
        runner = BenchmarkRunner()
        scenario = get_light_workload(seed=42)

        sched_config = SchedulerConfig(
            max_concurrency=2,
            batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=0.0),
        )

        result = await runner.run(
            scenario=scenario,
            backend=backend,
            scheduler_config=sched_config,
        )
        assert result.completed_requests == 10
        assert result.total_requests == 10
        assert result.total_batches > 0
        assert result.avg_latency_ms > 0

    async def test_burst_arrival_execution(self) -> None:
        backend = MockBackend(default_latency_sec=0.001)
        runner = BenchmarkRunner()
        scenario = get_burst_workload(seed=42)

        result = await runner.run(scenario=scenario, backend=backend)
        assert result.completed_requests == len(scenario.requests)
        assert result.failed_requests == 0
        assert result.peak_queue_depth > 0

    async def test_fixed_rate_arrival_execution(self) -> None:
        backend = MockBackend(default_latency_sec=0.001)
        runner = BenchmarkRunner()
        # Small 5-request fixed rate scenario for fast unit test
        cfg = WorkloadConfig(
            scenario_name="fixed-rate-fast",
            num_requests=5,
            arrival_pattern=ArrivalPattern.FIXED_RATE,
            arrival_rate_rps=100.0,
            max_tokens=16,
        )
        reqs = tuple(
            WorkloadRequestSpec(request_id=f"r-{i}", prompt=f"Prompt {i}") for i in range(5)
        )
        scenario = WorkloadScenario(scenario_name="fixed-rate-fast", config=cfg, requests=reqs)

        result = await runner.run(scenario=scenario, backend=backend)
        assert result.completed_requests == 5

    async def test_failing_backend_benchmark(self) -> None:
        backend = FailingBackend()
        runner = BenchmarkRunner()
        scenario = get_light_workload(seed=42)

        result = await runner.run(scenario=scenario, backend=backend)
        assert result.total_requests == 10
        assert result.completed_requests == 0
        assert result.failed_requests == 10

    async def test_non_batch_backend_benchmark(self) -> None:
        backend = NonBatchMockBackend()
        runner = BenchmarkRunner()
        scenario = get_light_workload(seed=42)

        result = await runner.run(scenario=scenario, backend=backend)
        assert result.completed_requests == 10
        assert result.backend_name == "non-batch-mock"
