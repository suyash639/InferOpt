"""Benchmark runner executing synthetic workloads against InferOpt schedulers and backends."""

import asyncio
import time
import uuid
from typing import Any

from inferopt.backends.base import InferenceBackend
from inferopt.benchmarks.models import (
    ArrivalPattern,
    BenchmarkResult,
    WorkloadRequestSpec,
    WorkloadScenario,
)
from inferopt.core.models import InferenceResponse
from inferopt.scheduler.config import SchedulerConfig
from inferopt.scheduler.scheduler import Scheduler
from inferopt.telemetry.collector import MetricsCollector


class BenchmarkRunner:
    """Orchestrates benchmark runs against InferOpt serving pipelines.

    Coordinates workload dispatch according to specified arrival patterns,
    collects telemetry snapshots, and formats reproducible BenchmarkResults.
    """

    def __init__(self, collector: MetricsCollector | None = None) -> None:
        """Initialize benchmark runner with an optional metrics collector.

        Args:
            collector: Optional MetricsCollector instance.
        """
        self._collector = collector

    async def run(
        self,
        scenario: WorkloadScenario,
        backend: InferenceBackend,
        scheduler_config: SchedulerConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Execute a workload scenario against the provided backend and scheduler configuration.

        Args:
            scenario: WorkloadScenario containing requests and arrival configuration.
            backend: Target InferenceBackend implementation (e.g. MockBackend, MLXBackend).
            scheduler_config: Optional configuration for Scheduler and batching.
            metadata: Optional additional execution metadata or environment context.

        Returns:
            Immutable BenchmarkResult containing telemetry metrics and throughput summary.
        """
        config = scheduler_config or SchedulerConfig()
        collector = self._collector or MetricsCollector()
        collector.reset()

        start_time = time.perf_counter()

        async with Scheduler(backend=backend, config=config, collector=collector) as scheduler:
            await self._dispatch_workload(scheduler, scenario)

        duration_sec = max(0.0001, time.perf_counter() - start_time)
        snapshot = collector.snapshot(metadata=metadata)

        # Compute throughput metrics across the benchmark run duration
        completed = snapshot.requests.completed_requests
        req_per_sec = completed / duration_sec if duration_sec > 0 else 0.0
        batch_per_sec = snapshot.batches.total_batches / duration_sec if duration_sec > 0 else 0.0
        total_tokens = (
            snapshot.throughput.total_input_tokens + snapshot.throughput.total_output_tokens
        )
        tokens_per_sec = total_tokens / duration_sec if duration_sec > 0 else 0.0

        backend_name = getattr(backend, "backend_name", "unknown")

        return BenchmarkResult(
            benchmark_id=f"bench-{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            scenario_name=scenario.scenario_name,
            workload_config=scenario.config,
            backend_name=backend_name,
            scheduler_config=config,
            batch_config=config.batch_config,
            telemetry_snapshot=snapshot,
            duration_sec=duration_sec,
            total_requests=snapshot.requests.total_requests,
            completed_requests=completed,
            failed_requests=snapshot.requests.failed_requests,
            cancelled_requests=snapshot.requests.cancelled_requests,
            requests_per_sec=req_per_sec,
            batches_per_sec=batch_per_sec,
            tokens_per_sec=tokens_per_sec,
            avg_latency_ms=snapshot.requests.avg_total_latency_ms,
            p50_latency_ms=snapshot.requests.p50_total_latency_ms,
            p95_latency_ms=snapshot.requests.p95_total_latency_ms,
            p99_latency_ms=snapshot.requests.p99_total_latency_ms,
            avg_queue_wait_ms=snapshot.requests.avg_queue_wait_ms,
            avg_execution_ms=snapshot.requests.avg_execution_ms,
            peak_queue_depth=snapshot.queue.peak_queue_depth,
            peak_active_requests=snapshot.queue.peak_active_requests,
            total_batches=snapshot.batches.total_batches,
            avg_batch_size=snapshot.batches.avg_batch_size,
            max_batch_size=snapshot.batches.max_batch_size,
            metadata=dict(metadata or {}),
        )

    async def _dispatch_workload(
        self, scheduler: Scheduler, scenario: WorkloadScenario
    ) -> list[InferenceResponse | Exception]:
        """Dispatch workload requests according to the configured arrival pattern."""
        pattern = scenario.config.arrival_pattern

        if pattern == ArrivalPattern.SEQUENTIAL:
            return await self._dispatch_sequential(scheduler, scenario.requests)
        elif pattern == ArrivalPattern.CONCURRENT:
            return await self._dispatch_concurrent(scheduler, scenario.requests)
        elif pattern == ArrivalPattern.BURST:
            return await self._dispatch_burst(scheduler, scenario.requests)
        elif pattern == ArrivalPattern.FIXED_RATE:
            rate_rps = scenario.config.arrival_rate_rps or 10.0
            return await self._dispatch_fixed_rate(scheduler, scenario.requests, rate_rps)
        else:
            return await self._dispatch_concurrent(scheduler, scenario.requests)

    async def _dispatch_sequential(
        self, scheduler: Scheduler, requests: tuple[WorkloadRequestSpec, ...]
    ) -> list[InferenceResponse | Exception]:
        """Dispatch requests sequentially, awaiting each before submitting the next."""
        results: list[InferenceResponse | Exception] = []
        for req_spec in requests:
            try:
                resp = await scheduler.submit(req_spec.to_inference_request())
                results.append(resp)
            except Exception as exc:
                results.append(exc)
        return results

    async def _dispatch_concurrent(
        self, scheduler: Scheduler, requests: tuple[WorkloadRequestSpec, ...]
    ) -> list[InferenceResponse | Exception]:
        """Dispatch all requests concurrently."""

        async def _submit(spec: WorkloadRequestSpec) -> InferenceResponse | Exception:
            try:
                return await scheduler.submit(spec.to_inference_request())
            except Exception as exc:
                return exc

        tasks = [_submit(spec) for spec in requests]
        res = await asyncio.gather(*tasks)
        return list(res)

    async def _dispatch_burst(
        self, scheduler: Scheduler, requests: tuple[WorkloadRequestSpec, ...]
    ) -> list[InferenceResponse | Exception]:
        """Dispatch requests in a synchronized burst using an asyncio.Event barrier."""
        barrier = asyncio.Event()
        ready_count = 0
        ready_lock = asyncio.Lock()
        all_ready = asyncio.Event()

        async def _burst_submit(spec: WorkloadRequestSpec) -> InferenceResponse | Exception:
            nonlocal ready_count
            async with ready_lock:
                ready_count += 1
                if ready_count == len(requests):
                    all_ready.set()
            await barrier.wait()
            try:
                return await scheduler.submit(spec.to_inference_request())
            except Exception as exc:
                return exc

        tasks = [asyncio.create_task(_burst_submit(spec)) for spec in requests]
        await all_ready.wait()
        barrier.set()
        res = await asyncio.gather(*tasks)
        return list(res)

    async def _dispatch_fixed_rate(
        self,
        scheduler: Scheduler,
        requests: tuple[WorkloadRequestSpec, ...],
        rate_rps: float,
    ) -> list[InferenceResponse | Exception]:
        """Dispatch requests at approximate fixed-rate intervals."""
        start_benchmark = time.perf_counter()

        async def _scheduled_submit(
            spec: WorkloadRequestSpec, target_offset_sec: float
        ) -> InferenceResponse | Exception:
            elapsed = time.perf_counter() - start_benchmark
            remaining = target_offset_sec - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            try:
                return await scheduler.submit(spec.to_inference_request())
            except Exception as exc:
                return exc

        tasks = [
            asyncio.create_task(_scheduled_submit(spec, target_offset_sec=(i / rate_rps)))
            for i, spec in enumerate(requests)
        ]
        res = await asyncio.gather(*tasks)
        return list(res)
