"""Unit tests for safe runtime scheduler configuration updates and worker scaling."""

import asyncio

import pytest

from inferopt.backends.mock import MockBackend
from inferopt.core.models import InferenceRequest
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.scheduler.scheduler import Scheduler


@pytest.mark.asyncio
class TestSchedulerRuntimeConfig:
    """Tests for scheduler.apply_config runtime updates."""

    async def test_apply_valid_config_updates_parameters(self) -> None:
        backend = MockBackend(default_latency_sec=0.001)
        initial_cfg = SchedulerConfig(
            max_concurrency=2,
            batch_config=BatchConfig(max_batch_size=2, batch_wait_ms=0.0),
        )
        scheduler = Scheduler(backend=backend, config=initial_cfg)
        await scheduler.start()

        try:
            assert scheduler.config.max_concurrency == 2
            assert scheduler.batch_config.max_batch_size == 2
            assert scheduler.batch_config.batch_wait_ms == 0.0

            new_cfg = SchedulerConfig(
                max_concurrency=4,
                batch_config=BatchConfig(max_batch_size=8, batch_wait_ms=10.0),
            )
            scheduler.apply_config(new_cfg)

            assert scheduler.config.max_concurrency == 4
            assert scheduler.batch_config.max_batch_size == 8
            assert scheduler.batch_config.batch_wait_ms == 10.0
        finally:
            await scheduler.shutdown()

    async def test_apply_invalid_config_type_raises_error(self) -> None:
        backend = MockBackend()
        scheduler = Scheduler(backend=backend)
        with pytest.raises(TypeError, match="Expected SchedulerConfig"):
            scheduler.apply_config("invalid_config")  # type: ignore[arg-type]

    async def test_apply_config_running_requests_unaffected(self) -> None:
        backend = MockBackend(default_latency_sec=0.030)
        scheduler = Scheduler(
            backend=backend,
            config=SchedulerConfig(
                max_concurrency=2,
                batch_config=BatchConfig(max_batch_size=2, batch_wait_ms=0.0),
            ),
        )
        await scheduler.start()

        try:
            # Submit two requests that start running
            t1 = asyncio.create_task(
                scheduler.submit(
                    InferenceRequest(
                        request_id="r1",
                        model="mock-model",
                        prompt="Prompt 1",
                        max_tokens=10,
                    )
                )
            )
            t2 = asyncio.create_task(
                scheduler.submit(
                    InferenceRequest(
                        request_id="r2",
                        model="mock-model",
                        prompt="Prompt 2",
                        max_tokens=10,
                    )
                )
            )
            await asyncio.sleep(0.005)  # Let workers pick up batch

            # Update configuration while requests are executing
            new_cfg = SchedulerConfig(
                max_concurrency=4,
                batch_config=BatchConfig(max_batch_size=16, batch_wait_ms=5.0),
            )
            scheduler.apply_config(new_cfg)

            # Both original requests must complete successfully
            res1, res2 = await asyncio.gather(t1, t2)
            assert res1.request_id == "r1"
            assert res2.request_id == "r2"
            assert res1.generated_text != ""
            assert res2.generated_text != ""
        finally:
            await scheduler.shutdown()

    async def test_apply_config_queued_requests_preserved(self) -> None:
        backend = MockBackend(default_latency_sec=0.020)
        scheduler = Scheduler(
            backend=backend,
            config=SchedulerConfig(
                max_concurrency=1,
                batch_config=BatchConfig(max_batch_size=1, batch_wait_ms=0.0),
            ),
        )
        await scheduler.start()

        try:
            # Submit 5 requests: first runs, rest are queued
            tasks = [
                asyncio.create_task(
                    scheduler.submit(
                        InferenceRequest(
                            request_id=f"req-{i}",
                            model="mock-model",
                            prompt=f"Prompt {i}",
                            priority=i,
                            max_tokens=10,
                        )
                    )
                )
                for i in range(5)
            ]
            await asyncio.sleep(0.005)

            # Update batch config so queued items are batched together
            new_cfg = SchedulerConfig(
                max_concurrency=1,
                batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=0.0),
            )
            scheduler.apply_config(new_cfg)

            responses = await asyncio.gather(*tasks)
            assert len(responses) == 5
            assert all(r.generated_text != "" for r in responses)

            # Verify batch metrics recorded larger batch formation
            batches = scheduler.collector.get_recent_batches()
            assert any(b.size > 1 for b in batches)
        finally:
            await scheduler.shutdown()

    async def test_dynamic_worker_scale_up_and_scale_down(self) -> None:
        backend = MockBackend(default_latency_sec=0.015)
        scheduler = Scheduler(
            backend=backend,
            config=SchedulerConfig(
                max_concurrency=1,
                batch_config=BatchConfig(max_batch_size=1, batch_wait_ms=0.0),
            ),
        )
        await scheduler.start()

        try:
            # Scale up to 4 workers
            scheduler.apply_config(
                SchedulerConfig(
                    max_concurrency=4,
                    batch_config=BatchConfig(max_batch_size=1, batch_wait_ms=0.0),
                )
            )
            assert scheduler.config.max_concurrency == 4

            # Submit 4 requests concurrently; with concurrency 4 they should execute simultaneously
            t_start = asyncio.get_running_loop().time()
            tasks = [
                asyncio.create_task(
                    scheduler.submit(
                        InferenceRequest(
                            request_id=f"scale-{i}",
                            model="mock-model",
                            prompt="Test",
                            max_tokens=5,
                        )
                    )
                )
                for i in range(4)
            ]
            responses = await asyncio.gather(*tasks)
            t_elapsed = asyncio.get_running_loop().time() - t_start
            assert len(responses) == 4
            # If sequential, would take 4 * 15ms = 60ms; concurrent ~ 15-30ms
            assert t_elapsed < 0.055

            # Scale down to 1 worker
            scheduler.apply_config(
                SchedulerConfig(
                    max_concurrency=1,
                    batch_config=BatchConfig(max_batch_size=1, batch_wait_ms=0.0),
                )
            )
            assert scheduler.config.max_concurrency == 1
        finally:
            await scheduler.shutdown()
