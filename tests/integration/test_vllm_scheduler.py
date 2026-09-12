"""End-to-end integration tests validating Scheduler -> Dynamic Batching -> VLLMBackend pipeline."""

import asyncio
import os
import sys
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from inferopt.backends.vllm import DEFAULT_VLLM_MODEL_ID, VLLMBackend, VLLMConfig
from inferopt.core.exceptions import QueueFullError, SchedulerShutdownError
from inferopt.core.models import InferenceRequest
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.scheduler.lifecycle import RequestStatus
from inferopt.scheduler.scheduler import AsyncScheduler, Scheduler
from inferopt.telemetry.collector import MetricsCollector

HAS_VLLM: bool = "vllm" in sys.modules
RUN_VLLM_EXPLICIT: bool = os.environ.get("INFEROPT_RUN_VLLM_TESTS") == "1"


def _create_mock_vllm_output(
    prompt: str = "Test prompt",
    text: str = "Test response from vLLM backend",
    prompt_tokens: list[int] | None = None,
    output_tokens: list[int] | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    """Construct a mock vLLM RequestOutput matching the internal vLLM data structure."""
    req_out = MagicMock()
    req_out.prompt = prompt
    req_out.prompt_token_ids = prompt_tokens if prompt_tokens is not None else [101, 102, 103, 104]

    comp_out = MagicMock()
    comp_out.text = text
    comp_out.token_ids = (
        output_tokens if output_tokens is not None else [201, 202, 203, 204, 205, 206]
    )
    comp_out.finish_reason = finish_reason

    req_out.outputs = [comp_out]
    return req_out


def _setup_mock_vllm_engine(delay: float = 0.0) -> tuple[MagicMock, list[list[str]]]:
    """Configure a realistic mock vLLM module that records dispatched batches."""
    dispatched_batches: list[list[str]] = []
    mock_llm_instance = MagicMock()

    def _mock_generate(
        prompts: list[str],
        sampling_params: Any = None,
        use_tqdm: bool = False,
    ) -> list[MagicMock]:
        dispatched_batches.append(list(prompts))
        if delay > 0:
            time.sleep(delay)
        return [
            _create_mock_vllm_output(
                prompt=p,
                text=f"Completed response for: {p}",
                prompt_tokens=[1, 2, 3, 4],
                output_tokens=[10, 11, 12, 13, 14, 15, 16, 17],
            )
            for p in prompts
        ]

    mock_llm_instance.generate.side_effect = _mock_generate

    mock_llm_cls = MagicMock(return_value=mock_llm_instance)
    mock_sampling_cls = MagicMock(side_effect=lambda **kwargs: MagicMock(**kwargs))

    mock_vllm = MagicMock()
    mock_vllm.LLM = mock_llm_cls
    mock_vllm.SamplingParams = mock_sampling_cls

    return mock_vllm, dispatched_batches


class TestVLLMSchedulerSingleRequest:
    """Integration test validating single request end-to-end execution through VLLMBackend."""

    @pytest.mark.asyncio
    async def test_single_request_end_to_end(self) -> None:
        mock_vllm, _ = _setup_mock_vllm_engine()
        config = VLLMConfig(model=DEFAULT_VLLM_MODEL_ID)
        backend = VLLMBackend(config=config)
        collector = MetricsCollector()
        scheduler_config = SchedulerConfig(
            max_concurrency=2,
            batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=0.0),
        )

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            async with Scheduler(
                backend=backend, config=scheduler_config, collector=collector
            ) as scheduler:
                req = InferenceRequest(
                    request_id="vllm-e2e-single-1",
                    model=DEFAULT_VLLM_MODEL_ID,
                    prompt="Explain what InferOpt does in one sentence.",
                    max_tokens=32,
                    temperature=0.0,
                )

                response = await scheduler.submit(req)

                # 1. Request completed successfully
                assert response is not None
                # 2. Request ID matches
                assert response.request_id == "vllm-e2e-single-1"
                # 3. Output is non-empty
                assert response.generated_text != ""
                assert "Completed response for:" in response.generated_text
                # 4. Input and output token counts are valid (> 0)
                assert response.input_tokens == 4
                assert response.output_tokens == 8
                total_tokens = (response.input_tokens or 0) + (response.output_tokens or 0)
                assert total_tokens == 12
                # 5. Backend identifier matches
                assert response.backend_name == "vllm"

                # Verify telemetry recorded
                snapshot = collector.snapshot()
                assert snapshot.requests.total_requests == 1
                assert snapshot.requests.completed_requests == 1
                assert snapshot.requests.failed_requests == 0

    @pytest.mark.asyncio
    async def test_single_request_eager_diagnostic_mode(self) -> None:
        """Verify pipeline execution with enforce_eager diagnostic mode enabled."""
        mock_vllm, _ = _setup_mock_vllm_engine()
        config = VLLMConfig(model=DEFAULT_VLLM_MODEL_ID, enforce_eager=True)
        backend = VLLMBackend(config=config)
        collector = MetricsCollector()
        scheduler_config = SchedulerConfig(
            max_concurrency=2,
            batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=0.0),
        )

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            async with Scheduler(
                backend=backend, config=scheduler_config, collector=collector
            ) as scheduler:
                req = InferenceRequest(
                    request_id="vllm-e2e-eager-1",
                    model=DEFAULT_VLLM_MODEL_ID,
                    prompt="Testing eager diagnostic mode.",
                    max_tokens=32,
                    temperature=0.0,
                )

                response = await scheduler.submit(req)
                assert response.request_id == "vllm-e2e-eager-1"
                assert response.output_tokens == 8
                assert mock_vllm.LLM.call_args.kwargs.get("enforce_eager") is True


class TestVLLMSchedulerDynamicBatching:
    """Integration tests validating dynamic batch formation and request coalescing."""

    @pytest.mark.asyncio
    async def test_deterministic_batch_formation(self) -> None:
        """Verify 4 concurrent requests coalesce into a single batch of 4 with wait window."""
        mock_vllm, dispatched_batches = _setup_mock_vllm_engine(delay=0.01)
        config = VLLMConfig(model=DEFAULT_VLLM_MODEL_ID)
        backend = VLLMBackend(config=config)
        collector = MetricsCollector()

        # Generous batch_wait_ms to guarantee all 4 requests are grouped into 1 batch
        scheduler_config = SchedulerConfig(
            max_concurrency=2,
            batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=100.0),
        )

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            async with Scheduler(
                backend=backend, config=scheduler_config, collector=collector
            ) as scheduler:
                requests = [
                    InferenceRequest(
                        request_id=f"det-req-{i}",
                        model=DEFAULT_VLLM_MODEL_ID,
                        prompt=f"Deterministic prompt {i}",
                        max_tokens=32,
                    )
                    for i in range(4)
                ]

                # Submit all 4 requests concurrently
                responses = await asyncio.gather(*[scheduler.submit(r) for r in requests])

                # 1. Exactly 4 requests completed
                assert len(responses) == 4

                # 2. All request IDs are preserved without loss or duplicates
                resp_ids = [r.request_id for r in responses]
                assert resp_ids == [f"det-req-{i}" for i in range(4)]
                assert len(set(resp_ids)) == 4

                # 3. All outputs are non-empty
                for resp in responses:
                    assert resp.generated_text != ""
                    assert resp.backend_name == "vllm"
                    assert (resp.output_tokens or 0) > 0

                # 4. Engine received exactly 1 batch of 4
                assert len(dispatched_batches) == 1
                assert len(dispatched_batches[0]) == 4

                # 5. Telemetry validation
                snapshot = collector.snapshot()
                assert snapshot.requests.total_requests == 4
                assert snapshot.requests.completed_requests == 4
                assert snapshot.batches.total_batches == 1
                assert snapshot.batches.avg_batch_size == 4.0

                batch_metrics = collector.get_recent_batches()
                assert len(batch_metrics) == 1
                assert batch_metrics[0].size == 4
                assert list(batch_metrics[0].request_ids) == [f"det-req-{i}" for i in range(4)]

                # Verify each request telemetry contains the matching batch_id
                req_metrics = collector.get_recent_requests()
                assert len(req_metrics) == 4
                batch_id = batch_metrics[0].batch_id
                for rm in req_metrics:
                    assert rm.batch_id == batch_id
                    assert rm.status == RequestStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_normal_dynamic_batching_integration(self) -> None:
        """Verify normal dynamic batching with multiple requests preserves all invariants."""
        mock_vllm, dispatched_batches = _setup_mock_vllm_engine()
        config = VLLMConfig(model=DEFAULT_VLLM_MODEL_ID)
        backend = VLLMBackend(config=config)
        collector = MetricsCollector()

        scheduler_config = SchedulerConfig(
            max_concurrency=4,
            batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=0.0),
        )

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            async with AsyncScheduler(
                backend=backend, config=scheduler_config, collector=collector
            ) as scheduler:
                requests = [
                    InferenceRequest(
                        request_id=f"dyn-req-{i}",
                        model=DEFAULT_VLLM_MODEL_ID,
                        prompt=f"Dynamic prompt {i}",
                        max_tokens=32,
                    )
                    for i in range(4)
                ]

                responses = await asyncio.gather(*[scheduler.submit(r) for r in requests])

                assert len(responses) == 4
                resp_ids = {r.request_id for r in responses}
                assert resp_ids == {f"dyn-req-{i}" for i in range(4)}

                for resp in responses:
                    assert resp.generated_text != ""
                    assert resp.backend_name == "vllm"

                # Total prompts dispatched across batches equals 4
                total_prompts_dispatched = sum(len(b) for b in dispatched_batches)
                assert total_prompts_dispatched == 4

                snapshot = collector.snapshot()
                assert snapshot.requests.total_requests == 4
                assert snapshot.requests.completed_requests == 4
                assert snapshot.requests.failed_requests == 0


class TestVLLMSchedulerBatchLimits:
    """Integration tests verifying strict enforcement of max_batch_size limits."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_batch_size", [1, 2, 4])
    async def test_batch_size_limits_enforced(self, max_batch_size: int) -> None:
        """Verify scheduler never dispatches a batch larger than max_batch_size."""
        mock_vllm, dispatched_batches = _setup_mock_vllm_engine(delay=0.01)
        config = VLLMConfig(model=DEFAULT_VLLM_MODEL_ID)
        backend = VLLMBackend(config=config)
        collector = MetricsCollector()

        scheduler_config = SchedulerConfig(
            max_concurrency=4,
            batch_config=BatchConfig(max_batch_size=max_batch_size, batch_wait_ms=50.0),
        )

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            async with Scheduler(
                backend=backend, config=scheduler_config, collector=collector
            ) as scheduler:
                requests = [
                    InferenceRequest(
                        request_id=f"limit-req-{max_batch_size}-{i}",
                        model=DEFAULT_VLLM_MODEL_ID,
                        prompt=f"Limit prompt {i}",
                        max_tokens=32,
                    )
                    for i in range(4)
                ]

                responses = await asyncio.gather(*[scheduler.submit(r) for r in requests])

                assert len(responses) == 4

                # Assert that every dispatched batch respected the limit
                for batch in dispatched_batches:
                    assert len(batch) <= max_batch_size, (
                        f"Dispatched batch size {len(batch)} "
                        f"exceeds max_batch_size {max_batch_size}"
                    )

                # Batch telemetry also reflects max_batch_size bound
                for bm in collector.get_recent_batches():
                    assert bm.size <= max_batch_size


class TestVLLMSchedulerBackpressure:
    """Integration tests verifying scheduler backpressure, queue limits, and isolation."""

    @pytest.mark.asyncio
    async def test_queue_full_backpressure_and_backend_health(self) -> None:
        """Verify QueueFullError is raised on queue saturation without corrupting VLLMBackend."""
        mock_vllm, _ = _setup_mock_vllm_engine(delay=0.1)
        config = VLLMConfig(model=DEFAULT_VLLM_MODEL_ID)
        backend = VLLMBackend(config=config)
        collector = MetricsCollector()

        # Single worker, queue size = 1
        scheduler_config = SchedulerConfig(
            max_concurrency=1,
            max_queue_size=1,
            batch_config=BatchConfig(max_batch_size=1, batch_wait_ms=0.0),
        )

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            async with Scheduler(
                backend=backend, config=scheduler_config, collector=collector
            ) as scheduler:
                # 1. First request occupies worker (running)
                req1 = InferenceRequest(
                    request_id="bp-1",
                    model=DEFAULT_VLLM_MODEL_ID,
                    prompt="Prompt 1",
                )
                task1 = asyncio.create_task(scheduler.submit(req1))
                await asyncio.sleep(0.01)

                # 2. Second request fills the queue (queued, max_queue_size=1)
                req2 = InferenceRequest(
                    request_id="bp-2",
                    model=DEFAULT_VLLM_MODEL_ID,
                    prompt="Prompt 2",
                )
                task2 = asyncio.create_task(scheduler.submit(req2))
                await asyncio.sleep(0.01)

                # 3. Third request exceeds queue capacity -> QueueFullError
                req3 = InferenceRequest(
                    request_id="bp-3",
                    model=DEFAULT_VLLM_MODEL_ID,
                    prompt="Prompt 3",
                )
                with pytest.raises(QueueFullError):
                    await scheduler.submit(req3)

                # 4. First and second requests complete successfully
                resp1 = await task1
                resp2 = await task2

                assert resp1.request_id == "bp-1"
                assert resp2.request_id == "bp-2"
                assert resp1.generated_text != ""
                assert resp2.generated_text != ""

                # 5. VLLMBackend remains healthy for subsequent requests
                req4 = InferenceRequest(
                    request_id="bp-4",
                    model=DEFAULT_VLLM_MODEL_ID,
                    prompt="Prompt 4",
                )
                resp4 = await scheduler.submit(req4)
                assert resp4.request_id == "bp-4"
                assert resp4.generated_text != ""


class TestVLLMSchedulerLifecycleAndShutdown:
    """Integration tests verifying graceful shutdown, task cleanup, and backend unloading."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_and_backend_unload(self) -> None:
        """Verify scheduler shutdown drains/cancels cleanly and VLLMBackend unloads."""
        mock_vllm, _ = _setup_mock_vllm_engine(delay=0.05)
        config = VLLMConfig(model=DEFAULT_VLLM_MODEL_ID)
        backend = VLLMBackend(config=config)
        collector = MetricsCollector()

        scheduler = Scheduler(
            backend=backend,
            config=SchedulerConfig(
                max_concurrency=1,
                batch_config=BatchConfig(max_batch_size=1, batch_wait_ms=0.0),
            ),
            collector=collector,
        )

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            await scheduler.start()

            # Submit first request (will be running)
            req1 = InferenceRequest(
                request_id="shut-1",
                model=DEFAULT_VLLM_MODEL_ID,
                prompt="Shutdown prompt 1",
            )
            task1 = asyncio.create_task(scheduler.submit(req1))
            await asyncio.sleep(0.01)

            # Submit second request (will be queued)
            req2 = InferenceRequest(
                request_id="shut-2",
                model=DEFAULT_VLLM_MODEL_ID,
                prompt="Shutdown prompt 2",
            )
            task2 = asyncio.create_task(scheduler.submit(req2))
            await asyncio.sleep(0.01)

            # Initiate shutdown: wait for running task1, cancel queued task2
            await scheduler.shutdown(wait_running=True, cancel_queued=True, timeout=2.0)

            # Task 1 completed
            resp1 = await task1
            assert resp1.request_id == "shut-1"

            # Task 2 cancelled due to shutdown
            with pytest.raises(SchedulerShutdownError):
                await task2

            # Verify no worker task leaks
            assert scheduler.is_running is False
            assert len(scheduler._workers) == 0

            # Verify backend unload
            assert backend.is_loaded is True
            await backend.unload_model()
            assert backend.is_loaded is False


class TestVLLMSchedulerTelemetryValidation:
    """Integration tests validating telemetry consistency and timing relationships."""

    @pytest.mark.asyncio
    async def test_telemetry_semantics_and_timing_consistency(self) -> None:
        """Verify RequestMetrics and BatchMetrics timing semantics and fields."""
        mock_vllm, _ = _setup_mock_vllm_engine(delay=0.02)
        config = VLLMConfig(model=DEFAULT_VLLM_MODEL_ID)
        backend = VLLMBackend(config=config)
        collector = MetricsCollector()

        scheduler_config = SchedulerConfig(
            max_concurrency=2,
            batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=50.0),
        )

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            async with Scheduler(
                backend=backend, config=scheduler_config, collector=collector
            ) as scheduler:
                requests = [
                    InferenceRequest(
                        request_id=f"telem-req-{i}",
                        model=DEFAULT_VLLM_MODEL_ID,
                        prompt=f"Telemetry prompt {i}",
                        max_tokens=32,
                    )
                    for i in range(4)
                ]

                responses = await asyncio.gather(*[scheduler.submit(r) for r in requests])
                assert len(responses) == 4

        # Validate BatchMetrics
        batches = collector.get_recent_batches()
        assert len(batches) == 1
        bm = batches[0]
        assert bm.size == 4
        assert bm.backend_name == "vllm"
        assert bm.batch_formation_wait_ms >= 0.0
        assert bm.execution_ms >= 0.0
        assert set(bm.request_ids) == {f"telem-req-{i}" for i in range(4)}
        assert bm.completed_request_count == 4
        assert bm.failed_request_count == 0

        # Validate RequestMetrics
        reqs = collector.get_recent_requests()
        assert len(reqs) == 4
        for rm in reqs:
            assert rm.status == RequestStatus.COMPLETED
            assert rm.backend_name == "vllm"
            assert rm.batch_id == bm.batch_id
            assert rm.input_tokens == 4
            assert rm.output_tokens == 8
            assert rm.queue_wait_ms >= 0.0
            assert rm.execution_ms >= 0.0
            # Total latency must be greater than or equal to queue wait time
            assert rm.total_latency_ms >= rm.queue_wait_ms


class TestLiveVLLMSchedulerIntegration:
    """Hardware integration tests using real vLLM on NVIDIA GPUs when available."""

    @pytest.mark.skipif(
        not (HAS_VLLM or RUN_VLLM_EXPLICIT),
        reason="Real vLLM integration tests require an NVIDIA GPU with vLLM installed.",
    )
    @pytest.mark.asyncio
    async def test_live_vllm_scheduler_e2e(self) -> None:
        """Execute real 4-request dynamic batching end-to-end on NVIDIA GPU."""
        config = VLLMConfig(
            model=DEFAULT_VLLM_MODEL_ID,
            default_max_tokens=16,
            default_temperature=0.0,
        )
        backend = VLLMBackend(config=config)
        collector = MetricsCollector()
        scheduler_config = SchedulerConfig(
            max_concurrency=2,
            batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=50.0),
        )

        try:
            async with Scheduler(
                backend=backend, config=scheduler_config, collector=collector
            ) as scheduler:
                requests = [
                    InferenceRequest(
                        request_id=f"live-vllm-req-{i}",
                        model=DEFAULT_VLLM_MODEL_ID,
                        prompt=f"Count to {i + 1}:",
                        max_tokens=16,
                    )
                    for i in range(4)
                ]

                responses = await asyncio.gather(*[scheduler.submit(r) for r in requests])

                assert len(responses) == 4
                for resp in responses:
                    assert resp.generated_text != ""
                    assert resp.backend_name == "vllm"
                    assert (resp.output_tokens or 0) > 0

                snapshot = collector.snapshot()
                assert snapshot.requests.total_requests == 4
                assert snapshot.requests.completed_requests == 4
        finally:
            await backend.unload_model()
