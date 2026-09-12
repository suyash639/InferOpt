"""Unit and mock integration tests for VLLMBackend."""

import asyncio
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from inferopt.backends.base import BatchInferenceBackend, InferenceBackend
from inferopt.backends.vllm import DEFAULT_VLLM_MODEL_ID, VLLMBackend, VLLMConfig
from inferopt.core.exceptions import BackendError, InferenceError
from inferopt.core.models import InferenceBatch, InferenceRequest
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.scheduler.scheduler import AsyncScheduler

HAS_VLLM: bool = "vllm" in sys.modules
RUN_VLLM_EXPLICIT: bool = os.environ.get("INFEROPT_RUN_VLLM_TESTS") == "1"


def _create_mock_vllm_output(
    prompt: str = "Test prompt",
    text: str = "Test completion output",
    prompt_tokens: list[int] | None = None,
    output_tokens: list[int] | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    """Helper to construct a realistic mock RequestOutput object matching vLLM structure."""
    req_out = MagicMock()
    req_out.prompt = prompt
    req_out.prompt_token_ids = prompt_tokens if prompt_tokens is not None else [1, 2, 3, 4]

    comp_out = MagicMock()
    comp_out.text = text
    comp_out.token_ids = output_tokens if output_tokens is not None else [10, 11, 12, 13, 14]
    comp_out.finish_reason = finish_reason

    req_out.outputs = [comp_out]
    return req_out


class TestVLLMConfig:
    """Unit tests for VLLMConfig validation and immutability."""

    def test_default_configuration(self) -> None:
        config = VLLMConfig()
        assert config.model == DEFAULT_VLLM_MODEL_ID
        assert config.max_model_len is None
        assert config.gpu_memory_utilization == 0.9
        assert config.tensor_parallel_size == 1
        assert config.dtype == "auto"
        assert config.trust_remote_code is False
        assert config.seed == 42
        assert config.enforce_eager is False
        assert config.default_temperature == 0.0
        assert config.default_max_tokens == 128
        assert config.extra_engine_args == {}
        assert config.extra_sampling_args == {}

    def test_custom_valid_configuration(self) -> None:
        config = VLLMConfig(
            model="meta-llama/Llama-3.2-1B-Instruct",
            max_model_len=4096,
            gpu_memory_utilization=0.75,
            tensor_parallel_size=2,
            dtype="float16",
            trust_remote_code=True,
            seed=123,
            enforce_eager=True,
            default_temperature=0.5,
            default_max_tokens=256,
            extra_engine_args={"gpu_memory_utilization": 0.8},
            extra_sampling_args={"top_p": 0.95},
        )
        assert config.model == "meta-llama/Llama-3.2-1B-Instruct"
        assert config.max_model_len == 4096
        assert config.gpu_memory_utilization == 0.75
        assert config.tensor_parallel_size == 2
        assert config.dtype == "float16"
        assert config.trust_remote_code is True
        assert config.seed == 123
        assert config.enforce_eager is True
        assert config.default_temperature == 0.5
        assert config.default_max_tokens == 256
        assert config.extra_engine_args == {"gpu_memory_utilization": 0.8}
        assert config.extra_sampling_args == {"top_p": 0.95}

    def test_invalid_gpu_memory_utilization(self) -> None:
        with pytest.raises(ValidationError):
            VLLMConfig(gpu_memory_utilization=0.0)

        with pytest.raises(ValidationError):
            VLLMConfig(gpu_memory_utilization=1.1)

    def test_invalid_tensor_parallel_size(self) -> None:
        with pytest.raises(ValidationError):
            VLLMConfig(tensor_parallel_size=0)

    def test_invalid_temperature_bounds(self) -> None:
        with pytest.raises(ValidationError):
            VLLMConfig(default_temperature=-0.1)

        with pytest.raises(ValidationError):
            VLLMConfig(default_temperature=2.1)

    def test_invalid_max_tokens(self) -> None:
        with pytest.raises(ValidationError):
            VLLMConfig(default_max_tokens=0)

    def test_config_immutability(self) -> None:
        config = VLLMConfig()
        with pytest.raises(ValidationError):
            config.model = "new-model"

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            VLLMConfig(unknown_field="value")  # type: ignore[call-arg]


class TestVLLMBackendProtocolConformance:
    """Tests ensuring VLLMBackend strictly conforms to InferenceBackend protocols."""

    def test_protocol_implementation(self) -> None:
        backend = VLLMBackend()
        assert isinstance(backend, InferenceBackend)
        assert isinstance(backend, BatchInferenceBackend)
        assert backend.backend_name == "vllm"
        assert backend.model_id == DEFAULT_VLLM_MODEL_ID
        assert backend.default_temperature == 0.0
        assert backend.default_max_tokens == 128
        assert backend.is_loaded is False
        assert backend.model_load_time_ms == 0.0


class TestVLLMBackendMissingDependency:
    """Tests handling when optional vLLM dependency is not installed."""

    @pytest.mark.asyncio
    async def test_missing_vllm_import_raises_backend_error(self) -> None:
        backend = VLLMBackend()
        with patch.dict(sys.modules, {"vllm": None}):
            with pytest.raises(BackendError) as exc_info:
                await backend.load_model()
            assert "vLLM dependencies are not installed" in str(exc_info.value)
            assert "pip install 'inferopt[vllm]'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_vllm_generate_raises_backend_error(self) -> None:
        backend = VLLMBackend()
        req = InferenceRequest(model="vllm", prompt="Hello world")
        with patch.dict(sys.modules, {"vllm": None}), pytest.raises(BackendError):
            await backend.generate(req)


class TestVLLMBackendInitializationAndLifecycle:
    """Tests lazy engine initialization, idempotency, failure mapping, and unload."""

    @pytest.mark.asyncio
    async def test_lazy_initialization_success(self) -> None:
        mock_llm_cls = MagicMock()
        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance

        mock_vllm = MagicMock()
        mock_vllm.LLM = mock_llm_cls

        config = VLLMConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_model_len=2048,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=1,
            dtype="float16",
            trust_remote_code=True,
            seed=42,
        )
        backend = VLLMBackend(config=config)
        assert backend.is_loaded is False

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            await backend.load_model()

            assert backend.is_loaded is True
            assert backend.model_load_time_ms >= 0.0
            mock_llm_cls.assert_called_once_with(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                gpu_memory_utilization=0.85,
                tensor_parallel_size=1,
                dtype="float16",
                trust_remote_code=True,
                max_model_len=2048,
                seed=42,
            )

            # Idempotent: second call reuses loaded engine without re-initializing
            await backend.load_model()
            assert mock_llm_cls.call_count == 1

            # Unload releases engine instance
            await backend.unload_model()
            assert backend.is_loaded is False
            assert backend.model_load_time_ms == 0.0

    @pytest.mark.asyncio
    async def test_enforce_eager_mode_propagation(self) -> None:
        mock_llm_cls = MagicMock()
        mock_llm_cls.return_value = MagicMock()

        mock_vllm = MagicMock()
        mock_vllm.LLM = mock_llm_cls

        config_eager = VLLMConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype="float16",
            enforce_eager=True,
        )
        backend = VLLMBackend(config=config_eager)

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            await backend.load_model()
            assert backend.is_loaded is True
            mock_llm_cls.assert_called_once_with(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                gpu_memory_utilization=0.9,
                tensor_parallel_size=1,
                dtype="float16",
                trust_remote_code=False,
                seed=42,
                enforce_eager=True,
            )

    @pytest.mark.asyncio
    async def test_initialization_failure_raises_backend_error(self) -> None:
        mock_llm_cls = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        mock_vllm = MagicMock()
        mock_vllm.LLM = mock_llm_cls

        backend = VLLMBackend()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with pytest.raises(BackendError) as exc_info:
                await backend.load_model()
            assert "Failed to initialize vLLM engine" in str(exc_info.value)
            assert "CUDA out of memory" in str(exc_info.value)


class TestVLLMBackendSingleGeneration:
    """Tests single request generation, parameter mapping, and token accounting."""

    @pytest.mark.asyncio
    async def test_single_request_success(self) -> None:
        mock_output = _create_mock_vllm_output(
            prompt="What is dynamic batching?",
            text="Dynamic batching groups inference requests.",
            prompt_tokens=[10, 20, 30],
            output_tokens=[100, 101, 102, 103, 104],
            finish_reason="stop",
        )
        mock_llm = MagicMock()
        mock_llm.generate.return_value = [mock_output]

        mock_sampling_cls = MagicMock(return_value=MagicMock())
        mock_vllm = MagicMock()
        mock_vllm.LLM.return_value = mock_llm
        mock_vllm.SamplingParams = mock_sampling_cls

        backend = VLLMBackend(default_temperature=0.0, default_max_tokens=64)

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            req = InferenceRequest(
                request_id="vllm-req-1",
                model="vllm-model",
                prompt="What is dynamic batching?",
                max_tokens=32,
                temperature=0.0,
                metadata={"client": "test"},
            )
            response = await backend.generate(req)

            assert response.request_id == "vllm-req-1"
            assert response.generated_text == "Dynamic batching groups inference requests."
            assert response.input_tokens == 3
            assert response.output_tokens == 5
            assert response.latency_ms >= 0.0
            assert response.backend_name == "vllm"
            assert response.metadata.get("client") == "test"
            assert response.metadata.get("finish_reason") == "stop"

            mock_llm.generate.assert_called_once()
            mock_sampling_cls.assert_called_once_with(temperature=0.0, max_tokens=32)

    @pytest.mark.asyncio
    async def test_single_request_failure_raises_inference_error(self) -> None:
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("GPU kernel execution failed")

        mock_vllm = MagicMock()
        mock_vllm.LLM.return_value = mock_llm
        mock_vllm.SamplingParams = MagicMock()

        backend = VLLMBackend()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            req = InferenceRequest(request_id="fail-1", model="vllm", prompt="Test")
            with pytest.raises(InferenceError) as exc_info:
                await backend.generate(req)
            assert "vLLM generation failed for request 'fail-1'" in str(exc_info.value)


class TestVLLMBackendBatchedGeneration:
    """Tests native batched execution, 1:1 request ID mapping, and sequence order preservation."""

    @pytest.mark.asyncio
    async def test_batch_generation_success(self) -> None:
        out1 = _create_mock_vllm_output(
            prompt="Prompt 1",
            text="Response 1",
            prompt_tokens=[1, 2],
            output_tokens=[10, 11, 12],
            finish_reason="stop",
        )
        out2 = _create_mock_vllm_output(
            prompt="Prompt 2",
            text="Response 2",
            prompt_tokens=[3, 4, 5],
            output_tokens=[20, 21],
            finish_reason="length",
        )
        out3 = _create_mock_vllm_output(
            prompt="Prompt 3",
            text="Response 3",
            prompt_tokens=[6],
            output_tokens=[30, 31, 32, 33],
            finish_reason="stop",
        )

        mock_llm = MagicMock()
        mock_llm.generate.return_value = [out1, out2, out3]

        mock_vllm = MagicMock()
        mock_vllm.LLM.return_value = mock_llm
        mock_vllm.SamplingParams = MagicMock()

        backend = VLLMBackend()
        batch = InferenceBatch(
            requests=(
                InferenceRequest(request_id="b-1", model="vllm", prompt="Prompt 1", max_tokens=16),
                InferenceRequest(request_id="b-2", model="vllm", prompt="Prompt 2", max_tokens=32),
                InferenceRequest(request_id="b-3", model="vllm", prompt="Prompt 3", max_tokens=64),
            )
        )

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            responses = await backend.generate_batch(batch)

            assert len(responses) == 3
            # Exact input ordering and request ID mapping preserved
            assert responses[0].request_id == "b-1"
            assert responses[0].generated_text == "Response 1"
            assert responses[0].input_tokens == 2
            assert responses[0].output_tokens == 3
            assert responses[0].metadata.get("finish_reason") == "stop"

            assert responses[1].request_id == "b-2"
            assert responses[1].generated_text == "Response 2"
            assert responses[1].input_tokens == 3
            assert responses[1].output_tokens == 2
            assert responses[1].metadata.get("finish_reason") == "length"

            assert responses[2].request_id == "b-3"
            assert responses[2].generated_text == "Response 3"
            assert responses[2].input_tokens == 1
            assert responses[2].output_tokens == 4

            # Native batching passed all prompts in a single LLM.generate invocation
            mock_llm.generate.assert_called_once()
            called_kwargs = mock_llm.generate.call_args.kwargs
            assert called_kwargs.get("prompts") == ["Prompt 1", "Prompt 2", "Prompt 3"]
            assert len(called_kwargs.get("sampling_params", [])) == 3

    @pytest.mark.asyncio
    async def test_batch_generation_count_mismatch_raises_inference_error(self) -> None:
        mock_llm = MagicMock()
        # Returns 1 output for a 2-request batch
        mock_llm.generate.return_value = [_create_mock_vllm_output()]

        mock_vllm = MagicMock()
        mock_vllm.LLM.return_value = mock_llm
        mock_vllm.SamplingParams = MagicMock()

        backend = VLLMBackend()
        batch = InferenceBatch(
            requests=(
                InferenceRequest(request_id="b-1", model="vllm", prompt="Prompt 1"),
                InferenceRequest(request_id="b-2", model="vllm", prompt="Prompt 2"),
            )
        )

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with pytest.raises(InferenceError) as exc_info:
                await backend.generate_batch(batch)
            assert "vLLM batch returned 1 outputs, expected 2" in str(exc_info.value)


class TestVLLMBackendSchedulerIntegration:
    """End-to-end integration testing of VLLMBackend within AsyncScheduler dynamic batching."""

    @pytest.mark.asyncio
    async def test_scheduler_dynamic_batching_with_vllm_backend(self) -> None:
        mock_llm = MagicMock()

        def _mock_generate(
            prompts: list[str],
            sampling_params: Any,
            use_tqdm: bool = False,
        ) -> list[MagicMock]:
            return [
                _create_mock_vllm_output(
                    prompt=p,
                    text=f"Completion for {p}",
                    prompt_tokens=[1, 2],
                    output_tokens=[10, 11],
                )
                for p in prompts
            ]

        mock_llm.generate.side_effect = _mock_generate
        mock_vllm = MagicMock()
        mock_vllm.LLM.return_value = mock_llm
        mock_vllm.SamplingParams = MagicMock()

        backend = VLLMBackend()
        scheduler_config = SchedulerConfig(
            max_concurrency=2,
            batch_config=BatchConfig(max_batch_size=4, batch_wait_ms=20.0),
        )

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            async with AsyncScheduler(backend=backend, config=scheduler_config) as scheduler:
                reqs = [
                    InferenceRequest(
                        request_id=f"sched-vllm-{i}",
                        model="vllm",
                        prompt=f"Prompt {i}",
                        max_tokens=16,
                    )
                    for i in range(4)
                ]

                tasks = [asyncio.create_task(scheduler.submit(r)) for r in reqs]
                responses = await asyncio.gather(*tasks)

                assert len(responses) == 4
                for i, resp in enumerate(responses):
                    assert resp.request_id == f"sched-vllm-{i}"
                    assert resp.backend_name == "vllm"
                    assert resp.generated_text == f"Completion for Prompt {i}"


@pytest.mark.skipif(
    not (HAS_VLLM or RUN_VLLM_EXPLICIT),
    reason="Requires NVIDIA GPU and vLLM (set INFEROPT_RUN_VLLM_TESTS=1 to enforce)",
)
class TestLiveVLLMIntegration:
    """Opt-in live integration tests on real NVIDIA GPU hardware."""

    @pytest.mark.asyncio
    async def test_live_vllm_generation(self) -> None:
        backend = VLLMBackend()
        await backend.load_model()

        req = InferenceRequest(
            request_id="live-vllm-1",
            model=backend.model_id,
            prompt="What is InferOpt in one sentence?",
            max_tokens=32,
            temperature=0.0,
        )
        response = await backend.generate(req)
        assert response.request_id == "live-vllm-1"
        assert len(response.generated_text.strip()) > 0
        assert response.output_tokens is not None and response.output_tokens > 0
