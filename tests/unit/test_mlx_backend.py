"""Unit and integration tests for MLXBackend on Apple Silicon."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from inferopt.backends.base import BatchInferenceBackend, InferenceBackend
from inferopt.backends.mlx import DEFAULT_MODEL_ID, MLXBackend
from inferopt.core.exceptions import BackendError, InferenceError
from inferopt.core.models import InferenceBatch, InferenceRequest

try:
    import mlx.core as mx  # noqa: F401
    import mlx_lm  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class TestMLXBackendProtocolAndConfig:
    """Tests verifying MLXBackend protocols, defaults, and configuration options."""

    def test_implements_protocol(self) -> None:
        backend = MLXBackend()
        assert isinstance(backend, InferenceBackend)
        assert isinstance(backend, BatchInferenceBackend)
        assert backend.backend_name == "mlx"
        assert backend.model_id == DEFAULT_MODEL_ID
        assert backend.default_temperature == 0.0
        assert backend.default_max_tokens == 128
        assert backend.is_loaded is False

    def test_custom_configuration(self) -> None:
        backend = MLXBackend(
            model_id="custom-org/custom-model",
            default_temperature=0.7,
            default_max_tokens=64,
            tokenizer_config={"trust_remote_code": True},
            model_config={"dtype": "float16"},
            adapter_path="/path/to/adapters",
        )
        assert backend.model_id == "custom-org/custom-model"
        assert backend.default_temperature == 0.7
        assert backend.default_max_tokens == 64


@pytest.mark.asyncio
class TestMLXBackendMockedExecution:
    """Tests verifying execution flow, response mapping, and error handling with mocks."""

    async def test_missing_mlx_raises_backend_error(self) -> None:
        backend = MLXBackend()
        with (
            patch.dict(sys.modules, {"mlx_lm": None}),
            pytest.raises(BackendError, match="MLX dependencies are not installed"),
        ):
            await backend.load_model()

    async def test_mocked_generate_success(self) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text: (
            [1, 2, 3, 4] if "Prompt" in text else [5, 6]
        )

        backend = MLXBackend()
        backend._model = mock_model
        backend._tokenizer = mock_tokenizer

        with patch("mlx_lm.generate", return_value="Generated completion"):
            req = InferenceRequest(
                request_id="req-mock-1",
                model="mock-model",
                prompt="Prompt text",
                max_tokens=20,
            )
            resp = await backend.generate(req)
            assert resp.request_id == "req-mock-1"
            assert resp.generated_text == "Generated completion"
            assert resp.input_tokens == 4
            assert resp.output_tokens == 2
            assert resp.latency_ms >= 0.0
            assert resp.backend_name == "mlx"

    async def test_mocked_generate_error_wrapped_in_inference_error(self) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2]

        backend = MLXBackend()
        backend._model = mock_model
        backend._tokenizer = mock_tokenizer

        with patch("mlx_lm.generate", side_effect=RuntimeError("Metal device out of memory")):
            req = InferenceRequest(
                request_id="req-fail",
                model="mock-model",
                prompt="Prompt text",
            )
            with pytest.raises(InferenceError, match="MLX generation failed"):
                await backend.generate(req)

    async def test_mocked_batch_generate_success(self) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text: [1, 2, 3] if "1" in text else [4, 5]

        mock_batch_resp = MagicMock()
        mock_batch_resp.texts = ["Answer 1", "Answer 2"]

        backend = MLXBackend()
        backend._model = mock_model
        backend._tokenizer = mock_tokenizer

        with patch("mlx_lm.batch_generate", return_value=mock_batch_resp):
            batch = InferenceBatch(
                requests=(
                    InferenceRequest(request_id="r1", model="m", prompt="Prompt 1"),
                    InferenceRequest(request_id="r2", model="m", prompt="Prompt 2"),
                )
            )
            responses = await backend.generate_batch(batch)
            assert len(responses) == 2
            assert responses[0].request_id == "r1"
            assert responses[0].generated_text == "Answer 1"
            assert responses[1].request_id == "r2"
            assert responses[1].generated_text == "Answer 2"

    async def test_unload_model(self) -> None:
        backend = MLXBackend()
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()
        assert backend.is_loaded is True

        await backend.unload_model()
        assert backend.is_loaded is False


@pytest.mark.skipif(not HAS_MLX, reason="MLX and mlx-lm must be installed on Apple Silicon")
@pytest.mark.asyncio
class TestMLXBackendLiveIntegration:
    """Live inference integration tests executing against Qwen2.5-0.5B on Apple Silicon."""

    async def test_live_single_generation(self) -> None:
        backend = MLXBackend(
            model_id=DEFAULT_MODEL_ID,
            default_temperature=0.0,
            default_max_tokens=15,
        )
        await backend.load_model()
        assert backend.is_loaded is True

        req = InferenceRequest(
            request_id="live-req-1",
            model=DEFAULT_MODEL_ID,
            prompt="Hello! What is 2 + 2?",
            max_tokens=10,
            temperature=0.0,
        )
        resp = await backend.generate(req)
        assert resp.request_id == "live-req-1"
        assert resp.backend_name == "mlx"
        assert resp.generated_text != ""
        assert resp.input_tokens is not None and resp.input_tokens > 0
        assert resp.output_tokens is not None and resp.output_tokens > 0
        assert resp.latency_ms > 0.0

    async def test_live_deterministic_generation(self) -> None:
        backend = MLXBackend(model_id=DEFAULT_MODEL_ID)
        await backend.load_model()

        req1 = InferenceRequest(
            request_id="det-1",
            model=DEFAULT_MODEL_ID,
            prompt="Write the first 3 letters of the alphabet.",
            max_tokens=10,
            temperature=0.0,
        )
        req2 = InferenceRequest(
            request_id="det-2",
            model=DEFAULT_MODEL_ID,
            prompt="Write the first 3 letters of the alphabet.",
            max_tokens=10,
            temperature=0.0,
        )

        resp1 = await backend.generate(req1)
        resp2 = await backend.generate(req2)
        # Greedy decoding with temperature 0.0 must produce identical text
        assert resp1.generated_text == resp2.generated_text

    async def test_live_batch_generation(self) -> None:
        backend = MLXBackend(model_id=DEFAULT_MODEL_ID)
        await backend.load_model()

        batch = InferenceBatch(
            requests=(
                InferenceRequest(
                    request_id="b1",
                    model=DEFAULT_MODEL_ID,
                    prompt="What color is the sky on a clear day?",
                    max_tokens=8,
                ),
                InferenceRequest(
                    request_id="b2",
                    model=DEFAULT_MODEL_ID,
                    prompt="Name the capital of France.",
                    max_tokens=8,
                ),
            )
        )
        responses = await backend.generate_batch(batch)
        assert len(responses) == 2
        assert responses[0].request_id == "b1"
        assert responses[1].request_id == "b2"
        assert responses[0].generated_text != ""
        assert responses[1].generated_text != ""
        assert responses[0].input_tokens is not None and responses[0].input_tokens > 0
        assert responses[1].input_tokens is not None and responses[1].input_tokens > 0
