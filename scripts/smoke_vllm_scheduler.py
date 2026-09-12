"""Standalone smoke test script for InferOpt Scheduler -> Dynamic Batching -> VLLMBackend."""

import argparse
import asyncio
import sys
import time

from inferopt.backends.vllm import DEFAULT_VLLM_MODEL_ID, VLLMBackend, VLLMConfig
from inferopt.core.models import InferenceRequest
from inferopt.scheduler.config import BatchConfig, SchedulerConfig
from inferopt.scheduler.scheduler import Scheduler
from inferopt.telemetry.collector import MetricsCollector


async def run_vllm_scheduler_smoke_test(
    model_id: str = DEFAULT_VLLM_MODEL_ID,
    max_batch_size: int = 4,
    batch_wait_ms: float = 50.0,
    max_concurrency: int = 2,
    max_tokens: int = 32,
    temperature: float = 0.0,
    enforce_eager: bool = False,
) -> None:
    """Execute a 4-request end-to-end smoke test through Scheduler and VLLMBackend."""
    print("=" * 60)
    print("InferOpt VLLM E2E Smoke Test")
    print("=" * 60)
    print("Backend:                 VLLMBackend")
    print(f"Model ID:                {model_id}")
    print(f"Max Batch Size:          {max_batch_size}")
    print(f"Batch Wait Window:       {batch_wait_ms} ms")
    print(f"Max Concurrency:         {max_concurrency}")
    print(f"Enforce Eager:           {enforce_eager}")
    print("-" * 60)
    print("Initializing VLLMBackend...")

    config = VLLMConfig(
        model=model_id,
        default_max_tokens=max_tokens,
        default_temperature=temperature,
        enforce_eager=enforce_eager,
    )
    backend = VLLMBackend(config=config)

    collector = MetricsCollector()
    scheduler_config = SchedulerConfig(
        max_concurrency=max_concurrency,
        batch_config=BatchConfig(
            max_batch_size=max_batch_size,
            batch_wait_ms=batch_wait_ms,
        ),
    )

    prompts = [
        "Explain what InferOpt does in one sentence.",
        "What are the benefits of dynamic batching in LLM serving?",
        "How does speculative decoding improve inference throughput?",
        "Describe the role of key-value cache in transformer generation.",
    ]

    requests = [
        InferenceRequest(
            request_id=f"vllm-e2e-smoke-{i + 1}",
            model=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for i, prompt in enumerate(prompts)
    ]

    print(f"Submitting {len(requests)} requests to Scheduler...")
    t_start = time.perf_counter()

    async with Scheduler(
        backend=backend,
        config=scheduler_config,
        collector=collector,
    ) as scheduler:
        responses = await asyncio.gather(*[scheduler.submit(r) for r in requests])

    t_total_sec = max(0.0001, time.perf_counter() - t_start)
    snapshot = collector.snapshot()
    req_stats = snapshot.requests
    batch_stats = snapshot.batches

    print("-" * 60)
    print("InferOpt VLLM E2E Smoke Test")
    print("----------------------------")
    print("Backend: VLLMBackend")
    print(f"Requests: {req_stats.total_requests}")
    print(f"Completed: {req_stats.completed_requests}")
    print(f"Failed: {req_stats.failed_requests}")
    print(f"Batches: {batch_stats.total_batches}")
    print(f"Average batch size: {batch_stats.avg_batch_size:.1f}")
    print(f"p50 latency: {req_stats.p50_total_latency_ms:.2f} ms")
    print(f"p95 latency: {req_stats.p95_total_latency_ms:.2f} ms")
    throughput = req_stats.completed_requests / t_total_sec
    print(f"Throughput: {throughput:.2f} req/s")
    print("=" * 60)

    for resp in responses:
        print(
            f"[{resp.request_id}] (tokens: in={resp.input_tokens}, "
            f"out={resp.output_tokens}, lat={resp.latency_ms:.1f}ms):"
        )
        print(f"  {resp.generated_text.strip()[:100]}...\n")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the vLLM scheduler smoke test."""
    parser = argparse.ArgumentParser(
        description="InferOpt Scheduler + Dynamic Batching -> VLLMBackend Smoke Test"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_VLLM_MODEL_ID,
        help=f"Target model identifier (default: {DEFAULT_VLLM_MODEL_ID})",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="Max batch size for dynamic batching (default: 4)",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=50.0,
        help="Dynamic batch formation wait window in ms (default: 50.0)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max concurrency for scheduler (default: 2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max generation tokens (default: 32)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help=(
            "Enforce eager execution mode in vLLM (disables CUDA graphs, diagnostic mode). "
            "Note: Results in eager mode are not directly comparable to default compiled vLLM runs."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    try:
        asyncio.run(
            run_vllm_scheduler_smoke_test(
                model_id=args.model,
                max_batch_size=args.max_batch_size,
                batch_wait_ms=args.batch_wait_ms,
                max_concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                enforce_eager=args.enforce_eager,
            )
        )
    except Exception as exc:
        print(f"Smoke test encountered error: {exc}", file=sys.stderr)
        sys.exit(1)
