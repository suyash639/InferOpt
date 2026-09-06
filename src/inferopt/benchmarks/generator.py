"""Deterministic synthetic workload generator with realistic prompt categories."""

import random
from collections.abc import Callable
from typing import Final

from inferopt.benchmarks.models import (
    ArrivalPattern,
    PromptCategory,
    WorkloadConfig,
    WorkloadRequestSpec,
    WorkloadScenario,
)

_FACTUAL_PROMPTS: Final[tuple[str, ...]] = (
    "What is the primary function of an asynchronous event loop in Python?",
    "Explain the difference between process and thread execution models in OS.",
    "What is Amdahl's Law and how does it apply to parallel computing limits?",
    "Describe the role of the Key-Value (KV) cache in transformer inference.",
    "List three standard cache replacement policies and describe their trade-offs.",
    "How does priority queueing prevent head-of-line blocking in schedulers?",
    "What is the distinction between prefill and decode during LLM generation?",
    "Define backpressure in distributed queueing architectures.",
)

_SHORT_PROMPTS: Final[tuple[str, ...]] = (
    "Summarize the benefits of dynamic batching in three bullet points.",
    "Provide a one-sentence definition of Time-To-First-Token (TTFT).",
    "Explain the concept of Inter-Token Latency (ITL) for streaming inference.",
    "What is the time complexity of binary heap insertion and extraction?",
    "Why does high temperature increase output entropy during token sampling?",
    "What is memory bandwidth saturation in modern GPU architectures?",
)

_MEDIUM_PROMPTS: Final[tuple[str, ...]] = (
    (
        "Explain how continuous batching differs from static batching in LLM serving. "
        "Include sequence lengths, memory allocation, and KV-cache management."
    ),
    (
        "Describe the architectural trade-offs between TTFT and ITL "
        "when serving multi-tenant workloads with heterogeneous prompt lengths."
    ),
    (
        "Write a detailed design explanation for a token-bucket rate limiter, "
        "discussing burst capacity handling, thread-safety, and backpressure."
    ),
    (
        "Analyze how memory bandwidth saturation limits decoding throughput in LLMs "
        "compared to compute saturation during the prefill phase."
    ),
    (
        "Discuss strategies for prefix caching in multi-turn conversational AI, "
        "focusing on radix trees and cache eviction policies under bounded memory."
    ),
)

_LONG_PROMPTS: Final[tuple[str, ...]] = (
    (
        "Context: Large Language Model serving infrastructure must balance multiple "
        "conflicting objectives: minimizing latency for interactive users, maximizing "
        "aggregate token throughput for background batch jobs, and maintaining predictable "
        "tail latencies (p95/p99) under fluctuating request volumes. Traditional static "
        "batching buffers requests until a fixed batch size is reached, introducing queue "
        "delay during low-traffic periods. Dynamic batching introduces a configurable wait "
        "window to amortize kernel launch overhead. In contrast, token-level continuous "
        "batching operates at the iteration granularity, dynamically inserting newly "
        "arrived requests into the active generation pool and retiring finished sequences "
        "immediately. However, continuous batching requires sophisticated paged memory "
        "management to eliminate external memory fragmentation. "
        "Question: Contrast the queue delay and memory fragmentation characteristics of "
        "static batching, dynamic batching, and iteration-level continuous batching."
    ),
    (
        "Context: In distributed inference systems, model routing and load balancing play "
        "a critical role in optimizing hardware utilization across heterogeneous clusters. "
        "Naive round-robin or least-connections balancers fail to account for KV cache "
        "locality, leading to redundant prefill computation when identical prompt prefixes "
        "are routed to arbitrary instances. Cache-aware routing algorithms construct global "
        "prefix indices to steer incoming requests toward worker instances that already hold "
        "relevant KV cache pages. However, strict cache affinity can cause load imbalance "
        "if a popular prefix creates a hotspot on a single worker node. Hybrid routing "
        "policies combine prefix affinity scoring with real-time queue depth and memory "
        "utilization telemetry to achieve optimal global performance. "
        "Question: Analyze how hybrid cache-aware routing balances prefix reuse with load "
        "distribution, and identify failure modes when popularity skew creates hotspots."
    ),
)

_REASONING_PROMPTS: Final[tuple[str, ...]] = (
    (
        "Solve this scheduling problem step-by-step: An inference server processes "
        "batches of up to 4 requests. Each batch takes exactly 40ms to execute. If "
        "requests arrive at a Poisson rate of 80 requests/sec, what is the minimum "
        "number of parallel worker slots required to maintain a stable queue where "
        "arrival rate does not exceed maximum service rate? Show derivation."
    ),
    (
        "Analyze the algorithmic complexity and concurrency implications of maintaining "
        "an in-memory priority min-heap with monotonic sequence IDs for FIFO tie-breaking "
        "under high async insertion rates. Discuss lock contention vs lock-free designs."
    ),
    (
        "Compare two scheduling policies for LLM serving: Shortest-Job-First "
        "(approximated by prompt length + max_tokens) versus strict First-In-First-Out "
        "(FIFO). Prove why SJF minimizes average turnaround time while discussing "
        "starvation mitigation mechanisms."
    ),
)

_SUMMARIZATION_PROMPTS: Final[tuple[str, ...]] = (
    (
        "Text: Speculative decoding accelerates autoregressive generation by using a "
        "lightweight draft model to generate candidate tokens in parallel, which are "
        "verified by the target model in a single forward pass. If draft tokens match "
        "the target model distribution, multiple tokens are accepted in one step, "
        "significantly improving throughput without altering output quality. "
        "Task: Provide a concise two-sentence summary outlining the core mechanism "
        "of speculative decoding."
    ),
    (
        "Text: FlashAttention and FlashAttention-2 reformulate exact attention "
        "computation to optimize GPU SRAM and HBM memory transfers. By tiling query, "
        "key, and value matrices and performing softmax reduction incrementally in "
        "fast on-chip shared memory, FlashAttention avoids materializing the quadratic "
        "intermediate attention matrix in high-bandwidth memory. "
        "Task: Summarize how tiling and incremental softmax enable linear memory "
        "complexity in FlashAttention."
    ),
)

PROMPT_TEMPLATES: Final[dict[PromptCategory, tuple[str, ...]]] = {
    PromptCategory.FACTUAL: _FACTUAL_PROMPTS,
    PromptCategory.SHORT: _SHORT_PROMPTS,
    PromptCategory.MEDIUM: _MEDIUM_PROMPTS,
    PromptCategory.LONG: _LONG_PROMPTS,
    PromptCategory.REASONING: _REASONING_PROMPTS,
    PromptCategory.SUMMARIZATION: _SUMMARIZATION_PROMPTS,
}


def generate_workload(config: WorkloadConfig) -> WorkloadScenario:
    """Deterministically generate a synthetic workload scenario from configuration parameters.

    The generator is completely reproducible: identical configuration and random seed
    will always produce the exact same sequence of requests, prompt contents, token counts,
    and priority distributions.

    Args:
        config: Validated WorkloadConfig specifying parameters and seed.

    Returns:
        Immutable WorkloadScenario containing the generated request specifications.
    """
    rng = random.Random(config.seed)
    requests: list[WorkloadRequestSpec] = []

    for i in range(config.num_requests):
        req_id = f"req-{config.scenario_name}-{i:04d}"
        category = rng.choice(config.prompt_categories)
        templates = PROMPT_TEMPLATES[category]
        prompt_text = rng.choice(templates)
        priority = rng.choice(config.priority_levels)

        # Compute arrival timing offset for fixed-rate workloads
        delay_ms = 0.0
        if (
            config.arrival_pattern == ArrivalPattern.FIXED_RATE
            and config.arrival_rate_rps is not None
            and config.arrival_rate_rps > 0
        ):
            delay_ms = (i / config.arrival_rate_rps) * 1000.0

        requests.append(
            WorkloadRequestSpec(
                request_id=req_id,
                model="benchmark-model",
                prompt=prompt_text,
                priority=priority,
                max_tokens=config.max_tokens,
                temperature=0.7,
                metadata={
                    "category": category.value,
                    "index": i,
                    "seed": config.seed,
                },
                scheduled_delay_ms=delay_ms,
            )
        )

    return WorkloadScenario(
        scenario_name=config.scenario_name,
        description=config.description,
        config=config,
        requests=tuple(requests),
    )


def get_light_workload(seed: int = 42) -> WorkloadScenario:
    """Light workload: 10 requests with short factual prompts, concurrent arrival."""
    config = WorkloadConfig(
        scenario_name="light",
        description="Light workload: 10 requests with short factual queries",
        num_requests=10,
        arrival_pattern=ArrivalPattern.CONCURRENT,
        seed=seed,
        prompt_categories=(PromptCategory.SHORT, PromptCategory.FACTUAL),
        max_tokens=32,
        priority_levels=(0,),
    )
    return generate_workload(config)


def get_medium_workload(seed: int = 42) -> WorkloadScenario:
    """Medium workload: 50 requests with mixed short/medium prompts, fixed rate 10 RPS."""
    config = WorkloadConfig(
        scenario_name="medium",
        description="Medium workload: 50 requests with mixed short/medium prompts at 10 RPS",
        num_requests=50,
        arrival_pattern=ArrivalPattern.FIXED_RATE,
        arrival_rate_rps=10.0,
        seed=seed,
        prompt_categories=(PromptCategory.SHORT, PromptCategory.MEDIUM, PromptCategory.FACTUAL),
        max_tokens=64,
        priority_levels=(0,),
    )
    return generate_workload(config)


def get_heavy_workload(seed: int = 42) -> WorkloadScenario:
    """Heavy workload: 200 requests with mixed prompt lengths, concurrent arrival."""
    config = WorkloadConfig(
        scenario_name="heavy",
        description="Heavy workload: 200 requests with mixed short/medium/long prompts",
        num_requests=200,
        arrival_pattern=ArrivalPattern.CONCURRENT,
        seed=seed,
        prompt_categories=(
            PromptCategory.SHORT,
            PromptCategory.MEDIUM,
            PromptCategory.LONG,
            PromptCategory.FACTUAL,
        ),
        max_tokens=128,
        priority_levels=(0,),
    )
    return generate_workload(config)


def get_burst_workload(seed: int = 42) -> WorkloadScenario:
    """Burst workload: 50 requests released simultaneously via synchronization barrier."""
    config = WorkloadConfig(
        scenario_name="burst",
        description="Burst workload: 50 requests released simultaneously to evaluate queueing",
        num_requests=50,
        arrival_pattern=ArrivalPattern.BURST,
        seed=seed,
        prompt_categories=(PromptCategory.SHORT, PromptCategory.MEDIUM),
        max_tokens=64,
        priority_levels=(0,),
    )
    return generate_workload(config)


def get_mixed_workload(seed: int = 42) -> WorkloadScenario:
    """Mixed workload: 60 requests with heterogeneous priorities and multiple categories."""
    config = WorkloadConfig(
        scenario_name="mixed",
        description="Mixed workload: 60 requests with mixed priorities and categories",
        num_requests=60,
        arrival_pattern=ArrivalPattern.CONCURRENT,
        seed=seed,
        prompt_categories=(
            PromptCategory.SHORT,
            PromptCategory.MEDIUM,
            PromptCategory.REASONING,
            PromptCategory.SUMMARIZATION,
        ),
        max_tokens=64,
        priority_levels=(-1, 0, 1, 2),
    )
    return generate_workload(config)


def get_long_context_workload(seed: int = 42) -> WorkloadScenario:
    """Long context workload: 20 requests with long context prompts (500+ tokens)."""
    config = WorkloadConfig(
        scenario_name="long_context",
        description="Long context workload: 20 requests with multi-paragraph long prompts",
        num_requests=20,
        arrival_pattern=ArrivalPattern.CONCURRENT,
        seed=seed,
        prompt_categories=(PromptCategory.LONG,),
        max_tokens=256,
        priority_levels=(0,),
    )
    return generate_workload(config)


PRESET_SCENARIOS: Final[dict[str, Callable[[int], WorkloadScenario]]] = {
    "light": get_light_workload,
    "medium": get_medium_workload,
    "heavy": get_heavy_workload,
    "burst": get_burst_workload,
    "mixed": get_mixed_workload,
    "long_context": get_long_context_workload,
}
