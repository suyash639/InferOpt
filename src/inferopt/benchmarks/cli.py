"""Command-line interface for running reproducible InferOpt benchmarks."""

import argparse
import asyncio
import sys
from collections.abc import Sequence

from inferopt.backends.mock import MockBackend
from inferopt.benchmarks.generator import PRESET_SCENARIOS, generate_workload
from inferopt.benchmarks.models import WorkloadConfig
from inferopt.benchmarks.runner import BenchmarkRunner
from inferopt.scheduler.config import BatchConfig, SchedulerConfig


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser for the benchmark harness."""
    parser = argparse.ArgumentParser(
        prog="inferopt-bench",
        description="InferOpt Benchmarking Harness - Run controlled inference serving benchmarks",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        choices=list(PRESET_SCENARIOS.keys()),
        default="light",
        help="Pre-configured workload scenario to execute (default: light).",
    )
    parser.add_argument(
        "--num-requests",
        "-n",
        type=int,
        default=None,
        help="Override total request count in scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible workload generation (default: 42).",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=None,
        help="Scheduler max_concurrency override.",
    )
    parser.add_argument(
        "--max-batch-size",
        "-b",
        type=int,
        default=None,
        help="Scheduler batch_config.max_batch_size override.",
    )
    parser.add_argument(
        "--batch-wait-ms",
        "-w",
        type=float,
        default=None,
        help="Scheduler batch_config.batch_wait_ms override.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="File path to save the JSON benchmark result.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed benchmark output.",
    )
    return parser


async def run_benchmark_cli(args: argparse.Namespace) -> int:
    """Execute benchmark run with arguments parsed from CLI."""
    scenario_factory = PRESET_SCENARIOS.get(args.scenario)
    if scenario_factory is None:
        print(f"Error: Unknown scenario '{args.scenario}'", file=sys.stderr)
        return 1

    scenario = scenario_factory(args.seed)

    # Apply request count override if specified
    if args.num_requests is not None and args.num_requests > 0:
        cfg_dict = scenario.config.model_dump()
        cfg_dict["num_requests"] = args.num_requests
        scenario = generate_workload(WorkloadConfig.model_validate(cfg_dict))

    # Build scheduler config overrides
    default_sched = SchedulerConfig()
    default_batch = default_sched.batch_config

    max_batch_size = (
        args.max_batch_size if args.max_batch_size is not None else default_batch.max_batch_size
    )
    batch_wait_ms = (
        args.batch_wait_ms if args.batch_wait_ms is not None else default_batch.batch_wait_ms
    )
    max_concurrency = (
        args.concurrency if args.concurrency is not None else default_sched.max_concurrency
    )

    batch_cfg = BatchConfig(max_batch_size=max_batch_size, batch_wait_ms=batch_wait_ms)
    sched_config = SchedulerConfig(max_concurrency=max_concurrency, batch_config=batch_cfg)

    backend = MockBackend(default_latency_sec=0.005)
    runner = BenchmarkRunner()

    if args.verbose:
        print(f"Executing scenario: {scenario.scenario_name} ({len(scenario.requests)} reqs)...")

    result = await runner.run(
        scenario=scenario,
        backend=backend,
        scheduler_config=sched_config,
        metadata={"cli": True, "seed": args.seed},
    )

    separator = "=" * 60
    sub_sep = "-" * 60

    print()
    print(separator)
    print(f"  InferOpt Benchmark Results: {result.scenario_name.upper()}")
    print(separator)
    req_summary = f"  Requests:       {result.completed_requests}/{result.total_requests} completed"
    if result.failed_requests > 0:
        req_summary += f" ({result.failed_requests} failed)"
    print(req_summary)
    print(f"  Duration:       {result.duration_sec:.3f} s")
    tput_str = (
        f"  Throughput:     {result.requests_per_sec:.2f} req/s | "
        f"{result.batches_per_sec:.2f} batch/s | "
        f"{result.tokens_per_sec:.1f} tokens/s"
    )
    print(tput_str)
    print(sub_sep)
    print(f"  Latency (avg):  {result.avg_latency_ms:.2f} ms")
    print(f"  Latency (p50):  {result.p50_latency_ms:.2f} ms")
    print(f"  Latency (p95):  {result.p95_latency_ms:.2f} ms")
    print(f"  Latency (p99):  {result.p99_latency_ms:.2f} ms")
    print(f"  Queue Wait:     {result.avg_queue_wait_ms:.2f} ms")
    print(f"  Execution:      {result.avg_execution_ms:.2f} ms")
    print(sub_sep)
    batch_str = (
        f"  Total Batches:  {result.total_batches} "
        f"(avg size {result.avg_batch_size:.2f}, max {result.max_batch_size})"
    )
    print(batch_str)
    print(f"  Peak Queue:     {result.peak_queue_depth} requests")
    print(f"  Peak Active:    {result.peak_active_requests} requests")
    print(separator)
    print()

    if args.output:
        result.save_json(args.output)
        if args.verbose:
            print(f"Saved benchmark result to: {args.output}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main entrypoint for CLI execution."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(run_benchmark_cli(args))


if __name__ == "__main__":
    sys.exit(main())
