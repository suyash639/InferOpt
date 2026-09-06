"""Unit tests for Benchmark CLI argument parsing and execution."""

from pathlib import Path

from inferopt.benchmarks.cli import build_parser, main
from inferopt.benchmarks.models import BenchmarkResult


class TestBenchmarkCLI:
    """Tests for CLI options and execution."""

    def test_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.scenario == "light"
        assert args.num_requests is None
        assert args.seed == 42
        assert args.concurrency is None
        assert args.max_batch_size is None
        assert args.batch_wait_ms is None
        assert args.output is None
        assert args.verbose is False

    def test_parser_custom_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--scenario",
                "mixed",
                "--num-requests",
                "25",
                "--seed",
                "99",
                "--concurrency",
                "8",
                "--max-batch-size",
                "16",
                "--batch-wait-ms",
                "25.5",
                "--output",
                "/tmp/test_result.json",
                "--verbose",
            ]
        )
        assert args.scenario == "mixed"
        assert args.num_requests == 25
        assert args.seed == 99
        assert args.concurrency == 8
        assert args.max_batch_size == 16
        assert args.batch_wait_ms == 25.5
        assert args.output == "/tmp/test_result.json"
        assert args.verbose is True

    def test_cli_execution_with_output_file(self, tmp_path: Path) -> None:
        out_file = tmp_path / "bench_out.json"
        exit_code = main(
            [
                "--scenario",
                "light",
                "--num-requests",
                "4",
                "--output",
                str(out_file),
                "--verbose",
            ]
        )
        assert exit_code == 0
        assert out_file.exists()

        loaded = BenchmarkResult.load_json(out_file)
        assert loaded.scenario_name == "light"
        assert loaded.completed_requests == 4
