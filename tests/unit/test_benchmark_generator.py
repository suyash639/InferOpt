"""Unit tests for deterministic workload generation and preset profiles."""

from inferopt.benchmarks.generator import (
    PRESET_SCENARIOS,
    generate_workload,
    get_burst_workload,
    get_heavy_workload,
    get_light_workload,
    get_long_context_workload,
    get_medium_workload,
    get_mixed_workload,
)
from inferopt.benchmarks.models import (
    ArrivalPattern,
    PromptCategory,
    WorkloadConfig,
)


class TestWorkloadGeneratorDeterminism:
    """Tests verifying strict reproducibility and determinism of generated workloads."""

    def test_identical_seed_produces_identical_workload(self) -> None:
        cfg1 = WorkloadConfig(
            scenario_name="eval",
            num_requests=20,
            seed=12345,
            prompt_categories=(
                PromptCategory.SHORT,
                PromptCategory.MEDIUM,
                PromptCategory.REASONING,
            ),
            max_tokens=64,
            priority_levels=(-1, 0, 1),
        )
        cfg2 = WorkloadConfig(
            scenario_name="eval",
            num_requests=20,
            seed=12345,
            prompt_categories=(
                PromptCategory.SHORT,
                PromptCategory.MEDIUM,
                PromptCategory.REASONING,
            ),
            max_tokens=64,
            priority_levels=(-1, 0, 1),
        )

        w1 = generate_workload(cfg1)
        w2 = generate_workload(cfg2)

        assert len(w1.requests) == 20
        assert len(w2.requests) == 20
        for r1, r2 in zip(w1.requests, w2.requests, strict=True):
            assert r1.request_id == r2.request_id
            assert r1.prompt == r2.prompt
            assert r1.priority == r2.priority
            assert r1.max_tokens == r2.max_tokens
            assert r1.scheduled_delay_ms == r2.scheduled_delay_ms

    def test_different_seed_produces_distinct_workload(self) -> None:
        cfg1 = WorkloadConfig(scenario_name="eval", num_requests=20, seed=111)
        cfg2 = WorkloadConfig(scenario_name="eval", num_requests=20, seed=222)

        w1 = generate_workload(cfg1)
        w2 = generate_workload(cfg2)

        prompts1 = [r.prompt for r in w1.requests]
        prompts2 = [r.prompt for r in w2.requests]
        assert prompts1 != prompts2


class TestPresetScenarios:
    """Tests for standard preset benchmark scenario profiles."""

    def test_light_workload(self) -> None:
        w = get_light_workload(seed=42)
        assert w.scenario_name == "light"
        assert len(w.requests) == 10
        assert w.config.arrival_pattern == ArrivalPattern.CONCURRENT
        assert w.config.max_tokens == 32

    def test_medium_workload(self) -> None:
        w = get_medium_workload(seed=42)
        assert w.scenario_name == "medium"
        assert len(w.requests) == 50
        assert w.config.arrival_pattern == ArrivalPattern.FIXED_RATE
        assert w.config.arrival_rate_rps == 10.0
        assert w.requests[1].scheduled_delay_ms == 100.0  # 1 / 10 rps = 100ms

    def test_heavy_workload(self) -> None:
        w = get_heavy_workload(seed=42)
        assert w.scenario_name == "heavy"
        assert len(w.requests) == 200
        assert w.config.arrival_pattern == ArrivalPattern.CONCURRENT

    def test_burst_workload(self) -> None:
        w = get_burst_workload(seed=42)
        assert w.scenario_name == "burst"
        assert len(w.requests) == 50
        assert w.config.arrival_pattern == ArrivalPattern.BURST

    def test_mixed_workload(self) -> None:
        w = get_mixed_workload(seed=42)
        assert w.scenario_name == "mixed"
        assert len(w.requests) == 60
        priorities = {r.priority for r in w.requests}
        assert len(priorities) > 1

    def test_long_context_workload(self) -> None:
        w = get_long_context_workload(seed=42)
        assert w.scenario_name == "long_context"
        assert len(w.requests) == 20
        assert all(len(r.prompt) > 200 for r in w.requests)

    def test_preset_scenarios_registry(self) -> None:
        expected = {
            "single",
            "concurrent_4",
            "concurrent_8",
            "concurrent_16",
            "light",
            "medium",
            "heavy",
            "burst",
            "mixed",
            "long_context",
        }
        assert set(PRESET_SCENARIOS.keys()) == expected
        for name, factory in PRESET_SCENARIOS.items():
            w = factory(42)
            assert w.scenario_name == name
            assert len(w.requests) > 0
