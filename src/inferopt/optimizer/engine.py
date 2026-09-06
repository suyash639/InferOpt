"""Deterministic optimization engine for evaluating candidate configurations against objectives."""

import uuid
from collections.abc import Sequence
from typing import Any

from inferopt.benchmarks.models import BenchmarkResult
from inferopt.optimizer.models import (
    CandidateEvaluation,
    CandidateSpace,
    ObjectiveConfig,
    OptimizationConstraints,
    OptimizationObjectiveType,
    OptimizationResult,
    TunableConfig,
)
from inferopt.scheduler.config import SchedulerConfig


def calculate_objective_score(
    result: BenchmarkResult, objective: ObjectiveConfig
) -> tuple[float, str]:
    """Compute numerical score and explanation for a benchmark result given an objective."""
    if objective.objective_type == OptimizationObjectiveType.THROUGHPUT:
        score = result.requests_per_sec
        explanation = f"Throughput: {result.requests_per_sec:.2f} req/s"
        return score, explanation

    elif objective.objective_type == OptimizationObjectiveType.LATENCY:
        # Negative p95 latency so higher score corresponds to lower latency
        score = -result.p95_latency_ms
        explanation = f"p95 Latency: {result.p95_latency_ms:.2f}ms (Score: {score:.2f})"
        return score, explanation

    elif objective.objective_type == OptimizationObjectiveType.BALANCED:
        target_tput = objective.target_throughput_rps or 100.0
        target_p95 = objective.target_p95_latency_ms or 50.0

        norm_tput = result.requests_per_sec / target_tput
        norm_lat = result.p95_latency_ms / target_p95

        score = (objective.throughput_weight * norm_tput) - (objective.latency_weight * norm_lat)
        explanation = (
            f"Balanced score: {score:.4f} (throughput {result.requests_per_sec:.2f} req/s "
            f"[w={objective.throughput_weight}], p95 {result.p95_latency_ms:.2f}ms "
            f"[w={objective.latency_weight}])"
        )
        return score, explanation

    else:
        score = result.requests_per_sec
        return score, f"Default throughput score: {score:.2f} req/s"


class DeterministicOptimizer:
    """Deterministic, backend-agnostic optimization engine.

    Evaluates measured benchmark results against explicit SLA constraints and objectives,
    ranks feasible candidate configurations using deterministic tie-breaking rules,
    and produces structured experiment plans.
    """

    def __init__(self) -> None:
        """Initialize deterministic optimization engine."""

    def create_experiment_plan(
        self, candidate_space: CandidateSpace
    ) -> tuple[SchedulerConfig, ...]:
        """Generate a deterministic sequence of SchedulerConfig candidate configurations.

        Args:
            candidate_space: Search space defining concurrency, batch size, and wait window grids.

        Returns:
            Ordered tuple of SchedulerConfig instances to be evaluated in experiments.
        """
        candidates = candidate_space.generate_candidates()
        return tuple(c.to_scheduler_config() for c in candidates)

    def evaluate_results(
        self,
        results: Sequence[BenchmarkResult],
        objective: ObjectiveConfig,
        constraints: OptimizationConstraints | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Evaluate historical benchmark evidence against objectives and constraints.

        Args:
            results: Sequence of measured BenchmarkResult instances from actual runs.
            objective: User-specified objective function (THROUGHPUT, LATENCY, BALANCED).
            constraints: Optional SLA and resource constraints. Defaults to unconstrained.
            metadata: Optional user context or tags attached to the optimization result.

        Returns:
            OptimizationResult containing candidate evaluations and optimal recommendation.
        """
        active_constraints = constraints or OptimizationConstraints()
        evaluations: list[CandidateEvaluation] = []

        for res in results:
            config = TunableConfig.from_scheduler_config(res.scheduler_config)
            violations = active_constraints.evaluate(config, res)
            is_feasible = len(violations) == 0

            score, score_explanation = calculate_objective_score(res, objective)

            if is_feasible:
                explanation = f"Feasible. {score_explanation}"
            else:
                violation_str = "; ".join(violations)
                explanation = f"Infeasible. Violations: {violation_str}. {score_explanation}"

            evaluations.append(
                CandidateEvaluation(
                    config=config,
                    is_feasible=is_feasible,
                    objective_score=score,
                    constraint_violations=violations,
                    metrics=res,
                    explanation=explanation,
                )
            )

        feasible_evals = [e for e in evaluations if e.is_feasible]
        opt_id = f"opt-{uuid.uuid4().hex[:8]}"

        if not feasible_evals:
            summary = (
                f"Optimization failed: No candidate configurations satisfied all constraints. "
                f"Evaluated {len(evaluations)} candidates, 0 feasible."
            )
            return OptimizationResult(
                optimization_id=opt_id,
                objective=objective,
                constraints=active_constraints,
                recommended_config=None,
                best_score=None,
                is_feasible=False,
                total_candidates=len(evaluations),
                feasible_candidates=0,
                evaluations=tuple(evaluations),
                summary_explanation=summary,
                metadata=dict(metadata or {}),
            )

        # Deterministic tie-breaking hierarchy:
        # 1. Higher objective score (-score ascending)
        # 2. Lower p95 latency (ascending)
        # 3. Smaller batch_wait_ms (ascending)
        # 4. Smaller max_batch_size (ascending)
        # 5. Lower max_concurrency (ascending)
        def tie_breaker_key(ev: CandidateEvaluation) -> tuple[float, float, float, int, int]:
            p95 = ev.metrics.p95_latency_ms if ev.metrics else 0.0
            return (
                -ev.objective_score,
                p95,
                ev.config.batch_wait_ms,
                ev.config.max_batch_size,
                ev.config.max_concurrency,
            )

        sorted_feasible = sorted(feasible_evals, key=tie_breaker_key)
        winner = sorted_feasible[0]

        summary = (
            f"Recommended candidate (concurrency={winner.config.max_concurrency}, "
            f"batch_size={winner.config.max_batch_size}, "
            f"batch_wait_ms={winner.config.batch_wait_ms:.1f}) achieved best score "
            f"{winner.objective_score:.4f} satisfying all SLA constraints. "
            f"Evaluated {len(evaluations)} candidates ({len(feasible_evals)} feasible)."
        )

        return OptimizationResult(
            optimization_id=opt_id,
            objective=objective,
            constraints=active_constraints,
            recommended_config=winner.config,
            best_score=winner.objective_score,
            is_feasible=True,
            total_candidates=len(evaluations),
            feasible_candidates=len(feasible_evals),
            evaluations=tuple(evaluations),
            summary_explanation=summary,
            metadata=dict(metadata or {}),
        )
