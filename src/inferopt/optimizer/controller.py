"""Closed-loop adaptive controller evaluating evidence and modulating scheduler configurations."""

import time
from collections.abc import Sequence

from inferopt.benchmarks.models import BenchmarkResult, WorkloadConfig
from inferopt.optimizer.adaptation_models import (
    AdaptationDecision,
    AdaptationDecisionType,
    AdaptationPolicy,
    AdaptationRecord,
)
from inferopt.optimizer.engine import DeterministicOptimizer, calculate_objective_score
from inferopt.optimizer.models import OptimizationObjectiveType, TunableConfig
from inferopt.scheduler.config import SchedulerConfig
from inferopt.scheduler.scheduler import Scheduler
from inferopt.telemetry.models import AdaptationEvent, MetricsSnapshot


def _make_benchmark_result_from_snapshot(
    snapshot: MetricsSnapshot,
    config: TunableConfig,
) -> BenchmarkResult:
    """Synthesize a BenchmarkResult from a live MetricsSnapshot and TunableConfig."""
    sched_cfg = config.to_scheduler_config()
    elapsed = max(0.001, snapshot.throughput.elapsed_sec)
    completed_reqs = snapshot.requests.completed_requests
    reqs_per_sec = (
        snapshot.throughput.requests_per_sec
        if snapshot.throughput.requests_per_sec > 0
        else (completed_reqs / elapsed)
    )

    return BenchmarkResult(
        benchmark_id=f"snap-{config.max_concurrency}-{config.max_batch_size}-{int(time.time())}",
        scenario_name="live_observation",
        workload_config=WorkloadConfig(
            scenario_name="live_observation",
        ),
        backend_name="observed",
        scheduler_config=sched_cfg,
        batch_config=sched_cfg.batch_config,
        telemetry_snapshot=snapshot,
        duration_sec=elapsed,
        total_requests=snapshot.requests.total_requests,
        completed_requests=completed_reqs,
        failed_requests=snapshot.requests.failed_requests,
        cancelled_requests=snapshot.requests.cancelled_requests,
        requests_per_sec=reqs_per_sec,
        batches_per_sec=snapshot.throughput.batches_per_sec,
        tokens_per_sec=snapshot.throughput.tokens_per_sec,
        avg_latency_ms=snapshot.requests.avg_total_latency_ms,
        p50_latency_ms=snapshot.requests.p50_total_latency_ms,
        p95_latency_ms=snapshot.requests.p95_total_latency_ms,
        p99_latency_ms=snapshot.requests.p99_total_latency_ms,
        avg_queue_wait_ms=snapshot.requests.avg_queue_wait_ms,
        avg_execution_ms=snapshot.requests.avg_execution_ms,
        peak_queue_depth=snapshot.queue.peak_queue_depth,
        peak_active_requests=snapshot.queue.peak_active_requests,
        total_batches=snapshot.batches.total_batches,
        avg_batch_size=snapshot.batches.avg_batch_size,
        max_batch_size=snapshot.batches.max_batch_size,
    )


class AdaptiveController:
    """Closed-loop adaptive controller for inference scheduling and dynamic batching.

    Evaluates observed telemetry metrics against explicit optimization policies,
    ranks candidate configurations via DeterministicOptimizer, enforces evaluation
    windows, cooldowns, and hysteresis gates, and safely applies configuration
    updates to a running Scheduler.
    """

    def __init__(
        self,
        policy: AdaptationPolicy | None = None,
        optimizer: DeterministicOptimizer | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        """Initialize the adaptive controller.

        Args:
            policy: Adaptation policy governing objectives, constraints, and stability gates.
            optimizer: Deterministic optimizer engine for candidate evaluation and ranking.
            scheduler: Optional attached running Scheduler instance to modulate.
        """
        self._policy = policy or AdaptationPolicy()
        self._optimizer = optimizer or DeterministicOptimizer()
        self._scheduler = scheduler

        self._window_counter: int = 0
        self._cooldown_counter: int = 0
        self._known_good_stack: list[tuple[TunableConfig, float]] = []
        self._history: list[AdaptationRecord] = []

        if self._scheduler is not None:
            initial_cfg = TunableConfig.from_scheduler_config(self._scheduler.config)
            self._known_good_stack.append((initial_cfg, 0.0))

    @property
    def policy(self) -> AdaptationPolicy:
        """Active adaptation policy."""
        return self._policy

    @property
    def optimizer(self) -> DeterministicOptimizer:
        """Underlying deterministic optimization engine."""
        return self._optimizer

    @property
    def scheduler(self) -> Scheduler | None:
        """Attached Scheduler instance, if any."""
        return self._scheduler

    @property
    def cooldown_remaining(self) -> int:
        """Number of remaining evaluation windows in cooldown."""
        return self._cooldown_counter

    @property
    def current_window_id(self) -> int:
        """Current evaluation window index."""
        return self._window_counter

    @property
    def history(self) -> tuple[AdaptationRecord, ...]:
        """Chronological tuple of all evaluated adaptation decision records."""
        return tuple(self._history)

    @property
    def known_good_configs(self) -> tuple[TunableConfig, ...]:
        """Tuple of known-good configurations available for rollback."""
        return tuple(cfg for cfg, _ in self._known_good_stack)

    def attach_scheduler(self, scheduler: Scheduler) -> None:
        """Attach a Scheduler instance to the controller."""
        self._scheduler = scheduler
        initial_cfg = TunableConfig.from_scheduler_config(scheduler.config)
        if not self._known_good_stack:
            self._known_good_stack.append((initial_cfg, 0.0))

    def reset(self) -> None:
        """Reset internal counters, cooldowns, and decision history."""
        self._window_counter = 0
        self._cooldown_counter = 0
        self._known_good_stack.clear()
        self._history.clear()
        if self._scheduler is not None:
            initial_cfg = TunableConfig.from_scheduler_config(self._scheduler.config)
            self._known_good_stack.append((initial_cfg, 0.0))

    def evaluate(
        self,
        current_evidence: BenchmarkResult | MetricsSnapshot,
        candidate_evidence: Sequence[BenchmarkResult] | None = None,
        current_config: TunableConfig | SchedulerConfig | None = None,
    ) -> AdaptationDecision:
        """Evaluate current telemetry evidence and candidate configurations against policy.

        Args:
            current_evidence: Measured metrics from the current evaluation window.
            candidate_evidence: Sequence of measured candidate benchmark runs to evaluate.
            current_config: Active configuration during the window (inferred if None).

        Returns:
            Immutable AdaptationDecision detailing the evaluation outcome and justification.
        """
        self._window_counter += 1
        window_id = self._window_counter

        # 1. Resolve current active configuration
        resolved_config: TunableConfig
        if current_config is not None:
            if isinstance(current_config, SchedulerConfig):
                resolved_config = TunableConfig.from_scheduler_config(current_config)
            else:
                resolved_config = current_config
        elif self._scheduler is not None:
            resolved_config = TunableConfig.from_scheduler_config(self._scheduler.config)
        elif isinstance(current_evidence, BenchmarkResult):
            resolved_config = TunableConfig.from_scheduler_config(current_evidence.scheduler_config)
        else:
            resolved_config = TunableConfig()

        # 2. Convert evidence to standardized BenchmarkResult
        if isinstance(current_evidence, MetricsSnapshot):
            current_res = _make_benchmark_result_from_snapshot(current_evidence, resolved_config)
        else:
            current_res = current_evidence

        # 3. Evaluation Window Sufficiency Gate
        reqs = current_res.completed_requests
        batches = current_res.total_batches
        min_reqs = self._policy.min_completed_requests
        min_batches = self._policy.min_completed_batches

        if reqs < min_reqs or batches < min_batches:
            reason = (
                f"Insufficient evidence in window {window_id}: {reqs}/{min_reqs} completed "
                f"requests, {batches}/{min_batches} batches."
            )
            decision = AdaptationDecision(
                window_id=window_id,
                decision_type=AdaptationDecisionType.INSUFFICIENT_DATA,
                current_config=resolved_config,
                reason=reason,
            )
            self._history.append(AdaptationRecord.from_decision(decision, is_applied=False))
            return decision

        # 4. Compute baseline objective score
        curr_score, _ = calculate_objective_score(current_res, self._policy.objective)

        # 5. Rollback Gate (Constraint Violation / Severe Degradation Check)
        if self._policy.enable_rollback and self._known_good_stack:
            target_cfg, target_score = self._known_good_stack[-1]
            if target_cfg != resolved_config:
                # Check hard SLA constraint violations
                if self._policy.constraints is not None:
                    violations = self._policy.constraints.evaluate(resolved_config, current_res)
                    if violations:
                        reason = (
                            f"Current configuration {resolved_config} violated constraints: "
                            f"{'; '.join(violations)}. Rolling back to known-good "
                            f"configuration {target_cfg}."
                        )
                        decision = AdaptationDecision(
                            window_id=window_id,
                            decision_type=AdaptationDecisionType.ROLLBACK,
                            current_config=resolved_config,
                            proposed_config=target_cfg,
                            current_score=curr_score,
                            proposed_score=target_score,
                            reason=reason,
                        )
                        self._history.append(
                            AdaptationRecord.from_decision(decision, is_applied=False)
                        )
                        return decision

                # Check relative performance degradation
                if target_score != 0.0 and curr_score < target_score:
                    degradation_pct = (target_score - curr_score) / abs(target_score) * 100.0
                    if degradation_pct >= self._policy.rollback_degradation_pct:
                        reason = (
                            f"Current configuration {resolved_config} degraded score by "
                            f"{degradation_pct:.1f}% below baseline ({target_score:.4f} -> "
                            f"{curr_score:.4f}). Rolling back to known-good config {target_cfg}."
                        )
                        decision = AdaptationDecision(
                            window_id=window_id,
                            decision_type=AdaptationDecisionType.ROLLBACK,
                            current_config=resolved_config,
                            proposed_config=target_cfg,
                            current_score=curr_score,
                            proposed_score=target_score,
                            improvement_pct=-degradation_pct,
                            reason=reason,
                        )
                        self._history.append(
                            AdaptationRecord.from_decision(decision, is_applied=False)
                        )
                        return decision

        # 6. Cooldown Gate
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            reason = (
                f"Adaptation suppressed: cooldown active ({self._cooldown_counter} "
                f"window(s) remaining)."
            )
            decision = AdaptationDecision(
                window_id=window_id,
                decision_type=AdaptationDecisionType.COOLDOWN,
                current_config=resolved_config,
                current_score=curr_score,
                reason=reason,
            )
            self._history.append(AdaptationRecord.from_decision(decision, is_applied=False))
            return decision

        # 7. Candidate Evidence Check
        if not candidate_evidence:
            decision = AdaptationDecision(
                window_id=window_id,
                decision_type=AdaptationDecisionType.NO_CHANGE,
                current_config=resolved_config,
                current_score=curr_score,
                reason="No candidate configurations provided for evaluation.",
            )
            self._history.append(AdaptationRecord.from_decision(decision, is_applied=False))
            return decision

        # 8. Candidate Evaluation & Ranking via Optimizer
        opt_res = self._optimizer.evaluate_results(
            results=candidate_evidence,
            objective=self._policy.objective,
            constraints=self._policy.constraints,
        )

        if not opt_res.is_feasible or opt_res.recommended_config is None:
            reason = (
                f"All candidate configurations were infeasible or violated constraints: "
                f"{opt_res.summary_explanation}"
            )
            decision = AdaptationDecision(
                window_id=window_id,
                decision_type=AdaptationDecisionType.INFEASIBLE,
                current_config=resolved_config,
                current_score=curr_score,
                reason=reason,
            )
            self._history.append(AdaptationRecord.from_decision(decision, is_applied=False))
            return decision

        best_cand = opt_res.recommended_config
        best_score = opt_res.best_score or 0.0

        # 9. Policy Safety Bounds Check
        within_bounds, bounds_msg = self._policy.is_within_bounds(best_cand)
        if not within_bounds:
            reason = f"Candidate {best_cand} violates policy bounds: {bounds_msg}"
            decision = AdaptationDecision(
                window_id=window_id,
                decision_type=AdaptationDecisionType.INFEASIBLE,
                current_config=resolved_config,
                proposed_config=best_cand,
                current_score=curr_score,
                proposed_score=best_score,
                reason=reason,
            )
            self._history.append(AdaptationRecord.from_decision(decision, is_applied=False))
            return decision

        # 10. Check if best candidate is already active
        if best_cand == resolved_config:
            decision = AdaptationDecision(
                window_id=window_id,
                decision_type=AdaptationDecisionType.NO_CHANGE,
                current_config=resolved_config,
                proposed_config=best_cand,
                current_score=curr_score,
                proposed_score=best_score,
                improvement_pct=0.0,
                reason="Current configuration is already the highest-scoring feasible candidate.",
            )
            self._history.append(AdaptationRecord.from_decision(decision, is_applied=False))
            return decision

        # 11. Improvement Threshold & Hysteresis Gate
        obj_type = self._policy.objective.objective_type
        if obj_type == OptimizationObjectiveType.LATENCY:
            # Latency scores are negative p95 (e.g., -40ms vs -30ms).
            # Improvement is positive when new latency is lower.
            curr_lat = -curr_score if curr_score < 0 else 0.001
            cand_lat = -best_score if best_score < 0 else 0.001
            improvement_pct = (curr_lat - cand_lat) / curr_lat * 100.0
            abs_improvement = best_score - curr_score
        else:
            abs_improvement = best_score - curr_score
            if curr_score != 0.0:
                improvement_pct = (best_score - curr_score) / abs(curr_score) * 100.0
            else:
                improvement_pct = 100.0 if best_score > 0.0 else 0.0

        if (
            improvement_pct < self._policy.min_improvement_pct
            or abs_improvement < self._policy.min_improvement_abs
        ):
            reason = (
                f"Candidate improvement {improvement_pct:.2f}% is below required threshold "
                f"{self._policy.min_improvement_pct:.2f}% (score {curr_score:.4f} -> "
                f"{best_score:.4f})."
            )
            decision = AdaptationDecision(
                window_id=window_id,
                decision_type=AdaptationDecisionType.REJECT,
                current_config=resolved_config,
                proposed_config=best_cand,
                current_score=curr_score,
                proposed_score=best_score,
                improvement_pct=improvement_pct,
                reason=reason,
            )
            self._history.append(AdaptationRecord.from_decision(decision, is_applied=False))
            return decision

        # 12. Candidate Accepted (APPLY)
        reason = (
            f"Candidate {best_cand} improves {obj_type} by {improvement_pct:.2f}% "
            f"({curr_score:.4f} -> {best_score:.4f}) while satisfying all constraints."
        )
        decision = AdaptationDecision(
            window_id=window_id,
            decision_type=AdaptationDecisionType.APPLY,
            current_config=resolved_config,
            proposed_config=best_cand,
            current_score=curr_score,
            proposed_score=best_score,
            improvement_pct=improvement_pct,
            reason=reason,
        )
        self._history.append(AdaptationRecord.from_decision(decision, is_applied=False))
        return decision

    def apply_decision(self, decision: AdaptationDecision) -> bool:
        """Apply an accepted APPLY or ROLLBACK decision to the scheduler.

        Args:
            decision: Validated AdaptationDecision instance.

        Returns:
            True if the configuration change was applied, False otherwise.
        """
        if decision.decision_type not in (
            AdaptationDecisionType.APPLY,
            AdaptationDecisionType.ROLLBACK,
        ):
            return False

        if decision.proposed_config is None:
            return False

        target_tunable = decision.proposed_config

        # 1. Apply to attached scheduler if available
        if self._scheduler is not None:
            self._scheduler.apply_config(target_tunable.to_scheduler_config())

        # 2. Update known-good stack and cooldown
        if decision.decision_type == AdaptationDecisionType.APPLY:
            self._known_good_stack.append((decision.current_config, decision.current_score or 0.0))
            if len(self._known_good_stack) > 10:
                self._known_good_stack = self._known_good_stack[-10:]
            self._cooldown_counter = self._policy.cooldown_windows
        elif decision.decision_type == AdaptationDecisionType.ROLLBACK:
            self._cooldown_counter = self._policy.cooldown_windows
            # If we rolled back to the top known-good config, retain it

        # 3. Record telemetry event if collector is available
        collector = self._scheduler.collector if self._scheduler is not None else None
        if collector is not None:
            event = AdaptationEvent(
                event_id=f"evt-{decision.decision_id}",
                window_id=decision.window_id,
                decision=decision.decision_type.value,
                previous_config=decision.current_config.model_dump(),
                new_config=target_tunable.model_dump(),
                current_score=decision.current_score,
                proposed_score=decision.proposed_score,
                improvement_pct=decision.improvement_pct,
                reason=decision.reason,
                is_applied=True,
            )
            collector.record_adaptation(event)

        # 4. Mark corresponding record in history as applied
        for idx in range(len(self._history) - 1, -1, -1):
            if self._history[idx].window_id == decision.window_id:
                old_rec = self._history[idx]
                self._history[idx] = AdaptationRecord(
                    record_id=old_rec.record_id,
                    window_id=old_rec.window_id,
                    decision_type=old_rec.decision_type,
                    previous_config=old_rec.previous_config,
                    target_config=old_rec.target_config,
                    current_score=old_rec.current_score,
                    target_score=old_rec.target_score,
                    improvement_pct=old_rec.improvement_pct,
                    reason=old_rec.reason,
                    is_applied=True,
                    timestamp=old_rec.timestamp,
                )
                break

        return True

    def step(
        self,
        current_evidence: BenchmarkResult | MetricsSnapshot,
        candidate_evidence: Sequence[BenchmarkResult] | None = None,
        current_config: TunableConfig | SchedulerConfig | None = None,
    ) -> AdaptationDecision:
        """Perform evaluation and automatically apply configuration update if accepted.

        Args:
            current_evidence: Measured metrics from the current evaluation window.
            candidate_evidence: Sequence of candidate benchmark runs.
            current_config: Optional override of active configuration.

        Returns:
            The evaluated AdaptationDecision (with is_applied reflected).
        """
        decision = self.evaluate(current_evidence, candidate_evidence, current_config)
        if decision.decision_type in (
            AdaptationDecisionType.APPLY,
            AdaptationDecisionType.ROLLBACK,
        ):
            applied = self.apply_decision(decision)
            if applied:
                decision = AdaptationDecision(
                    decision_id=decision.decision_id,
                    window_id=decision.window_id,
                    decision_type=decision.decision_type,
                    current_config=decision.current_config,
                    proposed_config=decision.proposed_config,
                    current_score=decision.current_score,
                    proposed_score=decision.proposed_score,
                    improvement_pct=decision.improvement_pct,
                    reason=decision.reason,
                    timestamp=decision.timestamp,
                    is_applied=True,
                )
        return decision
