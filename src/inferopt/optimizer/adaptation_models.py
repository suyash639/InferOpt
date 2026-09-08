"""Domain models for closed-loop adaptation policies, decisions, and historical records."""

import time
import uuid
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inferopt.optimizer.models import (
    ObjectiveConfig,
    OptimizationConstraints,
    TunableConfig,
)


class AdaptationDecisionType(StrEnum):
    """Categorization of adaptive control evaluation outcomes."""

    APPLY = "APPLY"
    REJECT = "REJECT"
    COOLDOWN = "COOLDOWN"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INFEASIBLE = "INFEASIBLE"
    NO_CHANGE = "NO_CHANGE"
    ROLLBACK = "ROLLBACK"


class AdaptationPolicy(BaseModel):
    """Configuration governing closed-loop adaptation decisions and safety boundaries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective: ObjectiveConfig = Field(
        default_factory=ObjectiveConfig,
        description="Target optimization objective and weighting.",
    )
    constraints: OptimizationConstraints | None = Field(
        default=None,
        description="Optional SLA and operational constraints for candidate feasibility.",
    )
    min_improvement_pct: float = Field(
        default=5.0,
        ge=0.0,
        description="Minimum percentage improvement required over current score to accept switch.",
    )
    min_improvement_abs: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum absolute score improvement required to accept switch.",
    )
    cooldown_windows: int = Field(
        default=2,
        ge=0,
        description="Number of evaluation windows to wait after a switch before re-adapting.",
    )
    min_completed_requests: int = Field(
        default=20,
        ge=1,
        description="Minimum completed requests required in an evaluation window.",
    )
    min_completed_batches: int = Field(
        default=5,
        ge=1,
        description="Minimum completed batches required in an evaluation window.",
    )
    min_concurrency: int = Field(
        default=1,
        gt=0,
        description="Lower bound on allowable max_concurrency.",
    )
    max_concurrency: int = Field(
        default=16,
        gt=0,
        description="Upper bound on allowable max_concurrency.",
    )
    min_batch_size: int = Field(
        default=1,
        gt=0,
        description="Lower bound on allowable max_batch_size.",
    )
    max_batch_size: int = Field(
        default=32,
        gt=0,
        description="Upper bound on allowable max_batch_size.",
    )
    min_batch_wait_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Lower bound on allowable batch_wait_ms.",
    )
    max_batch_wait_ms: float = Field(
        default=100.0,
        ge=0.0,
        description="Upper bound on allowable batch_wait_ms.",
    )
    enable_rollback: bool = Field(
        default=True,
        description="Enable automatic rollback to known-good config on severe SLA violation.",
    )
    rollback_degradation_pct: float = Field(
        default=20.0,
        ge=0.0,
        description="Degradation percentage threshold relative to baseline triggering rollback.",
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "AdaptationPolicy":
        """Verify that lower bounds do not exceed upper bounds."""
        if self.min_concurrency > self.max_concurrency:
            raise ValueError(
                f"min_concurrency ({self.min_concurrency}) must not exceed "
                f"max_concurrency ({self.max_concurrency})"
            )
        if self.min_batch_size > self.max_batch_size:
            raise ValueError(
                f"min_batch_size ({self.min_batch_size}) must not exceed "
                f"max_batch_size ({self.max_batch_size})"
            )
        if self.min_batch_wait_ms > self.max_batch_wait_ms:
            raise ValueError(
                f"min_batch_wait_ms ({self.min_batch_wait_ms}) must not exceed "
                f"max_batch_wait_ms ({self.max_batch_wait_ms})"
            )
        return self

    def is_within_bounds(self, config: TunableConfig) -> tuple[bool, str]:
        """Check whether a tunable configuration satisfies all policy parameter bounds."""
        if config.max_concurrency < self.min_concurrency:
            return (
                False,
                f"Concurrency {config.max_concurrency} < min {self.min_concurrency}",
            )
        if config.max_concurrency > self.max_concurrency:
            return (
                False,
                f"Concurrency {config.max_concurrency} > max {self.max_concurrency}",
            )
        if config.max_batch_size < self.min_batch_size:
            return (
                False,
                f"Batch size {config.max_batch_size} < min {self.min_batch_size}",
            )
        if config.max_batch_size > self.max_batch_size:
            return (
                False,
                f"Batch size {config.max_batch_size} > max {self.max_batch_size}",
            )
        if config.batch_wait_ms < self.min_batch_wait_ms:
            return (
                False,
                f"Batch wait {config.batch_wait_ms}ms < min {self.min_batch_wait_ms}ms",
            )
        if config.batch_wait_ms > self.max_batch_wait_ms:
            return (
                False,
                f"Batch wait {config.batch_wait_ms}ms > max {self.max_batch_wait_ms}ms",
            )
        return True, "Configuration satisfies policy bounds."


class AdaptationDecision(BaseModel):
    """Immutable outcome of an adaptive evaluation step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str = Field(
        default_factory=lambda: f"dec-{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this decision.",
    )
    window_id: int = Field(
        default=0,
        ge=0,
        description="Index of the evaluation window.",
    )
    decision_type: AdaptationDecisionType = Field(
        ...,
        description="Type of decision reached.",
    )
    current_config: TunableConfig = Field(
        ...,
        description="Active scheduler configuration during evaluation.",
    )
    proposed_config: TunableConfig | None = Field(
        default=None,
        description="Proposed configuration if accepted or evaluated.",
    )
    current_score: float | None = Field(
        default=None,
        description="Objective score computed for current configuration.",
    )
    proposed_score: float | None = Field(
        default=None,
        description="Objective score computed for proposed candidate.",
    )
    improvement_pct: float | None = Field(
        default=None,
        description="Relative improvement percentage.",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Explainable human-readable justification for the decision.",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Wall-clock timestamp of decision creation.",
    )
    is_applied: bool = Field(
        default=False,
        description="Whether this decision was applied to the active scheduler.",
    )


class AdaptationRecord(BaseModel):
    """Historical immutable record of an evaluated or applied adaptation decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(
        ...,
        min_length=1,
        description="Unique record identifier.",
    )
    window_id: int = Field(
        ...,
        ge=0,
        description="Evaluation window index.",
    )
    decision_type: AdaptationDecisionType = Field(
        ...,
        description="Decision type.",
    )
    previous_config: TunableConfig = Field(
        ...,
        description="Configuration before decision.",
    )
    target_config: TunableConfig | None = Field(
        default=None,
        description="Target configuration proposed or rolled back to.",
    )
    current_score: float | None = Field(
        default=None,
        description="Baseline score before adaptation.",
    )
    target_score: float | None = Field(
        default=None,
        description="Target score estimated or evaluated.",
    )
    improvement_pct: float | None = Field(
        default=None,
        description="Calculated improvement percentage.",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Explanatory reason.",
    )
    is_applied: bool = Field(
        default=False,
        description="Whether the configuration was applied to the scheduler.",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Wall-clock timestamp.",
    )

    @classmethod
    def from_decision(
        cls, decision: AdaptationDecision, is_applied: bool | None = None
    ) -> "AdaptationRecord":
        """Construct historical record from an AdaptationDecision."""
        applied = decision.is_applied if is_applied is None else is_applied
        return cls(
            record_id=f"rec-{decision.decision_id}",
            window_id=decision.window_id,
            decision_type=decision.decision_type,
            previous_config=decision.current_config,
            target_config=decision.proposed_config,
            current_score=decision.current_score,
            target_score=decision.proposed_score,
            improvement_pct=decision.improvement_pct,
            reason=decision.reason,
            is_applied=applied,
            timestamp=decision.timestamp,
        )
