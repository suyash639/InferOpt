"""Deterministic workload regime detection subsystem for closed-loop adaptive control."""

import time
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from inferopt.telemetry.models import MetricsSnapshot


class WorkloadRegime(StrEnum):
    """Categorization of observed workload traffic regimes."""

    LIGHT = "LIGHT"
    BURSTY = "BURSTY"
    SATURATED = "SATURATED"
    UNKNOWN = "UNKNOWN"


class RegimeDetectionConfig(BaseModel):
    """Configuration thresholds for deterministic workload regime detection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_requests_for_detection: int = Field(
        default=2,
        ge=1,
        description="Minimum completed or active requests in window required to classify.",
    )
    light_max_arrival_rate: float = Field(
        default=2.5,
        ge=0.0,
        description="Maximum request arrival rate (req/s) for LIGHT regime.",
    )
    light_max_queue_depth: int = Field(
        default=1,
        ge=0,
        description="Maximum peak queue depth for LIGHT regime.",
    )
    light_max_concurrency: int = Field(
        default=2,
        ge=1,
        description="Maximum peak active requests for LIGHT regime.",
    )
    bursty_min_queue_depth: int = Field(
        default=3,
        ge=1,
        description="Minimum peak queue depth indicating burst accumulation.",
    )
    saturated_min_concurrency: int = Field(
        default=5,
        ge=1,
        description="Minimum active concurrency indicating SATURATED regime.",
    )
    saturated_min_queue_depth: int = Field(
        default=4,
        ge=1,
        description="Minimum sustained queue depth indicating SATURATED regime.",
    )


class RegimeDetectionResult(BaseModel):
    """Immutable output of a deterministic regime detection evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    regime: WorkloadRegime = Field(description="Classified workload regime.")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the deterministic classification.",
    )
    arrival_rate_rps: float = Field(
        default=0.0,
        ge=0.0,
        description="Observed request arrival/completion rate (req/s).",
    )
    current_queue_depth: int = Field(
        default=0,
        ge=0,
        description="Instantaneous queue depth at snapshot time.",
    )
    peak_queue_depth: int = Field(
        default=0,
        ge=0,
        description="Peak queue depth observed in window.",
    )
    active_concurrency: int = Field(
        default=0,
        ge=0,
        description="Peak or current active concurrent requests in window.",
    )
    avg_batch_size: float = Field(
        default=1.0,
        ge=0.0,
        description="Average batch size formed during the window.",
    )
    p95_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Observed p95 latency in milliseconds.",
    )
    reason: str = Field(
        description="Deterministic explanation and metrics justifying the classification."
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Timestamp when regime detection was evaluated.",
    )


class DeterministicRegimeDetector:
    """Deterministic rule-based workload regime detector operating on telemetry snapshots.

    Evaluates queue dynamics, arrival/completion throughput, active concurrency,
    and latency distributions against explicit configuration thresholds to classify
    workloads into LIGHT, BURSTY, or SATURATED regimes.
    """

    def __init__(self, config: RegimeDetectionConfig | None = None) -> None:
        """Initialize detector with deterministic thresholds."""
        self._config = config or RegimeDetectionConfig()

    @property
    def config(self) -> RegimeDetectionConfig:
        """Active detection configuration thresholds."""
        return self._config

    def detect(
        self,
        snapshot: MetricsSnapshot,
        previous_snapshot: MetricsSnapshot | None = None,
    ) -> RegimeDetectionResult:
        """Deterministically classify workload regime from a telemetry snapshot.

        Args:
            snapshot: Current telemetry metrics snapshot.
            previous_snapshot: Optional previous snapshot to compute sliding-window deltas.

        Returns:
            RegimeDetectionResult with classified regime and justification.
        """
        # Extract metrics
        if previous_snapshot is not None:
            # Window delta calculations
            dt = max(0.001, snapshot.timestamp - previous_snapshot.timestamp)
            d_reqs = max(
                0,
                snapshot.requests.completed_requests
                - previous_snapshot.requests.completed_requests,
            )
            arrival_rps = d_reqs / dt if dt > 0 else 0.0
            peak_q = max(snapshot.queue.peak_queue_depth, snapshot.queue.current_queue_depth)
            curr_q = snapshot.queue.current_queue_depth
            peak_c = max(
                snapshot.queue.peak_active_requests,
                snapshot.queue.current_active_requests,
            )
            total_activity = d_reqs + snapshot.queue.current_active_requests
        else:
            elapsed = max(0.001, snapshot.throughput.elapsed_sec)
            arrival_rps = (
                snapshot.throughput.requests_per_sec
                if snapshot.throughput.requests_per_sec > 0
                else (snapshot.requests.completed_requests / elapsed)
            )
            peak_q = snapshot.queue.peak_queue_depth
            curr_q = snapshot.queue.current_queue_depth
            peak_c = snapshot.queue.peak_active_requests
            total_activity = (
                snapshot.requests.completed_requests + snapshot.queue.current_active_requests
            )

        p95_lat = snapshot.requests.p95_total_latency_ms
        avg_batch = snapshot.batches.avg_batch_size if snapshot.batches.total_batches > 0 else 1.0

        # Check sufficiency gate
        if total_activity < self._config.min_requests_for_detection:
            return RegimeDetectionResult(
                regime=WorkloadRegime.UNKNOWN,
                confidence=0.0,
                arrival_rate_rps=arrival_rps,
                current_queue_depth=curr_q,
                peak_queue_depth=peak_q,
                active_concurrency=peak_c,
                avg_batch_size=avg_batch,
                p95_latency_ms=p95_lat,
                reason=(
                    f"Insufficient requests in window ({total_activity} < "
                    f"{self._config.min_requests_for_detection})."
                ),
            )

        # Rule 1: Saturated Regime
        # High sustained concurrency (>= 5) or high sustained queue (>= 4 with concurrency >= 4)
        if peak_c >= self._config.saturated_min_concurrency or (
            peak_q >= self._config.saturated_min_queue_depth and peak_c >= 4
        ):
            return RegimeDetectionResult(
                regime=WorkloadRegime.SATURATED,
                confidence=1.0,
                arrival_rate_rps=arrival_rps,
                current_queue_depth=curr_q,
                peak_queue_depth=peak_q,
                active_concurrency=peak_c,
                avg_batch_size=avg_batch,
                p95_latency_ms=p95_lat,
                reason=(
                    f"Saturated traffic detected: peak concurrency={peak_c} "
                    f"(>= {self._config.saturated_min_concurrency}), "
                    f"peak queue={peak_q} (>= {self._config.saturated_min_queue_depth})."
                ),
            )

        # Rule 2: Bursty Regime
        # Sudden queue depth spike (>= 3) while concurrency is moderate (< 5)
        if peak_q >= self._config.bursty_min_queue_depth:
            return RegimeDetectionResult(
                regime=WorkloadRegime.BURSTY,
                confidence=1.0,
                arrival_rate_rps=arrival_rps,
                current_queue_depth=curr_q,
                peak_queue_depth=peak_q,
                active_concurrency=peak_c,
                avg_batch_size=avg_batch,
                p95_latency_ms=p95_lat,
                reason=(
                    f"Bursty traffic spike detected: peak queue={peak_q} "
                    f"(>= {self._config.bursty_min_queue_depth}) with concurrency={peak_c}."
                ),
            )

        # Rule 3: Light Regime
        # Low arrival rate, minimal queue depth (<= 1), low concurrency (<= 2)
        if (
            peak_q <= self._config.light_max_queue_depth
            and peak_c <= self._config.light_max_concurrency
            and arrival_rps <= self._config.light_max_arrival_rate
        ):
            return RegimeDetectionResult(
                regime=WorkloadRegime.LIGHT,
                confidence=1.0,
                arrival_rate_rps=arrival_rps,
                current_queue_depth=curr_q,
                peak_queue_depth=peak_q,
                active_concurrency=peak_c,
                avg_batch_size=avg_batch,
                p95_latency_ms=p95_lat,
                reason=(
                    f"Light traffic detected: peak queue={peak_q} "
                    f"(<= {self._config.light_max_queue_depth}), "
                    f"concurrency={peak_c} (<= {self._config.light_max_concurrency}), "
                    f"arrival rate={arrival_rps:.2f} rps "
                    f"(<= {self._config.light_max_arrival_rate})."
                ),
            )

        # Rule 4: Moderate/Fallback classification
        if peak_c >= 4:
            regime = WorkloadRegime.SATURATED
            reason = f"Moderate-high concurrency {peak_c} mapped to SATURATED regime."
        elif peak_q >= 2:
            regime = WorkloadRegime.BURSTY
            reason = f"Queue pressure {peak_q} mapped to BURSTY regime."
        else:
            regime = WorkloadRegime.LIGHT
            reason = f"Low queue {peak_q} and concurrency {peak_c} mapped to LIGHT regime."

        return RegimeDetectionResult(
            regime=regime,
            confidence=0.85,
            arrival_rate_rps=arrival_rps,
            current_queue_depth=curr_q,
            peak_queue_depth=peak_q,
            active_concurrency=peak_c,
            avg_batch_size=avg_batch,
            p95_latency_ms=p95_lat,
            reason=reason,
        )
