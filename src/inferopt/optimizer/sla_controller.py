"""SLA-aware online adaptive controller with deadband hysteresis and Pareto guardrailing."""

import time
from collections.abc import Sequence
from typing import Final

from inferopt.optimizer.models import TunableConfig
from inferopt.optimizer.sla_models import (
    DEFAULT_MITIGATION_DWELL_SEC,
    SLAMode,
    SLAStatusRecord,
    Step14SLAAdaptationEventRecord,
    TargetSLO,
)
from inferopt.scheduler.scheduler import Scheduler
from inferopt.telemetry.models import MetricsSnapshot

# Standard Pareto candidate ladder sorted by increasing batch capacity and concurrency
DEFAULT_SLA_CANDIDATE_LADDER: Final[tuple[TunableConfig, ...]] = (
    # Level 0: Min latency / conservative
    TunableConfig(max_concurrency=1, max_batch_size=1, batch_wait_ms=50.0),
    # Level 1: Light workload sweet spot
    TunableConfig(max_concurrency=1, max_batch_size=2, batch_wait_ms=50.0),
    # Level 2: Moderate concurrency
    TunableConfig(max_concurrency=4, max_batch_size=4, batch_wait_ms=50.0),
    # Level 3: Max batch throughput
    TunableConfig(max_concurrency=8, max_batch_size=8, batch_wait_ms=50.0),
)


class SLAConstrainedAdaptiveController:
    """Closed-loop adaptive controller regulating scheduler config for tail-latency SLOs.

    Evaluates live telemetry snapshots against an explicit TargetSLO (p95 <= T_target).
    Applies deadband hysteresis to maintain stability:
    1. MITIGATE: When p95 > T_target, downshifts concurrency/batching to clear queue.
    2. EXPAND: When p95 < alpha * T_target and traffic demands capacity, upshifts along ladder.
    3. HOLD: When alpha * T_target <= p95 <= T_target, holds steady (deadband).
    """

    def __init__(
        self,
        target_slo: TargetSLO,
        candidate_ladder: Sequence[TunableConfig] = DEFAULT_SLA_CANDIDATE_LADDER,
        min_dwell_time_sec: float = DEFAULT_MITIGATION_DWELL_SEC,
    ) -> None:
        """Initialize the SLA-constrained adaptive controller.

        Args:
            target_slo: Service Level Objective config with target p95 and headroom ratio.
            candidate_ladder: Ordered candidate configs from lowest latency to highest throughput.
            min_dwell_time_sec: Minimum dwell duration before allowing subsequent reconfigurations.
        """
        if not candidate_ladder:
            raise ValueError("candidate_ladder must contain at least one TunableConfig")

        self._target_slo = target_slo
        self._ladder = tuple(candidate_ladder)
        self._min_dwell_time_sec = min_dwell_time_sec

        # State tracking
        self._last_adaptation_time: float = 0.0
        self._last_mode: SLAMode = SLAMode.HOLD
        self._last_direction: int = 0  # +1 for upshift, -1 for downshift
        self._oscillation_count: int = 0
        self._adaptation_events: list[Step14SLAAdaptationEventRecord] = []
        self._evaluation_history: list[SLAStatusRecord] = []

    @property
    def target_slo(self) -> TargetSLO:
        """Get the active TargetSLO configuration."""
        return self._target_slo

    @property
    def candidate_ladder(self) -> tuple[TunableConfig, ...]:
        """Get the ordered candidate Pareto ladder."""
        return self._ladder

    @property
    def oscillation_count(self) -> int:
        """Get the count of rapid directional oscillations recorded."""
        return self._oscillation_count

    @property
    def adaptation_events(self) -> tuple[Step14SLAAdaptationEventRecord, ...]:
        """Get chronological adaptation events executed by this controller."""
        return tuple(self._adaptation_events)

    def _find_ladder_index(self, config: TunableConfig) -> int:
        """Find the index of the matching or closest candidate in the Pareto ladder."""
        for i, cand in enumerate(self._ladder):
            if (
                cand.max_concurrency == config.max_concurrency
                and cand.max_batch_size == config.max_batch_size
            ):
                return i

        # Fallback: match by concurrency
        for i, cand in enumerate(self._ladder):
            if cand.max_concurrency == config.max_concurrency:
                return i

        return 0

    def evaluate(
        self,
        snapshot: MetricsSnapshot,
        active_config: TunableConfig,
        current_time: float | None = None,
    ) -> SLAStatusRecord:
        """Evaluate telemetry snapshot against target SLO and determine next configuration.

        Args:
            snapshot: Current telemetry snapshot containing queue, latency, and throughput metrics.
            active_config: Currently active tunable configuration on the scheduler.
            current_time: Optional explicit timestamp (defaults to time.time()).

        Returns:
            SLAStatusRecord containing evaluated mode, headroom, and recommended configuration.
        """
        now = current_time if current_time is not None else time.time()
        active_idx = self._find_ladder_index(active_config)

        # Extract observed p95 total latency and queue wait
        obs_p95 = snapshot.requests.p95_total_latency_ms
        obs_q_wait = snapshot.requests.avg_queue_wait_ms
        peak_q = snapshot.queue.peak_queue_depth
        arrival_rate = snapshot.throughput.requests_per_sec

        target_p95 = self._target_slo.p95_latency_ms
        headroom_ms = target_p95 - obs_p95
        headroom_thresh = self._target_slo.headroom_ratio * target_p95

        mode: SLAMode
        target_idx: int
        reason: str
        slo_violated: bool = obs_p95 > target_p95

        # 1. Mitigation check: if p95 exceeds SLO or queue wait violates threshold
        if slo_violated or (
            self._target_slo.max_queue_wait_ms is not None
            and obs_q_wait > self._target_slo.max_queue_wait_ms
        ):
            mode = SLAMode.MITIGATE
            target_idx = max(0, active_idx - 1)
            reason = (
                f"p95 latency ({obs_p95:.2f}ms) exceeds target SLO ({target_p95:.2f}ms) "
                f"[queue_wait={obs_q_wait:.2f}ms] - downscaling capacity to mitigate tail latency."
            )
        # 2. Expansion check: if latency is safely below headroom and traffic demands capacity
        elif obs_p95 > 0.0 and obs_p95 < headroom_thresh and (peak_q >= 2 or arrival_rate > 2.0):
            mode = SLAMode.EXPAND
            target_idx = min(len(self._ladder) - 1, active_idx + 1)
            reason = (
                f"Latency headroom available ({obs_p95:.2f}ms < {headroom_thresh:.2f}ms) "
                f"with demand (peak_q={peak_q}, rate={arrival_rate:.1f} rps) - upscaling capacity."
            )
        # 3. Deadband hold: operating safely within [alpha * SLO, SLO]
        else:
            mode = SLAMode.HOLD
            target_idx = active_idx
            reason = (
                f"Operating within stable SLO deadband ({obs_p95:.2f}ms in "
                f"[{headroom_thresh:.1f}ms, {target_p95:.1f}ms]) - holding configuration steady."
            )

        # Check dwell time gating
        target_cfg = self._ladder[target_idx]
        rec_cfg: TunableConfig

        if target_idx != active_idx:
            time_since_last = now - self._last_adaptation_time
            if time_since_last < self._min_dwell_time_sec and self._last_adaptation_time > 0.0:
                rec_cfg = active_config
                reason += (
                    f" [Dwell gate: {time_since_last:.2f}s < "
                    f"{self._min_dwell_time_sec:.2f}s cooldown - held]"
                )
            else:
                rec_cfg = target_cfg
        else:
            rec_cfg = active_config

        record = SLAStatusRecord(
            timestamp=now,
            target_p95_ms=target_p95,
            observed_p95_ms=obs_p95,
            observed_queue_wait_ms=obs_q_wait,
            headroom_ms=headroom_ms,
            mode=mode,
            slo_violated=slo_violated,
            active_config=active_config,
            recommended_config=rec_cfg,
            reason=reason,
        )

        self._evaluation_history.append(record)
        return record

    def apply_decision(
        self,
        status: SLAStatusRecord,
        scheduler: Scheduler,
        current_time: float | None = None,
        phase_index: int = 0,
        phase_name: str = "",
        timestamp_offset_sec: float = 0.0,
    ) -> bool:
        """Apply recommended configuration to running scheduler if a reconfiguration is warranted.

        Args:
            status: The SLAStatusRecord generated by evaluate().
            scheduler: Running Scheduler instance to dynamically reconfigure.
            current_time: Optional explicit timestamp.
            phase_index: Optional phase sequence index.
            phase_name: Optional phase identifier.
            timestamp_offset_sec: Optional elapsed seconds from start of benchmark.

        Returns:
            True if a dynamic reconfiguration was applied, False otherwise.
        """
        now = current_time if current_time is not None else time.time()
        rec_cfg = status.recommended_config
        active_cfg = status.active_config

        if (
            rec_cfg.max_concurrency == active_cfg.max_concurrency
            and rec_cfg.max_batch_size == active_cfg.max_batch_size
            and rec_cfg.batch_wait_ms == active_cfg.batch_wait_ms
        ):
            return False

        # Determine direction (+1 for upshift, -1 for downshift)
        old_idx = self._find_ladder_index(active_cfg)
        new_idx = self._find_ladder_index(rec_cfg)
        direction = 1 if new_idx > old_idx else -1

        if self._last_direction != 0 and direction != self._last_direction:
            self._oscillation_count += 1

        self._last_direction = direction
        self._last_mode = status.mode
        self._last_adaptation_time = now

        # Apply to live scheduler without dropping requests or restarting engine
        t_before = time.perf_counter()
        scheduler.apply_config(rec_cfg.to_scheduler_config())
        t_adapt_ms = (time.perf_counter() - t_before) * 1000.0

        event = Step14SLAAdaptationEventRecord(
            event_id=status.record_id,
            timestamp_offset_sec=timestamp_offset_sec,
            phase_index=phase_index,
            phase_name=phase_name,
            mode=status.mode.value,
            old_config=f"c={active_cfg.max_concurrency}, b={active_cfg.max_batch_size}",
            new_config=f"c={rec_cfg.max_concurrency}, b={rec_cfg.max_batch_size}",
            observed_p95_ms=status.observed_p95_ms,
            target_slo_p95_ms=status.target_p95_ms,
            observed_queue_wait_ms=status.observed_queue_wait_ms,
            reason=status.reason,
            time_to_detect_ms=10.0,
            time_to_adapt_ms=t_adapt_ms,
            in_flight_requests_at_change=scheduler.active_count,
            queue_depth_at_change=scheduler.queued_count,
        )
        self._adaptation_events.append(event)
        return True
