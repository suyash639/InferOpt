"""Scheduler-specific exception definitions."""

from inferopt.core.exceptions import (
    QueueFullError,
    RequestCancelledError,
    SchedulerError,
    SchedulerNotRunningError,
    SchedulerShutdownError,
)

__all__ = [
    "QueueFullError",
    "RequestCancelledError",
    "SchedulerError",
    "SchedulerNotRunningError",
    "SchedulerShutdownError",
]
