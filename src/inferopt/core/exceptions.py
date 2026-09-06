"""Core exceptions for the InferOpt serving system."""


class InferOptError(Exception):
    """Base exception for all InferOpt system errors."""


class BackendError(InferOptError):
    """Exception raised when an inference backend engine encounters an error."""


class InferenceError(BackendError):
    """Exception raised when an inference request execution fails."""


class SchedulerError(InferOptError):
    """Base exception for all request scheduler errors."""


class QueueFullError(SchedulerError):
    """Exception raised when the scheduler request queue is at maximum capacity."""


class SchedulerNotRunningError(SchedulerError):
    """Exception raised when a request is submitted to an unstarted or inactive scheduler."""


class SchedulerShutdownError(SchedulerError):
    """Exception raised when a request is rejected or cancelled due to scheduler shutdown."""


class RequestCancelledError(SchedulerError):
    """Exception raised when an in-flight or queued request is explicitly cancelled."""
