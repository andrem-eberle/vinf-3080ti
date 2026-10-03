class VinfError(Exception):
    """Base error for engine failures."""


class ConfigurationError(VinfError):
    """Raised when engine or generation configuration is invalid."""


class UnsupportedHardwareError(VinfError):
    """Raised when the selected hardware cannot run the requested executor."""


class UnsupportedModelError(VinfError):
    """Raised when model metadata does not match supported engine shapes."""


class ExecutorUnavailableError(VinfError):
    """Raised when a requested execution backend is not available."""


class DecodeError(VinfError):
    """Raised for runtime decode failures."""



class InsufficientMemoryError(VinfError):
    """Raised when VRAM or host memory cannot hold the requested runtime state."""
