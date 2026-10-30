"""PaperPilot runtime errors."""


class PaperPilotError(RuntimeError):
    """Base exception for user-facing runtime failures."""


class ConfigurationError(PaperPilotError):
    """Raised when a required local setting or secret is unavailable."""
