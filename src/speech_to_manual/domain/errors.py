class PipelineError(Exception):
    """Base pipeline error."""


class DomainValidationError(PipelineError):
    """Invalid domain/config data."""


class PlanValidationError(DomainValidationError):
    """JSON plan has invalid structure."""


class LlmGenerationError(PipelineError):
    """LLM generation failed after retries."""


class SttError(PipelineError):
    """Speech-to-text backend failed."""


class LatexValidationError(DomainValidationError):
    """Generated LaTeX is invalid."""


class JsonRepairError(DomainValidationError):
    """Failed to repair/parse JSON response."""
