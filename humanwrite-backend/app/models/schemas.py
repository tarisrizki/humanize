"""Request / response schemas for the HumanWrite API."""

from pydantic import BaseModel, Field


# ── Process endpoint ──────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    """Request body for the /process endpoint."""
    draft: str = Field(..., min_length=1, max_length=20000, description="The draft text to rewrite")
    style_mode: str = Field(default="populer", description="The chosen style mode")

class ProcessedText(BaseModel):
    """Structured output returned by the writing engine."""
    final_text: str = Field(..., description="The rewritten text matching the global style")
    changes_made: list[str] = Field(
        default_factory=list,
        description="Human-readable list of changes applied to the draft",
    )


# ── Generic ───────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
