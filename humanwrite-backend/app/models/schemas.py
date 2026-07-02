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

class EvaluateRunRequest(BaseModel):
    style_mode: str
    language: str = "id"
    original_text: str
    output_text: str

class EvaluateRunResponse(BaseModel):
    record_id: int
    trigram_overlap: float

class EvaluationRecord(BaseModel):
    id: int
    timestamp: str
    style_mode: str
    language: str
    trigram_overlap: float | None = None
    enhanced_judge_score: float | None = None
    gptzero_enh_human: float | None = None

class HistoryResponse(BaseModel):
    data: list[EvaluationRecord]
