"""Evaluation endpoints for HumanWrite AI."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.models.schemas import EvaluateRunRequest, EvaluateRunResponse, HistoryResponse
from app.core.evaluator import save_evaluation, get_history, update_gptzero_score
# Import the existing overlap checker
try:
    from app.core.text_utils import check_trigram_overlap
except ImportError:
    # Fallback if it's nested differently
    from app.core.text_utils.metrics import check_trigram_overlap

router = APIRouter()

@router.post("/run", response_model=EvaluateRunResponse)
async def run_evaluation(request: EvaluateRunRequest) -> EvaluateRunResponse:
    """Evaluate a processed text and save it to the database."""
    try:
        # Compute trigram overlap
        overlap = check_trigram_overlap(request.original_text, request.output_text)
        
        # Save to SQLite
        record_id = save_evaluation(
            style_mode=request.style_mode,
            language=request.language,
            original_text=request.original_text,
            output_text=request.output_text,
            trigram_overlap=overlap,
            details={}
        )
        
        return EvaluateRunResponse(record_id=record_id, trigram_overlap=overlap)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=HistoryResponse)
async def get_evaluation_history(limit: int = 50) -> HistoryResponse:
    """Fetch recent evaluation runs for the dashboard."""
    try:
        rows = get_history(limit=limit)
        return HistoryResponse(data=rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class GPTZeroUpdateRequest(BaseModel):
    gptzero_enh_human: float

@router.patch("/{record_id}/gptzero")
async def update_gptzero(record_id: int, request: GPTZeroUpdateRequest):
    """Update GPTZero score manually."""
    success = update_gptzero_score(record_id, request.gptzero_enh_human)
    if not success:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"status": "success"}
