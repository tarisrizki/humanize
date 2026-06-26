from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from app.core.evaluator import SQLiteEvaluator, EvaluationRecord, run_llm_judge
from pathlib import Path

router = APIRouter()

evaluator = SQLiteEvaluator(db_path=str(Path("data") / "evaluations.db"))

class EvaluationRequest(BaseModel):
    style_mode: str
    language: str
    original_text: str
    output_text: str
    burstiness: Optional[float] = None
    content_preservation: Optional[float] = None
    ai_word_reduction: Optional[float] = None
    paragraph_integrity: Optional[float] = None
    eyd_score: Optional[float] = None
    judge_score: Optional[float] = None
    judge_feedback: Optional[str] = None
    gptzero_before: Optional[float] = None
    gptzero_after: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class GPTZeroUpdateRequest(BaseModel):
    gptzero_before: Optional[float] = None
    gptzero_after: Optional[float] = None

class JudgeRequest(BaseModel):
    record_id: int          # FK ke evaluation result
    original_text: str
    humanized_text: str
    style_mode: str
    language: str = "id"

@router.post("/run", summary="Log a new evaluation result")
async def run_evaluation(request: EvaluationRequest):
    """
    Saves an evaluation record (from NLP or LLM as a Judge) into the SQLite database.
    """
    try:
        record = EvaluationRecord(
            style_mode=request.style_mode,
            language=request.language,
            original_text=request.original_text,
            output_text=request.output_text,
            burstiness=request.burstiness,
            content_preservation=request.content_preservation,
            ai_word_reduction=request.ai_word_reduction,
            paragraph_integrity=request.paragraph_integrity,
            eyd_score=request.eyd_score,
            judge_score=request.judge_score,
            judge_feedback=request.judge_feedback,
            gptzero_before=request.gptzero_before,
            gptzero_after=request.gptzero_after,
            metadata=request.metadata
        )
        record_id = evaluator.log_evaluation(record)
        return {"status": "success", "record_id": record_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{record_id}/gptzero", summary="Update GPTZero score manually")
async def update_gptzero(record_id: int, request: GPTZeroUpdateRequest):
    """
    Updates the GPTZero score for a specific evaluation record.
    """
    try:
        success = evaluator.update_gptzero_score(
            record_id, 
            gptzero_before=request.gptzero_before, 
            gptzero_after=request.gptzero_after
        )
        if not success:
            raise HTTPException(status_code=404, detail="Record not found or no updates provided")
        return {"status": "success", "message": "GPTZero score updated"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", summary="Get evaluation history")
async def get_history(limit: int = 100):
    """
    Retrieves the most recent evaluation records from the database.
    """
    try:
        records = evaluator.get_evaluations(limit=limit)
        return {"status": "success", "data": [record.model_dump() for record in records]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/judge", summary="Run LLM as a Judge evaluation")
async def run_judge(request: JudgeRequest):
    """
    Jalankan Llama-3.3-70b (via Groq) sebagai judge untuk menilai kualitas output.
    Hasilnya disimpan sebagai judge_score dan judge_feedback di DB.
    """
    import json
    
    result = run_llm_judge(
        original_text=request.original_text,
        humanized_text=request.humanized_text,
        style_mode=request.style_mode,
        language=request.language,
    )

    if not result.get("success"):
        raise HTTPException(500, detail=result.get("error", "Unknown error in LLM Judge"))

    # Format feedback untuk disimpan ke DB
    feedback = {
        "naturalness":         result["naturalness"],
        "register_compliance": result["register_compliance"],
        "content_fidelity":    result["content_fidelity"],
        "eyd_grammar":         result["eyd_grammar"],
        "anti_detection":      result["anti_detection"],
        "critical_issues":     result["critical_issues"],
        "highlight":           result["highlight"],
        "worst_sentence":      result["worst_sentence"],
    }

    # Update DB
    evaluator.update_judge_result(
        record_id=request.record_id,
        judge_score=result["overall_score"],
        judge_feedback=json.dumps(feedback, ensure_ascii=False),
    )

    return {
        "status":        "success",
        "overall_score": result["overall_score"],
        "breakdown":     feedback,
    }
