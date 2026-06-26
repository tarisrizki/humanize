"""Process endpoint — rewrites a draft to match the user's writing style."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.core.writing_engine import apply_style, apply_style_stream
from app.models.schemas import ProcessedText, ProcessRequest
from app.models.style_profile import StyleProfile
from app.storage.json_store import load_json, save_json

router = APIRouter()


@router.get("/style", response_model=StyleProfile)
async def get_global_style(mode: str = "populer") -> StyleProfile:
    """Return the pre-trained Global StyleProfile."""
    profile_path = settings.profiles_path / f"{mode}_style.json"
    if not profile_path.exists():
        profile_path = settings.profiles_path / "global_style.json"
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="Style profile not found.")
    data = load_json(profile_path)
    data["style_mode"] = mode
    return StyleProfile(**data)


@router.post("/process")
async def process_draft(request: ProcessRequest) -> StreamingResponse:
    """Rewrite a draft to match the Global StyleProfile and stream the result.

    Requires that the global style has been trained offline.
    """
    mode = request.style_mode or "populer"
    profile_path = settings.profiles_path / f"{mode}_style.json"
    
    if not profile_path.exists():
        profile_path = settings.profiles_path / "global_style.json"
        
    if not profile_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Style profile not found. Run training script first.",
        )

    # Load the style profile
    try:
        profile_data = load_json(profile_path)
        profile_data["style_mode"] = request.style_mode
        style = StyleProfile(**profile_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load global style profile: {e}",
        )

    return StreamingResponse(
        apply_style_stream(request.draft, style), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )



