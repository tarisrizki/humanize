"""Process endpoint — rewrites a draft to match the user's writing style."""

from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.core.pipeline import apply_style, apply_style_stream
from app.models.schemas import ProcessedText, ProcessRequest
from app.models.style_profile import StyleProfile
from app.storage.json_store import load_json, save_json

router = APIRouter()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

@router.get("/style", response_model=StyleProfile)
async def get_global_style(mode: str = "populer", lang: str = "id") -> StyleProfile:
    """Return the pre-trained Global StyleProfile."""
    profile_path = settings.profiles_path / f"{mode}_{lang}_style.json"
    if not profile_path.exists():
        # Fallback to older format or ID
        profile_path = settings.profiles_path / f"{mode}_style.json"
        
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="Style profile not found.")
    data = load_json(profile_path)
    data["style_mode"] = mode
    data["language"] = lang
    return StyleProfile(**data)


@router.post("/process")
async def process_draft(request: ProcessRequest, api_key: str = Depends(verify_api_key)) -> StreamingResponse:
    """Rewrite a draft to match the Global StyleProfile and stream the result.

    Requires that the global style has been trained offline.
    """
    mode = request.style_mode or "populer"
    
    from langdetect import detect
    try:
        input_lang = "id" if detect(request.draft[:2000]) in ("id", "ms") else "en"
    except:
        input_lang = "id"
        
    profile_path = settings.profiles_path / f"{mode}_{input_lang}_style.json"
    
    if not profile_path.exists():
        # Fallback
        profile_path = settings.profiles_path / f"{mode}_style.json"
        
    if not profile_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Style profile for {mode} ({input_lang}) not found. Run training script first.",
        )

    # Load the style profile
    try:
        profile_data = load_json(profile_path)
        profile_data["style_mode"] = mode
        profile_data["language"] = input_lang
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



