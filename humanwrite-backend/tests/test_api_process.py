import pytest
from unittest.mock import patch
from app.models.schemas import ProcessedText
from app.models.style_profile import StyleProfile
from app.storage.json_store import save_json
from app.config import settings

@patch("app.api.v1.endpoints.process.apply_style_stream")
def test_process_endpoint(mock_apply_style_stream, client):
    # Setup mock generator
    import json
    
    async def async_generator(*args, **kwargs):
        yield "This is "
        yield "the rewritten text."
        yield "\n\n___METRICS___\n\n"
        yield json.dumps({"voice_match_score": 95.0, "changes_made": ["Changed tone to formal"]})
        
    mock_apply_style_stream.side_effect = async_generator

    # Setup dummy global profile
    profile = StyleProfile(user_id="global", language="en")
    profile_path = settings.profiles_path / "global_style.json"
    save_json(profile_path, profile.model_dump())

    # Call process
    response = client.post("/api/v1/process", json={"draft": "Draft text"})
    
    assert response.status_code == 200
    
    # Read streaming response
    content = response.text
    assert "This is the rewritten text." in content
    assert "___METRICS___" in content
    
    # Extract JSON part
    json_part = content.split("___METRICS___")[1].strip()
    data = json.loads(json_part)
    
    assert data["voice_match_score"] == 95.0
    assert data["changes_made"] == ["Changed tone to formal"]

def test_process_no_profile(client):
    # Ensure profile is removed if it exists
    profile_path = settings.profiles_path / "global_style.json"
    if profile_path.exists():
        profile_path.unlink()
        
    response = client.post("/api/v1/process", json={"draft": "Draft"})
    assert response.status_code == 503
    assert "Style profile not found" in response.json()["detail"]
