import pytest
import json
from unittest.mock import patch
from app.models.schemas import ProcessedText
from app.models.style_profile import StyleProfile
from app.storage.json_store import save_json
from app.config import settings

@patch("app.api.v1.endpoints.process.apply_style_stream")
def test_process_endpoint(mock_apply_style_stream, client):
    """Test that /process returns valid SSE stream with text and metrics events."""

    async def async_generator(*args, **kwargs):
        # Simulate SSE format used by apply_style_stream
        yield 'event: text\ndata: "Hello "\n\n'
        yield 'event: text\ndata: "world."\n\n'
        yield f'event: metrics\ndata: {json.dumps({"changes_made": ["Changed tone to formal"]})}\n\n'

    mock_apply_style_stream.side_effect = async_generator

    # Setup dummy global profile
    profile = StyleProfile(user_id="global", language="en")
    profile_path = settings.profiles_path / "global_style.json"
    save_json(profile_path, profile.model_dump())

    # Call process
    response = client.post("/api/v1/process", json={"draft": "Draft text"})

    assert response.status_code == 200

    # Read streaming response — should contain SSE events
    content = response.text
    assert "event: text" in content
    assert "Hello " in content
    assert "event: metrics" in content
    assert "changes_made" in content

def test_process_no_profile(client):
    """Test that /process returns 503 when no style profile exists."""
    profile_path = settings.profiles_path / "global_style.json"
    if profile_path.exists():
        profile_path.unlink()

    response = client.post("/api/v1/process", json={"draft": "Draft"})
    assert response.status_code == 503
    assert "Style profile not found" in response.json()["detail"]
