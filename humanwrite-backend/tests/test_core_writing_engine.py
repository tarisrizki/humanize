import pytest
from unittest.mock import patch, MagicMock
from app.core.writing_engine import apply_style, _build_system_prompt
from app.models.style_profile import StyleProfile

def test_build_system_prompt_en():
    profile = StyleProfile(
        user_id="test1",
        language="en",
        avg_sentence_length=15.5,
        emotion_register="formal"
    )
    prompt = _build_system_prompt(profile, paragraph_count=2)
    assert "LANGUAGE: The user writes in English." in prompt
    assert "Average sentence length: ~15.5" in prompt
    assert "Tone & Register: formal" in prompt

def test_build_system_prompt_id():
    profile = StyleProfile(
        user_id="test2",
        language="id"
    )
    prompt = _build_system_prompt(profile, paragraph_count=1)
    assert "LANGUAGE: The user writes in Indonesian" in prompt

@pytest.mark.asyncio
@patch("app.core.writing_engine.Agent")
async def test_apply_style(mock_agent_class):
    # Mock the Agent instance and its run method
    mock_agent_instance = MagicMock()
    mock_agent_class.return_value = mock_agent_instance
    
    mock_result = MagicMock()
    mock_result.output.final_text = "Rewritten text"
    mock_result.output.voice_match_score = 90.0
    mock_result.output.changes_made = ["Made it better"]
    
    async def async_run(*args, **kwargs):
        return mock_result
        
    mock_agent_instance.run.side_effect = async_run
    
    profile = StyleProfile(user_id="test3")
    result = await apply_style("Original draft", profile)
    
    assert result.final_text == "Rewritten text"
    assert result.voice_match_score == 90.0
