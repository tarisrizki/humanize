import pytest
from unittest.mock import patch, MagicMock
from app.core.pipeline import apply_style
from app.core.prompts import _build_system_prompt
from app.models.style_profile import StyleProfile

def test_build_system_prompt_en():
    profile = StyleProfile(
        user_id="test1",
        language="en",
        avg_sentence_length=15.5,
        emotion_register="formal"
    )
    prompt = _build_system_prompt(profile, paragraph_count=2)
    assert "Write in natural, idiomatic English only" in prompt

def test_build_system_prompt_id():
    profile = StyleProfile(
        user_id="test2",
        language="id"
    )
    prompt = _build_system_prompt(profile, paragraph_count=1)
    assert "Tulis HANYA dalam Bahasa Indonesia" in prompt

@pytest.mark.asyncio
@patch("app.core.pipeline.Agent")
async def test_apply_style(mock_agent_class):
    # Mock the Agent instance and its run method
    mock_agent_instance = MagicMock()
    mock_agent_class.return_value = mock_agent_instance
    
    mock_result = MagicMock()
    mock_result.output = "Rewritten text"
    
    async def async_run(*args, **kwargs):
        return mock_result
        
    mock_agent_instance.run.side_effect = async_run
    
    profile = StyleProfile(user_id="test3")
    result = await apply_style("Original draft", profile)
    
    assert result.final_text == "Rewritten text"
    assert len(result.changes_made) > 0
