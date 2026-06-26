import pytest
from app.core.style_analyzer import analyze_style

def test_analyze_style_empty():
    profile = analyze_style([], [])
    assert profile.user_id == "anonymous"
    assert profile.avg_sentence_length == 0.0

def test_analyze_style_en(sample_text_en):
    profile = analyze_style([], [sample_text_en], user_id="test_en")
    assert profile.user_id == "test_en"
    assert profile.avg_sentence_length > 0
    assert "however" in profile.top_transitions

def test_analyze_style_id(sample_text_id):
    profile = analyze_style([sample_text_id], [], user_id="test_id")
    assert profile.user_id == "test_id"
    assert profile.language == "id"
    assert profile.avg_sentence_length > 0
    assert "namun" in profile.top_transitions
