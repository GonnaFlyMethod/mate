"""
Tests for the moderation module using OpenAI Moderation API.
"""
import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)

from src.processing.toxicity import (
    analyze_toxicity,
    analyze_segments,
    aggregate_moderation,
    ModerationScore
)


class TestModerationScoring:
    def test_safe_text(self):
        """Safe text should get low scores."""
        result = analyze_toxicity("Hello, how are you today?")
        assert result.max_score < 0.3
        assert result.flagged is False
        assert result.severity == "safe"
    
    def test_empty_text(self):
        """Empty text should return zero scores."""
        result = analyze_toxicity("")
        assert result.max_score == 0.0
        assert result.flagged is False
    
    def test_whitespace_only(self):
        """Whitespace-only text should return zero scores."""
        result = analyze_toxicity("   \n\t  ")
        assert result.max_score == 0.0
    
    def test_to_dict(self):
        """Result should convert to dictionary."""
        result = analyze_toxicity("Test message")
        d = result.to_dict()
        assert "hate" in d
        assert "harassment" in d
        assert "max_score" in d
        assert "flagged" in d
        assert "severity" in d


class TestSegmentAnalysis:
    def test_analyze_multiple_segments(self):
        """Should analyze multiple text segments."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello everyone"},
            {"start": 5.0, "end": 10.0, "text": "Welcome to the show"},
        ]
        
        results = analyze_segments(segments)
        assert len(results) == 2
        assert results[0].start == 0.0
    
    def test_aggregate_moderation_safe(self):
        """Safe segments should aggregate to safe summary."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello"},
            {"start": 5.0, "end": 10.0, "text": "Goodbye"},
        ]
        
        analyzed = analyze_segments(segments)
        summary = aggregate_moderation(analyzed)
        
        assert summary["max_score"] < 0.5
        assert summary["overall_severity"] == "safe"
    
    def test_aggregate_empty(self):
        """Empty list should return safe defaults."""
        summary = aggregate_moderation([])
        assert summary["max_score"] == 0.0
        assert summary["total_segments"] == 0
