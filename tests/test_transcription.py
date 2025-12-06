"""
Tests for the transcription module using OpenAI Whisper API.
"""
import os
import pytest
import tempfile
import subprocess

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)

from src.processing.transcription import (
    transcribe_audio,
    transcribe_with_timestamps,
    TranscriptionResult,
    TranscriptionSegment
)


@pytest.fixture
def sample_audio_file():
    """Create a simple test audio file using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = f.name
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "sine=frequency=440:duration=2",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        yield output_path
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


class TestTranscription:
    def test_transcribe_audio(self, sample_audio_file):
        """Should transcribe audio file."""
        result = transcribe_audio(sample_audio_file)
        assert isinstance(result, TranscriptionResult)
        assert isinstance(result.text, str)
        assert isinstance(result.segments, list)
    
    def test_transcribe_nonexistent_file(self):
        """Should raise error for missing file."""
        with pytest.raises(FileNotFoundError):
            transcribe_audio("/nonexistent/path/audio.wav")
    
    def test_transcribe_with_timestamps(self, sample_audio_file):
        """Should return segments with timestamps."""
        result = transcribe_with_timestamps(sample_audio_file)
        assert isinstance(result, list)


class TestTranscriptionResult:
    def test_result_attributes(self):
        """TranscriptionResult should have expected attributes."""
        segments = [
            TranscriptionSegment(start=0.0, end=5.0, text="Hello world")
        ]
        
        result = TranscriptionResult(
            text="Hello world",
            segments=segments,
            language="en",
            duration=5.0
        )
        
        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert result.language == "en"
        assert result.duration == 5.0
