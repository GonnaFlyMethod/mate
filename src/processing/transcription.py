"""
Speech-to-text transcription using OpenAI Whisper API.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


@dataclass
class TranscriptionSegment:
    """A single segment of transcribed text with timing."""
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    segments: list[TranscriptionSegment]
    language: str
    duration: float


def get_openai_client() -> OpenAI:
    """Get OpenAI client using API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def transcribe_audio(
    audio_path: str,
    language: str | None = None
) -> TranscriptionResult:
    """
    Transcribe audio file to text using OpenAI Whisper API.
    
    Args:
        audio_path: Path to audio file (wav, mp3, etc.)
        language: Optional language code (e.g., "en", "es"). Auto-detected if None.
    
    Returns:
        TranscriptionResult with full text, segments, and metadata
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    client = get_openai_client()
    
    # Get transcription with timestamps
    with open(audio_path, "rb") as audio_file:
        kwargs = {
            "model": "whisper-1",
            "file": audio_file,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"]
        }
        if language:
            kwargs["language"] = language
            
        response = client.audio.transcriptions.create(**kwargs)
    
    # Parse segments
    segments = []
    if hasattr(response, 'segments') and response.segments:
        for seg in response.segments:
            # Handle both dict and Pydantic model responses
            if hasattr(seg, 'start'):
                # Pydantic model
                segments.append(TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip() if seg.text else ""
                ))
            else:
                # Dict response
                segments.append(TranscriptionSegment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", "").strip()
                ))
    
    # Calculate duration from last segment or response
    duration = 0.0
    if segments:
        duration = segments[-1].end
    elif hasattr(response, 'duration'):
        duration = response.duration
    
    return TranscriptionResult(
        text=response.text.strip() if response.text else "",
        segments=segments,
        language=response.language if hasattr(response, 'language') else "unknown",
        duration=duration
    )


def transcribe_with_timestamps(audio_path: str) -> list[dict]:
    """
    Transcribe audio and return segments with timestamps.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        List of dicts with 'start', 'end', 'text' keys
    """
    result = transcribe_audio(audio_path)
    return [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        }
        for seg in result.segments
    ]
