"""
Processing modules for video safety analysis using OpenAI APIs.
"""
from .audio import extract_audio, get_video_duration_seconds
from .transcription import transcribe_audio, TranscriptionResult, TranscriptionSegment
from .toxicity import (
    analyze_toxicity, analyze_segments, aggregate_moderation,
    ModerationScore, SegmentModeration
)
from .visual_moderation import (
    analyze_video_frames, quick_visual_check,
    FrameAnalysis, VisualModerationResult
)
from .safety_analyzer import (
    analyze_video_safety, get_safety_summary,
    SafetyReport
)

__all__ = [
    # Audio
    "extract_audio",
    "get_video_duration_seconds",
    # Transcription
    "transcribe_audio",
    "TranscriptionResult",
    "TranscriptionSegment",
    # Toxicity/Moderation
    "analyze_toxicity",
    "analyze_segments",
    "aggregate_moderation",
    "ModerationScore",
    "SegmentModeration",
    # Visual moderation
    "analyze_video_frames",
    "quick_visual_check",
    "FrameAnalysis",
    "VisualModerationResult",
    # Safety analyzer
    "analyze_video_safety",
    "get_safety_summary",
    "SafetyReport",
]
