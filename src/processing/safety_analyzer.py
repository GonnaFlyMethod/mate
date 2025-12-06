"""
Video Safety Analyzer - Orchestrates transcription, toxicity scoring, and visual moderation.

Uses OpenAI APIs for all analysis:
- Whisper API for transcription
- Moderation API for text content
- Vision API for visual content
"""
import os
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from .audio import extract_audio
from .transcription import transcribe_audio, TranscriptionResult
from .toxicity import analyze_segments, aggregate_moderation, SegmentModeration
from .visual_moderation import analyze_video_frames, quick_visual_check, VisualModerationResult


@dataclass
class SafetyReport:
    """Complete safety analysis report for a video."""
    video_path: str
    analyzed_at: str
    duration_seconds: float
    
    # Transcription results
    transcript: str
    transcript_segments: list[dict]
    detected_language: str
    
    # Moderation results
    moderation_summary: dict
    flagged_segments: list[dict]
    
    # Visual moderation results
    visual_summary: dict
    flagged_frames: list[dict]
    
    # Overall assessment
    overall_safety_score: float
    overall_verdict: str
    flags: list[str]
    recommendations: list[str]
    
    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "video_path": self.video_path,
            "analyzed_at": self.analyzed_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "transcription": {
                "full_text": self.transcript,
                "segments": self.transcript_segments,
                "language": self.detected_language
            },
            "moderation": {
                "summary": self.moderation_summary,
                "flagged_segments": self.flagged_segments
            },
            "visual_moderation": {
                "summary": self.visual_summary,
                "flagged_frames": self.flagged_frames
            },
            "overall_assessment": {
                "safety_score": round(self.overall_safety_score, 4),
                "verdict": self.overall_verdict,
                "flags": self.flags,
                "recommendations": self.recommendations
            }
        }


def analyze_video_safety(
    video_path: str,
    audio_path: Optional[str] = None,
    visual_fps: float = 0.2,
    max_visual_frames: int = 10,
    moderation_threshold: float = 0.5,
    visual_threshold: float = 0.5,
    skip_transcription: bool = False,
    skip_visual: bool = False
) -> SafetyReport:
    """
    Perform complete safety analysis on a video using OpenAI APIs.
    
    Args:
        video_path: Path to video file
        audio_path: Optional pre-extracted audio path
        visual_fps: Frames per second to analyze
        max_visual_frames: Maximum frames to analyze
        moderation_threshold: Threshold for flagging text content
        visual_threshold: Threshold for flagging visual content
        skip_transcription: Skip audio analysis
        skip_visual: Skip visual analysis
    
    Returns:
        SafetyReport with complete analysis
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    flags = []
    recommendations = []
    
    # === TRANSCRIPTION & MODERATION ===
    transcript = ""
    transcript_segments = []
    detected_language = "unknown"
    moderation_summary = {}
    flagged_segments = []
    duration_seconds = 0.0
    
    if not skip_transcription:
        # Extract audio if not provided
        if audio_path is None or not os.path.exists(audio_path):
            import tempfile
            temp_audio = tempfile.mktemp(suffix=".wav")
            audio_path = extract_audio(video_path, temp_audio.replace(".wav", ""), "wav")
        
        # Transcribe using Whisper API
        transcription_result = transcribe_audio(audio_path)
        transcript = transcription_result.text
        detected_language = transcription_result.language
        duration_seconds = transcription_result.duration
        
        transcript_segments = [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in transcription_result.segments
        ]
        
        # Analyze moderation for each segment
        if transcript_segments:
            segment_moderation = analyze_segments(transcript_segments)
            moderation_summary = aggregate_moderation(segment_moderation)
            
            # Extract flagged segments
            flagged_segments = [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "scores": seg.scores.to_dict()
                }
                for seg in segment_moderation
                if seg.scores.max_score >= moderation_threshold
            ]
            
            # Add flags based on moderation
            max_mod = moderation_summary.get("max_score", 0)
            if max_mod >= 0.9:
                flags.append("EXTREME_CONTENT")
                recommendations.append("Content contains extremely harmful material - requires immediate review")
            elif max_mod >= 0.7:
                flags.append("HIGH_RISK_CONTENT")
                recommendations.append("Content contains high-risk material - consider restrictions")
            elif max_mod >= 0.5:
                flags.append("MODERATE_RISK_CONTENT")
                recommendations.append("Content contains moderate-risk material - age restriction suggested")
    
    # === VISUAL CONTENT MODERATION ===
    visual_summary = {}
    flagged_frames = []
    
    if not skip_visual:
        visual_result = analyze_video_frames(
            video_path,
            fps=visual_fps,
            max_frames=max_visual_frames,
            threshold=visual_threshold
        )
        
        visual_summary = {
            "total_frames_analyzed": visual_result.total_frames_analyzed,
            "flagged_frames": visual_result.flagged_frames,
            "min_safety_score": round(visual_result.min_safety_score, 4),
            "avg_safety_score": round(visual_result.avg_safety_score, 4),
            "overall_safety": visual_result.overall_safety
        }
        
        flagged_frames = [
            frame.to_dict()
            for frame in visual_result.frame_analyses
            if frame.safety_score < visual_threshold
        ]
        
        # Add flags based on visual content
        min_safety = visual_result.min_safety_score
        if min_safety < 0.2:
            flags.append("EXPLICIT_VISUAL_CONTENT")
            recommendations.append("Video contains explicit visual content - requires removal or strong restrictions")
        elif min_safety < 0.4:
            flags.append("NSFW_VISUAL_CONTENT")
            recommendations.append("Video contains NSFW visual content - age gate recommended")
        elif min_safety < 0.6:
            flags.append("QUESTIONABLE_VISUAL_CONTENT")
            recommendations.append("Video contains questionable content - consider age restrictions")
    
    # === CALCULATE OVERALL SAFETY SCORE ===
    text_safety = 1.0 - moderation_summary.get("max_score", 0)
    visual_safety = visual_summary.get("min_safety_score", 1.0)
    
    if skip_transcription:
        overall_safety_score = visual_safety
    elif skip_visual:
        overall_safety_score = text_safety
    else:
        overall_safety_score = min(text_safety, visual_safety)
    
    # Determine verdict
    if overall_safety_score >= 0.8:
        verdict = "SAFE"
    elif overall_safety_score >= 0.6:
        verdict = "CAUTION"
        if not recommendations:
            recommendations.append("Content may not be suitable for all audiences")
    elif overall_safety_score >= 0.4:
        verdict = "RESTRICTED"
        if not recommendations:
            recommendations.append("Content should be age-restricted")
    elif overall_safety_score >= 0.2:
        verdict = "UNSAFE"
        recommendations.append("Content violates safety guidelines - requires review")
    else:
        verdict = "BLOCKED"
        recommendations.append("Content severely violates safety guidelines - should not be published")
    
    return SafetyReport(
        video_path=video_path,
        analyzed_at=datetime.utcnow().isoformat() + "Z",
        duration_seconds=duration_seconds,
        transcript=transcript,
        transcript_segments=transcript_segments,
        detected_language=detected_language,
        moderation_summary=moderation_summary,
        flagged_segments=flagged_segments,
        visual_summary=visual_summary,
        flagged_frames=flagged_frames,
        overall_safety_score=overall_safety_score,
        overall_verdict=verdict,
        flags=flags,
        recommendations=recommendations
    )


def get_safety_summary(report: SafetyReport) -> str:
    """Generate a human-readable safety summary from a report."""
    lines = [
        f"=== VIDEO SAFETY REPORT ===",
        f"Analyzed: {report.analyzed_at}",
        f"Duration: {report.duration_seconds:.1f} seconds",
        f"",
        f"OVERALL VERDICT: {report.overall_verdict}",
        f"Safety Score: {report.overall_safety_score:.1%}",
        f"",
    ]
    
    if report.flags:
        lines.append("FLAGS:")
        for flag in report.flags:
            lines.append(f"  ⚠️  {flag}")
        lines.append("")
    
    if report.moderation_summary:
        lines.append("AUDIO/TEXT ANALYSIS:")
        lines.append(f"  Language: {report.detected_language}")
        lines.append(f"  Max Risk Score: {report.moderation_summary.get('max_score', 0):.1%}")
        lines.append(f"  Flagged Segments: {report.moderation_summary.get('flagged_count', 0)}")
        lines.append("")
    
    if report.visual_summary:
        lines.append("VISUAL ANALYSIS:")
        lines.append(f"  Frames Analyzed: {report.visual_summary.get('total_frames_analyzed', 0)}")
        lines.append(f"  Flagged Frames: {report.visual_summary.get('flagged_frames', 0)}")
        lines.append(f"  Min Safety Score: {report.visual_summary.get('min_safety_score', 1.0):.1%}")
        lines.append("")
    
    if report.recommendations:
        lines.append("RECOMMENDATIONS:")
        for rec in report.recommendations:
            lines.append(f"  → {rec}")
    
    return "\n".join(lines)
