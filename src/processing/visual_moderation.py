"""
Visual content moderation for video frames using OpenAI Vision API.

Analyzes frames for:
- NSFW content
- Violence
- Other inappropriate visual content
"""
import os
import subprocess
import tempfile
import base64
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


@dataclass
class FrameAnalysis:
    """Analysis result for a single video frame."""
    timestamp: float
    frame_number: int
    is_safe: bool
    safety_score: float  # 0.0 (unsafe) to 1.0 (safe)
    concerns: list[str]
    description: str
    
    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 2),
            "frame_number": self.frame_number,
            "is_safe": self.is_safe,
            "safety_score": round(self.safety_score, 4),
            "concerns": self.concerns,
            "description": self.description
        }


@dataclass
class VisualModerationResult:
    """Complete visual moderation result for a video."""
    total_frames_analyzed: int
    flagged_frames: int
    min_safety_score: float
    avg_safety_score: float
    flagged_timestamps: list[float]
    frame_analyses: list[FrameAnalysis]
    overall_safety: str
    
    def to_dict(self) -> dict:
        return {
            "total_frames_analyzed": self.total_frames_analyzed,
            "flagged_frames": self.flagged_frames,
            "min_safety_score": round(self.min_safety_score, 4),
            "avg_safety_score": round(self.avg_safety_score, 4),
            "flagged_timestamps": [round(t, 2) for t in self.flagged_timestamps],
            "flagged_percentage": round(
                self.flagged_frames / self.total_frames_analyzed * 100, 2
            ) if self.total_frames_analyzed > 0 else 0,
            "overall_safety": self.overall_safety,
            "frame_analyses": [f.to_dict() for f in self.frame_analyses]
        }


def get_openai_client() -> OpenAI:
    """Get OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 0.5,
    max_frames: int = 10
) -> list[tuple[str, float]]:
    """
    Extract frames from video at specified FPS.
    
    Returns:
        List of tuples (frame_path, timestamp)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        output_pattern
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    frames = []
    for i, frame_file in enumerate(sorted(Path(output_dir).glob("frame_*.jpg"))):
        timestamp = i / fps
        frames.append((str(frame_file), timestamp))
        if len(frames) >= max_frames:
            break
    
    return frames


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_frame(image_path: str) -> dict:
    """
    Analyze a single frame using OpenAI Vision API.
    
    Returns dict with is_safe, safety_score, concerns, description
    """
    client = get_openai_client()
    
    base64_image = encode_image_base64(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a content safety analyzer. Analyze images for inappropriate content.
                
Respond in this exact JSON format:
{
    "is_safe": true/false,
    "safety_score": 0.0-1.0 (1.0 = completely safe),
    "concerns": ["list", "of", "concerns"] or [],
    "description": "brief neutral description"
}

Flag concerns for: nudity, sexual content, violence, gore, weapons, drugs, hate symbols, self-harm.
Be strict but fair - normal everyday content should score 0.9-1.0."""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Analyze this frame for content safety. Respond with JSON only."
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    # Parse response
    content = response.choices[0].message.content.strip()
    
    # Try to extract JSON
    import json
    try:
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content)
        return {
            "is_safe": result.get("is_safe", True),
            "safety_score": float(result.get("safety_score", 1.0)),
            "concerns": result.get("concerns", []),
            "description": result.get("description", "")
        }
    except (json.JSONDecodeError, KeyError):
        # Default to safe if parsing fails
        return {
            "is_safe": True,
            "safety_score": 0.8,
            "concerns": [],
            "description": "Unable to analyze"
        }


def analyze_video_frames(
    video_path: str,
    fps: float = 0.2,
    max_frames: int = 10,
    threshold: float = 0.5
) -> VisualModerationResult:
    """
    Analyze video for inappropriate visual content.
    
    Args:
        video_path: Path to video file
        fps: Frames per second to analyze (lower = fewer API calls)
        max_frames: Maximum frames to analyze
        threshold: Safety score threshold for flagging (below = flagged)
    
    Returns:
        VisualModerationResult with complete analysis
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        frames = extract_frames(video_path, temp_dir, fps, max_frames)
        
        if not frames:
            return VisualModerationResult(
                total_frames_analyzed=0,
                flagged_frames=0,
                min_safety_score=1.0,
                avg_safety_score=1.0,
                flagged_timestamps=[],
                frame_analyses=[],
                overall_safety="safe"
            )
        
        analyses = []
        safety_scores = []
        flagged_timestamps = []
        
        for i, (frame_path, timestamp) in enumerate(frames):
            result = analyze_frame(frame_path)
            
            is_flagged = result["safety_score"] < threshold
            
            analysis = FrameAnalysis(
                timestamp=timestamp,
                frame_number=i,
                is_safe=result["is_safe"],
                safety_score=result["safety_score"],
                concerns=result["concerns"],
                description=result["description"]
            )
            analyses.append(analysis)
            safety_scores.append(result["safety_score"])
            
            if is_flagged:
                flagged_timestamps.append(timestamp)
        
        min_safety = min(safety_scores)
        avg_safety = sum(safety_scores) / len(safety_scores)
        
        # Determine overall safety
        if min_safety >= 0.8:
            overall_safety = "safe"
        elif min_safety >= 0.6:
            overall_safety = "mild_concern"
        elif min_safety >= 0.4:
            overall_safety = "moderate_concern"
        elif min_safety >= 0.2:
            overall_safety = "high_concern"
        else:
            overall_safety = "severe_concern"
        
        return VisualModerationResult(
            total_frames_analyzed=len(analyses),
            flagged_frames=len(flagged_timestamps),
            min_safety_score=min_safety,
            avg_safety_score=avg_safety,
            flagged_timestamps=flagged_timestamps,
            frame_analyses=analyses,
            overall_safety=overall_safety
        )


def quick_visual_check(
    video_path: str,
    sample_frames: int = 3
) -> dict:
    """
    Quick visual safety check with minimal frames.
    
    Args:
        video_path: Path to video
        sample_frames: Number of frames to sample (default 3 for cost)
    
    Returns:
        Quick check result dict
    """
    # Get video duration
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())
    
    fps = sample_frames / duration if duration > 0 else 1.0
    
    result = analyze_video_frames(
        video_path,
        fps=fps,
        max_frames=sample_frames,
        threshold=0.5
    )
    
    return {
        "is_safe": result.flagged_frames == 0,
        "min_safety_score": round(result.min_safety_score, 4),
        "flagged_frames": result.flagged_frames,
        "samples_checked": result.total_frames_analyzed,
        "overall_safety": result.overall_safety
    }
