"""
Audio extraction from video files using ffmpeg.
"""
import subprocess
import os


def extract_audio(video_path: str, output_path: str, format: str = "wav") -> str:
    """
    Extract audio from video file.
    
    Args:
        video_path: Path to input video file
        output_path: Path for output audio file (without extension)
        format: Output format - "wav" (best quality) or "mp3" (smaller size)
    
    Returns:
        Path to the extracted audio file
    """
    if format == "mp3":
        audio_path = f"{output_path}.mp3"
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",          # No video
            "-ac", "1",     # Mono
            "-ar", "16000", # 16kHz sample rate
            "-b:a", "32k",  # 32kbps bitrate (small file)
            audio_path
        ]
    else:  # wav (default)
        audio_path = f"{output_path}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",          # No video
            "-ac", "1",     # Mono
            "-ar", "16000", # 16kHz sample rate
            "-f", "wav",
            audio_path
        ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def get_video_duration_seconds(video_path: str) -> float:
    """
    Get video duration in seconds using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

