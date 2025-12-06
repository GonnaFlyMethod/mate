"""
Video Safety Analyzer API

Analyze videos for content safety using OpenAI APIs:
- Whisper API for transcription
- Moderation API for text toxicity
- Vision API for visual content moderation
"""
import os
import uuid
import shutil
import tempfile
import asyncio
from pathlib import Path
from functools import partial
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Import processing modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from processing.audio import extract_audio, get_video_duration_seconds
from processing.safety_analyzer import analyze_video_safety, get_safety_summary
from processing.visual_moderation import quick_visual_check

# Configuration
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploaded_videos"
UPLOAD_DIR.mkdir(exist_ok=True)

AUDIO_DIR = Path(__file__).parent.parent.parent / "extracted_audio"
AUDIO_DIR.mkdir(exist_ok=True)

MAX_VIDEO_DURATION_MINUTES = int(os.getenv("MAX_VIDEO_DURATION_MINUTES", "60"))

# FastAPI app
app = FastAPI(
    title="Video Safety Analyzer",
    description="""
Analyze videos for content safety:

- **Transcription**: Speech-to-text using OpenAI Whisper API
- **Text Moderation**: Detect harmful language using OpenAI Moderation API  
- **Visual Moderation**: Detect inappropriate content using GPT-4 Vision
    """,
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (replace with database for production)
jobs: dict = {}


class SafetyAnalysisConfig(BaseModel):
    """Configuration for safety analysis."""
    visual_fps: float = 0.2
    max_visual_frames: int = 10
    moderation_threshold: float = 0.5
    visual_threshold: float = 0.5
    skip_transcription: bool = False
    skip_visual: bool = False


@app.get("/")
async def root():
    """API information and available endpoints."""
    return {
        "name": "Video Safety Analyzer",
        "version": "1.0.0",
        "description": "Analyze videos for transcription, toxicity, and visual content moderation",
        "endpoints": {
            "POST /upload": "Upload a video for processing",
            "POST /analyze/{job_id}": "Run safety analysis on uploaded video",
            "POST /quick-check/{job_id}": "Quick visual safety pre-screening",
            "GET /status/{job_id}": "Get job status and results",
            "GET /report/{job_id}": "Get safety report (json or text format)",
            "GET /transcript/{job_id}": "Get transcription only",
            "GET /audio/{job_id}": "Download extracted audio",
            "DELETE /job/{job_id}": "Delete job and associated files"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    extract_audio_flag: bool = True,
    run_safety_analysis: bool = False
):
    """
    Upload a video file for processing.
    
    - Validates video duration (max 60 minutes by default)
    - Saves video to disk
    - Optionally extracts audio in background
    - Optionally runs full safety analysis
    
    Returns a job_id to track processing status.
    """
    # Validate file type
    valid_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Supported: {', '.join(valid_extensions)}"
        )
    
    # Save to temp file first (for validation)
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        while chunk := await file.read(1024 * 1024):
            tmp.write(chunk)
    
    # Check video duration
    try:
        loop = asyncio.get_running_loop()
        duration_seconds = await loop.run_in_executor(
            None, partial(get_video_duration_seconds, temp_path)
        )
        duration_minutes = duration_seconds / 60
        
        if duration_minutes > MAX_VIDEO_DURATION_MINUTES:
            os.remove(temp_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Video too long. Max duration: {MAX_VIDEO_DURATION_MINUTES} minutes"
            )
    except Exception as e:
        os.remove(temp_path)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Could not read video: {str(e)}")
    
    # Move to final location
    job_id = str(uuid.uuid4())
    video_filename = f"{job_id}{suffix}"
    video_path = UPLOAD_DIR / video_filename
    shutil.move(temp_path, video_path)
    
    # Create job record
    jobs[job_id] = {
        "status": "uploaded",
        "video_path": str(video_path),
        "video_filename": video_filename,
        "duration_seconds": duration_seconds,
        "audio_path": None,
        "safety_report": None,
        "error": None
    }
    
    # Extract audio in background if requested
    if extract_audio_flag:
        jobs[job_id]["status"] = "processing_audio"
        background_tasks.add_task(process_audio_extraction, job_id, str(video_path))
    
    # Run safety analysis if requested
    if run_safety_analysis:
        jobs[job_id]["status"] = "analyzing_safety"
        background_tasks.add_task(
            process_safety_analysis, 
            job_id, 
            str(video_path),
            SafetyAnalysisConfig()
        )
    
    return {
        "job_id": job_id,
        "status": jobs[job_id]["status"],
        "duration_seconds": duration_seconds,
        "message": "Video uploaded successfully"
    }


async def process_audio_extraction(job_id: str, video_path: str):
    """Background task to extract audio from video."""
    try:
        duration = jobs[job_id]["duration_seconds"]
        audio_format = "mp3" if duration > 600 else "wav"
        
        output_path = str(AUDIO_DIR / job_id)
        loop = asyncio.get_running_loop()
        audio_path = await loop.run_in_executor(
            None, partial(extract_audio, video_path, output_path, audio_format)
        )
        
        jobs[job_id]["audio_path"] = audio_path
        
        if jobs[job_id]["status"] == "processing_audio":
            jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


async def process_safety_analysis(
    job_id: str, 
    video_path: str,
    config: SafetyAnalysisConfig
):
    """Background task to run safety analysis."""
    try:
        jobs[job_id]["status"] = "analyzing_safety"
        
        audio_path = jobs[job_id].get("audio_path")
        
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            None,
            partial(
                analyze_video_safety,
                video_path,
                audio_path=audio_path,
                visual_fps=config.visual_fps,
                max_visual_frames=config.max_visual_frames,
                moderation_threshold=config.moderation_threshold,
                visual_threshold=config.visual_threshold,
                skip_transcription=config.skip_transcription,
                skip_visual=config.skip_visual
            )
        )
        
        jobs[job_id]["safety_report"] = report.to_dict()
        jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/analyze/{job_id}")
async def analyze_video(
    job_id: str,
    background_tasks: BackgroundTasks,
    config: SafetyAnalysisConfig = SafetyAnalysisConfig()
):
    """
    Run safety analysis on an uploaded video.
    
    Performs:
    - Speech-to-text transcription (Whisper API)
    - Toxicity scoring on transcribed text (Moderation API)
    - Visual content moderation on video frames (Vision API)
    
    Results are stored with the job and retrieved via GET /status/{job_id}
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] == "analyzing_safety":
        raise HTTPException(status_code=400, detail="Analysis already in progress")
    
    video_path = job["video_path"]
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    background_tasks.add_task(process_safety_analysis, job_id, video_path, config)
    
    return {
        "job_id": job_id,
        "status": "analyzing_safety",
        "message": "Safety analysis started"
    }


@app.post("/quick-check/{job_id}")
async def quick_check(
    job_id: str, 
    sample_frames: int = Query(default=5, ge=1, le=20)
):
    """
    Quick visual safety pre-screening.
    
    Samples a few frames for obvious issues before running full analysis.
    Good for filtering obviously problematic content.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    video_path = job["video_path"]
    
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        partial(quick_visual_check, video_path, sample_frames)
    )
    
    return {"job_id": job_id, **result}


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job, including safety report if available."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@app.get("/report/{job_id}")
async def get_safety_report(
    job_id: str, 
    format: str = Query(default="json", pattern="^(json|text)$")
):
    """
    Get the safety report for a video.
    
    - format=json: Returns structured JSON report
    - format=text: Returns human-readable text summary
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if not job.get("safety_report"):
        raise HTTPException(
            status_code=400, 
            detail="Safety report not available. Run analysis first via POST /analyze/{job_id}"
        )
    
    if format == "text":
        from processing.safety_analyzer import SafetyReport
        report_data = job["safety_report"]
        
        class ReportProxy:
            pass
        
        proxy = ReportProxy()
        proxy.analyzed_at = report_data.get("analyzed_at", "")
        proxy.duration_seconds = report_data.get("duration_seconds", 0)
        proxy.overall_verdict = report_data["overall_assessment"]["verdict"]
        proxy.overall_safety_score = report_data["overall_assessment"]["safety_score"]
        proxy.flags = report_data["overall_assessment"]["flags"]
        proxy.recommendations = report_data["overall_assessment"]["recommendations"]
        proxy.detected_language = report_data["transcription"]["language"]
        proxy.moderation_summary = report_data["moderation"]["summary"]
        proxy.visual_summary = report_data["visual_moderation"]["summary"]
        
        summary = get_safety_summary(proxy)
        return {"job_id": job_id, "report": summary}
    
    return {"job_id": job_id, "report": job["safety_report"]}


@app.get("/transcript/{job_id}")
async def get_transcript(job_id: str):
    """Get just the transcription from a safety report."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if not job.get("safety_report"):
        raise HTTPException(
            status_code=400,
            detail="Transcript not available. Run analysis first via POST /analyze/{job_id}"
        )
    
    transcription = job["safety_report"].get("transcription", {})
    
    return {
        "job_id": job_id,
        "text": transcription.get("full_text", ""),
        "language": transcription.get("language", "unknown"),
        "segments": transcription.get("segments", [])
    }


@app.get("/audio/{job_id}")
async def download_audio(job_id: str):
    """Download the extracted audio file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if not job["audio_path"] or not os.path.exists(job["audio_path"]):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    media_type = "audio/wav" if job["audio_path"].endswith(".wav") else "audio/mpeg"
    ext = "wav" if job["audio_path"].endswith(".wav") else "mp3"
    
    return FileResponse(
        job["audio_path"],
        media_type=media_type,
        filename=f"{job_id}.{ext}"
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Delete video file
    if job["video_path"] and os.path.exists(job["video_path"]):
        os.remove(job["video_path"])
    
    # Delete audio file
    if job["audio_path"] and os.path.exists(job["audio_path"]):
        os.remove(job["audio_path"])
    
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}
