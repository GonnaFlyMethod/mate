"""
Mate API - Video upload and audio extraction service.
"""
import os
import uuid
import shutil
import tempfile
import asyncio
from pathlib import Path
from functools import partial
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Import processing modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from processing.audio import extract_audio, get_video_duration_seconds

# Configuration
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploaded_videos"
UPLOAD_DIR.mkdir(exist_ok=True)

AUDIO_DIR = Path(__file__).parent.parent.parent / "extracted_audio"
AUDIO_DIR.mkdir(exist_ok=True)

MAX_VIDEO_DURATION_MINUTES = int(os.getenv("MAX_VIDEO_DURATION_MINUTES", "60"))

# FastAPI app
app = FastAPI(
    title="Mate API",
    description="Upload videos and extract audio",
    version="0.1.0"
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    extract_audio_flag: bool = True
):
    """
    Upload a video file.
    
    - Validates video duration
    - Saves video to disk
    - Optionally extracts audio in background
    
    Returns a job_id to track processing status.
    """
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: mp4, mov, avi, mkv, webm")
    
    # Save to temp file first (for validation)
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            tmp.write(chunk)
    
    # Check video duration (run in executor to not block)
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
        "error": None
    }
    
    # Extract audio in background if requested
    if extract_audio_flag:
        jobs[job_id]["status"] = "processing"
        background_tasks.add_task(process_audio_extraction, job_id, str(video_path))
    
    return {
        "job_id": job_id,
        "status": jobs[job_id]["status"],
        "duration_seconds": duration_seconds,
        "message": "Video uploaded successfully"
    }


async def process_audio_extraction(job_id: str, video_path: str):
    """Background task to extract audio from video."""
    try:
        # Determine format based on duration
        duration = jobs[job_id]["duration_seconds"]
        audio_format = "mp3" if duration > 600 else "wav"  # MP3 for videos > 10 min
        
        # Extract audio
        output_path = str(AUDIO_DIR / job_id)
        loop = asyncio.get_running_loop()
        audio_path = await loop.run_in_executor(
            None, partial(extract_audio, video_path, output_path, audio_format)
        )
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["audio_path"] = audio_path
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@app.get("/audio/{job_id}")
async def download_audio(job_id: str):
    """Download the extracted audio file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Audio not ready. Status: {job['status']}")
    
    if not job["audio_path"] or not os.path.exists(job["audio_path"]):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        job["audio_path"],
        media_type="audio/wav" if job["audio_path"].endswith(".wav") else "audio/mpeg",
        filename=f"{job_id}.{'wav' if job['audio_path'].endswith('.wav') else 'mp3'}"
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

