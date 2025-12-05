# Mate

Video upload and audio extraction API.

## Features

- 📹 Upload video files (MP4, MOV, AVI, MKV, WebM)
- 🎵 Automatic audio extraction using FFmpeg
- ⚡ Background processing
- 📁 Download extracted audio (WAV or MP3)

## Requirements

- Python 3.11+
- FFmpeg installed on system
- Poetry for dependency management

## Installation

```bash
# Clone the repo
git clone https://github.com/GonnaFlyMethod/mate.git
cd mate

# Install dependencies
poetry install

# Copy environment file
cp .env.example .env
```

## Running

```bash
# Start the server
poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Or from the src directory:
```bash
cd src
poetry run uvicorn api.main:app --reload
```

## API Endpoints

### Health Check
```
GET /health
```

### Upload Video
```
POST /upload
Content-Type: multipart/form-data

file: <video file>
extract_audio_flag: true (optional, default: true)
```

Response:
```json
{
  "job_id": "uuid",
  "status": "processing",
  "duration_seconds": 120.5,
  "message": "Video uploaded successfully"
}
```

### Check Status
```
GET /status/{job_id}
```

Response:
```json
{
  "status": "completed",
  "video_path": "...",
  "audio_path": "...",
  "duration_seconds": 120.5,
  "error": null
}
```

### Download Audio
```
GET /audio/{job_id}
```

Returns the audio file (WAV for videos ≤10min, MP3 for longer videos).

### Delete Job
```
DELETE /job/{job_id}
```

## Configuration

Environment variables (`.env`):

```
MAX_VIDEO_DURATION_MINUTES=60
```

## License

MIT

