# Video Safety Analyzer

Analyze videos for content safety with transcription, toxicity scoring, and visual content moderation using OpenAI APIs.

## Features

- 🎙️ **Transcription** - Speech-to-text using OpenAI Whisper API
- ☢️ **Toxicity Detection** - Detect harmful language using OpenAI Moderation API
- 👁️ **Visual Moderation** - Detect inappropriate content using GPT-4 Vision
- 📊 **Safety Reports** - Comprehensive reports with severity ratings and recommendations

## Requirements

- Python 3.11+
- FFmpeg installed on system
- OpenAI API key

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/mate.git
cd mate

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

## Quick Start

```bash
# Start the API server
uvicorn src.api.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/upload` | Upload a video |
| POST | `/analyze/{job_id}` | Run safety analysis |
| POST | `/quick-check/{job_id}` | Quick visual screening |
| GET | `/status/{job_id}` | Get job status |
| GET | `/report/{job_id}` | Get safety report |
| GET | `/transcript/{job_id}` | Get transcription |
| GET | `/audio/{job_id}` | Download audio |
| DELETE | `/job/{job_id}` | Delete job |

## Usage Example

```bash
# 1. Upload a video
curl -X POST "http://localhost:8000/upload" \
  -F "file=@video.mp4"

# Response: {"job_id": "abc-123", "status": "uploaded", ...}

# 2. Run safety analysis
curl -X POST "http://localhost:8000/analyze/abc-123"

# 3. Get the report
curl "http://localhost:8000/report/abc-123"
```

## Safety Report Structure

```json
{
  "transcription": {
    "full_text": "...",
    "segments": [...],
    "language": "english"
  },
  "moderation": {
    "summary": {
      "max_score": 0.12,
      "flagged_count": 0,
      "overall_severity": "safe"
    },
    "flagged_segments": []
  },
  "visual_moderation": {
    "summary": {
      "total_frames_analyzed": 10,
      "flagged_frames": 0,
      "min_safety_score": 0.95
    },
    "flagged_frames": []
  },
  "overall_assessment": {
    "safety_score": 0.88,
    "verdict": "SAFE",
    "flags": [],
    "recommendations": []
  }
}
```

## Verdicts

| Verdict | Safety Score | Description |
|---------|--------------|-------------|
| SAFE | ≥80% | Safe for general audiences |
| CAUTION | 60-80% | May not suit all audiences |
| RESTRICTED | 40-60% | Should be age-restricted |
| UNSAFE | 20-40% | Violates safety guidelines |
| BLOCKED | <20% | Severely violates guidelines |

## Configuration

Environment variables in `.env`:

```
OPENAI_API_KEY=sk-...          # Required
MAX_VIDEO_DURATION_MINUTES=60  # Optional (default: 60)
```

## Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
pytest tests/ -v
```

## License

MIT
