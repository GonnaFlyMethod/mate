"""
Tests for the FastAPI endpoints.
"""
import os
import pytest
import tempfile
import subprocess
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_video_file():
    """Create a simple test video file using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        output_path = f.name
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=black:s=320x240:d=2",
        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", "2",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        yield output_path
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestRootEndpoint:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "endpoints" in data


class TestUploadEndpoint:
    def test_upload_video(self, client, sample_video_file):
        with open(sample_video_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("test.mp4", f, "video/mp4")},
                data={"extract_audio_flag": "false"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] in ["uploaded", "processing_audio"]
        
        # Cleanup
        client.delete(f"/job/{data['job_id']}")
    
    def test_upload_invalid_file_type(self, client):
        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"not a video", "text/plain")}
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]


class TestStatusEndpoint:
    def test_status_not_found(self, client):
        response = client.get("/status/nonexistent-job-id")
        assert response.status_code == 404
    
    def test_status_after_upload(self, client, sample_video_file):
        with open(sample_video_file, "rb") as f:
            upload_response = client.post(
                "/upload",
                files={"file": ("test.mp4", f, "video/mp4")},
                data={"extract_audio_flag": "false"}
            )
        
        job_id = upload_response.json()["job_id"]
        response = client.get(f"/status/{job_id}")
        
        assert response.status_code == 200
        client.delete(f"/job/{job_id}")


class TestDeleteEndpoint:
    def test_delete_not_found(self, client):
        response = client.delete("/job/nonexistent-job-id")
        assert response.status_code == 404
    
    def test_delete_job(self, client, sample_video_file):
        with open(sample_video_file, "rb") as f:
            upload_response = client.post(
                "/upload",
                files={"file": ("test.mp4", f, "video/mp4")},
                data={"extract_audio_flag": "false"}
            )
        
        job_id = upload_response.json()["job_id"]
        response = client.delete(f"/job/{job_id}")
        
        assert response.status_code == 200
        assert client.get(f"/status/{job_id}").status_code == 404


class TestReportEndpoint:
    def test_report_no_analysis(self, client, sample_video_file):
        with open(sample_video_file, "rb") as f:
            upload_response = client.post(
                "/upload",
                files={"file": ("test.mp4", f, "video/mp4")},
                data={"extract_audio_flag": "false"}
            )
        
        job_id = upload_response.json()["job_id"]
        response = client.get(f"/report/{job_id}")
        
        assert response.status_code == 400
        assert "not available" in response.json()["detail"]
        
        client.delete(f"/job/{job_id}")
