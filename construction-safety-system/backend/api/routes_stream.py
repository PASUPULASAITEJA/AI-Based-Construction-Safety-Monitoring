"""
REST API: Video Upload & Live Stream Session Management
Handles video file uploads (MP4/AVI), stream source switching (Webcam / File / RTSP),
and MJPEG live stream encoding.
"""
from typing import Optional
import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import StreamingResponse
import cv2

import config
from core.pipeline import SafetyVisionPipeline

router = APIRouter(prefix="/api/stream", tags=["Video Stream Management"])

# Global stream state
class StreamManager:
    def __init__(self):
        self.active_source: str = "0"  # Default webcam device
        self.is_running: bool = False
        self.pipeline = SafetyVisionPipeline()
        self.current_filename: Optional[str] = None

stream_mgr = StreamManager()

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a recorded video file (MP4/AVI) for safety pipeline processing."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported video format. Allowed formats: MP4, AVI, MOV, MKV, WEBM."
        )

    file_id = uuid.uuid4().hex[:10]
    dest_filename = f"video_{file_id}{ext}"
    dest_path = os.path.join(config.UPLOADS_DIR, dest_filename)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    stream_mgr.active_source = dest_path
    stream_mgr.current_filename = file.filename

    return {
        "status": "SUCCESS",
        "file_id": file_id,
        "filename": file.filename,
        "source_path": dest_path,
        "message": "Video successfully uploaded and set as active stream source."
    }

@router.post("/source")
def set_stream_source(source_type: str = Form(...), source_url: Optional[str] = Form(None)):
    """Configures the active video stream source: 'webcam', 'demo', or custom RTSP/HTTP URL."""
    if source_type == "webcam":
        stream_mgr.active_source = "0"
        stream_mgr.current_filename = "Live Hardware Webcam"
    elif source_type == "rtsp" and source_url:
        stream_mgr.active_source = source_url
        stream_mgr.current_filename = f"RTSP: {source_url}"
    elif source_type == "demo":
        stream_mgr.active_source = "demo"
        stream_mgr.current_filename = "Synthetic Test Simulator"
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid source parameters.")

    return {"status": "SUCCESS", "active_source": stream_mgr.active_source, "source_label": stream_mgr.current_filename}

@router.get("/status")
def get_stream_status():
    """Returns active video stream details and pipeline FPS."""
    return {
        "active_source": stream_mgr.current_filename or "Default Camera",
        "fps": round(stream_mgr.pipeline.fps, 1),
        "is_custom_model": stream_mgr.pipeline.detector.is_custom_model,
        "model_name": stream_mgr.pipeline.detector.model_name
    }
