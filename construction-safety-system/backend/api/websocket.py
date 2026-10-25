"""
WebSocket Real-Time Telemetry & Video Stream Broadcaster
Streams live annotated JPEG frames (base64) at 15 FPS alongside real-time
telemetry payloads (active workers, live violations, zone breaches).
Persists confirmed incidents to SQLite database in real-time.
"""
from typing import List, Dict, Any, Set
import asyncio
import base64
import json
import logging
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from database import SessionLocal
from models import RestrictedZone, ViolationIncident
from core.pipeline import SafetyVisionPipeline
from api.routes_stream import stream_mgr
import config

logger = logging.getLogger("SafetyPipeline.WebSocket")
router = APIRouter(tags=["WebSocket Stream"])

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast_json(self, message: Dict[str, Any]):
        dead_connections = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                dead_connections.add(connection)
        for dc in dead_connections:
            self.active_connections.discard(dc)

manager = ConnectionManager()

def get_active_zones() -> List[Dict[str, Any]]:
    """Loads active polygon zones from DB for spatial evaluation."""
    db = SessionLocal()
    try:
        zones = db.query(RestrictedZone).filter(RestrictedZone.is_active == True).all()
        result = []
        for z in zones:
            coords = json.loads(z.coordinates) if isinstance(z.coordinates, str) else z.coordinates
            result.append({
                "id": z.id,
                "name": z.name,
                "severity": z.severity,
                "coordinates": coords
            })
        return result
    finally:
        db.close()

def persist_incident(incident_data: Dict[str, Any]):
    """Persists newly confirmed temporal incident to database."""
    db = SessionLocal()
    try:
        record = ViolationIncident(
            incident_code=incident_data["incident_code"],
            track_id=incident_data["track_id"],
            violation_type=incident_data["violation_type"],
            severity=incident_data["severity"],
            status="ACTIVE",
            zone_id=incident_data.get("zone_id"),
            zone_name=incident_data.get("zone_name"),
            snapshot_path=incident_data.get("snapshot_path"),
            confidence=incident_data.get("confidence", 0.85),
            description=incident_data.get("description")
        )
        db.add(record)
        db.commit()
    except Exception as e:
        logger.error(f"Error persisting incident: {e}")
        db.rollback()
    finally:
        db.close()

def resolve_incident(incident_code: str):
    """Marks latched incident as resolved once worker demonstrates compliance."""
    db = SessionLocal()
    try:
        inc = db.query(ViolationIncident).filter(ViolationIncident.incident_code == incident_code).first()
        if inc and inc.status == "ACTIVE":
            inc.status = "RESOLVED"
            db.commit()
    except Exception as e:
        logger.error(f"Error resolving incident: {e}")
        db.rollback()
    finally:
        db.close()

# Background stream generator
async def stream_broadcaster_loop():
    """Continuous background loop processing frames from active source and broadcasting via WebSocket."""
    logger.info("Initializing Video Stream Broadcaster Loop...")
    pipeline = stream_mgr.pipeline
    cap = None
    last_source = None

    while True:
        # Check active clients
        if len(manager.active_connections) == 0:
            await asyncio.sleep(0.5)
            continue

        # Check source change
        current_source = stream_mgr.active_source
        if current_source != last_source or cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()

            if current_source.isdigit():
                cap = cv2.VideoCapture(int(current_source))
            elif current_source == "demo":
                cap = None
            else:
                cap = cv2.VideoCapture(current_source)
            last_source = current_source

        # Read Frame
        frame = None
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Loop video files if end reached
                if not current_source.isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
        
        # Synthetic simulator frame if no webcam/video
        if frame is None:
            frame = np.full((480, 640, 3), 20, dtype=np.uint8)
            # Draw synthetic construction worker silhouette
            cv2.rectangle(frame, (240, 160), (400, 420), (160, 130, 90), -1)
            cv2.putText(frame, "SIMULATOR STREAM (STANDBY)", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 255), 2)

        # Ingest active polygon zones
        zones = get_active_zones()

        # Execute Safety Vision Pipeline
        annotated_frame, enriched_workers, new_incidents, resolved_incidents, stats = pipeline.process_frame(frame, zones)

        # Persist new confirmed incidents to SQLite
        for inc in new_incidents:
            persist_incident(inc)

        # Resolve unlatched incidents in SQLite
        for res in resolved_incidents:
            if res.get("incident_code"):
                resolve_incident(res["incident_code"])

        # Encode JPEG to Base64
        _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        b64_frame = base64.b64encode(buffer).decode("utf-8")

        # Telemetry Payload
        payload = {
            "type": "FRAME_TELEMETRY",
            "frame": f"data:image/jpeg;base64,{b64_frame}",
            "stats": stats,
            "workers": [
                {
                    "track_id": w["track_id"],
                    "bbox": w["bbox"],
                    "has_helmet": w.get("has_helmet", False),
                    "has_vest": w.get("has_vest", False),
                    "is_in_zone": w.get("is_in_zone", False),
                    "active_violations": w.get("active_violations", [])
                }
                for w in enriched_workers
            ],
            "new_alerts": new_incidents
        }

        # Broadcast to all connected WebSockets
        await manager.broadcast_json(payload)

        # Regulate 15 FPS
        await asyncio.sleep(1.0 / config.STREAM_FPS)

@router.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """Client WebSocket connection endpoint."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep-alive heartbeat listener
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket connection error: {e}")
        manager.disconnect(websocket)
