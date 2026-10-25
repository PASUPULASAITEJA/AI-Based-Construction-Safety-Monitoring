"""
SiteGuard AI - FastAPI Application Entrypoint
Integrates REST APIs, WebSocket real-time telemetry, and computer vision pipeline.
"""
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import init_db, get_db
from models import ViolationIncident, RestrictedZone
from schemas import SystemKPIStats
from api.routes_zones import router as zones_router
from api.routes_incidents import router as incidents_router
from api.routes_stream import router as stream_router, stream_mgr
from api.websocket import router as ws_router, stream_broadcaster_loop
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SiteGuard.Main")

# Lifespan context manager for background streaming task
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Database Tables
    logger.info("Initializing SiteGuard AI Database...")
    init_db()

    # Launch background WebSocket broadcaster loop
    broadcaster_task = asyncio.create_task(stream_broadcaster_loop())
    logger.info("Background video stream broadcaster task spawned.")

    yield

    # Shutdown
    broadcaster_task.cancel()
    try:
        await broadcaster_task
    except asyncio.CancelledError:
        pass
    logger.info("SiteGuard AI Server shut down cleanly.")

app = FastAPI(
    title="SiteGuard AI - Construction Safety Monitoring API",
    description="Real-time computer vision safety monitoring: YOLOv8, ByteTrack, Anatomical PPE matching, and Polygon Geofencing.",
    version="2.5.0",
    lifespan=lifespan
)

# Enable CORS for local Vite development & Docker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount media directories
app.mount("/data/snapshots", StaticFiles(directory=str(config.SNAPSHOTS_DIR)), name="snapshots")
app.mount("/data/uploads", StaticFiles(directory=str(config.UPLOADS_DIR)), name="uploads")

# Include API Routers
app.include_router(zones_router)
app.include_router(incidents_router)
app.include_router(stream_router)
app.include_router(ws_router)

@app.get("/health", tags=["Health"])
def health_check():
    """System liveness probe."""
    return {
        "status": "HEALTHY",
        "model": stream_mgr.pipeline.detector.model_name,
        "is_custom_model": stream_mgr.pipeline.detector.is_custom_model,
        "active_source": stream_mgr.current_filename or stream_mgr.active_source
    }

@app.get("/api/stats", response_model=SystemKPIStats, tags=["Telemetry"])
def get_kpi_stats(db: Session = Depends(get_db)):
    """Retrieves operational safety metrics from database and active pipeline."""
    pipeline_stats = {
        "active_workers": 0,
        "compliant_workers": 0,
        "active_violations": 0,
        "helmet_violations": 0,
        "vest_violations": 0,
        "zone_intrusions": 0,
        "fps": round(stream_mgr.pipeline.fps, 1),
        "compliance_rate": 100.0
    }

    # Query active violations from DB
    active_incidents = db.query(ViolationIncident).filter(ViolationIncident.status == "ACTIVE").all()
    active_viols = len(active_incidents)
    h_viols = sum(1 for i in active_incidents if i.violation_type == "NO_HELMET")
    v_viols = sum(1 for i in active_incidents if i.violation_type == "NO_VEST")
    z_viols = sum(1 for i in active_incidents if i.violation_type == "ZONE_INTRUSION")

    pipeline_stats["active_violations"] = active_viols
    pipeline_stats["helmet_violations"] = h_viols
    pipeline_stats["vest_violations"] = v_viols
    pipeline_stats["zone_intrusions"] = z_viols

    return pipeline_stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
