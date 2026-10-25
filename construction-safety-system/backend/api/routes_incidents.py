"""
REST API: Incident Management & Evidence Retrieval
Provides incident log queries, status transitions, and cropped snapshot image delivery.
"""
from typing import List, Optional
import os
import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import get_db
from models import ViolationIncident
from schemas import IncidentOut, IncidentStatusUpdate
import config

router = APIRouter(prefix="/api/incidents", tags=["Incidents & Violations"])

@router.get("", response_model=List[IncidentOut])
def list_incidents(
    status_filter: Optional[str] = Query(None, alias="status"),
    severity: Optional[str] = Query(None),
    violation_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Retrieves paginated and filtered historical safety incidents."""
    query = db.query(ViolationIncident)

    if status_filter:
        query = query.filter(ViolationIncident.status == status_filter.upper())
    if severity:
        query = query.filter(ViolationIncident.severity == severity.upper())
    if violation_type:
        query = query.filter(ViolationIncident.violation_type == violation_type.upper())

    incidents = query.order_by(ViolationIncident.id.desc()).limit(limit).all()
    return incidents

@router.get("/{incident_id}", response_model=IncidentOut)
def get_incident(incident_id: int, db: Session = Depends(get_db)):
    """Retrieves a single incident record by ID."""
    incident = db.query(ViolationIncident).filter(ViolationIncident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")
    return incident

@router.patch("/{incident_id}/status", response_model=IncidentOut)
def update_incident_status(incident_id: int, payload: IncidentStatusUpdate, db: Session = Depends(get_db)):
    """Transitions incident lifecycle (e.g., ACTIVE -> ACKNOWLEDGED -> RESOLVED)."""
    incident = db.query(ViolationIncident).filter(ViolationIncident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")

    new_status = payload.status.upper()
    if new_status not in ["ACTIVE", "ACKNOWLEDGED", "RESOLVED"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status value.")

    incident.status = new_status
    if new_status == "RESOLVED":
        incident.resolved_at = datetime.datetime.utcnow()

    db.commit()
    db.refresh(incident)
    return incident

@router.get("/{incident_id}/snapshot")
def get_incident_snapshot(incident_id: int, db: Session = Depends(get_db)):
    """Serves the cropped visual evidence snapshot associated with this incident."""
    incident = db.query(ViolationIncident).filter(ViolationIncident.id == incident_id).first()
    if not incident or not incident.snapshot_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Snapshot not found")

    # Resolve filename
    filename = os.path.basename(incident.snapshot_path)
    full_path = os.path.join(config.SNAPSHOTS_DIR, filename)

    if not os.path.exists(full_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Snapshot file missing on disk")

    return FileResponse(full_path, media_type="image/jpeg")
