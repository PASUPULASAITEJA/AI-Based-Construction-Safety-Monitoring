"""
REST API: Polygon Safety Zones Management
Provides CRUD endpoints for 2D spatial restricted and hazard zones.
"""
from typing import List
import json
import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models import RestrictedZone
from schemas import ZoneCreate, ZoneOut

router = APIRouter(prefix="/api/zones", tags=["Restricted Zones"])

@router.get("", response_model=List[ZoneOut])
def list_zones(db: Session = Depends(get_db)):
    """Retrieves all registered polygon safety zones."""
    zones = db.query(RestrictedZone).filter(RestrictedZone.is_active == True).all()
    results = []
    for z in zones:
        coords = json.loads(z.coordinates) if isinstance(z.coordinates, str) else z.coordinates
        results.append({
            "id": z.id,
            "name": z.name,
            "severity": z.severity,
            "coordinates": coords,
            "is_active": z.is_active,
            "created_at": z.created_at,
            "updated_at": z.updated_at
        })
    return results

@router.post("", response_model=ZoneOut, status_code=status.HTTP_201_CREATED)
def create_zone(payload: ZoneCreate, db: Session = Depends(get_db)):
    """Creates a new polygon safety zone."""
    if len(payload.coordinates) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A polygon zone requires at least 3 vertices."
        )

    coords_json = json.dumps(payload.coordinates)
    zone = RestrictedZone(
        name=payload.name,
        severity=payload.severity.upper(),
        coordinates=coords_json,
        is_active=payload.is_active,
        created_at=datetime.datetime.utcnow()
    )
    db.add(zone)
    db.commit()
    db.refresh(zone)

    return {
        "id": zone.id,
        "name": zone.name,
        "severity": zone.severity,
        "coordinates": payload.coordinates,
        "is_active": zone.is_active,
        "created_at": zone.created_at,
        "updated_at": zone.updated_at
    }

@router.delete("/{zone_id}", status_code=status.HTTP_200_OK)
def delete_zone(zone_id: int, db: Session = Depends(get_db)):
    """Deletes a polygon zone by ID."""
    zone = db.query(RestrictedZone).filter(RestrictedZone.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Zone not found")

    db.delete(zone)
    db.commit()
    return {"message": f"Zone {zone_id} deleted successfully."}
