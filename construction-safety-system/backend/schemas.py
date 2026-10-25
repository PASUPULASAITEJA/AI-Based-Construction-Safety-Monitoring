"""
Pydantic Schemas for Request & Response Data Validation
"""
import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class ZoneBase(BaseModel):
    name: str = Field(..., example="Excavation Zone 1")
    severity: str = Field(default="DANGER", example="DANGER")
    coordinates: List[List[float]] = Field(..., example=[[100, 100], [400, 100], [400, 350], [100, 350]])
    is_active: bool = True

class ZoneCreate(ZoneBase):
    pass

class ZoneOut(ZoneBase):
    id: int
    created_at: datetime.datetime
    updated_at: Optional[datetime.datetime] = None

    class Config:
        from_attributes = True

class IncidentOut(BaseModel):
    id: int
    incident_code: str
    track_id: int
    violation_type: str
    severity: str
    status: str
    zone_id: Optional[int] = None
    zone_name: Optional[str] = None
    snapshot_path: Optional[str] = None
    started_at: datetime.datetime
    resolved_at: Optional[datetime.datetime] = None
    duration_frames: int
    confidence: float
    description: Optional[str] = None

    class Config:
        from_attributes = True

class IncidentStatusUpdate(BaseModel):
    status: str = Field(..., example="RESOLVED")  # ACKNOWLEDGED, RESOLVED

class SystemKPIStats(BaseModel):
    active_workers: int
    compliant_workers: int
    active_violations: int
    helmet_violations: int
    vest_violations: int
    zone_intrusions: int
    fps: float
    compliance_rate: float
