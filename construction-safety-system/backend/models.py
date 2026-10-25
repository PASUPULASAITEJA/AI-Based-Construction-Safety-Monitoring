"""
SQLAlchemy ORM Data Models
"""
import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class RestrictedZone(Base):
    __tablename__ = "restricted_zones"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    severity = Column(String(20), default="DANGER")  # DANGER, WARNING, CAUTION
    coordinates = Column(Text, nullable=False)        # JSON serialized string: [[x1, y1], [x2, y2], ...]
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    incidents = relationship("ViolationIncident", back_populates="zone", cascade="all, delete-orphan")


class ViolationIncident(Base):
    __tablename__ = "violation_incidents"

    id = Column(Integer, primary_key=True, index=True)
    incident_code = Column(String(50), unique=True, index=True, nullable=False)
    track_id = Column(Integer, index=True, nullable=False)
    violation_type = Column(String(50), index=True, nullable=False)  # NO_HELMET, NO_VEST, ZONE_INTRUSION
    severity = Column(String(20), default="HIGH")                   # CRITICAL, HIGH, MEDIUM, LOW
    status = Column(String(20), default="ACTIVE", index=True)       # ACTIVE, RESOLVED, ACKNOWLEDGED
    
    zone_id = Column(Integer, ForeignKey("restricted_zones.id", ondelete="SET NULL"), nullable=True)
    zone_name = Column(String(100), nullable=True)
    
    snapshot_path = Column(String(255), nullable=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    resolved_at = Column(DateTime, nullable=True)
    duration_frames = Column(Integer, default=1)
    confidence = Column(Float, default=0.85)
    description = Column(Text, nullable=True)

    zone = relationship("RestrictedZone", back_populates="incidents")


class SystemLog(Base):
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    level = Column(String(20), default="INFO")
    source = Column(String(50), default="PIPELINE")
    message = Column(Text, nullable=False)
