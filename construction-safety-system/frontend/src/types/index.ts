export interface Zone {
  id: number;
  name: string;
  severity: 'DANGER' | 'WARNING' | 'CAUTION';
  coordinates: [number, number][];
  is_active: boolean;
  created_at: string;
}

export interface Incident {
  id: number;
  incident_code: string;
  track_id: number;
  violation_type: 'NO_HELMET' | 'NO_VEST' | 'ZONE_INTRUSION';
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  status: 'ACTIVE' | 'RESOLVED' | 'ACKNOWLEDGED';
  zone_id?: number | null;
  zone_name?: string | null;
  snapshot_path?: string | null;
  started_at: string;
  resolved_at?: string | null;
  duration_frames: number;
  confidence: number;
  description?: string | null;
}

export interface WorkerState {
  track_id: number;
  bbox: [number, number, number, number];
  has_helmet: boolean;
  has_vest: boolean;
  is_in_zone: boolean;
  active_violations: string[];
}

export interface KPIStats {
  active_workers: number;
  compliant_workers: number;
  active_violations: number;
  helmet_violations: number;
  vest_violations: number;
  zone_intrusions: number;
  fps: number;
  compliance_rate: number;
}

export interface TelemetryMessage {
  type: string;
  frame: string; // Base64 data URI
  stats: KPIStats;
  workers: WorkerState[];
  new_alerts: any[];
}
