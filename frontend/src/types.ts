export interface KPIStats {
  observed_workers: number;
  compliant_workers: number;
  violations_today: number;
  open_alerts: number;
  resolved_incidents: number;
  compliance_pct: number;
  active_cameras: number;
  system_fps: number;
}

export interface Incident {
  id: number;
  incident_code: string;
  title?: string;
  violation_type: string;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  status: 'OPEN' | 'ACKNOWLEDGED' | 'UNDER_REVIEW' | 'RESOLVED' | 'FALSE_POSITIVE';
  timestamp: string;
  worker_id?: number;
  site_name?: string;
  camera_name?: string;
  description?: string;
  resolution_notes?: string;
}

export interface Violation {
  id: number;
  violation_code: string;
  worker_id: number;
  worker_label: string;
  violation_type: string;
  severity: string;
  confidence: number;
  timestamp: string;
  source: string;
  description?: string;
}

export interface WorkerTrack {
  id: number;
  track_code: string;
  label: string;
  total_observations: number;
  compliant_observations: number;
  violation_count: number;
  compliance_rate: number;
  compliance_pct: number;
  grade: string;
  first_seen: string;
  last_seen: string;
}

export interface Zone {
  id: number;
  site_id: number;
  name: string;
  zone_type: 'RESTRICTED' | 'HAZARD' | 'SAFE';
  coordinates: number[][];
  created_at: string;
}

export interface Site {
  id: number;
  site_code: string;
  name: string;
  location: string;
  project_name: string;
  safety_officer: string;
}

export interface AnalysisJob {
  id: number;
  job_id: string;
  original_filename: string;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  progress: number;
  current_frame: number;
  total_frames: number;
  violations_count: number;
  processing_duration_sec: number;
  output_video_path?: string;
  error_message?: string;
  created_at: string;
}

export interface SystemHealth {
  status: string;
  database_connected: boolean;
  database_latency_ms: number;
  model_loaded: boolean;
  model_file: string;
  free_disk_gb: number;
  total_disk_gb: number;
  cv_pipeline_ready: boolean;
}
