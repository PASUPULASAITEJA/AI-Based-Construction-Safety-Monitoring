import React, { useState, useRef, useEffect } from 'react';
import { 
  Play, 
  Square, 
  Trash2, 
  Plus, 
  ShieldAlert, 
  Radio, 
  CheckCircle2, 
  AlertTriangle,
  RefreshCw
} from 'lucide-react';
import { Zone } from '../types';

export const MonitoringView: React.FC = () => {
  const [zones, setZones] = useState<Zone[]>([]);
  const [activeWorkers, setActiveWorkers] = useState<any[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [polygonPoints, setPolygonPoints] = useState<[number, number][]>([]);
  const [zoneName, setZoneName] = useState('');
  const [zoneType, setZoneType] = useState<'RESTRICTED' | 'HAZARD' | 'SAFE'>('RESTRICTED');
  const [isSaving, setIsSaving] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Load existing zones
  const loadZones = async () => {
    try {
      const res = await fetch('/api/zones');
      if (res.ok) {
        const data = await res.json();
        setZones(data);
      }
    } catch (e) {
      console.error('Failed to load zones:', e);
    }
  };

  // Poll live detections
  const fetchDetections = async () => {
    try {
      const res = await fetch('/api/monitor/detections');
      if (res.ok) {
        const data = await res.json();
        setActiveWorkers(data.workers || []);
      }
    } catch (e) {
      // Non-blocking telemetry
    }
  };

  useEffect(() => {
    loadZones();
    const interval = setInterval(fetchDetections, 2000);
    return () => clearInterval(interval);
  }, []);

  // Handle canvas clicks to draw polygon vertices
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    setPolygonPoints(prev => [...prev, [x, y]]);
  };

  // Redraw canvas with existing zones and in-progress polygon
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw existing zones
    zones.forEach(z => {
      if (!z.coordinates || z.coordinates.length < 3) return;
      ctx.beginPath();
      ctx.moveTo(z.coordinates[0][0], z.coordinates[0][1]);
      for (let i = 1; i < z.coordinates.length; i++) {
        ctx.lineTo(z.coordinates[i][0], z.coordinates[i][1]);
      }
      ctx.closePath();

      if (z.zone_type === 'RESTRICTED') {
        ctx.fillStyle = 'rgba(239, 68, 68, 0.25)';
        ctx.strokeStyle = '#EF4444';
      } else if (z.zone_type === 'HAZARD') {
        ctx.fillStyle = 'rgba(245, 158, 11, 0.25)';
        ctx.strokeStyle = '#F59E0B';
      } else {
        ctx.fillStyle = 'rgba(16, 185, 129, 0.25)';
        ctx.strokeStyle = '#10B981';
      }
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();

      // Label
      ctx.fillStyle = '#FFFFFF';
      ctx.font = 'bold 12px monospace';
      ctx.fillText(`[${z.zone_type}] ${z.name}`, z.coordinates[0][0] + 6, z.coordinates[0][1] - 6);
    });

    // Draw current polygon in progress
    if (polygonPoints.length > 0) {
      ctx.beginPath();
      ctx.moveTo(polygonPoints[0][0], polygonPoints[0][1]);
      for (let i = 1; i < polygonPoints.length; i++) {
        ctx.lineTo(polygonPoints[i][0], polygonPoints[i][1]);
      }
      if (polygonPoints.length >= 3) {
        ctx.closePath();
        ctx.fillStyle = 'rgba(249, 115, 22, 0.3)';
        ctx.fill();
      }
      ctx.strokeStyle = '#F97316';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw vertex handles
      polygonPoints.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = '#F97316';
        ctx.fill();
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1;
        ctx.stroke();
      });
    }
  }, [zones, polygonPoints]);

  const handleSaveZone = async () => {
    if (!zoneName.trim() || polygonPoints.length < 3) return;
    setIsSaving(true);
    try {
      const res = await fetch('/api/zones', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: zoneName.trim(),
          zone_type: zoneType,
          coordinates: polygonPoints,
          site_id: 1
        })
      });
      if (res.ok) {
        setPolygonPoints([]);
        setZoneName('');
        setIsDrawing(false);
        loadZones();
      }
    } catch (e) {
      console.error('Error saving zone:', e);
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteZone = async (id: number) => {
    try {
      const res = await fetch(`/api/zones/${id}`, { method: 'DELETE' });
      if (res.ok) {
        loadZones();
      }
    } catch (e) {
      console.error('Error deleting zone:', e);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-orange-500/15 border border-orange-500/30 flex items-center justify-center">
            <Radio className="w-5 h-5 text-orange-400 animate-pulse" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 font-mono">LIVE SAFETY MONITORING WORKSTATION</h3>
            <p className="text-xs text-slate-400 font-mono">Primary Optical Stream • CAM-01 • 640x480 Reference Frame</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {!isDrawing ? (
            <button
              onClick={() => { setIsDrawing(true); setPolygonPoints([]); }}
              className="px-3.5 py-1.5 rounded-lg bg-orange-500 hover:bg-orange-600 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-lg shadow-orange-500/20"
            >
              <Plus className="w-3.5 h-3.5" />
              Draw Safety Zone
            </button>
          ) : (
            <button
              onClick={() => { setIsDrawing(false); setPolygonPoints([]); }}
              className="px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center gap-1.5 transition-colors"
            >
              <Square className="w-3.5 h-3.5" />
              Cancel Drawing
            </button>
          )}

          <button
            onClick={loadZones}
            className="p-2 rounded-lg bg-[#141B2D] hover:bg-[#1C263B] border border-[#232F47] text-slate-400 hover:text-slate-200"
            title="Reload Zones"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Editor & Stream Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Stream & Drawing Canvas (8 cols) */}
        <div className="lg:col-span-8 bg-[#0E131F] border border-[#1E283D] rounded-xl overflow-hidden flex flex-col">
          <div className="relative aspect-video bg-black flex items-center justify-center select-none overflow-hidden">
            {/* Live MJPEG Feed */}
            <img 
              src="/api/monitor/feed" 
              alt="Live Construction Stream" 
              className="w-full h-full object-cover"
            />

            {/* Transparent Polygon Drawing Overlay Canvas */}
            <canvas
              ref={canvasRef}
              width={640}
              height={480}
              onClick={handleCanvasClick}
              className={`absolute inset-0 w-full h-full z-10 ${isDrawing ? 'cursor-crosshair' : 'pointer-events-none'}`}
            />

            {/* Drawing Prompt Overlay */}
            {isDrawing && (
              <div className="absolute top-4 left-4 z-20 bg-black/80 backdrop-blur-md border border-orange-500/50 rounded-lg p-2.5 text-xs text-orange-300 font-mono">
                Click anywhere on the stream to place polygon vertices ({polygonPoints.length} points placed).
              </div>
            )}
          </div>

          {/* Drawing Controls Drawer */}
          {isDrawing && (
            <div className="p-4 bg-[#0A0D15] border-t border-[#1E283D] flex flex-wrap items-center justify-between gap-3 font-mono text-xs">
              <div className="flex items-center gap-3">
                <input
                  type="text"
                  placeholder="Zone Name (e.g., Excavation Trench #2)"
                  value={zoneName}
                  onChange={(e) => setZoneName(e.target.value)}
                  className="px-3 py-1.5 rounded-lg bg-[#141B2D] border border-[#232F47] text-slate-200 focus:outline-none focus:border-orange-500"
                />
                <select
                  value={zoneType}
                  onChange={(e) => setZoneType(e.target.value as any)}
                  className="px-3 py-1.5 rounded-lg bg-[#141B2D] border border-[#232F47] text-slate-200 focus:outline-none focus:border-orange-500"
                >
                  <option value="RESTRICTED">RESTRICTED (Red)</option>
                  <option value="HAZARD">HAZARD (Amber)</option>
                  <option value="SAFE">SAFE (Green)</option>
                </select>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPolygonPoints([])}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white"
                >
                  Clear Points
                </button>
                <button
                  onClick={handleSaveZone}
                  disabled={isSaving || polygonPoints.length < 3 || !zoneName.trim()}
                  className="px-4 py-1.5 rounded-lg bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white font-semibold transition-colors"
                >
                  {isSaving ? 'Saving...' : 'Save Geofence'}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Active Zones & Live Worker Roster (4 cols) */}
        <div className="lg:col-span-4 space-y-5">
          {/* Active Geofence Zones List */}
          <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4">
            <h4 className="text-xs font-bold text-slate-200 font-mono mb-3 flex items-center justify-between">
              <span>CONFIGURED GEOFENCES ({zones.length})</span>
            </h4>
            <div className="space-y-2 max-h-56 overflow-y-auto">
              {zones.length === 0 ? (
                <p className="text-xs text-slate-500 font-mono py-2">No polygon zones defined yet.</p>
              ) : (
                zones.map(z => (
                  <div key={z.id} className="p-2.5 rounded-lg bg-[#141B2D] border border-[#232F47] flex items-center justify-between">
                    <div>
                      <p className="text-xs font-bold text-slate-200">{z.name}</p>
                      <span className={`text-[10px] font-mono px-1.5 py-0.2 rounded font-semibold ${
                        z.zone_type === 'RESTRICTED' ? 'bg-red-500/20 text-red-400' :
                        z.zone_type === 'HAZARD' ? 'bg-amber-500/20 text-amber-400' :
                        'bg-emerald-500/20 text-emerald-400'
                      }`}>
                        {z.zone_type}
                      </span>
                    </div>
                    <button
                      onClick={() => handleDeleteZone(z.id)}
                      className="p-1.5 rounded text-slate-400 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                      title="Delete Zone"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Active Worker Roster on Stream */}
          <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4">
            <h4 className="text-xs font-bold text-slate-200 font-mono mb-3">
              CURRENT DETECTED WORKERS ({activeWorkers.length})
            </h4>
            <div className="space-y-2 max-h-64 overflow-y-auto font-mono text-xs">
              {activeWorkers.length === 0 ? (
                <p className="text-xs text-slate-500 py-3">Scanning scene for active personnel...</p>
              ) : (
                activeWorkers.map((w, idx) => (
                  <div key={idx} className="p-2.5 rounded-lg bg-[#141B2D] border border-[#232F47] flex items-center justify-between">
                    <span className="font-bold text-slate-200">Worker #{w.worker_id || idx + 1}</span>
                    <div className="flex items-center gap-1.5">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        w.has_helmet ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                      }`}>
                        {w.has_helmet ? 'Hardhat OK' : 'No Hardhat'}
                      </span>
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        w.has_vest ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                      }`}>
                        {w.has_vest ? 'Vest OK' : 'No Vest'}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
