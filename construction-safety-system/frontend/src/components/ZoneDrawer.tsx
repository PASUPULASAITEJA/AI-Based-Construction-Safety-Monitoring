import React, { useState, useRef, useEffect } from 'react';
import { Plus, Trash2, Check, X, ShieldAlert, AlertTriangle } from 'lucide-react';
import { Zone } from '../types';
import { api } from '../services/api';

interface ZoneDrawerProps {
  zones: Zone[];
  onZonesUpdated: () => void;
}

export const ZoneDrawer: React.FC<ZoneDrawerProps> = ({ zones, onZonesUpdated }) => {
  const [isDrawing, setIsDrawing] = useState(false);
  const [points, setPoints] = useState<[number, number][]>([]);
  const [zoneName, setZoneName] = useState('');
  const [severity, setSeverity] = useState<'DANGER' | 'WARNING'>('DANGER');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Redraw canvas with polygon preview
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw existing zones
    zones.forEach((z) => {
      if (z.coordinates.length < 3) return;
      ctx.beginPath();
      ctx.moveTo(z.coordinates[0][0], z.coordinates[0][1]);
      for (let i = 1; i < z.coordinates.length; i++) {
        ctx.lineTo(z.coordinates[i][0], z.coordinates[i][1]);
      }
      ctx.closePath();

      ctx.fillStyle = z.severity === 'DANGER' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)';
      ctx.strokeStyle = z.severity === 'DANGER' ? '#EF4444' : '#F59E0B';
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();

      // Label
      ctx.fillStyle = '#FFFFFF';
      ctx.font = 'bold 11px monospace';
      ctx.fillText(`[${z.severity}] ${z.name}`, z.coordinates[0][0] + 6, z.coordinates[0][1] - 6);
    });

    // Draw active drawing polygon
    if (points.length > 0) {
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0], points[i][1]);
      }
      if (points.length >= 3) {
        ctx.closePath();
        ctx.fillStyle = severity === 'DANGER' ? 'rgba(239, 68, 68, 0.25)' : 'rgba(245, 158, 11, 0.25)';
        ctx.fill();
      }
      ctx.strokeStyle = '#F97316';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw vertex dots
      points.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = '#F97316';
        ctx.fill();
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    }
  }, [points, zones, severity]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    setPoints((prev) => [...prev, [x, y]]);
  };

  const handleSaveZone = async () => {
    if (points.length < 3 || !zoneName.trim()) return;
    setIsSubmitting(true);
    try {
      await api.createZone({
        name: zoneName.trim(),
        severity,
        coordinates: points,
      });
      setPoints([]);
      setZoneName('');
      setIsDrawing(false);
      onZonesUpdated();
    } catch (e) {
      console.error('Failed to save zone:', e);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDeleteZone = async (id: number) => {
    try {
      await api.deleteZone(id);
      onZonesUpdated();
    } catch (e) {
      console.error('Failed to delete zone:', e);
    }
  };

  return (
    <div className="bg-[#0F1420] border border-[#1D263B] rounded-xl p-4 space-y-4 font-mono text-xs">
      <div className="flex items-center justify-between border-b border-[#1D263B] pb-3">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-orange-400" />
          <h3 className="text-slate-100 font-bold tracking-wide">RESTRICTED GEOFENCE ZONES</h3>
        </div>

        {!isDrawing ? (
          <button
            onClick={() => { setIsDrawing(true); setPoints([]); }}
            className="px-3 py-1.5 rounded-lg bg-orange-500 hover:bg-orange-600 text-white font-bold flex items-center gap-1.5 transition-colors shadow-lg shadow-orange-500/20"
          >
            <Plus className="w-3.5 h-3.5" /> Draw Zone
          </button>
        ) : (
          <button
            onClick={() => { setIsDrawing(false); setPoints([]); }}
            className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-bold flex items-center gap-1.5"
          >
            <X className="w-3.5 h-3.5" /> Cancel
          </button>
        )}
      </div>

      {/* Interactive Drawing Overlay Canvas */}
      <div className="relative aspect-video bg-black rounded-lg border border-[#232F47] overflow-hidden select-none">
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          onClick={handleCanvasClick}
          className={`w-full h-full ${isDrawing ? 'cursor-crosshair' : 'pointer-events-none'}`}
        />

        {isDrawing && (
          <div className="absolute top-2 left-2 bg-black/80 backdrop-blur-md border border-orange-500/40 rounded px-2.5 py-1 text-[11px] text-orange-300">
            Click to add vertex ({points.length} points). Form closed polygon & save.
          </div>
        )}
      </div>

      {/* Zone Creation Form */}
      {isDrawing && (
        <div className="p-3 bg-[#0A0D15] border border-[#1D263B] rounded-lg flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <input
              type="text"
              placeholder="Zone Name (e.g., Crane Swing Radius)"
              value={zoneName}
              onChange={(e) => setZoneName(e.target.value)}
              className="px-3 py-1.5 rounded bg-[#151D2E] border border-[#232F47] text-slate-200 focus:outline-none focus:border-orange-500"
            />
            <select
              value={severity}
              onChange={(e) => setSeverity(e.target.value as any)}
              className="px-3 py-1.5 rounded bg-[#151D2E] border border-[#232F47] text-slate-200 focus:outline-none focus:border-orange-500"
            >
              <option value="DANGER">DANGER (High Hazard)</option>
              <option value="WARNING">WARNING (Precautionary)</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setPoints([])}
              className="px-3 py-1.5 rounded bg-slate-800 text-slate-400 hover:text-white"
            >
              Reset Points
            </button>
            <button
              onClick={handleSaveZone}
              disabled={isSubmitting || points.length < 3 || !zoneName.trim()}
              className="px-4 py-1.5 rounded bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white font-bold transition-colors flex items-center gap-1.5"
            >
              <Check className="w-3.5 h-3.5" />
              {isSubmitting ? 'Saving...' : 'Save Zone'}
            </button>
          </div>
        </div>
      )}

      {/* List of Configured Zones */}
      <div className="space-y-2 max-h-48 overflow-y-auto">
        {zones.length === 0 ? (
          <p className="text-slate-500 text-center py-2">No active polygon geofences defined.</p>
        ) : (
          zones.map((z) => (
            <div key={z.id} className="p-2.5 rounded-lg bg-[#151D2E] border border-[#232F47] flex items-center justify-between">
              <div>
                <span className="font-bold text-slate-200">{z.name}</span>
                <span className={`ml-2 text-[10px] px-1.5 py-0.5 rounded font-bold ${
                  z.severity === 'DANGER' ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                }`}>
                  {z.severity}
                </span>
                <p className="text-[10px] text-slate-400 mt-0.5">{z.coordinates.length} vertices</p>
              </div>
              <button
                onClick={() => handleDeleteZone(z.id)}
                className="p-1.5 rounded text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                title="Delete Zone"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
