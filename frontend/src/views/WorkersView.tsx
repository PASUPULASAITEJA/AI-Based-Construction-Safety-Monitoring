import React, { useState, useEffect } from 'react';
import { Users, ShieldCheck, AlertTriangle, RefreshCw, Award } from 'lucide-react';
import { WorkerTrack } from '../types';

export const WorkersView: React.FC = () => {
  const [workers, setWorkers] = useState<WorkerTrack[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchWorkers = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/workers');
      if (res.ok) {
        const data = await res.json();
        setWorkers(data);
      }
    } catch (e) {
      console.error('Error fetching workers:', e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchWorkers();
  }, []);

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-blue-500/15 border border-blue-500/30 flex items-center justify-center">
            <Users className="w-5 h-5 text-blue-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 font-mono">ANONYMOUS WORKER TRACKING ROSTER</h3>
            <p className="text-xs text-slate-400 font-mono">
              Privacy-Preserving Tracking (No Facial Recognition) • Longitudinal PPE Compliance Scoring
            </p>
          </div>
        </div>
        <button 
          onClick={fetchWorkers}
          className="p-2 rounded-lg bg-[#141B2D] hover:bg-[#1C263B] border border-[#232F47] text-slate-400 hover:text-slate-200"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-orange-400' : ''}`} />
        </button>
      </div>

      {/* Workers Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {workers.map((w) => (
          <div key={w.id} className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex flex-col justify-between space-y-3 font-mono">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-orange-400">{w.track_code}</span>
              <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                w.grade.startsWith('A') ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
                w.grade === 'B' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' :
                'bg-red-500/20 text-red-400 border border-red-500/30'
              }`}>
                Grade: {w.grade}
              </span>
            </div>

            <div>
              <p className="text-sm font-bold text-slate-200">{w.label}</p>
              <div className="flex items-center justify-between text-xs text-slate-400 mt-2">
                <span>Compliance:</span>
                <strong className={w.compliance_pct >= 80 ? 'text-emerald-400' : 'text-red-400'}>
                  {w.compliance_pct}%
                </strong>
              </div>
              <div className="w-full bg-[#141B2D] h-2 rounded-full mt-1.5 overflow-hidden border border-[#232F47]">
                <div 
                  className={`h-full ${w.compliance_pct >= 80 ? 'bg-emerald-400' : 'bg-red-400'}`}
                  style={{ width: `${w.compliance_pct}%` }}
                />
              </div>
            </div>

            <div className="pt-2 border-t border-[#1E283D] text-[11px] text-slate-400 flex items-center justify-between">
              <span>Obs: {w.total_observations}</span>
              <span>Violations: {w.violation_count || 0}</span>
            </div>
          </div>
        ))}
      </div>

      {workers.length === 0 && (
        <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-8 text-center text-slate-500 font-mono text-xs">
          No worker tracking profiles stored yet. Tracks accumulate during video analysis and live camera sessions.
        </div>
      )}
    </div>
  );
};
