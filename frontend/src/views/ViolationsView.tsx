import React, { useState, useEffect } from 'react';
import { AlertTriangle, Search, Filter, RefreshCw, ShieldAlert } from 'lucide-react';
import { Violation } from '../types';

export const ViolationsView: React.FC = () => {
  const [violations, setViolations] = useState<Violation[]>([]);
  const [severityFilter, setSeverityFilter] = useState('ALL');
  const [searchTerm, setSearchTerm] = useState('');

  const fetchViolations = async () => {
    try {
      const url = severityFilter === 'ALL' ? '/api/violations' : `/api/violations?severity=${severityFilter}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setViolations(data);
      }
    } catch (e) {
      console.error('Failed to load violations:', e);
    }
  };

  useEffect(() => {
    fetchViolations();
  }, [severityFilter]);

  const filtered = violations.filter(v =>
    v.violation_code.toLowerCase().includes(searchTerm.toLowerCase()) ||
    v.violation_type.toLowerCase().includes(searchTerm.toLowerCase()) ||
    v.worker_label.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-red-500/15 border border-red-500/30 flex items-center justify-center">
            <AlertTriangle className="w-5 h-5 text-red-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 font-mono">VIOLATION TELEMETRY AUDIT TRAIL</h3>
            <p className="text-xs text-slate-400 font-mono">Immutable temporal violations recorded with confidence scores and worker tracking IDs</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {['ALL', 'CRITICAL', 'HIGH', 'MEDIUM'].map((sev) => (
            <button
              key={sev}
              onClick={() => setSeverityFilter(sev)}
              className={`px-3 py-1 rounded text-xs font-mono font-semibold transition-all ${
                severityFilter === sev
                  ? 'bg-orange-500 text-white'
                  : 'bg-[#141B2D] text-slate-400 hover:text-slate-200'
              }`}
            >
              {sev}
            </button>
          ))}
          <button 
            onClick={fetchViolations}
            className="p-1.5 rounded bg-[#141B2D] hover:bg-[#1C263B] text-slate-400 hover:text-slate-200"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Violations Table */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl overflow-hidden">
        <div className="p-4 border-b border-[#1E283D] flex items-center justify-between bg-[#0B0F19]">
          <div className="relative w-72">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Search code, type, worker..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-[#141B2D] border border-[#232F47] rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-orange-500 font-mono"
            />
          </div>
          <span className="text-xs font-mono text-slate-400">Total: {filtered.length}</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#080B12] text-slate-400 border-b border-[#1E283D]">
              <tr>
                <th className="py-3 px-4 font-semibold">CODE</th>
                <th className="py-3 px-4 font-semibold">WORKER TRACK</th>
                <th className="py-3 px-4 font-semibold">VIOLATION TYPE</th>
                <th className="py-3 px-4 font-semibold">SEVERITY</th>
                <th className="py-3 px-4 font-semibold">CONFIDENCE</th>
                <th className="py-3 px-4 font-semibold">SOURCE</th>
                <th className="py-3 px-4 font-semibold">TIMESTAMP</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#172033] text-slate-300">
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={7} className="text-center py-8 text-slate-500">No violation records logged.</td>
                </tr>
              ) : (
                filtered.map((v) => (
                  <tr key={v.id} className="hover:bg-[#121A2C] transition-colors">
                    <td className="py-3 px-4 text-orange-400 font-bold">{v.violation_code}</td>
                    <td className="py-3 px-4 text-slate-200 font-semibold">{v.worker_label || `Worker #${v.worker_id}`}</td>
                    <td className="py-3 px-4">{v.violation_type}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        v.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                        v.severity === 'HIGH' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                        'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                      }`}>
                        {v.severity}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-300">{(v.confidence * 100).toFixed(1)}%</td>
                    <td className="py-3 px-4 text-slate-400">{v.source}</td>
                    <td className="py-3 px-4 text-slate-400">{v.timestamp.replace('T', ' ').substring(0, 19)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
