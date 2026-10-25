import React, { useState, useEffect } from 'react';
import { 
  AlertOctagon, 
  CheckCircle2, 
  Clock, 
  Filter, 
  Search, 
  ChevronRight,
  ShieldAlert,
  XCircle,
  FileCheck
} from 'lucide-react';
import { Incident } from '../types';

export const IncidentsView: React.FC = () => {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [notes, setNotes] = useState('');
  const [isUpdating, setIsUpdating] = useState(false);

  const fetchIncidents = async () => {
    try {
      const url = statusFilter === 'ALL' ? '/api/incidents' : `/api/incidents?status=${statusFilter}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setIncidents(data);
      }
    } catch (e) {
      console.error('Failed to load incidents:', e);
    }
  };

  useEffect(() => {
    fetchIncidents();
  }, [statusFilter]);

  const handleTransition = async (newStatus: string) => {
    if (!selectedIncident) return;
    setIsUpdating(true);
    try {
      const res = await fetch(`/api/incidents/${selectedIncident.id}/transition`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          status: newStatus,
          notes: notes.trim(),
          actor: 'Lead Safety Officer'
        })
      });
      if (res.ok) {
        setSelectedIncident(null);
        setNotes('');
        fetchIncidents();
      }
    } catch (e) {
      console.error('Transition error:', e);
    } finally {
      setIsUpdating(false);
    }
  };

  const filtered = incidents.filter(i => 
    i.incident_code.toLowerCase().includes(searchTerm.toLowerCase()) ||
    i.violation_type.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-red-500/15 border border-red-500/30 flex items-center justify-center">
            <AlertOctagon className="w-5 h-5 text-red-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 font-mono">INCIDENT LIFECYCLE MANAGEMENT BOARD</h3>
            <p className="text-xs text-slate-400 font-mono">State Machine: OPEN → ACKNOWLEDGED → UNDER_REVIEW → RESOLVED</p>
          </div>
        </div>

        {/* Filter Tabs */}
        <div className="flex items-center gap-1.5 bg-[#080B12] p-1 rounded-lg border border-[#1E283D] text-xs font-mono">
          {['ALL', 'OPEN', 'ACKNOWLEDGED', 'UNDER_REVIEW', 'RESOLVED'].map((s) => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              className={`px-3 py-1 rounded font-semibold transition-all ${
                statusFilter === s 
                  ? 'bg-orange-500 text-white shadow-sm' 
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Incidents Table */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl overflow-hidden">
        <div className="p-4 border-b border-[#1E283D] flex items-center justify-between bg-[#0B0F19]">
          <div className="relative w-72">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Search code or violation type..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-[#141B2D] border border-[#232F47] rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-orange-500 font-mono"
            />
          </div>
          <span className="text-xs font-mono text-slate-400">Total Cases: {filtered.length}</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#080B12] text-slate-400 border-b border-[#1E283D]">
              <tr>
                <th className="py-3 px-4 font-semibold">CODE</th>
                <th className="py-3 px-4 font-semibold">DESCRIPTION</th>
                <th className="py-3 px-4 font-semibold">SEVERITY</th>
                <th className="py-3 px-4 font-semibold">STATUS</th>
                <th className="py-3 px-4 font-semibold">TIMESTAMP</th>
                <th className="py-3 px-4 font-semibold text-right">ACTION</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#172033] text-slate-300">
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center py-8 text-slate-500">No incident records matching criteria.</td>
                </tr>
              ) : (
                filtered.map((inc) => (
                  <tr key={inc.id} className="hover:bg-[#121A2C] transition-colors">
                    <td className="py-3 px-4 text-orange-400 font-bold">{inc.incident_code}</td>
                    <td className="py-3 px-4 text-slate-200">{inc.title || inc.violation_type}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        inc.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                        inc.severity === 'HIGH' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                        'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                      }`}>
                        {inc.severity}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        inc.status === 'RESOLVED' ? 'bg-emerald-500/20 text-emerald-400' :
                        inc.status === 'UNDER_REVIEW' ? 'bg-blue-500/20 text-blue-400' :
                        inc.status === 'ACKNOWLEDGED' ? 'bg-amber-500/20 text-amber-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {inc.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-400">{inc.timestamp.replace('T', ' ').substring(0, 19)}</td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => { setSelectedIncident(inc); setNotes(inc.resolution_notes || ''); }}
                        className="px-3 py-1 rounded bg-[#141B2D] hover:bg-[#1C263B] border border-[#232F47] text-slate-300 hover:text-orange-400 font-semibold transition-colors"
                      >
                        Manage
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Incident Review & Transition Modal */}
      {selectedIncident && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl max-w-lg w-full p-6 space-y-4 font-mono text-xs">
            <div className="flex items-center justify-between border-b border-[#1E283D] pb-3">
              <div>
                <h4 className="text-sm font-bold text-slate-100">{selectedIncident.incident_code}</h4>
                <p className="text-slate-400 text-[11px]">{selectedIncident.violation_type}</p>
              </div>
              <button 
                onClick={() => setSelectedIncident(null)} 
                className="text-slate-500 hover:text-slate-300"
              >
                ✕
              </button>
            </div>

            <div className="space-y-2 text-slate-300">
              <p>Current Status: <strong className="text-orange-400">{selectedIncident.status}</strong></p>
              <p>Severity Level: <strong className="text-red-400">{selectedIncident.severity}</strong></p>
              <p>Site Reference: <strong>{selectedIncident.site_name || 'Metro Hub'}</strong></p>
            </div>

            <div className="space-y-1.5">
              <label className="text-slate-400">Resolution & Audit Notes:</label>
              <textarea
                rows={3}
                placeholder="Enter root cause, corrective actions, or disciplinary measures..."
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                className="w-full bg-[#141B2D] border border-[#232F47] rounded-lg p-3 text-slate-200 focus:outline-none focus:border-orange-500 text-xs"
              />
            </div>

            {/* Transition Action Buttons */}
            <div className="flex flex-wrap gap-2 pt-2 border-t border-[#1E283D]">
              <button
                onClick={() => handleTransition('ACKNOWLEDGED')}
                disabled={isUpdating}
                className="px-3 py-1.5 rounded bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 font-semibold"
              >
                Acknowledge
              </button>
              <button
                onClick={() => handleTransition('UNDER_REVIEW')}
                disabled={isUpdating}
                className="px-3 py-1.5 rounded bg-blue-500/20 text-blue-300 hover:bg-blue-500/30 font-semibold"
              >
                Under Review
              </button>
              <button
                onClick={() => handleTransition('RESOLVED')}
                disabled={isUpdating}
                className="px-3 py-1.5 rounded bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30 font-semibold"
              >
                Resolve Incident
              </button>
              <button
                onClick={() => handleTransition('FALSE_POSITIVE')}
                disabled={isUpdating}
                className="px-3 py-1.5 rounded bg-slate-800 text-slate-400 hover:text-slate-200"
              >
                Close as False Alarm
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
