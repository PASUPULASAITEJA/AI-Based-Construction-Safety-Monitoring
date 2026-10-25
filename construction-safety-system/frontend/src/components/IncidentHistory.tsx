import React, { useState } from 'react';
import { Search, Image as ImageIcon, CheckCircle, ShieldAlert, X } from 'lucide-react';
import { Incident } from '../types';
import { api } from '../services/api';

interface IncidentHistoryProps {
  incidents: Incident[];
  onRefresh: () => void;
}

export const IncidentHistory: React.FC<IncidentHistoryProps> = ({ incidents, onRefresh }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [severityFilter, setSeverityFilter] = useState('ALL');
  const [selectedSnapshot, setSelectedSnapshot] = useState<{ id: number; code: string; type: string } | null>(null);

  const filtered = incidents.filter((i) => {
    const matchesSearch =
      i.incident_code.toLowerCase().includes(searchTerm.toLowerCase()) ||
      i.violation_type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      String(i.track_id).includes(searchTerm);
    const matchesSeverity = severityFilter === 'ALL' || i.severity === severityFilter;
    return matchesSearch && matchesSeverity;
  });

  const handleResolve = async (id: number) => {
    try {
      await api.updateIncidentStatus(id, 'RESOLVED');
      onRefresh();
    } catch (e) {
      console.error('Failed to resolve incident:', e);
    }
  };

  return (
    <div className="bg-[#0F1420] border border-[#1D263B] rounded-xl overflow-hidden font-mono text-xs">
      {/* Table Header & Search Filter */}
      <div className="p-4 bg-[#0B0F18] border-b border-[#1D263B] flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-orange-400" />
          <h3 className="font-bold text-slate-100 tracking-wide">INCIDENT AUDIT LOG</h3>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Search code, ID, type..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-8 pr-3 py-1.5 rounded-lg bg-[#151D2E] border border-[#232F47] text-slate-200 focus:outline-none focus:border-orange-500 text-xs w-52"
            />
          </div>

          <div className="flex items-center gap-1 bg-[#151D2E] p-1 rounded-lg border border-[#232F47]">
            {['ALL', 'CRITICAL', 'HIGH', 'MEDIUM'].map((sev) => (
              <button
                key={sev}
                onClick={() => setSeverityFilter(sev)}
                className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                  severityFilter === sev ? 'bg-orange-500 text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {sev}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Incidents Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead className="bg-[#0A0D14] text-slate-400 border-b border-[#1D263B]">
            <tr>
              <th className="py-2.5 px-4 font-semibold">INCIDENT CODE</th>
              <th className="py-2.5 px-4 font-semibold">TRACK ID</th>
              <th className="py-2.5 px-4 font-semibold">VIOLATION TYPE</th>
              <th className="py-2.5 px-4 font-semibold">SEVERITY</th>
              <th className="py-2.5 px-4 font-semibold">STATUS</th>
              <th className="py-2.5 px-4 font-semibold">TIMESTAMP</th>
              <th className="py-2.5 px-4 font-semibold text-right">EVIDENCE & ACTION</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#172033] text-slate-300">
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={7} className="text-center py-6 text-slate-500">
                  No incident records found.
                </td>
              </tr>
            ) : (
              filtered.map((inc) => (
                <tr key={inc.id} className="hover:bg-[#151D2E] transition-colors">
                  <td className="py-2.5 px-4 font-bold text-orange-400">{inc.incident_code}</td>
                  <td className="py-2.5 px-4 font-semibold text-slate-200">Worker #{inc.track_id}</td>
                  <td className="py-2.5 px-4">{inc.violation_type}</td>
                  <td className="py-2.5 px-4">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      inc.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                      inc.severity === 'HIGH' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                      'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                    }`}>
                      {inc.severity}
                    </span>
                  </td>
                  <td className="py-2.5 px-4">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                      inc.status === 'RESOLVED' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                    }`}>
                      {inc.status}
                    </span>
                  </td>
                  <td className="py-2.5 px-4 text-slate-400">
                    {inc.started_at ? inc.started_at.replace('T', ' ').substring(0, 19) : '--'}
                  </td>
                  <td className="py-2.5 px-4 text-right flex items-center justify-end gap-2">
                    {inc.snapshot_path && (
                      <button
                        onClick={() => setSelectedSnapshot({ id: inc.id, code: inc.incident_code, type: inc.violation_type })}
                        className="px-2 py-1 rounded bg-[#1D263B] hover:bg-[#25324E] text-slate-200 text-[10px] font-bold flex items-center gap-1 transition-colors"
                      >
                        <ImageIcon className="w-3 h-3 text-orange-400" /> Evidence
                      </button>
                    )}
                    {inc.status === 'ACTIVE' && (
                      <button
                        onClick={() => handleResolve(inc.id)}
                        className="px-2 py-1 rounded bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 text-[10px] font-bold flex items-center gap-1"
                      >
                        <CheckCircle className="w-3 h-3" /> Resolve
                      </button>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Snapshot Preview Modal */}
      {selectedSnapshot && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#0F1420] border border-[#1D263B] rounded-xl max-w-md w-full p-4 space-y-3 font-mono">
            <div className="flex items-center justify-between border-b border-[#1D263B] pb-2">
              <div>
                <h4 className="font-bold text-slate-100 text-xs">{selectedSnapshot.code}</h4>
                <p className="text-[10px] text-red-400 font-bold">{selectedSnapshot.type}</p>
              </div>
              <button
                onClick={() => setSelectedSnapshot(null)}
                className="text-slate-400 hover:text-slate-200"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="aspect-square bg-black rounded-lg overflow-hidden border border-[#232F47] flex items-center justify-center">
              <img
                src={api.getSnapshotUrl(selectedSnapshot.id)}
                alt="Cropped Evidence"
                className="w-full h-full object-contain"
                onError={(e) => {
                  (e.target as HTMLElement).style.display = 'none';
                }}
              />
            </div>
            <p className="text-[10px] text-slate-400 text-center">
              Cropped visual evidence saved during temporal trigger latching.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
