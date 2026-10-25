import React, { useState, useEffect } from 'react';
import { MapPin, Plus, Trash2, Shield, AlertTriangle } from 'lucide-react';
import { Site, Zone } from '../types';

export const SitesZonesView: React.FC = () => {
  const [sites, setSites] = useState<Site[]>([]);
  const [zones, setZones] = useState<Zone[]>([]);

  const fetchData = async () => {
    try {
      const [sRes, zRes] = await Promise.all([
        fetch('/api/sites'),
        fetch('/api/zones')
      ]);
      if (sRes.ok) setSites(await sRes.json());
      if (zRes.ok) setZones(await zRes.json());
    } catch (e) {
      console.error('Error fetching sites/zones:', e);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleDeleteZone = async (id: number) => {
    try {
      const res = await fetch(`/api/zones/${id}`, { method: 'DELETE' });
      if (res.ok) fetchData();
    } catch (e) {
      console.error('Error deleting zone:', e);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-orange-500/15 border border-orange-500/30 flex items-center justify-center">
            <MapPin className="w-5 h-5 text-orange-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 font-mono">CONSTRUCTION SITES & SAFETY GEOFENCES</h3>
            <p className="text-xs text-slate-400 font-mono">Polygon Spatial Geofences with Bottom-Center Ground Anchor Reasoning</p>
          </div>
        </div>
      </div>

      {/* Sites Section */}
      <div className="space-y-3">
        <h4 className="text-xs font-bold text-slate-400 font-mono tracking-wider">REGISTERED SITES</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {sites.map((s) => (
            <div key={s.id} className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 space-y-2 font-mono text-xs">
              <div className="flex items-center justify-between">
                <span className="font-bold text-orange-400">{s.site_code}</span>
                <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/30">
                  ACTIVE
                </span>
              </div>
              <p className="text-sm font-bold text-slate-200">{s.name}</p>
              <p className="text-slate-400 text-[11px]">{s.location || 'Urban Metro Area'}</p>
              <div className="pt-2 border-t border-[#1E283D] text-[11px] text-slate-400">
                Safety Lead: <strong className="text-slate-300">{s.safety_officer || 'Marcus Vance'}</strong>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Zones Section */}
      <div className="space-y-3">
        <h4 className="text-xs font-bold text-slate-400 font-mono tracking-wider">CONFIGURED SAFETY ZONES</h4>
        <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-[#080B12] text-slate-400 border-b border-[#1E283D]">
                <tr>
                  <th className="py-3 px-4 font-semibold">ZONE NAME</th>
                  <th className="py-3 px-4 font-semibold">TYPE</th>
                  <th className="py-3 px-4 font-semibold">POLYGON VERTICES</th>
                  <th className="py-3 px-4 font-semibold">SITE</th>
                  <th className="py-3 px-4 font-semibold text-right">ACTION</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#172033] text-slate-300">
                {zones.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="text-center py-6 text-slate-500">
                      No safety zones defined. Use the Live Monitoring workstation to draw new polygon geofences.
                    </td>
                  </tr>
                ) : (
                  zones.map((z) => (
                    <tr key={z.id} className="hover:bg-[#121A2C] transition-colors">
                      <td className="py-3 px-4 font-bold text-slate-200">{z.name}</td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          z.zone_type === 'RESTRICTED' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                          z.zone_type === 'HAZARD' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                          'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        }`}>
                          {z.zone_type}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-slate-400 truncate max-w-[240px]">
                        {JSON.stringify(z.coordinates)}
                      </td>
                      <td className="py-3 px-4 text-slate-300">Site #01</td>
                      <td className="py-3 px-4 text-right">
                        <button
                          onClick={() => handleDeleteZone(z.id)}
                          className="p-1.5 rounded text-slate-400 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                          title="Delete Zone"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
