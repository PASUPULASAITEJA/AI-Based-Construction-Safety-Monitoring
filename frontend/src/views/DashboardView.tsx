import React from 'react';
import { 
  Users, 
  ShieldCheck, 
  AlertTriangle, 
  BellRing, 
  CheckCircle2, 
  Percent, 
  ArrowUpRight, 
  Radio, 
  Clock, 
  ChevronRight,
  ShieldAlert
} from 'lucide-react';
import { KPIStats, Incident } from '../types';

interface DashboardViewProps {
  kpis: KPIStats | null;
  recentIncidents: Incident[];
  onNavigate: (view: string) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({ kpis, recentIncidents, onNavigate }) => {
  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* KPI Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {/* KPI 1 */}
        <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[11px] font-mono tracking-wider font-semibold">WORKERS SEEN</span>
            <Users className="w-4 h-4 text-blue-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-slate-100">{kpis?.observed_workers ?? '--'}</div>
            <p className="text-[10px] text-slate-400 mt-0.5">Tracked on site</p>
          </div>
        </div>

        {/* KPI 2 */}
        <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[11px] font-mono tracking-wider font-semibold">COMPLIANT</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-emerald-400">{kpis?.compliant_workers ?? '--'}</div>
            <p className="text-[10px] text-slate-400 mt-0.5">&gt;80% safety score</p>
          </div>
        </div>

        {/* KPI 3 */}
        <div className="bg-[#0E131F] border border-red-950/40 rounded-xl p-4 flex flex-col justify-between bg-gradient-to-b from-red-950/10 to-[#0E131F]">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[11px] font-mono tracking-wider font-semibold text-red-400">VIOLATIONS TODAY</span>
            <AlertTriangle className="w-4 h-4 text-red-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-red-400">{kpis?.violations_today ?? '--'}</div>
            <p className="text-[10px] text-slate-400 mt-0.5">Confirmed breaches</p>
          </div>
        </div>

        {/* KPI 4 */}
        <div className="bg-[#0E131F] border border-orange-950/40 rounded-xl p-4 flex flex-col justify-between bg-gradient-to-b from-orange-950/10 to-[#0E131F]">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[11px] font-mono tracking-wider font-semibold text-orange-400">OPEN ALERTS</span>
            <BellRing className="w-4 h-4 text-orange-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-orange-400">{kpis?.open_alerts ?? '--'}</div>
            <p className="text-[10px] text-slate-400 mt-0.5">Awaiting action</p>
          </div>
        </div>

        {/* KPI 5 */}
        <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[11px] font-mono tracking-wider font-semibold">RESOLVED</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-slate-100">{kpis?.resolved_incidents ?? '--'}</div>
            <p className="text-[10px] text-slate-400 mt-0.5">Mitigated cases</p>
          </div>
        </div>

        {/* KPI 6 */}
        <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[11px] font-mono tracking-wider font-semibold">COMPLIANCE</span>
            <Percent className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-amber-400">{kpis?.compliance_pct ?? '--'}%</div>
            <p className="text-[10px] text-slate-400 mt-0.5">Site safety index</p>
          </div>
        </div>
      </div>

      {/* Main Grid: Stream Preview & Operational Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Live Stream Banner (7 Cols) */}
        <div className="lg:col-span-7 bg-[#0E131F] border border-[#1E283D] rounded-xl overflow-hidden flex flex-col">
          <div className="p-4 border-b border-[#1E283D] flex items-center justify-between bg-[#0B0F19]">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <h3 className="text-xs font-bold text-slate-200 font-mono tracking-wide">PRIMARY CAMERA FEED (CAM-01)</h3>
            </div>
            <button 
              onClick={() => onNavigate('monitor')}
              className="text-xs text-orange-400 hover:text-orange-300 flex items-center gap-1 font-semibold"
            >
              Open Full Workstation <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
            <img 
              src="/api/monitor/feed" 
              alt="Live Construction Monitoring" 
              className="w-full h-full object-cover"
              onError={(e) => {
                // If feed fails or no camera, show aesthetic fallback
                (e.target as HTMLElement).style.display = 'none';
              }}
            />
            {/* Live Stream Overlay HUD */}
            <div className="absolute top-3 left-3 bg-black/70 backdrop-blur-sm border border-slate-700 rounded px-2.5 py-1 text-[11px] font-mono text-emerald-400 flex items-center gap-2">
              <Radio className="w-3 h-3 text-emerald-400 animate-pulse" />
              <span>FPS: {kpis?.system_fps ?? '15.0'} | Active Tracks</span>
            </div>
          </div>

          <div className="p-4 bg-[#0B0F19] flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>Zone Enforcement: Active (3 Geofences Configured)</span>
            <span>Hysteresis Filter: N=5 Frames</span>
          </div>
        </div>

        {/* Right Column: Quick Safety Actions & Guidelines (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-5">
            <h3 className="text-sm font-bold text-slate-200 mb-3 flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-orange-400" />
              Safety Operations Quick Actions
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <button 
                onClick={() => onNavigate('image-analysis')}
                className="p-3 rounded-lg bg-[#141B2D] hover:bg-[#1B243B] border border-[#232F47] text-left transition-all group"
              >
                <div className="text-xs font-bold text-slate-200 group-hover:text-orange-400 flex items-center justify-between">
                  Inspect Photo
                  <ChevronRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-orange-400" />
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Audit worker PPE from snapshot</p>
              </button>

              <button 
                onClick={() => onNavigate('video-analysis')}
                className="p-3 rounded-lg bg-[#141B2D] hover:bg-[#1B243B] border border-[#232F47] text-left transition-all group"
              >
                <div className="text-xs font-bold text-slate-200 group-hover:text-orange-400 flex items-center justify-between">
                  Process Video
                  <ChevronRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-orange-400" />
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Background AI batch job</p>
              </button>

              <button 
                onClick={() => onNavigate('sites-zones')}
                className="p-3 rounded-lg bg-[#141B2D] hover:bg-[#1B243B] border border-[#232F47] text-left transition-all group"
              >
                <div className="text-xs font-bold text-slate-200 group-hover:text-orange-400 flex items-center justify-between">
                  Polygon Zones
                  <ChevronRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-orange-400" />
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Draw & configure geofences</p>
              </button>

              <button 
                onClick={() => onNavigate('reports')}
                className="p-3 rounded-lg bg-[#141B2D] hover:bg-[#1B243B] border border-[#232F47] text-left transition-all group"
              >
                <div className="text-xs font-bold text-slate-200 group-hover:text-orange-400 flex items-center justify-between">
                  Audit Export
                  <ChevronRight className="w-3.5 h-3.5 text-slate-500 group-hover:text-orange-400" />
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Download CSV safety record</p>
              </button>
            </div>
          </div>

          <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-5">
            <h3 className="text-sm font-bold text-slate-200 mb-2 flex items-center gap-2">
              <Clock className="w-4 h-4 text-blue-400" />
              Engine Architecture Notes
            </h3>
            <div className="space-y-2 text-xs text-slate-400 leading-relaxed font-mono">
              <p>• <span className="text-slate-300 font-semibold">Detector:</span> YOLOv8 trained on 11 active PPE classes (Hardhat, Vest, Boots, Gloves, Goggles).</p>
              <p>• <span className="text-slate-300 font-semibold">Tracking:</span> ByteTrack assigns anonymous IDs (W-01, W-02) across temporal frames.</p>
              <p>• <span className="text-slate-300 font-semibold">Temporal Confirmation:</span> Eliminates frame flicker by verifying violations for 5 consecutive frames.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Incidents Table */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl overflow-hidden">
        <div className="p-4 border-b border-[#1E283D] flex items-center justify-between bg-[#0B0F19]">
          <h3 className="text-xs font-bold text-slate-200 font-mono tracking-wide">RECENT LOGGED INCIDENTS</h3>
          <button 
            onClick={() => onNavigate('incidents')}
            className="text-xs text-orange-400 hover:text-orange-300 font-semibold flex items-center gap-1"
          >
            View All Incidents <ArrowUpRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#080B12] text-slate-400 font-mono border-b border-[#1E283D]">
              <tr>
                <th className="py-3 px-4 font-semibold">CODE</th>
                <th className="py-3 px-4 font-semibold">VIOLATION TYPE</th>
                <th className="py-3 px-4 font-semibold">SEVERITY</th>
                <th className="py-3 px-4 font-semibold">STATUS</th>
                <th className="py-3 px-4 font-semibold">TIMESTAMP</th>
                <th className="py-3 px-4 font-semibold">CAMERA</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#172033] text-slate-300 font-mono">
              {recentIncidents.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center py-8 text-slate-500">
                    No safety incidents logged. Site is 100% compliant.
                  </td>
                </tr>
              ) : (
                recentIncidents.map((inc) => (
                  <tr key={inc.id} className="hover:bg-[#121A2C] transition-colors">
                    <td className="py-3 px-4 text-orange-400 font-semibold">{inc.incident_code}</td>
                    <td className="py-3 px-4 text-slate-200 font-medium">{inc.violation_type}</td>
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
                        inc.status === 'RESOLVED' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
                        inc.status === 'UNDER_REVIEW' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' :
                        'bg-red-500/20 text-red-400 border border-red-500/30'
                      }`}>
                        {inc.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-400">{inc.timestamp.replace('T', ' ').substring(0, 19)}</td>
                    <td className="py-3 px-4 text-slate-400">{inc.camera_name || 'CAM-01'}</td>
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
