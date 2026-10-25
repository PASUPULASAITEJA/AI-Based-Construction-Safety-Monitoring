import React from 'react';
import { BellRing, AlertTriangle, HardHat, ShieldAlert, Eye, Clock } from 'lucide-react';

interface AlertFeedProps {
  alerts: any[];
}

export const AlertFeed: React.FC<AlertFeedProps> = ({ alerts }) => {
  return (
    <div className="bg-[#0F1420] border border-[#1D263B] rounded-xl p-4 space-y-3 font-mono text-xs flex flex-col h-full">
      <div className="flex items-center justify-between border-b border-[#1D263B] pb-2.5">
        <div className="flex items-center gap-2 text-slate-200 font-bold">
          <BellRing className="w-4 h-4 text-orange-400 animate-bounce" />
          <span>REAL-TIME INCIDENT TICKER</span>
        </div>
        <span className="px-1.5 py-0.5 rounded text-[10px] bg-red-500/20 text-red-400 font-bold border border-red-500/30">
          {alerts.length} LATCHED
        </span>
      </div>

      <div className="space-y-2 flex-1 overflow-y-auto max-h-[360px] pr-1">
        {alerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-slate-500 text-center">
            <AlertTriangle className="w-8 h-8 mb-2 text-slate-600" />
            <p>Zero active safety violations detected.</p>
            <p className="text-[10px] text-slate-600 mt-1">All personnel compliant with PPE and perimeter rules.</p>
          </div>
        ) : (
          alerts.map((al, idx) => {
            const vType = al.violation_type || 'SAFETY_VIOLATION';
            const isHelmet = vType === 'NO_HELMET';
            const isVest = vType === 'NO_VEST';
            const isZone = vType === 'ZONE_INTRUSION';

            return (
              <div
                key={idx}
                className="p-3 rounded-lg bg-[#151D2E] border border-red-950/60 hover:border-red-500/50 transition-all space-y-1.5"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    {isHelmet && <HardHat className="w-4 h-4 text-orange-400" />}
                    {isVest && <ShieldAlert className="w-4 h-4 text-amber-400" />}
                    {isZone && <Eye className="w-4 h-4 text-red-400" />}
                    <span className="font-bold text-red-400">{vType}</span>
                  </div>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-300 font-bold">
                    {al.severity || 'HIGH'}
                  </span>
                </div>

                <p className="text-slate-300 text-[11px] leading-tight">
                  {al.description || `Worker ID #${al.track_id} safety breach`}
                </p>

                <div className="flex items-center justify-between text-[10px] text-slate-500 pt-1 border-t border-[#1F293D]">
                  <span>Track ID: <strong className="text-slate-300">#{al.track_id}</strong></span>
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    <span>Temporal N=22/30 Confirmed</span>
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
