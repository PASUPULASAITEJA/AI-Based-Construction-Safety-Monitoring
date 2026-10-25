import React from 'react';
import { Users, ShieldCheck, AlertTriangle, HardHat, Eye, Radio, Percent, ShieldAlert } from 'lucide-react';
import { KPIStats } from '../types';

interface MetricsOverviewProps {
  stats: KPIStats | null;
  isConnected: boolean;
}

export const MetricsOverview: React.FC<MetricsOverviewProps> = ({ stats, isConnected }) => {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3 mb-5 font-mono">
      {/* 1. Active Workers */}
      <div className="bg-[#0F1420] border border-[#1D263B] rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400">
          <span className="text-[10px] tracking-wider font-semibold">WORKERS</span>
          <Users className="w-3.5 h-3.5 text-blue-400" />
        </div>
        <div className="mt-1">
          <div className="text-2xl font-bold text-slate-100">{stats?.active_workers ?? 0}</div>
          <span className="text-[10px] text-slate-500">Tracked in scene</span>
        </div>
      </div>

      {/* 2. Compliant Workers */}
      <div className="bg-[#0F1420] border border-[#1D263B] rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400">
          <span className="text-[10px] tracking-wider font-semibold">COMPLIANT</span>
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
        </div>
        <div className="mt-1">
          <div className="text-2xl font-bold text-emerald-400">{stats?.compliant_workers ?? 0}</div>
          <span className="text-[10px] text-slate-500">100% PPE Score</span>
        </div>
      </div>

      {/* 3. Active Violations */}
      <div className="bg-[#0F1420] border border-red-950/40 rounded-xl p-3 flex flex-col justify-between bg-gradient-to-b from-red-950/20 to-[#0F1420]">
        <div className="flex items-center justify-between text-red-400">
          <span className="text-[10px] tracking-wider font-semibold">VIOLATIONS</span>
          <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
        </div>
        <div className="mt-1">
          <div className="text-2xl font-bold text-red-400">{stats?.active_violations ?? 0}</div>
          <span className="text-[10px] text-red-400/70">Latched incidents</span>
        </div>
      </div>

      {/* 4. Helmet Violations */}
      <div className="bg-[#0F1420] border border-[#1D263B] rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400">
          <span className="text-[10px] tracking-wider font-semibold">NO HELMET</span>
          <HardHat className="w-3.5 h-3.5 text-orange-400" />
        </div>
        <div className="mt-1">
          <div className="text-2xl font-bold text-orange-400">{stats?.helmet_violations ?? 0}</div>
          <span className="text-[10px] text-slate-500">Missing Hardhat</span>
        </div>
      </div>

      {/* 5. Vest Violations */}
      <div className="bg-[#0F1420] border border-[#1D263B] rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400">
          <span className="text-[10px] tracking-wider font-semibold">NO VEST</span>
          <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
        </div>
        <div className="mt-1">
          <div className="text-2xl font-bold text-amber-400">{stats?.vest_violations ?? 0}</div>
          <span className="text-[10px] text-slate-500">Missing High-Vis</span>
        </div>
      </div>

      {/* 6. Zone Intrusions */}
      <div className="bg-[#0F1420] border border-[#1D263B] rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400">
          <span className="text-[10px] tracking-wider font-semibold">ZONE BREACH</span>
          <Eye className="w-3.5 h-3.5 text-purple-400" />
        </div>
        <div className="mt-1">
          <div className="text-2xl font-bold text-purple-400">{stats?.zone_intrusions ?? 0}</div>
          <span className="text-[10px] text-slate-500">Restricted areas</span>
        </div>
      </div>

      {/* 7. Throughput & Telemetry FPS */}
      <div className="bg-[#0F1420] border border-[#1D263B] rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400">
          <span className="text-[10px] tracking-wider font-semibold">PIPELINE FPS</span>
          <Radio className={`w-3.5 h-3.5 ${isConnected ? 'text-emerald-400 animate-pulse' : 'text-slate-600'}`} />
        </div>
        <div className="mt-1">
          <div className="text-2xl font-bold text-slate-100">{stats?.fps ?? '0.0'}</div>
          <span className="text-[10px] text-slate-500">
            {isConnected ? 'Stream Active' : 'Disconnected'}
          </span>
        </div>
      </div>
    </div>
  );
};
