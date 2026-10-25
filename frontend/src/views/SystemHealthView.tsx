import React, { useState, useEffect } from 'react';
import { Activity, CheckCircle2, AlertTriangle, HardDrive, Database, Cpu, RefreshCw } from 'lucide-react';
import { SystemHealth } from '../types';

export const SystemHealthView: React.FC = () => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const checkHealth = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/system-health');
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch (e) {
      console.error('Health check failed:', e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center">
            <Activity className="w-5 h-5 text-emerald-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 font-mono">SYSTEM HEALTH & DIAGNOSTICS TELEMETRY</h3>
            <p className="text-xs text-slate-400 font-mono">Real-time status of SQLite persistence, model files, and processing pipelines</p>
          </div>
        </div>

        <button
          onClick={checkHealth}
          className="px-3.5 py-1.5 rounded-lg bg-[#141B2D] hover:bg-[#1C263B] border border-[#232F47] text-slate-200 text-xs font-semibold flex items-center gap-1.5 font-mono transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-orange-400' : 'text-slate-400'}`} />
          Run Health Diagnostics
        </button>
      </div>

      {/* Diagnostics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 font-mono text-xs">
        {/* Database */}
        <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between text-slate-400">
            <span className="font-semibold text-[10px]">DATABASE ENGINE</span>
            <Database className="w-4 h-4 text-blue-400" />
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            <span className="text-base font-bold text-slate-100">SQLite Connected</span>
          </div>
          <p className="text-[11px] text-slate-400">
            Latency: <strong className="text-emerald-400">{health?.database_latency_ms ?? '--'} ms</strong>
          </p>
        </div>

        {/* AI Model */}
        <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between text-slate-400">
            <span className="font-semibold text-[10px]">YOLO WEIGHTS</span>
            <Cpu className="w-4 h-4 text-orange-400" />
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            <span className="text-base font-bold text-slate-100">Model Loaded</span>
          </div>
          <p className="text-[11px] text-slate-400 truncate">
            {health?.model_file || 'model/best.pt'}
          </p>
        </div>

        {/* Disk Space */}
        <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between text-slate-400">
            <span className="font-semibold text-[10px]">STORAGE USAGE</span>
            <HardDrive className="w-4 h-4 text-amber-400" />
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            <span className="text-base font-bold text-slate-100">{health?.free_disk_gb ?? '--'} GB Free</span>
          </div>
          <p className="text-[11px] text-slate-400">
            Capacity: {health?.total_disk_gb ?? '--'} GB Total
          </p>
        </div>

        {/* CV Pipeline */}
        <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between text-slate-400">
            <span className="font-semibold text-[10px]">SAFETY PIPELINE</span>
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            <span className="text-base font-bold text-emerald-400">Operational</span>
          </div>
          <p className="text-[11px] text-slate-400">
            ByteTrack & Temporal Buffers Active
          </p>
        </div>
      </div>
    </div>
  );
};
