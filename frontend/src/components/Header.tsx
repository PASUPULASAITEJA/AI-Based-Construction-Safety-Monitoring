import React from 'react';
import { ShieldCheck, HardHat, RefreshCw, Radio } from 'lucide-react';

interface HeaderProps {
  title: string;
  subtitle?: string;
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

export const Header: React.FC<HeaderProps> = ({ title, subtitle, onRefresh, isRefreshing }) => {
  return (
    <header className="h-16 bg-[#0A0E17]/90 backdrop-blur-md border-b border-[#1B2436] px-6 flex items-center justify-between z-10 shrink-0">
      <div>
        <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
          {title}
        </h2>
        {subtitle && (
          <p className="text-xs text-slate-400 font-mono">{subtitle}</p>
        )}
      </div>

      <div className="flex items-center gap-4">
        {/* Active Site Indicator */}
        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#111827] border border-[#1F293D] text-xs">
          <HardHat className="w-3.5 h-3.5 text-orange-400" />
          <span className="text-slate-300 font-medium">Site #01: Metro Terminal Expansion</span>
        </div>

        {/* Engine Pipeline Status */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-950/30 border border-emerald-500/30 text-xs text-emerald-400 font-mono">
          <Radio className="w-3 h-3 text-emerald-400 animate-pulse" />
          <span>YOLOv8 & Tracking Live</span>
        </div>

        {/* Manual Refresh Button */}
        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="p-2 rounded-lg bg-[#111827] hover:bg-[#1C263B] border border-[#1F293D] text-slate-400 hover:text-slate-200 transition-colors"
            title="Refresh Data"
          >
            <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin text-orange-400' : ''}`} />
          </button>
        )}
      </div>
    </header>
  );
};
