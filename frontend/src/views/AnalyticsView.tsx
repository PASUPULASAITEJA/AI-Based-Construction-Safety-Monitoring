import React, { useState, useEffect } from 'react';
import { BarChart3, Clock, AlertTriangle, TrendingUp, Calendar } from 'lucide-react';

export const AnalyticsView: React.FC = () => {
  const [range, setRange] = useState('7d');
  const [analyticsData, setAnalyticsData] = useState<any | null>(null);

  const fetchAnalytics = async () => {
    try {
      const res = await fetch(`/api/analytics?range=${range}`);
      if (res.ok) {
        const data = await res.json();
        setAnalyticsData(data);
      }
    } catch (e) {
      console.error('Error fetching analytics:', e);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, [range]);

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-orange-500/15 border border-orange-500/30 flex items-center justify-center">
            <BarChart3 className="w-5 h-5 text-orange-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 font-mono">LONGITUDINAL SAFETY ANALYTICS & INSIGHTS</h3>
            <p className="text-xs text-slate-400 font-mono">Aggregated temporal violation trends and hourly distribution</p>
          </div>
        </div>

        {/* Range Switcher */}
        <div className="flex items-center gap-1.5 bg-[#080B12] p-1 rounded-lg border border-[#1E283D] text-xs font-mono">
          {['today', '7d', '30d', 'all'].map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`px-3 py-1 rounded uppercase font-semibold transition-all ${
                range === r 
                  ? 'bg-orange-500 text-white' 
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* Main Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Timeline Chart */}
        <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-5 space-y-4">
          <h4 className="text-xs font-bold text-slate-200 font-mono flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-orange-400" />
            VIOLATION FREQUENCY TIMELINE
          </h4>

          <div className="h-48 flex items-end gap-2 pt-6 pb-2 px-2 border-b border-[#1E283D]">
            {analyticsData?.timeline?.map((item: any, idx: number) => {
              const maxVal = Math.max(...analyticsData.timeline.map((t: any) => t.count), 1);
              const heightPct = Math.min(100, Math.max(10, (item.count / maxVal) * 100));
              return (
                <div key={idx} className="flex-1 flex flex-col items-center gap-1 group">
                  <span className="text-[10px] text-orange-400 font-mono opacity-0 group-hover:opacity-100 transition-opacity">
                    {item.count}
                  </span>
                  <div 
                    className="w-full bg-gradient-to-t from-orange-500/60 to-orange-400 rounded-t transition-all duration-300 group-hover:from-orange-500 group-hover:to-amber-400"
                    style={{ height: `${heightPct}%` }}
                  />
                  <span className="text-[9px] text-slate-400 font-mono truncate w-full text-center">
                    {item.day.slice(5)}
                  </span>
                </div>
              );
            })}
          </div>
          <p className="text-[11px] text-slate-400 font-mono text-center">
            Daily confirmed safety violations across all monitored cameras
          </p>
        </div>

        {/* Hourly Distribution */}
        <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-5 space-y-4">
          <h4 className="text-xs font-bold text-slate-200 font-mono flex items-center gap-2">
            <Clock className="w-4 h-4 text-blue-400" />
            HOURLY RISK DISTRIBUTION (24H)
          </h4>

          <div className="h-48 flex items-end gap-1.5 pt-6 pb-2 px-2 border-b border-[#1E283D]">
            {analyticsData?.hourly_distribution?.map((item: any, idx: number) => {
              const maxVal = Math.max(...analyticsData.hourly_distribution.map((h: any) => h.count), 1);
              const heightPct = Math.min(100, Math.max(8, (item.count / maxVal) * 100));
              return (
                <div key={idx} className="flex-1 flex flex-col items-center gap-1 group">
                  <span className="text-[9px] text-blue-400 font-mono opacity-0 group-hover:opacity-100 transition-opacity">
                    {item.count}
                  </span>
                  <div 
                    className="w-full bg-gradient-to-t from-blue-600/60 to-blue-400 rounded-t transition-all group-hover:from-blue-500"
                    style={{ height: `${heightPct}%` }}
                  />
                  <span className="text-[8px] text-slate-400 font-mono">
                    {item.hour}h
                  </span>
                </div>
              );
            })}
          </div>
          <p className="text-[11px] text-slate-400 font-mono text-center">
            Shift-level pattern analysis highlighting peak hazard hours
          </p>
        </div>
      </div>

      {/* PPE Category Breakdown */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-5">
        <h4 className="text-xs font-bold text-slate-200 font-mono mb-4">
          VIOLATION DISTRIBUTION BY HAZARD CATEGORY
        </h4>
        <div className="space-y-3 font-mono text-xs">
          {analyticsData?.type_breakdown?.map((item: any, idx: number) => (
            <div key={idx} className="space-y-1">
              <div className="flex items-center justify-between text-slate-300">
                <span>{item.type}</span>
                <span className="font-bold text-orange-400">{item.count} incidents</span>
              </div>
              <div className="w-full bg-[#141B2D] h-2 rounded-full overflow-hidden border border-[#232F47]">
                <div 
                  className="bg-orange-500 h-full rounded-full"
                  style={{ width: `${Math.min(100, item.count * 8)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
