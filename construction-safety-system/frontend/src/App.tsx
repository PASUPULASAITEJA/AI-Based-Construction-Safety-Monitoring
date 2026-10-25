import React, { useState, useEffect } from 'react';
import { ShieldAlert, Radio, Activity, RefreshCw } from 'lucide-react';
import { useWebSocket } from './hooks/useWebSocket';
import { MetricsOverview } from './components/MetricsOverview';
import { VideoPlayer } from './components/VideoPlayer';
import { ZoneDrawer } from './components/ZoneDrawer';
import { AlertFeed } from './components/AlertFeed';
import { IncidentHistory } from './components/IncidentHistory';
import { Zone, Incident } from './types';
import { api } from './services/api';

export function App() {
  const { telemetry, isConnected } = useWebSocket('/ws/telemetry');
  const [zones, setZones] = useState<Zone[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);

  const loadData = async () => {
    try {
      const [z, inc] = await Promise.all([
        api.getZones(),
        api.getIncidents()
      ]);
      setZones(z);
      setIncidents(inc);
    } catch (e) {
      console.error('Error fetching dashboard data:', e);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#070A10] text-slate-100 flex flex-col font-sans selection:bg-orange-500/30">
      {/* Top Navigation Bar */}
      <header className="h-14 bg-[#0B0F18]/90 backdrop-blur-md border-b border-[#1D263B] px-6 flex items-center justify-between z-20 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
            <ShieldAlert className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-slate-100 text-sm tracking-wide font-mono flex items-center gap-2">
              SITEGUARD AI <span className="text-[10px] text-orange-400 px-1.5 py-0.2 rounded bg-orange-500/10 border border-orange-500/30">VISION SAFETY</span>
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-3 font-mono text-xs">
          <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-lg bg-[#111726] border border-[#1D263B] text-slate-300">
            <Activity className="w-3.5 h-3.5 text-orange-400" />
            <span>Site #01: Construction Terminal</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-emerald-950/30 border border-emerald-500/30 text-emerald-400">
            <Radio className={`w-3 h-3 ${isConnected ? 'animate-pulse text-emerald-400' : 'text-slate-500'}`} />
            <span>{isConnected ? 'Telemetry Online' : 'Connecting...'}</span>
          </div>

          <button
            onClick={loadData}
            className="p-1.5 rounded-lg bg-[#111726] hover:bg-[#1C263B] border border-[#1D263B] text-slate-400 hover:text-slate-200 transition-colors"
            title="Refresh Log"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </header>

      {/* Main Control Room Workspace */}
      <main className="flex-1 p-5 max-w-[1600px] w-full mx-auto space-y-5">
        {/* Top KPI Metrics Overview Cards */}
        <MetricsOverview
          stats={telemetry?.stats || null}
          isConnected={isConnected}
        />

        {/* 2-Column Responsive Control Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Left Column: Live Video Stream & Incident Log (8 Cols) */}
          <div className="lg:col-span-8 space-y-5">
            <VideoPlayer
              frameSrc={telemetry?.frame || null}
              fps={telemetry?.stats?.fps || 0}
              onUploadSuccess={loadData}
            />

            <IncidentHistory
              incidents={incidents}
              onRefresh={loadData}
            />
          </div>

          {/* Right Column: Zone Drawer & Real-Time Alert Ticker (4 Cols) */}
          <div className="lg:col-span-4 space-y-5">
            <ZoneDrawer
              zones={zones}
              onZonesUpdated={loadData}
            />

            <AlertFeed
              alerts={telemetry?.new_alerts || []}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
