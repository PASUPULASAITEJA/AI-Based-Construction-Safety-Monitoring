import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { DashboardView } from './views/DashboardView';
import { MonitoringView } from './views/MonitoringView';
import { ImageAnalysisView } from './views/ImageAnalysisView';
import { VideoAnalysisView } from './views/VideoAnalysisView';
import { IncidentsView } from './views/IncidentsView';
import { ViolationsView } from './views/ViolationsView';
import { WorkersView } from './views/WorkersView';
import { SitesZonesView } from './views/SitesZonesView';
import { AnalyticsView } from './views/AnalyticsView';
import { ReportsView } from './views/ReportsView';
import { ModelResearchView } from './views/ModelResearchView';
import { SystemHealthView } from './views/SystemHealthView';
import { KPIStats, Incident } from './types';

export function App() {
  const [currentView, setCurrentView] = useState<string>('dashboard');
  const [kpis, setKpis] = useState<KPIStats | null>(null);
  const [recentIncidents, setRecentIncidents] = useState<Incident[]>([]);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchDashboardData = async () => {
    setIsRefreshing(true);
    try {
      const res = await fetch('/api/dashboard/stats');
      if (res.ok) {
        const data = await res.json();
        setKpis(data.kpis);
        setRecentIncidents(data.recent_incidents || []);
      }
    } catch (e) {
      console.error('Failed to load dashboard data:', e);
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 10000);
    return () => clearInterval(interval);
  }, []);

  const getTitleForView = (view: string) => {
    switch (view) {
      case 'dashboard': return 'Overview Dashboard';
      case 'monitor': return 'Live Monitoring Workstation';
      case 'image-analysis': return 'Optical Image Inspection';
      case 'video-analysis': return 'Background Video Analysis';
      case 'incidents': return 'Incidents & Hazard Log';
      case 'violations': return 'Violation Records';
      case 'workers': return 'Worker Roster & Safety Scores';
      case 'sites-zones': return 'Sites & Geofence Boundaries';
      case 'analytics': return 'Longitudinal Analytics';
      case 'reports': return 'Safety Audit Reports';
      case 'model-research': return 'AI Architecture & Benchmark';
      case 'system-health': return 'Diagnostics & Telemetry';
      default: return 'Safety Operations';
    }
  };

  return (
    <div className="flex h-screen bg-[#07090E] overflow-hidden">
      {/* Sidebar Navigation */}
      <Sidebar 
        currentView={currentView} 
        onNavigate={setCurrentView} 
        openAlertsCount={kpis?.open_alerts ?? 0}
      />

      {/* Main Workspace */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        <Header 
          title={getTitleForView(currentView)} 
          subtitle="SiteGuard AI • Real-time Construction Safety Verification"
          onRefresh={fetchDashboardData}
          isRefreshing={isRefreshing}
        />

        <main className="flex-1 overflow-y-auto bg-[#07090E]">
          {currentView === 'dashboard' && (
            <DashboardView 
              kpis={kpis} 
              recentIncidents={recentIncidents} 
              onNavigate={setCurrentView} 
            />
          )}
          {currentView === 'monitor' && <MonitoringView />}
          {currentView === 'image-analysis' && <ImageAnalysisView />}
          {currentView === 'video-analysis' && <VideoAnalysisView />}
          {currentView === 'incidents' && <IncidentsView />}
          {currentView === 'violations' && <ViolationsView />}
          {currentView === 'workers' && <WorkersView />}
          {currentView === 'sites-zones' && <SitesZonesView />}
          {currentView === 'analytics' && <AnalyticsView />}
          {currentView === 'reports' && <ReportsView />}
          {currentView === 'model-research' && <ModelResearchView />}
          {currentView === 'system-health' && <SystemHealthView />}
        </main>
      </div>
    </div>
  );
}

export default App;
