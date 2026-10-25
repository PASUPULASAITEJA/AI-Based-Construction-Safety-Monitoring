import React from 'react';
import { 
  ShieldAlert, 
  LayoutDashboard, 
  Video, 
  Camera, 
  FileVideo, 
  AlertTriangle, 
  AlertOctagon, 
  Users, 
  MapPin, 
  BarChart3, 
  FileText, 
  Cpu, 
  Activity, 
  Settings 
} from 'lucide-react';

interface SidebarProps {
  currentView: string;
  onNavigate: (view: string) => void;
  openAlertsCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentView, onNavigate, openAlertsCount }) => {
  const navSections = [
    {
      title: 'OPERATIONS',
      items: [
        { id: 'dashboard', label: 'Overview Dashboard', icon: LayoutDashboard },
        { id: 'monitor', label: 'Live Monitoring', icon: Video },
        { id: 'image-analysis', label: 'Image Analysis', icon: Camera },
        { id: 'video-analysis', label: 'Video Analysis', icon: FileVideo },
      ]
    },
    {
      title: 'SAFETY MANAGEMENT',
      items: [
        { id: 'incidents', label: 'Incidents Board', icon: AlertOctagon, badge: openAlertsCount > 0 ? openAlertsCount : undefined },
        { id: 'violations', label: 'Violations Log', icon: AlertTriangle },
        { id: 'workers', label: 'Worker Roster', icon: Users },
        { id: 'sites-zones', label: 'Sites & Geofences', icon: MapPin },
      ]
    },
    {
      title: 'INTELLIGENCE & AI',
      items: [
        { id: 'analytics', label: 'Safety Analytics', icon: BarChart3 },
        { id: 'reports', label: 'Audit Reports', icon: FileText },
        { id: 'model-research', label: 'AI Model & Research', icon: Cpu },
      ]
    },
    {
      title: 'ADMINISTRATION',
      items: [
        { id: 'system-health', label: 'System Health', icon: Activity },
      ]
    }
  ];

  return (
    <aside className="w-64 bg-[#0A0E17] border-r border-[#1B2436] flex flex-col h-screen select-none shrink-0">
      {/* Brand Header */}
      <div className="p-5 border-b border-[#1B2436] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
            <ShieldAlert className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-white text-base tracking-wide flex items-center gap-1.5">
              SiteGuard <span className="text-orange-500 text-xs px-1.5 py-0.5 rounded bg-orange-500/10 border border-orange-500/30">AI</span>
            </h1>
            <p className="text-[11px] text-slate-400 font-mono tracking-tight">Vision Safety Engine v2.4</p>
          </div>
        </div>
      </div>

      {/* Navigation List */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
        {navSections.map((sec) => (
          <div key={sec.title}>
            <div className="px-3 mb-2 text-[10px] font-bold text-slate-400 tracking-wider font-mono">
              {sec.title}
            </div>
            <div className="space-y-1">
              {sec.items.map((item) => {
                const Icon = item.icon;
                const isActive = currentView === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => onNavigate(item.id)}
                    className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                      isActive 
                        ? 'bg-orange-500/15 text-orange-400 border border-orange-500/30 font-semibold' 
                        : 'text-slate-300 hover:bg-[#121929] hover:text-slate-100'
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <Icon className={`w-4 h-4 ${isActive ? 'text-orange-400' : 'text-slate-400'}`} />
                      <span>{item.label}</span>
                    </div>
                    {item.badge && (
                      <span className="px-1.5 py-0.5 text-[10px] font-bold bg-red-500/20 text-red-400 border border-red-500/30 rounded-full">
                        {item.badge}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Operator Footer */}
      <div className="p-4 border-t border-[#1B2436] bg-[#070A11] flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-xs font-bold text-orange-400">
            SO
          </div>
          <div className="overflow-hidden">
            <p className="text-xs font-semibold text-slate-200 truncate">Safety Officer</p>
            <p className="text-[10px] text-emerald-400 flex items-center gap-1 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              ON-DUTY
            </p>
          </div>
        </div>
      </div>
    </aside>
  );
};
