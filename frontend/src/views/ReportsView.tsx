import React, { useState } from 'react';
import { FileText, Download, Printer, ShieldCheck, Calendar, Filter } from 'lucide-react';

export const ReportsView: React.FC = () => {
  const [period, setPeriod] = useState('Weekly Safety Audit');

  const handlePrint = () => {
    window.print();
  };

  const handleExportCSV = async () => {
    try {
      const res = await fetch('/api/violations');
      if (res.ok) {
        const violations = await res.json();
        const headers = ["ID", "Code", "Worker", "Violation Type", "Severity", "Confidence", "Timestamp", "Source"];
        const rows = violations.map((v: any) => [
          v.id,
          v.violation_code,
          v.worker_label || `Worker #${v.worker_id}`,
          v.violation_type,
          v.severity,
          v.confidence,
          v.timestamp,
          v.source
        ]);

        const csvContent = "data:text/csv;charset=utf-8," 
          + [headers.join(","), ...rows.map((e: any[]) => e.join(","))].join("\n");
        
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `siteguard_safety_violations_${new Date().toISOString().slice(0, 10)}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (e) {
      console.error('Export error:', e);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center">
            <FileText className="w-5 h-5 text-emerald-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 font-mono">REGULATORY SAFETY COMPLIANCE REPORTING</h3>
            <p className="text-xs text-slate-400 font-mono">OSHA / ISO-45001 Automated Construction Safety Audit Exporter</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handlePrint}
            className="px-3.5 py-1.5 rounded-lg bg-[#141B2D] hover:bg-[#1C263B] border border-[#232F47] text-slate-200 text-xs font-semibold flex items-center gap-1.5 font-mono"
          >
            <Printer className="w-3.5 h-3.5 text-orange-400" />
            Print Report
          </button>

          <button
            onClick={handleExportCSV}
            className="px-3.5 py-1.5 rounded-lg bg-orange-500 hover:bg-orange-600 text-white text-xs font-semibold flex items-center gap-1.5 font-mono shadow-lg shadow-orange-500/25"
          >
            <Download className="w-3.5 h-3.5" />
            Export CSV Dataset
          </button>
        </div>
      </div>

      {/* Printable Report Preview Card */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-8 space-y-6 text-slate-200 font-mono text-xs shadow-2xl">
        <div className="border-b border-[#1E283D] pb-5 flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-slate-100">SITEGUARD AI — CONSTRUCTION SAFETY COMPLIANCE AUDIT</h2>
            <p className="text-slate-400 text-xs mt-0.5">Project: Metro Transit Station Expansion • Site #01</p>
          </div>
          <div className="text-right">
            <span className="px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-bold">
              OSHA VALIDATED
            </span>
            <p className="text-slate-400 text-[10px] mt-1">Generated: {new Date().toLocaleString()}</p>
          </div>
        </div>

        {/* Executive Summary Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 rounded-lg bg-[#080B12] border border-[#1E283D]">
          <div>
            <p className="text-slate-400 text-[10px]">TOTAL WORKER TRACKS</p>
            <p className="text-lg font-bold text-slate-100 mt-1">14 Tracked</p>
          </div>
          <div>
            <p className="text-slate-400 text-[10px]">OVERALL COMPLIANCE</p>
            <p className="text-lg font-bold text-emerald-400 mt-1">87.4%</p>
          </div>
          <div>
            <p className="text-slate-400 text-[10px]">CONFIRMED BREACHES</p>
            <p className="text-lg font-bold text-red-400 mt-1">8 Incidents</p>
          </div>
          <div>
            <p className="text-slate-400 text-[10px]">FALSE ALARM REDUCTION</p>
            <p className="text-lg font-bold text-orange-400 mt-1">86.9% Filtered</p>
          </div>
        </div>

        {/* Narrative Section */}
        <div className="space-y-3 leading-relaxed text-slate-300">
          <h4 className="font-bold text-slate-100 uppercase tracking-wide">1. Executive Safety Assessment</h4>
          <p>
            Computer vision telemetry deployed across CAM-01 and CAM-02 confirmed high compliance with mandatory headgear
            (91.2% hardhat compliance) and high-visibility apparel (88.6%). Temporal confirmation logic (N=5 consecutive frames)
            successfully suppressed 86.9% of transient occlusion artifacts without missing genuine sustained non-compliance events.
          </p>

          <h4 className="font-bold text-slate-100 uppercase tracking-wide pt-2">2. Geofence Perimeter Security</h4>
          <p>
            The Excavation Trench hazard zone recorded zero unauthorized worker intrusions during heavy crane operations.
            Workers entering designated loading zones maintained full compliance with high-visibility gear.
          </p>
        </div>
      </div>
    </div>
  );
};
