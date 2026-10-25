import React, { useState, useEffect } from 'react';
import { Cpu, ShieldCheck, Zap, Activity, CheckCircle2, AlertTriangle, Layers } from 'lucide-react';

export const ModelResearchView: React.FC = () => {
  const [researchData, setResearchData] = useState<any | null>(null);
  const [robustnessData, setRobustnessData] = useState<any | null>(null);

  useEffect(() => {
    fetch('/api/research/evaluation')
      .then(res => res.json())
      .then(data => setResearchData(data))
      .catch(() => {});

    fetch('/api/research/robustness')
      .then(res => res.json())
      .then(data => setRobustnessData(data))
      .catch(() => {});
  }, []);

  const classes = [
    { id: 0, name: 'helmet', type: 'PPE' },
    { id: 1, name: 'gloves', type: 'PPE' },
    { id: 2, name: 'vest', type: 'PPE' },
    { id: 3, name: 'boots', type: 'PPE' },
    { id: 4, name: 'goggles', type: 'PPE' },
    { id: 5, name: 'none', type: 'Negative' },
    { id: 6, name: 'Person', type: 'Target' },
    { id: 7, name: 'no_helmet', type: 'Violation' },
    { id: 8, name: 'no_goggle', type: 'Violation' },
    { id: 9, name: 'no_gloves', type: 'Violation' },
    { id: 10, name: 'no_boots', type: 'Violation' },
  ];

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-orange-500/15 border border-orange-500/30 flex items-center justify-center">
            <Cpu className="w-5 h-5 text-orange-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 font-mono">AI MODEL ARCHITECTURE & RESEARCH BENCHMARKS</h3>
            <p className="text-xs text-slate-400 font-mono">
              Empirical Comparison: Baseline 1 (Frame-Level) vs Baseline 2 (Tracking) vs Context-Aware (Proposed)
            </p>
          </div>
        </div>
      </div>

      {/* Model Specs Card */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-5 space-y-4">
        <h4 className="text-xs font-bold text-slate-200 font-mono flex items-center gap-2">
          <Layers className="w-4 h-4 text-orange-400" />
          YOLOV8 CONVOLUTIONAL DETECTION ENGINE SPECIFICATION
        </h4>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono text-xs">
          <div className="p-3 bg-[#141B2D] border border-[#232F47] rounded-lg">
            <span className="text-slate-400 text-[10px]">WEIGHTS FILE</span>
            <p className="text-slate-100 font-bold mt-1">model/best.pt</p>
          </div>
          <div className="p-3 bg-[#141B2D] border border-[#232F47] rounded-lg">
            <span className="text-slate-400 text-[10px]">ACTIVE CLASSES</span>
            <p className="text-orange-400 font-bold mt-1">11 PPE Categories</p>
          </div>
          <div className="p-3 bg-[#141B2D] border border-[#232F47] rounded-lg">
            <span className="text-slate-400 text-[10px]">INPUT RESOLUTION</span>
            <p className="text-slate-100 font-bold mt-1">640 x 640 px</p>
          </div>
          <div className="p-3 bg-[#141B2D] border border-[#232F47] rounded-lg">
            <span className="text-slate-400 text-[10px]">INFERENCE ACCELERATOR</span>
            <p className="text-emerald-400 font-bold mt-1">CPU / CUDA Ready</p>
          </div>
        </div>

        {/* Classes Badges */}
        <div>
          <span className="text-[11px] font-mono text-slate-400 block mb-2">Supported Class Taxonomy:</span>
          <div className="flex flex-wrap gap-1.5 font-mono text-xs">
            {classes.map((c) => (
              <span key={c.id} className={`px-2 py-1 rounded text-[11px] font-semibold border ${
                c.type === 'PPE' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' :
                c.type === 'Target' ? 'bg-blue-500/10 text-blue-400 border-blue-500/30' :
                c.type === 'Violation' ? 'bg-red-500/10 text-red-400 border-red-500/30' :
                'bg-slate-800 text-slate-400 border-slate-700'
              }`}>
                {c.id}: {c.name}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Scientific Research Evaluation Table */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl overflow-hidden">
        <div className="p-4 border-b border-[#1E283D] bg-[#0B0F19]">
          <h4 className="text-xs font-bold text-slate-200 font-mono tracking-wide">
            EMPIRICAL SYSTEM EVALUATION (DATASET VALIDATION TEST SEQUENCE)
          </h4>
          <p className="text-[11px] text-slate-400 font-mono mt-0.5">
            Measured against 60 ground-truth construction validation frames
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#080B12] text-slate-400 border-b border-[#1E283D]">
              <tr>
                <th className="py-3 px-4 font-semibold">CONFIGURATION</th>
                <th className="py-3 px-4 font-semibold">TRACKING</th>
                <th className="py-3 px-4 font-semibold">GEOFENCING</th>
                <th className="py-3 px-4 font-semibold">TEMPORAL BUFFER</th>
                <th className="py-3 px-4 font-semibold">RAW ALERTS / INCIDENTS</th>
                <th className="py-3 px-4 font-semibold">THROUGHPUT</th>
                <th className="py-3 px-4 font-semibold">NOISE REDUCTION</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#172033] text-slate-300">
              <tr className="hover:bg-[#121A2C]">
                <td className="py-3.5 px-4 font-bold text-slate-300">Baseline 1 (Frame-Level)</td>
                <td className="py-3.5 px-4 text-slate-500">Disabled</td>
                <td className="py-3.5 px-4 text-slate-500">Disabled</td>
                <td className="py-3.5 px-4 text-slate-500">N = 1 (Immediate)</td>
                <td className="py-3.5 px-4 text-red-400 font-bold">61 Alerts (High Noise)</td>
                <td className="py-3.5 px-4">11.3 FPS</td>
                <td className="py-3.5 px-4 text-slate-500">0.0% (Reference)</td>
              </tr>
              <tr className="hover:bg-[#121A2C]">
                <td className="py-3.5 px-4 font-bold text-slate-300">Baseline 2 (Detection + Tracking)</td>
                <td className="py-3.5 px-4 text-blue-400 font-semibold">ByteTrack (Active)</td>
                <td className="py-3.5 px-4 text-slate-500">Disabled</td>
                <td className="py-3.5 px-4 text-slate-500">N = 1</td>
                <td className="py-3.5 px-4 text-red-400 font-bold">61 Alerts</td>
                <td className="py-3.5 px-4">15.7 FPS</td>
                <td className="py-3.5 px-4 text-slate-500">0.0%</td>
              </tr>
              <tr className="bg-orange-950/15 border-l-4 border-l-orange-500">
                <td className="py-3.5 px-4 font-bold text-orange-400">Proposed Context-Aware Framework</td>
                <td className="py-3.5 px-4 text-blue-400 font-semibold">ByteTrack (Active)</td>
                <td className="py-3.5 px-4 text-emerald-400 font-semibold">Polygon Geofences</td>
                <td className="py-3.5 px-4 text-orange-400 font-semibold">N = 5 Frames + Cooldown</td>
                <td className="py-3.5 px-4 text-emerald-400 font-bold">8 Confirmed Incidents</td>
                <td className="py-3.5 px-4 font-bold">16.2 FPS</td>
                <td className="py-3.5 px-4 text-emerald-400 font-bold">86.9% Filtered</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Environmental Robustness Degradation Evaluation */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl overflow-hidden">
        <div className="p-4 border-b border-[#1E283D] bg-[#0B0F19]">
          <h4 className="text-xs font-bold text-slate-200 font-mono tracking-wide">
            ENVIRONMENTAL ROBUSTNESS & DEGRADATION BENCHMARK (DELIVERABLE 10)
          </h4>
          <p className="text-[11px] text-slate-400 font-mono mt-0.5">
            Evaluates detector and temporal confirmation resilience under harsh construction site environmental stress
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#080B12] text-slate-400 border-b border-[#1E283D]">
              <tr>
                <th className="py-3 px-4 font-semibold">CONDITION</th>
                <th className="py-3 px-4 font-semibold">PHYSICAL SIMULATION</th>
                <th className="py-3 px-4 font-semibold">PERSONS DETECTED</th>
                <th className="py-3 px-4 font-semibold">RETENTION RATE</th>
                <th className="py-3 px-4 font-semibold">CONFIRMED INCIDENTS</th>
                <th className="py-3 px-4 font-semibold">MEAN FPS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#172033] text-slate-300">
              <tr className="hover:bg-[#121A2C]">
                <td className="py-3 px-4 font-bold text-slate-200">Standard Lighting (Clean)</td>
                <td className="py-3 px-4 text-slate-400">Unperturbed high-contrast reference</td>
                <td className="py-3 px-4">46</td>
                <td className="py-3 px-4 text-emerald-400 font-bold">100.0% (Baseline)</td>
                <td className="py-3 px-4 text-orange-400">4</td>
                <td className="py-3 px-4">7.54 FPS</td>
              </tr>
              <tr className="hover:bg-[#121A2C]">
                <td className="py-3 px-4 font-bold text-slate-200">Low Lighting & Dusk</td>
                <td className="py-3 px-4 text-slate-400">Gamma 1.8, 45% Brightness reduction</td>
                <td className="py-3 px-4">42</td>
                <td className="py-3 px-4 text-emerald-400 font-bold">91.3%</td>
                <td className="py-3 px-4 text-orange-400">4</td>
                <td className="py-3 px-4">12.28 FPS</td>
              </tr>
              <tr className="hover:bg-[#121A2C]">
                <td className="py-3 px-4 font-bold text-slate-200">Solar Glare & Overexposure</td>
                <td className="py-3 px-4 text-slate-400">1.4x Value Clip (Direct Noon Sunlight)</td>
                <td className="py-3 px-4">47</td>
                <td className="py-3 px-4 text-emerald-400 font-bold">102.2%</td>
                <td className="py-3 px-4 text-orange-400">4</td>
                <td className="py-3 px-4">15.14 FPS</td>
              </tr>
              <tr className="hover:bg-[#121A2C]">
                <td className="py-3 px-4 font-bold text-slate-200">Motion Blur & Vibration</td>
                <td className="py-3 px-4 text-slate-400">11px Linear kernel (Crane vibration)</td>
                <td className="py-3 px-4">44</td>
                <td className="py-3 px-4 text-emerald-400 font-bold">95.7%</td>
                <td className="py-3 px-4 text-orange-400">3</td>
                <td className="py-3 px-4">14.09 FPS</td>
              </tr>
              <tr className="hover:bg-[#121A2C]">
                <td className="py-3 px-4 font-bold text-slate-200">Structural Occlusion</td>
                <td className="py-3 px-4 text-slate-400">Rebar & Scaffolding vertical pipe masks</td>
                <td className="py-3 px-4">41</td>
                <td className="py-3 px-4 text-emerald-400 font-bold">89.1%</td>
                <td className="py-3 px-4 text-orange-400">3</td>
                <td className="py-3 px-4">15.09 FPS</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
