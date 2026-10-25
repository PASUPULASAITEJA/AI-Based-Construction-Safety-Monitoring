import React, { useState } from 'react';
import { 
  UploadCloud, 
  Camera, 
  Download, 
  ShieldCheck, 
  AlertTriangle, 
  Clock, 
  CheckCircle2, 
  ZoomIn,
  RefreshCw
} from 'lucide-react';

export const ImageAnalysisView: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('site_id', '1');
    formData.append('camera_id', '1');

    try {
      const res = await fetch('/api/upload/image', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Inference error: HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to process image');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-orange-500/15 border border-orange-500/30 flex items-center justify-center">
            <Camera className="w-5 h-5 text-orange-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 font-mono">OPTICAL PPE AUDIT & IMAGE INFERENCE</h3>
            <p className="text-xs text-slate-400 font-mono">Upload high-resolution construction photos for instant PPE compliance reasoning</p>
          </div>
        </div>
      </div>

      {/* Upload Dropzone */}
      {!result && (
        <div 
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          className="border-2 border-dashed border-[#26354D] hover:border-orange-500/50 bg-[#0A0E17] rounded-xl p-8 text-center transition-all cursor-pointer flex flex-col items-center justify-center min-h-[260px]"
        >
          <input 
            type="file" 
            id="image-file-input" 
            accept="image/*" 
            onChange={handleFileChange} 
            className="hidden" 
          />
          <label htmlFor="image-file-input" className="cursor-pointer flex flex-col items-center">
            <div className="w-14 h-14 rounded-full bg-[#141B2D] border border-[#232F47] flex items-center justify-center mb-3">
              <UploadCloud className="w-7 h-7 text-orange-400" />
            </div>
            <p className="text-sm font-bold text-slate-200">
              {selectedFile ? selectedFile.name : 'Drag & drop image here or click to browse'}
            </p>
            <p className="text-xs text-slate-400 mt-1 font-mono">
              Supports JPG, PNG, WEBP up to 25MB • Real YOLOv8 Analysis
            </p>
          </label>

          {selectedFile && (
            <div className="mt-5 flex items-center gap-3">
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                className="px-5 py-2 rounded-lg bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white font-semibold text-xs font-mono transition-colors shadow-lg shadow-orange-500/25 flex items-center gap-2"
              >
                {isAnalyzing ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Running Inference...
                  </>
                ) : (
                  'Run Safety Audit'
                )}
              </button>
            </div>
          )}

          {error && (
            <p className="text-xs text-red-400 font-mono mt-3">{error}</p>
          )}
        </div>
      )}

      {/* Results View: Side-by-Side Comparison & Table */}
      {result && (
        <div className="space-y-6">
          {/* Action Toolbar */}
          <div className="flex items-center justify-between bg-[#0E131F] border border-[#1E283D] rounded-xl p-3 px-4 font-mono text-xs">
            <div className="flex items-center gap-4 text-slate-300">
              <span>Latency: <strong className="text-orange-400">{result.inference_time_ms} ms</strong></span>
              <span>Workers: <strong className="text-slate-100">{result.summary?.persons_detected}</strong></span>
              <span>Violations: <strong className={result.violations?.length ? 'text-red-400' : 'text-emerald-400'}>{result.violations?.length}</strong></span>
            </div>
            <div className="flex items-center gap-2">
              <a
                href={result.annotated_image_url}
                download
                className="px-3 py-1.5 rounded-lg bg-[#141B2D] hover:bg-[#1B243B] border border-[#232F47] text-slate-200 text-xs font-semibold flex items-center gap-1.5 transition-colors"
              >
                <Download className="w-3.5 h-3.5 text-orange-400" />
                Download Annotated Image
              </a>
              <button
                onClick={() => { setResult(null); setSelectedFile(null); setPreviewUrl(null); }}
                className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white text-xs font-semibold"
              >
                Analyze Another Image
              </button>
            </div>
          </div>

          {/* 50/50 Split Images */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl overflow-hidden">
              <div className="p-3 bg-[#080B12] border-b border-[#1E283D] text-xs font-mono font-bold text-slate-300">
                ORIGINAL REFERENCE IMAGE
              </div>
              <div className="aspect-video bg-black flex items-center justify-center overflow-hidden">
                <img src={result.raw_image_url} alt="Original" className="w-full h-full object-contain" />
              </div>
            </div>

            <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl overflow-hidden">
              <div className="p-3 bg-[#080B12] border-b border-[#1E283D] text-xs font-mono font-bold text-orange-400 flex items-center justify-between">
                <span>YOLOV8 ANNOTATED OUTPUT (BOXES & HUD)</span>
                <span className="text-[10px] text-slate-400">1.8x Zoom Ready</span>
              </div>
              <div className="aspect-video bg-black flex items-center justify-center overflow-hidden">
                <img src={result.annotated_image_url} alt="Annotated" className="w-full h-full object-contain" />
              </div>
            </div>
          </div>

          {/* Detected Items & Violations Table */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Violations Summary */}
            <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4">
              <h4 className="text-xs font-bold text-slate-200 font-mono mb-3 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400" />
                CONFIRMED SAFETY VIOLATIONS ({result.violations?.length || 0})
              </h4>
              <div className="space-y-2 font-mono text-xs max-h-60 overflow-y-auto">
                {!result.violations || result.violations.length === 0 ? (
                  <div className="p-4 text-center text-emerald-400 bg-emerald-950/20 rounded-lg border border-emerald-500/30">
                    <ShieldCheck className="w-6 h-6 mx-auto mb-1 text-emerald-400" />
                    All observed workers are fully compliant with mandatory safety gear.
                  </div>
                ) : (
                  result.violations.map((v: any, idx: number) => (
                    <div key={idx} className="p-3 rounded-lg bg-[#141B2D] border border-red-950/50 flex items-center justify-between">
                      <div>
                        <p className="font-bold text-red-400">{v.violation_type}</p>
                        <p className="text-[11px] text-slate-400">{v.description}</p>
                      </div>
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-400 border border-red-500/30">
                        {v.severity}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Bounding Box Detections Log */}
            <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4">
              <h4 className="text-xs font-bold text-slate-200 font-mono mb-3">
                RAW YOLO BOUNDING BOXES ({result.detections?.length || 0})
              </h4>
              <div className="space-y-1.5 font-mono text-xs max-h-60 overflow-y-auto">
                {result.detections?.map((d: any, idx: number) => (
                  <div key={idx} className="p-2 rounded bg-[#141B2D] border border-[#232F47] flex items-center justify-between">
                    <span className="font-semibold text-slate-200">{d.class_name || d.category}</span>
                    <span className="text-orange-400 font-mono">{(d.confidence * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
