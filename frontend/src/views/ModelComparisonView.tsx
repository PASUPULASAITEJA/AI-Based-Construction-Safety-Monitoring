import React, { useState, useEffect } from 'react';
import { 
  Columns, 
  UploadCloud, 
  Download, 
  RefreshCw, 
  X, 
  CheckSquare, 
  Square, 
  ZoomIn, 
  Layers, 
  Zap, 
  Cpu, 
  Sliders, 
  AlertCircle,
  FileImage,
  ExternalLink
} from 'lucide-react';
import { ComparableModel, ModelComparisonResponse, ModelComparisonTile } from '../types';

export const ModelComparisonView: React.FC = () => {
  const [availableModels, setAvailableModels] = useState<ComparableModel[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>(['YOLOv8-n', 'YOLOv8-m', 'YOLOv8-x', 'YOLOv9-e']);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [filePreviews, setFilePreviews] = useState<string[]>([]);
  const [conf, setConf] = useState<number>(0.25);
  const [iou, setIou] = useState<number>(0.45);
  const [imgsz, setImgsz] = useState<number>(640);
  
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [comparisonResult, setComparisonResult] = useState<ModelComparisonResponse | null>(null);
  const [enlargedTile, setEnlargedTile] = useState<{ tile: ModelComparisonTile; filename: string } | null>(null);

  // Fetch registered models on mount
  useEffect(() => {
    fetch('/api/compare/models')
      .then((res) => res.json())
      .then((data) => {
        if (data.models && Array.isArray(data.models)) {
          setAvailableModels(data.models);
          setSelectedModels(data.models.map((m: ComparableModel) => m.name));
        }
      })
      .catch((err) => {
        console.error('Failed to load registered models:', err);
      });
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files).slice(0, 8);
      setSelectedFiles(newFiles);
      setFilePreviews(newFiles.map((f) => URL.createObjectURL(f)));
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files) {
      const newFiles = Array.from(e.dataTransfer.files).slice(0, 8);
      setSelectedFiles(newFiles);
      setFilePreviews(newFiles.map((f) => URL.createObjectURL(f)));
      setError(null);
    }
  };

  const removeFile = (idx: number) => {
    const updatedFiles = selectedFiles.filter((_, i) => i !== idx);
    const updatedPreviews = filePreviews.filter((_, i) => i !== idx);
    setSelectedFiles(updatedFiles);
    setFilePreviews(updatedPreviews);
  };

  const toggleModel = (name: string) => {
    if (selectedModels.includes(name)) {
      if (selectedModels.length === 1) return; // Keep at least one
      setSelectedModels(selectedModels.filter((m) => m !== name));
    } else {
      setSelectedModels([...selectedModels, name]);
    }
  };

  const handleRunComparison = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least 1 image (up to 8) to compare.');
      return;
    }
    if (selectedModels.length === 0) {
      setError('Please select at least one YOLO model.');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('models', JSON.stringify(selectedModels));
    formData.append('conf', conf.toString());
    formData.append('iou', iou.toString());
    formData.append('imgsz', imgsz.toString());
    formData.append('tile_width', '480');

    try {
      const res = await fetch('/api/compare/images', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned status ${res.status}`);
      }

      const data: ModelComparisonResponse = await res.json();
      setComparisonResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to complete model comparison.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-orange-500/15 border border-orange-500/30 flex items-center justify-center">
            <Columns className="w-5 h-5 text-orange-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 font-mono tracking-wide">
              MULTI-MODEL EMPIRICAL COMPARISON (SH17 FIG. 5)
            </h3>
            <p className="text-xs text-slate-400 font-mono">
              Synchronous side-by-side inference across YOLO architectures with clean paper-style annotations
            </p>
          </div>
        </div>

        {comparisonResult && (
          <div className="flex items-center gap-2">
            <a
              href={comparisonResult.grid_image_url}
              download
              className="px-3.5 py-1.5 rounded-lg bg-orange-500 hover:bg-orange-600 text-white font-mono text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-lg shadow-orange-500/25"
            >
              <Download className="w-3.5 h-3.5" />
              Download Grid PNG
            </a>
            <button
              onClick={() => {
                setComparisonResult(null);
              }}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-mono text-xs font-semibold transition-colors"
            >
              New Comparison
            </button>
          </div>
        )}
      </div>

      {/* Configuration & Upload Panel (if no active result) */}
      {!comparisonResult && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left 2 Cols: Multi-Image Upload */}
          <div className="lg:col-span-2 bg-[#0E131F] border border-[#1E283D] rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold text-slate-200 font-mono flex items-center gap-2">
                <FileImage className="w-4 h-4 text-orange-400" />
                SELECT INPUT IMAGES (1 TO 8 PHOTOS)
              </h4>
              <span className="text-[11px] font-mono text-slate-400">
                {selectedFiles.length} / 8 Selected
              </span>
            </div>

            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className="border-2 border-dashed border-[#232F47] hover:border-orange-500/50 bg-[#0A0E17] rounded-xl p-6 text-center transition-all cursor-pointer flex flex-col items-center justify-center min-h-[170px]"
            >
              <input
                type="file"
                id="multi-image-input"
                accept="image/*"
                multiple
                onChange={handleFileChange}
                className="hidden"
              />
              <label htmlFor="multi-image-input" className="cursor-pointer flex flex-col items-center">
                <div className="w-11 h-11 rounded-full bg-[#141B2D] border border-[#232F47] flex items-center justify-center mb-2">
                  <UploadCloud className="w-5 h-5 text-orange-400" />
                </div>
                <p className="text-xs font-bold text-slate-200 font-mono">
                  Drag & drop multiple construction photos or click to browse
                </p>
                <p className="text-[10px] text-slate-400 mt-1 font-mono">
                  Accepts JPG, PNG, WEBP (each image run through all selected models)
                </p>
              </label>
            </div>

            {/* Thumbnail Preview Strip */}
            {filePreviews.length > 0 && (
              <div className="grid grid-cols-4 sm:grid-cols-6 gap-2 pt-2">
                {filePreviews.map((url, i) => (
                  <div key={i} className="relative group aspect-square rounded-lg overflow-hidden border border-[#232F47] bg-black">
                    <img src={url} alt={`Preview ${i}`} className="w-full h-full object-cover" />
                    <button
                      onClick={() => removeFile(i)}
                      className="absolute top-1 right-1 p-1 bg-red-600/80 hover:bg-red-600 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <X className="w-3 h-3" />
                    </button>
                    <span className="absolute bottom-1 left-1 px-1 rounded bg-black/70 text-[9px] font-mono text-slate-300">
                      ({i + 1})
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Right Col: Model Checkboxes & Hyperparameters */}
          <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-5 space-y-5">
            <h4 className="text-xs font-bold text-slate-200 font-mono flex items-center gap-2">
              <Layers className="w-4 h-4 text-orange-400" />
              YOLO ARCHITECTURES (FIG. 5)
            </h4>

            {/* Model Checkbox Cards */}
            <div className="space-y-2">
              {availableModels.map((m) => {
                const isChecked = selectedModels.includes(m.name);
                return (
                  <div
                    key={m.name}
                    onClick={() => toggleModel(m.name)}
                    className={`p-2.5 rounded-lg border cursor-pointer transition-all flex items-center justify-between font-mono text-xs ${
                      isChecked
                        ? 'bg-orange-500/10 border-orange-500/40 text-slate-100'
                        : 'bg-[#141B2D] border-[#232F47] text-slate-400 hover:border-slate-600'
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      {isChecked ? (
                        <CheckSquare className="w-4 h-4 text-orange-400" />
                      ) : (
                        <Square className="w-4 h-4 text-slate-500" />
                      )}
                      <div>
                        <span className="font-bold">{m.name}</span>
                        <span className="text-[10px] text-slate-400 ml-1.5">({m.params_m}M)</span>
                      </div>
                    </div>

                    <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                      m.weights_exist
                        ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                        : 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
                    }`}>
                      {m.weights_exist ? 'Weights Ready' : 'Base Pretrained'}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Sliders */}
            <div className="space-y-3 font-mono text-xs pt-2 border-t border-[#1E283D]">
              <div>
                <div className="flex justify-between text-slate-400 text-[11px] mb-1">
                  <span>Confidence Threshold:</span>
                  <span className="text-orange-400 font-bold">{conf.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.90"
                  step="0.05"
                  value={conf}
                  onChange={(e) => setConf(parseFloat(e.target.value))}
                  className="w-full accent-orange-500 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-400 text-[11px] mb-1">
                  <span>NMS IoU Threshold:</span>
                  <span className="text-orange-400 font-bold">{iou.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.20"
                  max="0.80"
                  step="0.05"
                  value={iou}
                  onChange={(e) => setIou(parseFloat(e.target.value))}
                  className="w-full accent-orange-500 cursor-pointer"
                />
              </div>
            </div>

            {/* Action Button */}
            <button
              onClick={handleRunComparison}
              disabled={isLoading || selectedFiles.length === 0}
              className="w-full py-2.5 rounded-lg bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white font-mono font-bold text-xs transition-colors shadow-lg shadow-orange-500/25 flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Running Multi-Model Inference...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Run Comparison ({selectedModels.length} Models × {selectedFiles.length} Imgs)
                </>
              )}
            </button>

            {error && (
              <div className="p-3 bg-red-950/30 border border-red-500/40 rounded-lg text-xs font-mono text-red-400 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Render Results View: Publication-Grade SH17 Table Grid */}
      {comparisonResult && (
        <div className="space-y-6">
          {/* Summary Benchmark KPI Chips */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {comparisonResult.stats.map((s) => (
              <div key={s.model} className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-3.5 font-mono text-xs">
                <div className="flex items-center justify-between text-slate-400 text-[11px] mb-1">
                  <span className="font-bold text-slate-200">{s.model}</span>
                  <span className="text-orange-400">{s.params_m}M Params</span>
                </div>
                <div className="flex items-baseline justify-between mt-2">
                  <div>
                    <span className="text-[10px] text-slate-500 block">LATENCY</span>
                    <span className="text-sm font-bold text-emerald-400">{s.avg_inference_ms.toFixed(1)} ms</span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-slate-500 block">AVG DETECTIONS</span>
                    <span className="text-sm font-bold text-slate-200">{s.avg_detections.toFixed(1)} / img</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Master Comparison Table Grid */}
          <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl overflow-hidden shadow-2xl">
            <div className="p-4 border-b border-[#1E283D] bg-[#0A0E17] flex items-center justify-between">
              <div>
                <h4 className="text-xs font-bold text-slate-200 font-mono tracking-wide">
                  SYNCHRONOUS MULTI-MODEL DETECTION MATRIX
                </h4>
                <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                  Click any tile to enlarge • Clean per-class colored bounding boxes with labels
                </p>
              </div>
              <span className="text-xs font-mono text-slate-400">
                {comparisonResult.images_processed} Image Rows × {comparisonResult.model_names.length} Model Columns
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                {/* Column Headers */}
                <thead>
                  <tr className="bg-[#070A11] border-b border-[#1E283D] font-mono text-xs text-slate-300">
                    <th className="p-3 w-12 text-center text-slate-400 font-semibold border-r border-[#1E283D]">#</th>
                    {comparisonResult.model_names.map((m) => (
                      <th key={m} className="p-3 text-center border-r border-[#1E283D] last:border-r-0">
                        <div className="font-bold text-orange-400 text-sm">{m}</div>
                        <div className="text-[10px] text-slate-400 font-normal">
                          {availableModels.find((x) => x.name === m)?.params_m || ''}M Parameters
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>

                {/* Rows */}
                <tbody className="divide-y divide-[#1E283D]">
                  {comparisonResult.rows.map((row) => (
                    <tr key={row.row_index} className="hover:bg-[#121929] transition-colors">
                      {/* Row Label (1), (2)... */}
                      <td className="p-3 text-center font-mono text-xs font-bold text-slate-400 border-r border-[#1E283D] bg-[#0A0E17]">
                        ({row.row_index})
                      </td>

                      {/* Model Tiles */}
                      {row.tiles.map((tile) => (
                        <td key={tile.model} className="p-3 border-r border-[#1E283D] last:border-r-0 align-top">
                          <div className="space-y-2">
                            {/* Image Tile */}
                            <div
                              onClick={() => setEnlargedTile({ tile, filename: row.filename })}
                              className="relative group aspect-[4/3] bg-black rounded-lg overflow-hidden border border-[#232F47] hover:border-orange-500/70 cursor-pointer transition-all shadow-md"
                            >
                              <img
                                src={tile.image_url}
                                alt={`${tile.model} inference`}
                                className="w-full h-full object-contain"
                              />
                              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-1.5 text-white font-mono text-xs font-semibold">
                                <ZoomIn className="w-4 h-4 text-orange-400" />
                                <span>Enlarge</span>
                              </div>
                            </div>

                            {/* Caption Chip */}
                            <div className="flex items-center justify-between px-2 py-1 bg-[#141B2D] border border-[#232F47] rounded font-mono text-[11px]">
                              <span className="text-slate-300 font-semibold">{tile.detections} detections</span>
                              <span className="text-emerald-400">{tile.inference_ms.toFixed(1)} ms</span>
                            </div>
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Lightbox / Enlarged Tile Modal */}
      {enlargedTile && (
        <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col shadow-2xl">
            <div className="p-3.5 bg-[#070A11] border-b border-[#1E283D] flex items-center justify-between font-mono text-xs">
              <div className="flex items-center gap-3">
                <span className="font-bold text-orange-400 text-sm">{enlargedTile.tile.model}</span>
                <span className="text-slate-400">· {enlargedTile.tile.detections} detections</span>
                <span className="text-emerald-400">· {enlargedTile.tile.inference_ms.toFixed(1)} ms</span>
                <span className="text-slate-500">({enlargedTile.filename})</span>
              </div>
              <button
                onClick={() => setEnlargedTile(null)}
                className="p-1.5 rounded-lg bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="flex-1 bg-black p-2 flex items-center justify-center overflow-auto">
              <img
                src={enlargedTile.tile.image_url}
                alt="Enlarged inference preview"
                className="max-h-[75vh] max-w-full object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
