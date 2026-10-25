import React, { useRef, useState } from 'react';
import { Camera, Upload, Radio, Play, AlertCircle, Maximize2 } from 'lucide-react';
import { api } from '../services/api';

interface VideoPlayerProps {
  frameSrc: string | null;
  fps: number;
  onUploadSuccess: () => void;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({ frameSrc, fps, onUploadSuccess }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [sourceType, setSourceType] = useState('webcam');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSourceChange = async (type: string) => {
    setSourceType(type);
    try {
      await api.setStreamSource(type);
    } catch (e) {
      console.error('Failed to change source:', e);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setIsUploading(true);
      try {
        await api.uploadVideo(file);
        setSourceType('file');
        onUploadSuccess();
      } catch (err) {
        console.error('Upload failed:', err);
      } finally {
        setIsUploading(false);
      }
    }
  };

  return (
    <div className="bg-[#0F1420] border border-[#1D263B] rounded-xl overflow-hidden flex flex-col">
      {/* Stream Controls Header */}
      <div className="p-3 bg-[#0B0F18] border-b border-[#1D263B] flex flex-wrap items-center justify-between gap-3 font-mono text-xs">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span className="text-slate-200 font-semibold tracking-wide">LIVE VISION HUD STREAM</span>
          <span className="text-slate-500">•</span>
          <span className="text-slate-400">{fps.toFixed(1)} FPS</span>
        </div>

        {/* Source Selector Toolbar */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => handleSourceChange('webcam')}
            className={`px-2.5 py-1 rounded flex items-center gap-1.5 transition-colors ${
              sourceType === 'webcam' ? 'bg-orange-500 text-white font-bold' : 'bg-[#151D2E] text-slate-400 hover:text-slate-200'
            }`}
          >
            <Camera className="w-3.5 h-3.5" /> Webcam
          </button>

          <button
            onClick={() => handleSourceChange('demo')}
            className={`px-2.5 py-1 rounded flex items-center gap-1.5 transition-colors ${
              sourceType === 'demo' ? 'bg-orange-500 text-white font-bold' : 'bg-[#151D2E] text-slate-400 hover:text-slate-200'
            }`}
          >
            <Play className="w-3.5 h-3.5" /> Simulator
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileUpload}
            className="hidden"
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className={`px-2.5 py-1 rounded flex items-center gap-1.5 transition-colors ${
              sourceType === 'file' ? 'bg-orange-500 text-white font-bold' : 'bg-[#151D2E] text-slate-400 hover:text-slate-200'
            }`}
          >
            <Upload className="w-3.5 h-3.5" />
            {isUploading ? 'Uploading...' : 'Upload MP4'}
          </button>
        </div>
      </div>

      {/* Primary Video Canvas Feed */}
      <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden select-none">
        {frameSrc ? (
          <img
            src={frameSrc}
            alt="Safety Stream"
            className="w-full h-full object-contain"
          />
        ) : (
          <div className="flex flex-col items-center justify-center text-slate-500 font-mono text-xs space-y-2">
            <Radio className="w-8 h-8 text-orange-400 animate-pulse" />
            <p>Connecting to Video Telemetry Feed...</p>
          </div>
        )}

        {/* Floating Metadata Indicator */}
        <div className="absolute top-3 left-3 bg-black/75 backdrop-blur-md border border-[#2A3750] rounded px-2.5 py-1 text-[11px] font-mono text-emerald-400 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
          <span>ByteTrack Active • Anchor: P_feet (Bottom-Center)</span>
        </div>
      </div>
    </div>
  );
};
