import React, { useState, useEffect } from 'react';
import { 
  FileVideo, 
  UploadCloud, 
  Download, 
  RefreshCw, 
  CheckCircle2, 
  AlertTriangle, 
  Clock, 
  Play 
} from 'lucide-react';
import { AnalysisJob } from '../types';

export const VideoAnalysisView: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<AnalysisJob | null>(null);
  const [recentJobs, setRecentJobs] = useState<AnalysisJob[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchRecentJobs = async () => {
    try {
      const res = await fetch('/api/jobs');
      if (res.ok) {
        const data = await res.json();
        setRecentJobs(data);
      }
    } catch (e) {
      // Non-blocking
    }
  };

  useEffect(() => {
    fetchRecentJobs();
  }, []);

  // Poll active job status
  useEffect(() => {
    if (!activeJobId) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${activeJobId}`);
        if (res.ok) {
          const data = await res.json();
          setJobStatus(data);
          if (data.status === 'COMPLETED' || data.status === 'FAILED') {
            clearInterval(interval);
            fetchRecentJobs();
          }
        }
      } catch (e) {
        console.error('Job polling error:', e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [activeJobId]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('site_id', '1');
    formData.append('camera_id', '1');

    try {
      const res = await fetch('/api/upload/video', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Upload failed: HTTP ${res.status}`);
      }

      const data = await res.json();
      setActiveJobId(data.job_id);
    } catch (err: any) {
      setError(err.message || 'Failed to submit video');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-orange-500/15 border border-orange-500/30 flex items-center justify-center">
            <FileVideo className="w-5 h-5 text-orange-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100 font-mono">ASYNCHRONOUS VIDEO PROCESSING PIPELINE</h3>
            <p className="text-xs text-slate-400 font-mono">ByteTrack multi-worker tracking & 5-frame temporal confirmation buffer</p>
          </div>
        </div>
      </div>

      {/* Upload Dropzone */}
      <div className="border-2 border-dashed border-[#26354D] hover:border-orange-500/50 bg-[#0A0E17] rounded-xl p-8 text-center transition-all cursor-pointer flex flex-col items-center justify-center min-h-[220px]">
        <input 
          type="file" 
          id="video-file-input" 
          accept="video/*" 
          onChange={handleFileChange} 
          className="hidden" 
        />
        <label htmlFor="video-file-input" className="cursor-pointer flex flex-col items-center">
          <div className="w-14 h-14 rounded-full bg-[#141B2D] border border-[#232F47] flex items-center justify-center mb-3">
            <UploadCloud className="w-7 h-7 text-orange-400" />
          </div>
          <p className="text-sm font-bold text-slate-200">
            {selectedFile ? selectedFile.name : 'Select or drop video footage here'}
          </p>
          <p className="text-xs text-slate-400 mt-1 font-mono">
            MP4, AVI, MOV up to 500MB • Background Asynchronous Worker Execution
          </p>
        </label>

        {selectedFile && !activeJobId && (
          <div className="mt-4">
            <button
              onClick={handleUpload}
              disabled={isUploading}
              className="px-5 py-2 rounded-lg bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white font-semibold text-xs font-mono transition-colors shadow-lg shadow-orange-500/25 flex items-center gap-2"
            >
              {isUploading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  Uploading Video...
                </>
              ) : (
                'Queue Background Analysis'
              )}
            </button>
          </div>
        )}

        {error && (
          <p className="text-xs text-red-400 font-mono mt-3">{error}</p>
        )}
      </div>

      {/* Active Job Progress Card */}
      {jobStatus && (
        <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between font-mono text-xs">
            <span className="font-bold text-slate-200">JOB #{jobStatus.job_id} ({jobStatus.original_filename})</span>
            <span className={`px-2 py-0.5 rounded font-bold ${
              jobStatus.status === 'COMPLETED' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
              jobStatus.status === 'PROCESSING' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30 animate-pulse' :
              'bg-blue-500/20 text-blue-400 border border-blue-500/30'
            }`}>
              {jobStatus.status}
            </span>
          </div>

          {/* Animated Progress Bar */}
          <div className="space-y-1.5">
            <div className="w-full bg-[#141B2D] h-3 rounded-full overflow-hidden border border-[#232F47]">
              <div 
                className="bg-gradient-to-r from-orange-500 to-amber-500 h-full transition-all duration-300"
                style={{ width: `${jobStatus.progress || 0}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
              <span>Frame {jobStatus.current_frame} / {jobStatus.total_frames || 'Calculating...'}</span>
              <span>{jobStatus.progress}% Complete</span>
            </div>
          </div>

          {/* Completed Video Player */}
          {jobStatus.status === 'COMPLETED' && jobStatus.output_video_path && (
            <div className="pt-3 border-t border-[#1E283D] space-y-4">
              <div className="aspect-video bg-black rounded-lg overflow-hidden border border-[#232F47]">
                <video 
                  controls 
                  src={jobStatus.output_video_path} 
                  className="w-full h-full object-contain"
                />
              </div>

              <div className="flex items-center justify-between font-mono text-xs">
                <span className="text-slate-300">
                  Total Violations Confirmed: <strong className="text-red-400">{jobStatus.violations_count}</strong>
                </span>
                <a
                  href={jobStatus.output_video_path}
                  download
                  className="px-3.5 py-1.5 rounded-lg bg-orange-500 hover:bg-orange-600 text-white font-semibold flex items-center gap-1.5 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download Annotated Video
                </a>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Recent Video Jobs Table */}
      <div className="bg-[#0E131F] border border-[#1E283D] rounded-xl overflow-hidden">
        <div className="p-4 border-b border-[#1E283D] flex items-center justify-between bg-[#0B0F19]">
          <h4 className="text-xs font-bold text-slate-200 font-mono">RECENT PROCESSING JOBS</h4>
          <button onClick={fetchRecentJobs} className="text-xs text-orange-400 hover:text-orange-300 flex items-center gap-1 font-mono">
            <RefreshCw className="w-3 h-3" /> Refresh
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#080B12] text-slate-400 border-b border-[#1E283D]">
              <tr>
                <th className="py-3 px-4 font-semibold">JOB ID</th>
                <th className="py-3 px-4 font-semibold">FILENAME</th>
                <th className="py-3 px-4 font-semibold">STATUS</th>
                <th className="py-3 px-4 font-semibold">VIOLATIONS</th>
                <th className="py-3 px-4 font-semibold">DURATION</th>
                <th className="py-3 px-4 font-semibold">ACTIONS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#172033] text-slate-300">
              {recentJobs.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center py-6 text-slate-500">No previous video jobs recorded.</td>
                </tr>
              ) : (
                recentJobs.map((j) => (
                  <tr key={j.id} className="hover:bg-[#121A2C] transition-colors">
                    <td className="py-3 px-4 text-orange-400 font-bold">{j.job_id}</td>
                    <td className="py-3 px-4 truncate max-w-[200px]">{j.original_filename}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        j.status === 'COMPLETED' ? 'bg-emerald-500/20 text-emerald-400' :
                        j.status === 'PROCESSING' ? 'bg-orange-500/20 text-orange-400 animate-pulse' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {j.status}
                      </span>
                    </td>
                    <td className="py-3 px-4">{j.violations_count || 0}</td>
                    <td className="py-3 px-4">{j.processing_duration_sec ? `${j.processing_duration_sec}s` : '--'}</td>
                    <td className="py-3 px-4">
                      {j.output_video_path ? (
                        <a 
                          href={j.output_video_path} 
                          download 
                          className="text-orange-400 hover:text-orange-300 flex items-center gap-1 font-semibold"
                        >
                          <Download className="w-3.5 h-3.5" /> Video
                        </a>
                      ) : (
                        <span className="text-slate-500">--</span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
