"use client";

import { useState, useRef, useEffect } from 'react';
import Button from '@/components/ui/Button';

// Backend Orchestrator Endpoint
const ANALYZE_VIDEO_URL = 'http://localhost:3005/api/analyze/video';
const EXTRACTION_FPS = 25;

// Helper to keep sample JSON readable by collapsing 512-dim signature arrays
const formatSampleJson = (frame) => {
  if (!frame) return "";
  return JSON.stringify(
    frame,
    (key, value) => {
      if (key === 'signature' && Array.isArray(value)) {
        return `[${value.length} floats]`;
      }
      return value;
    },
    2
  );
};

export default function BackendTrackerTest() {
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // Pipeline States
  const [status, setStatus] = useState("idle"); // idle, processing, done, error
  const [progressMsg, setProgressMsg] = useState("");
  const [trackedFrames, setTrackedFrames] = useState([]);
  const [processingTime, setProcessingTime] = useState(null);
  const [error, setError] = useState(null);

  // Fullscreen state
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Refs for visual playback
  const fileInputRef = useRef(null);
  const visualVideoRef = useRef(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  // --- Drag & Drop Handlers ---
  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleFileSelection = (selectedFile) => {
    if (!selectedFile.type.startsWith('video/')) {
      setError("Please select a valid video file.");
      return;
    }
    setFile(selectedFile);
    setVideoUrl(URL.createObjectURL(selectedFile));
    setTrackedFrames([]);
    setProcessingTime(null);
    setStatus("idle");
    setError(null);
    setIsFullscreen(false);
  };

  // --- Core Pipeline: Send full video to Backend Orchestrator ---
  const handleRunPipeline = async () => {
    if (!file) return;

    setStatus("processing");
    setProgressMsg("Processing video on server (extracting frames, detecting & tracking)...");
    setError(null);
    setTrackedFrames([]);
    setProcessingTime(null);

    const startTime = performance.now();

    try {
      const formData = new FormData();
      formData.append('video', file);
      formData.append('fps', EXTRACTION_FPS.toString());

      // Do not set Content-Type manually so the browser attaches the multipart boundary
      const response = await fetch(ANALYZE_VIDEO_URL, {
        method: 'POST',
        body: formData,
      });

      let data = null;
      try {
        data = await response.json();
      } catch {
        // Fallback if server returned a non-JSON error body
      }

      if (!response.ok) {
        throw new Error(data?.error || `Server responded with status ${response.status}`);
      }

      const elapsedSeconds = ((performance.now() - startTime) / 1000).toFixed(2);
      setProcessingTime(elapsedSeconds);
      setTrackedFrames(data.trackedFrames || []);
      setStatus("done");
      setProgressMsg("Pipeline completed successfully!");
    } catch (err) {
      setError(err.message || "Pipeline execution failed.");
      setStatus("error");
    }
  };

  const handleClear = () => {
    setFile(null);
    setVideoUrl(null);
    setTrackedFrames([]);
    setProcessingTime(null);
    setStatus("idle");
    setError(null);
    setIsFullscreen(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
  };

  // --- Visualization Synchronization (Video + Canvas) ---
  const drawBoxesForCurrentTime = () => {
    if (!visualVideoRef.current || !canvasRef.current || trackedFrames.length === 0) return;

    const video = visualVideoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Ensure canvas internal resolution matches the video native resolution
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const currentMs = video.currentTime * 1000;

    let closestFrame = trackedFrames[0];
    let minDiff = Math.abs(closestFrame.timestamp_ms - currentMs);

    for (let i = 1; i < trackedFrames.length; i++) {
      const diff = Math.abs(trackedFrames[i].timestamp_ms - currentMs);
      if (diff < minDiff) {
        minDiff = diff;
        closestFrame = trackedFrames[i];
      }
    }

    // Drop drawing if frame is out of sync
    if (minDiff > (1000 / EXTRACTION_FPS) * 1.5) {
      animationRef.current = requestAnimationFrame(drawBoxesForCurrentTime);
      return;
    }

    // Draw Players with Tracking IDs
    ctx.lineWidth = 4;
    ctx.font = 'bold 24px monospace';

    closestFrame.players?.forEach(player => {
      const { x1, y1, x2, y2, tracking_id } = player;
      if (x1 === undefined) return;

      const width = x2 - x1;
      const height = y2 - y1;

      ctx.strokeStyle = '#3b82f6';
      ctx.strokeRect(x1, y1, width, height);

      const label = tracking_id !== null ? `ID: ${tracking_id}` : 'Detecting...';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = '#3b82f6';
      ctx.fillRect(x1, y1 - 30, textWidth + 10, 30);
      ctx.fillStyle = 'white';
      ctx.fillText(label, x1 + 5, y1 - 8);
    });

    animationRef.current = requestAnimationFrame(drawBoxesForCurrentTime);
  };

  useEffect(() => {
    if (status === "done" && visualVideoRef.current) {
      animationRef.current = requestAnimationFrame(drawBoxesForCurrentTime);
    }
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [status, trackedFrames]);

  return (
    <div className="w-full text-left">
      {!file ? (
        <div
          className={`border-2 border-dashed rounded-lg p-10 text-center transition-colors ${
            isDragging ? 'border-black bg-gray-50' : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="flex justify-center mb-4 text-gray-400">
            <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
          </div>
          <p className="text-sm font-medium text-gray-700 mb-1">Upload a short video for Backend Orchestrator Testing</p>
          <p className="text-xs text-gray-500 mb-6">MP4, WebM up to 20MB (Server-side FFmpeg + OD + Re-ID + Tracker)</p>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            className="hidden"
            accept="video/*"
          />
          <Button variant="outline" onClick={() => fileInputRef.current?.click()} className="mx-auto">
            Browse Videos
          </Button>
        </div>
      ) : (
        <div className="flex flex-col xl:flex-row gap-6 items-start">

          {/* Layout Wrapper for Video */}
          <div className="w-full xl:w-1/2 flex flex-col gap-4">

            {/* Morphs from a standard container into a fixed fullscreen modal without unmounting */}
            <div className={isFullscreen
              ? "fixed inset-0 z-[100] bg-black/95 flex items-center justify-center p-6"
              : "relative rounded-lg overflow-hidden border border-gray-200 shadow-sm bg-black flex items-center justify-center min-h-[300px]"
            }>

              <div className={`relative inline-block ${isFullscreen ? 'w-auto max-w-full max-h-full' : 'w-full'}`}>

                {/* Visual Video Player */}
                <video
                  ref={visualVideoRef}
                  src={videoUrl}
                  className={`block ${isFullscreen ? 'max-h-[90vh] w-auto' : 'w-full h-auto'}`}
                  controls={status === "done"}
                  controlsList="nofullscreen"
                  onPlay={() => {
                    if (status === "done") {
                      if (animationRef.current) cancelAnimationFrame(animationRef.current);
                      animationRef.current = requestAnimationFrame(drawBoxesForCurrentTime);
                    }
                  }}
                />

                {/* Transparent Canvas Overlay */}
                <canvas
                  ref={canvasRef}
                  className="absolute top-0 left-0 w-full h-full pointer-events-none"
                />

                {/* Custom Expand / Close Button */}
                {status === "done" && (
                  <button
                    type="button"
                    onClick={() => setIsFullscreen(!isFullscreen)}
                    className={`absolute z-20 bg-black/70 hover:bg-black text-white rounded-md transition-colors shadow-md backdrop-blur-sm border border-gray-600 cursor-pointer ${
                      isFullscreen ? "top-4 right-4 p-3" : "top-2 right-2 p-2"
                    }`}
                    title={isFullscreen ? "Close fullscreen" : "View fullscreen"}
                  >
                    {isFullscreen ? (
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                    ) : (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"></path></svg>
                    )}
                  </button>
                )}
              </div>
            </div>

            {/* Buttons show only when NOT in fullscreen */}
            {!isFullscreen && (
              <>
                <div className="flex gap-2">
                  <Button variant="outline" onClick={handleClear} disabled={status === "processing"} className="flex-1">
                    Clear
                  </Button>
                  <Button
                    variant="primary"
                    onClick={handleRunPipeline}
                    disabled={status === "processing"}
                    className="flex-1 bg-indigo-600 hover:bg-indigo-700 border-indigo-700 text-white"
                  >
                    {status === "processing" ? 'Processing...' : 'Run Pipeline'}
                  </Button>
                </div>

                {error && (
                  <div className="p-3 text-sm text-red-600 bg-red-50 rounded-md border border-red-100">
                    {error}
                  </div>
                )}
              </>
            )}
          </div>

          {/* Status & Results Area */}
          <div className="w-full xl:w-1/2">
            <div className="bg-gray-900 rounded-lg p-4 flex flex-col h-[400px]">
              <div className="flex justify-between items-center mb-2 pb-2 border-b border-gray-700 shrink-0">
                <span className="text-xs font-mono text-gray-400">Backend Orchestrator Status</span>
                {status === "done" && (
                  <span className="text-xs font-mono text-green-400">
                    Tracked {trackedFrames.length} Frames {processingTime && `• ${processingTime}s`}
                  </span>
                )}
              </div>

              <div className="flex-1 overflow-y-auto relative custom-scrollbar flex flex-col justify-center items-center p-6 text-center">

                {status === "idle" && (
                  <span className="text-gray-600 font-mono text-sm">Click 'Run Pipeline' to start processing</span>
                )}

                {status === "processing" && (
                  <div className="flex flex-col items-center gap-4">
                    <div className="w-8 h-8 border-4 border-gray-600 border-t-indigo-500 rounded-full animate-spin"></div>
                    <span className="text-indigo-400 font-mono text-sm font-semibold">{progressMsg}</span>
                    <span className="text-gray-500 font-mono text-xs max-w-xs mt-2">
                      Please do not close this tab. Waiting for backend orchestrator response.
                    </span>
                  </div>
                )}

                {status === "done" && (
                  <div className="flex flex-col items-center gap-3 w-full h-full justify-start mt-4">
                    <div className="w-12 h-12 bg-green-500/20 text-green-400 rounded-full flex items-center justify-center mb-2">
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                    </div>
                    <h3 className="text-green-400 font-bold text-lg">Tracking Completed</h3>
                    <p className="text-gray-400 text-sm mb-4">Play the video to see synchronized bounding boxes and Re-ID tracking IDs.</p>

                    <div className="bg-black/50 p-4 rounded-md w-full text-left overflow-y-auto custom-scrollbar max-h-[200px]">
                      <p className="text-xs text-gray-500 mb-2 sticky top-0">Sample Frame Data (Frame 1):</p>
                      <pre className="text-[10px] font-mono text-gray-300 whitespace-pre-wrap break-words">
                        {formatSampleJson(trackedFrames[0])}
                      </pre>
                    </div>
                  </div>
                )}

              </div>
            </div>
          </div>

        </div>
      )}
    </div>
  );
}