"use client";

import { useState, useRef } from 'react';
import Button from '@/components/ui/Button';

export default function ObjectDetection() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [imgDimensions, setImgDimensions] = useState({ width: 0, height: 0 });
  
  // State for fullscreen image modal
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // States for API call
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);

  // --- Drag & Drop Handlers ---
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

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
    if (!selectedFile.type.startsWith('image/')) {
      setError("Please select a valid image file.");
      return;
    }
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
    setImgDimensions({ width: 0, height: 0 });
  };

  // --- Image Load Handler ---
  const handleImageLoad = (e) => {
    // Save original image dimensions to calculate BBox percentages
    setImgDimensions({
      width: e.target.naturalWidth,
      height: e.target.naturalHeight
    });
  };

  // --- API Call Handler ---
  const handleRunDetection = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('image', file);
    formData.append('frame_number', '1');
    formData.append('timestamp_ms', '1000.0');

    try {
      const response = await fetch('http://localhost:8001/detect', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to connect to the Object Detection MS.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setImgDimensions({ width: 0, height: 0 });
    setIsFullscreen(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // --- Draw Bounding Boxes ---
  const renderBBoxes = () => {
    if (!result || imgDimensions.width === 0) return null;

    // Helper function to draw boxes based on JSON structure
    const drawBox = (item, color, label, key) => {
      const { x1, y1, x2, y2, confidence } = item;
      
      // Ensure coordinates exist before drawing
      if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) return null;
      
      const left = (x1 / imgDimensions.width) * 100;
      const top = (y1 / imgDimensions.height) * 100;
      const width = ((x2 - x1) / imgDimensions.width) * 100;
      const height = ((y2 - y1) / imgDimensions.height) * 100;

      // Format confidence to 1 decimal place if it exists
      const confText = confidence ? ` (${(confidence * 100).toFixed(1)}%)` : '';

      return (
        <div 
          key={key}
          className="absolute border-2 shadow-sm pointer-events-none"
          style={{ 
            left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%`,
            borderColor: color 
          }}
        >
          <span className="absolute -top-5 left-0 px-1 text-[10px] md:text-xs font-bold text-white whitespace-nowrap z-10" style={{ backgroundColor: color }}>
            {label}{confText}
          </span>
        </div>
      );
    };

    return (
      <>
        {result.players?.map((player, idx) => 
          drawBox(player, '#3b82f6', `Player ${idx}`, `player-${idx}`)
        )}
        
        {result.rims?.map((rim, idx) => 
          drawBox(rim, '#ef4444', 'Rim', `rim-${idx}`)
        )}

        {result.ball && drawBox(result.ball, '#f59e0b', 'Ball', 'ball-0')}
      </>
    );
  };

  // --- Reusable Image & BBox Component (Without buttons) ---
  const ImageWithBBoxes = ({ isExpanded = false }) => (
    <div className="relative inline-block">
      <img 
        src={previewUrl} 
        alt="Preview" 
        onLoad={handleImageLoad}
        className={`${isExpanded ? 'max-h-[90vh]' : 'max-h-[460px]'} w-auto block`} 
      />
      {renderBBoxes()}
    </div>
  );

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
            <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
          </div>
          <p className="text-sm font-medium text-gray-700 mb-1">Drag and drop an image here</p>
          <p className="text-xs text-gray-500 mb-6">JPEG, PNG up to 10MB</p>
          
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            className="hidden" 
            accept="image/*"
          />
          <Button variant="outline" onClick={() => fileInputRef.current?.click()} className="mx-auto">
            Browse Files
          </Button>
        </div>
      ) : (
        <div className="flex flex-col xl:flex-row gap-6 items-start">
          
          {/* Image & Controls Area */}
          <div className="w-full xl:w-1/2 flex flex-col gap-4">
            
            {/* 
              Image Container with absolute Expand button anchored to the container itself 
            */}
            <div className="relative rounded-lg overflow-hidden border border-gray-200 shadow-sm bg-gray-100 flex items-center justify-center h-[500px]">
              <ImageWithBBoxes isExpanded={false} />
              
              {/* Expand Button - Anchored to the top-right of the gray container */}
              <button 
                type="button"
                onClick={() => setIsFullscreen(true)}
                className="absolute top-3 right-3 p-2 bg-black/70 hover:bg-black text-white rounded-md transition-colors z-20 shadow-md backdrop-blur-sm border border-gray-600 cursor-pointer"
                title="View fullscreen"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"></path></svg>
              </button>
            </div>
            
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleClear} disabled={isLoading} className="flex-1">
                Clear
              </Button>
              <Button variant="primary" onClick={handleRunDetection} disabled={isLoading} className="flex-1">
                {isLoading ? 'Processing...' : 'Run Detection'}
              </Button>
            </div>
            
            {error && (
              <div className="p-3 text-sm text-red-600 bg-red-50 rounded-md border border-red-100">
                {error}
              </div>
            )}
          </div>

          {/* JSON Results Area */}
          <div className="w-full xl:w-1/2">
            <div className="bg-gray-900 rounded-lg p-4 flex flex-col h-[500px]">
              <div className="flex justify-between items-center mb-2 pb-2 border-b border-gray-700 shrink-0">
                <span className="text-xs font-mono text-gray-400">Response JSON</span>
                {result && (
                  <span className="text-xs font-mono text-green-400">
                    Found: {result.players?.length || 0} Players
                  </span>
                )}
              </div>
              
              <div className="flex-1 overflow-y-auto relative custom-scrollbar">
                {isLoading ? (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-gray-400 font-mono text-sm animate-pulse">Waiting for MS response...</span>
                  </div>
                ) : result ? (
                  <pre className="text-xs font-mono text-green-400 whitespace-pre-wrap break-words">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-gray-600 font-mono text-sm">Run detection to view JSON output</span>
                  </div>
                )}
              </div>
            </div>
          </div>
          
        </div>
      )}

      {/* Fullscreen Modal Overlay */}
      {isFullscreen && (
        <div className="fixed inset-0 z-50 bg-black/95 flex items-center justify-center p-4">
          <ImageWithBBoxes isExpanded={true} />
          
          {/* Close Button - Anchored to the top-right of the screen */}
          <button 
            type="button"
            onClick={() => setIsFullscreen(false)}
            className="absolute top-6 right-6 p-3 bg-gray-800 hover:bg-gray-700 text-white rounded-md transition-colors z-50 shadow-lg border border-gray-600 cursor-pointer"
            title="Close fullscreen"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
          </button>
        </div>
      )}

    </div>
  );
}