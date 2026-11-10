"use client";

import { useState, useRef } from 'react';
import Button from '@/components/ui/Button';

export default function TeamAssignment() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [imgDimensions, setImgDimensions] = useState({ width: 0, height: 0 });
  
  // State for fullscreen image modal
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // States for API calls and Pipeline
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(""); // "Detecting Objects..." -> "Assigning Teams..."
  const [result, setResult] = useState(null); // Will hold the FINAL json from Team Assignment
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
    setCurrentStep("");
    setImgDimensions({ width: 0, height: 0 });
  };

  // --- Image Load Handler ---
  const handleImageLoad = (e) => {
    setImgDimensions({
      width: e.target.naturalWidth,
      height: e.target.naturalHeight
    });
  };

  // --- Full Pipeline Handler (Object Detection -> Team Assignment) ---
  const handleRunPipeline = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      // Step 1: Object Detection
      setCurrentStep("Step 1: Detecting Objects (Port 8001)...");
      const detFormData = new FormData();
      detFormData.append('image', file);
      detFormData.append('frame_number', '1');
      detFormData.append('timestamp_ms', '1000.0');

      const detResponse = await fetch('http://localhost:8001/detect', {
        method: 'POST',
        body: detFormData,
      });

      if (!detResponse.ok) {
        throw new Error(`Object Detection MS failed with status ${detResponse.status}`);
      }

      const detectionData = await detResponse.json();

      // Step 2: Team Assignment
      setCurrentStep("Step 2: Assigning Teams (Port 8000)...");
      const teamFormData = new FormData();
      teamFormData.append('image', file);
      teamFormData.append('frame_data', JSON.stringify(detectionData)); // Pass the JSON from Step 1

      const teamResponse = await fetch('http://localhost:8000/assign-teams', {
        method: 'POST',
        body: teamFormData,
      });

      if (!teamResponse.ok) {
        throw new Error(`Team Assignment MS failed with status ${teamResponse.status}`);
      }

      const finalData = await teamResponse.json();
      
      // Pipeline complete
      setResult(finalData);
      setCurrentStep("");
      
    } catch (err) {
      setError(err.message || "Pipeline execution failed.");
      setCurrentStep("");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setCurrentStep("");
    setImgDimensions({ width: 0, height: 0 });
    setIsFullscreen(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // --- Draw Bounding Boxes with Team Colors ---
  const renderBBoxes = () => {
    if (!result || imgDimensions.width === 0) return null;

    // Define colors for different teams
    const teamColors = {
      0: '#6b7280', // Gray (e.g., Referees or unassigned)
      1: '#3b82f6', // Blue (Team 1)
      2: '#ef4444', // Red (Team 2)
      default: '#f59e0b' // Yellow (fallback)
    };

    const drawBox = (item, type, key) => {
      const { x1, y1, x2, y2, team, tracking_id } = item;
      
      if (x1 === undefined || y1 === undefined || x2 === undefined || y2 === undefined) return null;
      
      const left = (x1 / imgDimensions.width) * 100;
      const top = (y1 / imgDimensions.height) * 100;
      const width = ((x2 - x1) / imgDimensions.width) * 100;
      const height = ((y2 - y1) / imgDimensions.height) * 100;

      // Determine color and label based on item type and team assignment
      let color = teamColors.default;
      let label = type;

      if (type === 'Player') {
        color = teamColors[team] !== undefined ? teamColors[team] : teamColors.default;
        // Build label: e.g., "Team 1 (ID: 5)" or just "Team 1"
        label = team !== null ? `Team ${team}` : 'Player';
        if (tracking_id !== null) label += ` [${tracking_id}]`;
      } else if (type === 'Rim') {
        color = '#10b981'; // Green for rims
      } else if (type === 'Ball') {
        color = '#f59e0b'; // Orange/Yellow for ball
      }

      return (
        <div 
          key={key}
          className="absolute border-2 shadow-sm pointer-events-none transition-all duration-300"
          style={{ 
            left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%`,
            borderColor: color 
          }}
        >
          <span className="absolute -top-5 left-0 px-1.5 py-0.5 text-[10px] md:text-xs font-bold text-white whitespace-nowrap z-10 rounded-t-sm" style={{ backgroundColor: color }}>
            {label}
          </span>
        </div>
      );
    };

    return (
      <>
        {result.players?.map((player, idx) => 
          drawBox(player, 'Player', `player-${idx}`)
        )}
        {result.rims?.map((rim, idx) => 
          drawBox(rim, 'Rim', `rim-${idx}`)
        )}
        {result.ball && drawBox(result.ball, 'Ball', 'ball-0')}
      </>
    );
  };

  // --- Reusable Image & BBox Component (Matches ObjectDetection style) ---
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
            <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"></path></svg>
          </div>
          <p className="text-sm font-medium text-gray-700 mb-1">Upload image to run full Pipeline</p>
          <p className="text-xs text-gray-500 mb-6">Will trigger both Object Detection and Team Assignment</p>
          
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
            
            <div className="relative rounded-lg overflow-hidden border border-gray-200 shadow-sm bg-gray-100 flex items-center justify-center h-[500px]">
              <ImageWithBBoxes isExpanded={false} />
              
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
              <Button variant="primary" onClick={handleRunPipeline} disabled={isLoading} className="flex-1 bg-green-600 hover:bg-green-700 text-white border-green-700">
                {isLoading ? 'Running Pipeline...' : 'Run Pipeline'}
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
                <span className="text-xs font-mono text-gray-400">Pipeline Result (Clustered Teams)</span>
                {result && (
                  <span className="text-xs font-mono text-green-400">
                    Assigned: {result.players?.length || 0} Players
                  </span>
                )}
              </div>
              
              <div className="flex-1 overflow-y-auto relative custom-scrollbar">
                {isLoading ? (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                    <div className="w-6 h-6 border-2 border-gray-500 border-t-white rounded-full animate-spin"></div>
                    <span className="text-gray-400 font-mono text-sm animate-pulse">{currentStep}</span>
                  </div>
                ) : result ? (
                  <pre className="text-xs font-mono text-green-400 whitespace-pre-wrap break-words">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-gray-600 font-mono text-sm">Run pipeline to view JSON output</span>
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