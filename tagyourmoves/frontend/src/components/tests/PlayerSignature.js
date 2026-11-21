"use client";

import { useState, useRef, useEffect, useMemo } from 'react';
import Button from '@/components/ui/Button';

// API Endpoints
const OD_MS_URL = 'http://localhost:8001/detect';
const SIGNATURE_MS_URL = 'http://localhost:8004/extract-signatures';

// Dot product for L2-normalized vectors equals Cosine Similarity
const computeCosineSimilarity = (vecA, vecB) => {
  if (!Array.isArray(vecA) || !Array.isArray(vecB) || vecA.length !== vecB.length) {
    return null;
  }
  let dot = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
  }
  return dot;
};

export default function PlayerSignature() {
  // Image 1 (Query / Reference) states
  const [file1, setFile1] = useState(null);
  const [preview1, setPreview1] = useState(null);
  const [isDragging1, setIsDragging1] = useState(false);
  const [result1, setResult1] = useState(null);
  const [img1Loaded, setImg1Loaded] = useState(false);

  // Image 2 (Target / Comparison) states
  const [file2, setFile2] = useState(null);
  const [preview2, setPreview2] = useState(null);
  const [isDragging2, setIsDragging2] = useState(false);
  const [result2, setResult2] = useState(null);
  const [img2Loaded, setImg2Loaded] = useState(false);

  // Pipeline & interactive states
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState("");
  const [error, setError] = useState(null);
  const [selectedPlayerIdx, setSelectedPlayerIdx] = useState(null);

  // Refs
  const input1Ref = useRef(null);
  const input2Ref = useRef(null);
  const img1Ref = useRef(null);
  const img2Ref = useRef(null);
  const canvas1Ref = useRef(null);
  const canvas2Ref = useRef(null);

  // --- File Selection Helpers ---
  const handleFileSelection = (selectedFile, slot) => {
    if (!selectedFile || !selectedFile.type.startsWith('image/')) {
      setError("Please select a valid image file.");
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setError(null);
    setSelectedPlayerIdx(null);

    if (slot === 1) {
      setFile1(selectedFile);
      setPreview1(objectUrl);
      setResult1(null);
      setImg1Loaded(false);
    } else {
      setFile2(selectedFile);
      setPreview2(objectUrl);
      setResult2(null);
      setImg2Loaded(false);
    }
  };

  const handleClearAll = () => {
    setFile1(null);
    setPreview1(null);
    setResult1(null);
    setImg1Loaded(false);

    setFile2(null);
    setPreview2(null);
    setResult2(null);
    setImg2Loaded(false);

    setSelectedPlayerIdx(null);
    setError(null);
    setCurrentStep("");

    if (input1Ref.current) input1Ref.current.value = "";
    if (input2Ref.current) input2Ref.current.value = "";
  };

  // --- Pipeline Execution (OD -> Player Signature for both images) ---
  const processSingleImage = async (file, frameNumber, label) => {
    // Step A: Object Detection
    setCurrentStep(`${label}: Detecting objects (Port 8001)...`);
    const detFormData = new FormData();
    detFormData.append('image', file);
    detFormData.append('frame_number', frameNumber.toString());
    detFormData.append('timestamp_ms', (frameNumber * 1000).toString());

    const detResponse = await fetch(OD_MS_URL, {
      method: 'POST',
      body: detFormData,
    });

    if (!detResponse.ok) {
      throw new Error(`${label} - Object Detection failed (${detResponse.status})`);
    }
    const detectionData = await detResponse.json();

    // Step B: Player Signature Extraction
    setCurrentStep(`${label}: Extracting 512-dim Re-ID signatures (Port 8004)...`);
    const sigFormData = new FormData();
    sigFormData.append('image', file);
    sigFormData.append('frame_data', JSON.stringify(detectionData));

    const sigResponse = await fetch(SIGNATURE_MS_URL, {
      method: 'POST',
      body: sigFormData,
    });

    if (!sigResponse.ok) {
      const errText = await sigResponse.text();
      throw new Error(`${label} - Signature MS failed (${sigResponse.status}): ${errText}`);
    }

    return await sigResponse.json();
  };

  const handleRunPipeline = async () => {
    if (!file1 || !file2) return;

    setIsLoading(true);
    setError(null);
    setResult1(null);
    setResult2(null);
    setSelectedPlayerIdx(null);

    try {
      const res1 = await processSingleImage(file1, 1, "Image 1");
      const res2 = await processSingleImage(file2, 2, "Image 2");

      setResult1(res1);
      setResult2(res2);
      setCurrentStep("");
    } catch (err) {
      setError(err.message || "Pipeline execution failed.");
      setCurrentStep("");
    } finally {
      setIsLoading(false);
    }
  };

  // --- Similarity Calculation (Memoized) ---
  const comparisonData = useMemo(() => {
    if (selectedPlayerIdx === null || !result1?.players || !result2?.players) {
      return { scores: [], bestMatchIdx: null };
    }

    const queryPlayer = result1.players[selectedPlayerIdx];
    if (!queryPlayer?.signature) {
      return { scores: [], bestMatchIdx: null };
    }

    let bestScore = -Infinity;
    let bestMatchIdx = null;

    const scores = result2.players.map((targetPlayer, idx) => {
      const score = computeCosineSimilarity(queryPlayer.signature, targetPlayer.signature);
      if (score !== null && score > bestScore) {
        bestScore = score;
        bestMatchIdx = idx;
      }
      return score;
    });

    return { scores, bestMatchIdx };
  }, [selectedPlayerIdx, result1, result2]);

  // --- Canvas Drawing Helper ---
  const drawLabelBox = (ctx, x1, y1, x2, y2, color, label, lineWidth, fontSize) => {
    const width = x2 - x1;
    const height = y2 - y1;

    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, width, height);

    ctx.font = `bold ${fontSize}px monospace`;
    const textWidth = ctx.measureText(label).width;
    const padX = Math.round(fontSize * 0.35);
    const padY = Math.round(fontSize * 0.25);
    const boxHeight = fontSize + padY * 2;

    // Keep label inside top edge if box is near y = 0
    const labelY = y1 - boxHeight >= 0 ? y1 - boxHeight : y1;

    ctx.fillStyle = color;
    ctx.fillRect(x1, labelY, textWidth + padX * 2, boxHeight);

    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'top';
    ctx.fillText(label, x1 + padX, labelY + padY);
  };

  // --- Draw Canvas 1 (Query Image) ---
  useEffect(() => {
    const canvas = canvas1Ref.current;
    const img = img1Ref.current;
    if (!canvas || !img || !img1Loaded) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!result1?.players) return;

    const lineWidth = Math.max(2, Math.round(canvas.width / 350));
    const fontSize = Math.max(14, Math.round(canvas.width / 55));

    result1.players.forEach((player, idx) => {
      const { x1, y1, x2, y2 } = player;
      if (x1 === undefined) return;

      const isSelected = idx === selectedPlayerIdx;
      const color = isSelected ? '#2563eb' : '#52525b'; // Bright Blue if selected, Zinc Gray otherwise
      const label = isSelected ? `P${idx} (Selected)` : `P${idx}`;

      drawLabelBox(ctx, x1, y1, x2, y2, color, label, isSelected ? lineWidth * 1.6 : lineWidth, fontSize);
    });
  }, [result1, selectedPlayerIdx, img1Loaded]);

  // --- Draw Canvas 2 (Target Image with Similarity Scores) ---
  useEffect(() => {
    const canvas = canvas2Ref.current;
    const img = img2Ref.current;
    if (!canvas || !img || !img2Loaded) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!result2?.players) return;

    const lineWidth = Math.max(2, Math.round(canvas.width / 350));
    const fontSize = Math.max(14, Math.round(canvas.width / 55));

    result2.players.forEach((player, idx) => {
      const { x1, y1, x2, y2 } = player;
      if (x1 === undefined) return;

      const hasComparison = selectedPlayerIdx !== null && comparisonData.scores[idx] !== undefined;
      const score = hasComparison ? comparisonData.scores[idx] : null;
      const isBestMatch = hasComparison && idx === comparisonData.bestMatchIdx;

      let color = '#52525b'; // Default muted gray
      let label = `P${idx}`;

      if (hasComparison) {
        const pct = score !== null ? `${(score * 100).toFixed(1)}%` : 'N/A';
        label = `P${idx}: ${pct}`;
        color = isBestMatch ? '#16a34a' : '#3f3f46'; // Green for best match, dark gray for others
      }

      drawLabelBox(ctx, x1, y1, x2, y2, color, label, isBestMatch ? lineWidth * 1.8 : lineWidth, fontSize);
    });
  }, [result2, selectedPlayerIdx, comparisonData, img2Loaded]);

  // --- Interactive Click Handler on Canvas 1 ---
  const handleCanvas1Click = (e) => {
    if (!result1?.players || !canvas1Ref.current) return;

    const canvas = canvas1Ref.current;
    const rect = canvas.getBoundingClientRect();

    // Map click coordinates from CSS display space to native image pixel space
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    // Find all boxes containing the click; pick the smallest box if overlapping
    let matchedIdx = null;
    let smallestArea = Infinity;

    result1.players.forEach((player, idx) => {
      const { x1, y1, x2, y2 } = player;
      if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
        const area = (x2 - x1) * (y2 - y1);
        if (area < smallestArea) {
          smallestArea = area;
          matchedIdx = idx;
        }
      }
    });

    setSelectedPlayerIdx(matchedIdx);
  };

  // --- Reusable Upload Box Renderer ---
  const renderUploadSlot = (slot, file, preview, isDragging, setIsDragging, inputRef, imgRef, canvasRef, setImgLoaded, isInteractive) => (
    <div className="w-full xl:w-1/2 flex flex-col gap-2">
      <div className="flex justify-between items-center">
        <span className="text-xs font-mono font-bold uppercase tracking-wider text-gray-700">
          {slot === 1 ? 'Image 1 (Select Query Player)' : 'Image 2 (Similarity Targets)'}
        </span>
        {file && (
          <button
            type="button"
            disabled={isLoading}
            onClick={() => inputRef.current?.click()}
            className="text-xs font-mono text-gray-500 hover:text-black underline cursor-pointer disabled:opacity-40"
          >
            Replace
          </button>
        )}
      </div>

      <input
        type="file"
        ref={inputRef}
        onChange={(e) => e.target.files?.[0] && handleFileSelection(e.target.files[0], slot)}
        className="hidden"
        accept="image/*"
      />

      {!file ? (
        <div
          className={`border-2 border-dashed rounded-lg p-10 text-center transition-colors h-[380px] flex flex-col items-center justify-center ${
            isDragging ? 'border-black bg-gray-50' : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            if (e.dataTransfer.files?.[0]) handleFileSelection(e.dataTransfer.files[0], slot);
          }}
        >
          <svg className="w-9 h-9 text-gray-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <p className="text-sm font-medium text-gray-700 mb-1">Drag and drop Image {slot}</p>
          <p className="text-xs text-gray-500 mb-5">JPEG, PNG up to 10MB</p>
          <Button variant="outline" onClick={() => inputRef.current?.click()}>
            Browse Files
          </Button>
        </div>
      ) : (
        <div className="relative rounded-lg overflow-hidden border border-gray-200 bg-gray-100 flex items-center justify-center h-[380px]">
          <div className="relative inline-block max-h-full max-w-full">
            <img
              ref={imgRef}
              src={preview}
              alt={`Preview ${slot}`}
              onLoad={() => setImgLoaded(true)}
              className="max-h-[378px] w-auto block select-none"
            />
            <canvas
              ref={canvasRef}
              onClick={isInteractive ? handleCanvas1Click : undefined}
              className={`absolute inset-0 w-full h-full ${
                isInteractive && result1 ? 'cursor-pointer' : 'pointer-events-none'
              }`}
            />
          </div>
        </div>
      )}
    </div>
  );

  // Ranked similarity list for the DevTools panel
  const rankedMatches = useMemo(() => {
    if (selectedPlayerIdx === null || !comparisonData.scores.length) return [];
    return comparisonData.scores
      .map((score, idx) => ({ idx, score }))
      .filter((item) => item.score !== null)
      .sort((a, b) => b.score - a.score);
  }, [selectedPlayerIdx, comparisonData]);

  return (
    <div className="w-full text-left flex flex-col gap-5">
      {/* Dual Image Upload & Canvas Area */}
      <div className="flex flex-col xl:flex-row gap-6">
        {renderUploadSlot(1, file1, preview1, isDragging1, setIsDragging1, input1Ref, img1Ref, canvas1Ref, setImg1Loaded, true)}
        {renderUploadSlot(2, file2, preview2, isDragging2, setIsDragging2, input2Ref, img2Ref, canvas2Ref, setImg2Loaded, false)}
      </div>

      {/* Controls Bar */}
      <div className="flex flex-col sm:flex-row items-center gap-3">
        <Button
          variant="outline"
          onClick={handleClearAll}
          disabled={isLoading || (!file1 && !file2)}
          className="w-full sm:w-auto sm:min-w-[140px]"
        >
          Clear All
        </Button>
        <Button
          variant="primary"
          onClick={handleRunPipeline}
          disabled={isLoading || !file1 || !file2}
          className="w-full sm:flex-1"
        >
          {isLoading ? currentStep || 'Processing...' : 'Run Signature Extraction'}
        </Button>
      </div>

      {error && (
        <div className="p-3 text-sm text-red-600 bg-red-50 rounded-md border border-red-200 font-mono">
          {error}
        </div>
      )}

      {/* Interactive Status & Similarity Diagnostics Panel */}
      <div className="bg-gray-900 rounded-lg p-4 flex flex-col h-[220px]">
        <div className="flex justify-between items-center mb-2 pb-2 border-b border-gray-700 shrink-0">
          <span className="text-xs font-mono text-gray-400">
            Re-ID Cosine Similarity Diagnostics
          </span>
          {result1 && result2 && (
            <span className="text-xs font-mono text-gray-300">
              Image 1: {result1.players?.length || 0} players | Image 2: {result2.players?.length || 0} players
            </span>
          )}
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar font-mono text-xs">
          {isLoading ? (
            <div className="h-full flex items-center justify-center gap-3 text-gray-400">
              <div className="w-4 h-4 border-2 border-gray-600 border-t-white rounded-full animate-spin" />
              <span>{currentStep}</span>
            </div>
          ) : !result1 || !result2 ? (
            <div className="h-full flex items-center justify-center text-gray-500">
              Upload two images and run extraction to compare 512-dim L2-normalized embeddings.
            </div>
          ) : selectedPlayerIdx === null ? (
            <div className="h-full flex items-center justify-center text-gray-300">
              Click on any player bounding box in Image 1 to calculate cosine similarity against Image 2.
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              <div className="text-gray-300 pb-1">
                Query: <span className="text-blue-400 font-bold">Image 1 - Player {selectedPlayerIdx}</span>{' '}
                (dim: {result1.players[selectedPlayerIdx]?.signature?.length || 0})
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                {rankedMatches.map(({ idx, score }, rank) => {
                  const isTop = rank === 0;
                  return (
                    <div
                      key={idx}
                      className={`flex justify-between items-center px-3 py-1.5 rounded border ${
                        isTop
                          ? 'bg-green-950/50 border-green-600 text-green-400 font-bold'
                          : 'bg-gray-800/70 border-gray-700 text-gray-300'
                      }`}
                    >
                      <span>Image 2 - P{idx}</span>
                      <span>{(score * 100).toFixed(2)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}