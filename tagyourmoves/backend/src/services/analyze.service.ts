/**
 * @fileoverview Core video analysis orchestration service.
 * Coordinates frame extraction, bounded-concurrency stateless inference,
 * sequential stateful tracking, and guaranteed resource cleanup.
 */

import { randomUUID } from 'crypto';
import pLimit from 'p-limit';
import path from 'path';
import { detectObjectsInFrame } from '../clients/detection.client';
import { extractPlayerSignatures } from '../clients/signature.client';
import { deleteTrackerSession, trackFrame } from '../clients/tracker.client';
import { config } from '../config';
import { ExtractedFrame, Frame, VideoAnalysisResponse } from '../types';
import { cleanupSessionFiles, extractFrames } from '../utils/ffmpeg';

/**
 * Runs the stateless inference stages (object detection followed by player
 * Re-ID signature extraction) for a single extracted frame.
 *
 * @param extractedFrame Metadata and disk path of the extracted frame image.
 * @return A promise resolving to the Frame enriched with detections and signatures.
 */
async function processFrameStateless(
  extractedFrame: ExtractedFrame,
): Promise<Frame> {
  const detectedFrame = await detectObjectsInFrame(extractedFrame);
  return extractPlayerSignatures(extractedFrame.filePath, detectedFrame);
}

/**
 * Orchestrates the end-to-end analysis pipeline for an uploaded video file.
 * Guarantees removal of temporary disk files and remote tracker session state
 * regardless of whether processing succeeds or fails.
 *
 * @param videoPath Absolute filesystem path to the uploaded video file.
 * @param fps Target frame rate at which to sample and analyze the video.
 * @return A promise resolving to the complete video analysis result.
 * @throws Error if frame extraction yields no frames or any downstream service fails.
 */
export async function analyzeVideo(
  videoPath: string,
  fps: number,
): Promise<VideoAnalysisResponse> {
  const sessionId = randomUUID();
  const sessionDir = path.join(config.storage.tempDir, sessionId);
  const limit = pLimit(config.pipeline.concurrencyLimit);

  try {
    const extractedFrames = await extractFrames(videoPath, sessionDir, fps);

    if (extractedFrames.length === 0) {
      throw new Error('No frames could be extracted from the uploaded video.');
    }

    const enrichedFrames = await Promise.all(
      extractedFrames.map((extractedFrame) =>
        limit(() => processFrameStateless(extractedFrame)),
      ),
    );

    enrichedFrames.sort((a, b) => a.frame_number - b.frame_number);

    const trackedFrames: Frame[] = [];
    for (const frame of enrichedFrames) {
      const trackedFrame = await trackFrame(sessionId, frame);
      trackedFrames.push(trackedFrame);
    }

    return {
      session_id: sessionId,
      fps,
      total_frames: trackedFrames.length,
      trackedFrames,
    };
  } finally {
    await Promise.allSettled([
      cleanupSessionFiles(sessionDir, videoPath),
      deleteTrackerSession(sessionId).catch((err: unknown) => {
        console.warn(
          `[Orchestrator] Failed to delete tracker session ${sessionId}:`,
          err,
        );
      }),
    ]);
  }
}