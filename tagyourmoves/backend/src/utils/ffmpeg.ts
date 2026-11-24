/**
 * @fileoverview Utility functions for extracting video frames with FFmpeg
 * and cleaning up temporary session artifacts on disk.
 */

import { execFile } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { promisify } from 'util';
import { ExtractedFrame } from '../types';

const execFileAsync = promisify(execFile);

/**
 * Extracts sequential JPEG frames from a video file at the specified frame rate.
 *
 * @param videoPath Absolute path to the source video file.
 * @param outputDir Directory where extracted frame images will be written.
 * @param fps Target extraction frame rate (frames per second).
 * @return Ordered list of extracted frame metadata objects.
 * @throws Error if FFmpeg fails to execute or produces no output frames.
 */
export async function extractFrames(
  videoPath: string,
  outputDir: string,
  fps: number,
): Promise<ExtractedFrame[]> {
  await fs.mkdir(outputDir, { recursive: true });

  const outputPattern = path.join(outputDir, 'frame_%06d.jpg');
  const ffmpegArgs = [
    '-hide_banner',
    '-loglevel',
    'error',
    '-i',
    videoPath,
    '-vf',
    `fps=${fps}`,
    '-q:v',
    '2',
    '-start_number',
    '0',
    outputPattern,
  ];

  await execFileAsync('ffmpeg', ffmpegArgs);

  const files = await fs.readdir(outputDir);
  const frameFiles = files.filter((file) => file.endsWith('.jpg')).sort();

  return frameFiles.map((fileName, index) => ({
    frameNumber: index,
    timestampMs: Math.round((index * 1000) / fps),
    filePath: path.join(outputDir, fileName),
  }));
}

/**
 * Removes the temporary session frames directory and optional uploaded video file.
 * Uses `force: true` so missing paths do not raise errors during cleanup.
 *
 * @param sessionDir Path to the temporary directory holding extracted frames.
 * @param videoPath Optional path to the uploaded source video file.
 */
export async function cleanupSessionFiles(
  sessionDir: string,
  videoPath?: string,
): Promise<void> {
  const removals: Promise<void>[] = [
    fs.rm(sessionDir, { recursive: true, force: true }),
  ];

  if (videoPath) {
    removals.push(fs.rm(videoPath, { force: true }));
  }

  await Promise.allSettled(removals);
}