/**
 * @fileoverview HTTP client for communicating with the stateless
 * object-detection FastAPI microservice.
 */

import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';
import path from 'path';
import { config } from '../config';
import { ExtractedFrame, Frame } from '../types';

const detectionHttp = axios.create({
  baseURL: config.services.detectionUrl,
  timeout: config.pipeline.requestTimeoutMs,
  maxBodyLength: Infinity,
  maxContentLength: Infinity,
});

/**
 * Sends a single extracted frame image to the object-detection service
 * and returns the populated Frame contract.
 *
 * @param extractedFrame Metadata and disk path of the extracted frame image.
 * @return A promise resolving to the Frame containing detected ball, rims, and players.
 */
export async function detectObjectsInFrame(
  extractedFrame: ExtractedFrame,
): Promise<Frame> {
  const form = new FormData();
  form.append(
    'image',
    fs.createReadStream(extractedFrame.filePath),
    path.basename(extractedFrame.filePath),
  );
  form.append('frame_number', extractedFrame.frameNumber.toString());
  form.append('timestamp_ms', extractedFrame.timestampMs.toString());

  const response = await detectionHttp.post<Frame>('/detect', form, {
    headers: form.getHeaders(),
  });

  return {
    ...response.data,
    frame_number: extractedFrame.frameNumber,
    timestamp_ms: extractedFrame.timestampMs,
    ball: response.data.ball ?? null,
    rims: response.data.rims ?? [],
    players: response.data.players ?? [],
  };
}