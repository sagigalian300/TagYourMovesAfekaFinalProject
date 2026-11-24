/**
 * @fileoverview HTTP client for communicating with the stateless
 * player-signature FastAPI microservice.
 */

import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';
import path from 'path';
import { config } from '../config';
import { Frame } from '../types';

const signatureHttp = axios.create({
  baseURL: config.services.signatureUrl,
  timeout: config.pipeline.requestTimeoutMs,
  maxBodyLength: Infinity,
  maxContentLength: Infinity,
});

/**
 * Sends a frame image along with its detected objects to the player-signature
 * service to compute 512-dimensional Re-ID embeddings for each player.
 *
 * @param imagePath Absolute filesystem path to the extracted frame image.
 * @param frame The Frame object populated by the object-detection service.
 * @return A promise resolving to the Frame enriched with player signatures.
 */
export async function extractPlayerSignatures(
  imagePath: string,
  frame: Frame,
): Promise<Frame> {
  const form = new FormData();
  form.append(
    'image',
    fs.createReadStream(imagePath),
    path.basename(imagePath),
  );
  form.append('frame_data', JSON.stringify(frame));

  const response = await signatureHttp.post<Frame>(
    '/extract-signatures',
    form,
    {
      headers: form.getHeaders(),
    },
  );

  return {
    ...response.data,
    frame_number: frame.frame_number,
    timestamp_ms: frame.timestamp_ms,
    ball: response.data.ball ?? null,
    rims: response.data.rims ?? [],
    players: response.data.players ?? [],
  };
}