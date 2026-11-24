/**
 * @fileoverview HTTP client for communicating with the stateful
 * object-tracking FastAPI microservice.
 */

import axios from 'axios';
import { config } from '../config';
import { Frame, TrackRequest } from '../types';

const trackerHttp = axios.create({
  baseURL: config.services.trackerUrl,
  timeout: config.pipeline.requestTimeoutMs,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Sends a single enriched frame sequentially to the tracker microservice
 * to assign persistent tracking IDs across frames.
 *
 * @param sessionId Unique identifier isolating the current video session.
 * @param frame Frame containing object detections and player signatures.
 * @return A promise resolving to the Frame populated with tracking IDs.
 */
export async function trackFrame(
  sessionId: string,
  frame: Frame,
): Promise<Frame> {
  const payload: TrackRequest = {
    session_id: sessionId,
    ...frame,
  };

  const response = await trackerHttp.post<Frame>('/track', payload);

  return {
    ...response.data,
    frame_number: frame.frame_number,
    timestamp_ms: frame.timestamp_ms,
    ball: response.data.ball ?? null,
    rims: response.data.rims ?? [],
    players: response.data.players ?? [],
  };
}

/**
 * Deletes the in-memory tracking state associated with a session
 * on the tracker microservice.
 *
 * @param sessionId Unique identifier of the session to clean up.
 * @return A promise that resolves once the session cleanup request completes.
 */
export async function deleteTrackerSession(sessionId: string): Promise<void> {
  await trackerHttp.delete(`/session/${encodeURIComponent(sessionId)}`);
}