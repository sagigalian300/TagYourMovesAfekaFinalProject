/**
 * @fileoverview Centralized configuration module for the orchestrator service.
 * Loads environment variables and exports validated, strongly-typed settings.
 */

import dotenv from 'dotenv';
import path from 'path';

dotenv.config();

/**
 * Parses a string environment variable into a positive finite number.
 *
 * @param value The raw string value from `process.env`.
 * @param fallback The default value to use if parsing fails or is non-positive.
 * @return The parsed positive number, or the fallback value.
 */
function parsePositiveNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

/**
 * Application-wide configuration object populated from environment variables.
 */
export const config = {
  /** HTTP port on which the Express server listens. */
  port: parsePositiveNumber(process.env.PORT, 3005),

  /** Base URLs for downstream Python FastAPI microservices. */
  services: {
    detectionUrl: process.env.DETECTION_SERVICE_URL || 'http://localhost:8001',
    signatureUrl: process.env.SIGNATURE_SERVICE_URL || 'http://localhost:8002',
    trackerUrl: process.env.TRACKER_SERVICE_URL || 'http://localhost:8003',
  },

  /** Execution constraints for frame extraction and microservice requests. */
  pipeline: {
    /** Maximum number of frames processed concurrently across stateless services. */
    concurrencyLimit: parsePositiveNumber(process.env.CONCURRENCY_LIMIT, 5),
    /** Axios HTTP timeout in milliseconds to prevent GPU worker stalls. */
    requestTimeoutMs: parsePositiveNumber(process.env.REQUEST_TIMEOUT_MS, 30000),
    /** Default frame extraction rate when omitted by the client. */
    defaultFps: parsePositiveNumber(process.env.DEFAULT_FPS, 15),
    /** Upper bound allowed for client-requested extraction FPS. */
    maxFps: parsePositiveNumber(process.env.MAX_FPS, 30),
  },

  /** Temporary filesystem storage settings for uploaded videos and frames. */
  storage: {
    /** Absolute path to the root temporary directory. */
    tempDir: path.resolve(process.env.TEMP_DIR || './tmp'),
    /** Maximum allowed video upload size in bytes. */
    maxFileSizeBytes: parsePositiveNumber(process.env.MAX_FILE_SIZE_MB, 250) * 1024 * 1024,
  },
} as const;