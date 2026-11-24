/**
 * @fileoverview HTTP controller handling video analysis requests.
 * Validates incoming upload metadata and extraction parameters, delegates
 * processing to the analysis service, and formats the HTTP response.
 */

import { NextFunction, Request, Response } from 'express';
import { promises as fs } from 'fs';
import { config } from '../config';
import { HttpError } from '../middlewares/error.middleware';
import { analyzeVideo } from '../services/analyze.service';

/**
 * Parses and validates the requested extraction frame rate from the request.
 *
 * @param rawFps Raw FPS value provided in the request body or query string.
 * @return A validated integer frame rate within configured bounds.
 * @throws HttpError if the provided FPS value is not a valid positive integer or exceeds maxFps.
 */
function parseTargetFps(rawFps: unknown): number {
  if (rawFps === undefined || rawFps === null || rawFps === '') {
    return config.pipeline.defaultFps;
  }

  const parsed = Number(rawFps);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > config.pipeline.maxFps) {
    throw new HttpError(
      400,
      `Invalid 'fps' parameter. Expected an integer between 1 and ${config.pipeline.maxFps}.`,
    );
  }

  return parsed;
}

/**
 * Handles `POST /api/analyze/video` requests.
 * Extracts the uploaded video file and target FPS, runs the orchestration pipeline,
 * and returns the tracked frames payload.
 *
 * @param req Express request containing the uploaded `video` file and optional `fps`.
 * @param res Express response used to return the `VideoAnalysisResponse` JSON.
 * @param next Express next function forwarding errors to the centralized handler.
 */
export async function handleAnalyzeVideo(
  req: Request,
  res: Response,
  next: NextFunction,
): Promise<void> {
  try {
    if (!req.file) {
      throw new HttpError(400, "Missing video file. Upload a file under the 'video' field.");
    }

    let fps: number;
    try {
      const rawFps = req.body?.fps ?? req.query?.fps;
      fps = parseTargetFps(rawFps);
    } catch (validationErr) {
      await fs.rm(req.file.path, { force: true });
      throw validationErr;
    }

    const result = await analyzeVideo(req.file.path, fps);
    res.status(200).json(result);
  } catch (err) {
    next(err);
  }
}