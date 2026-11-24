/**
 * @fileoverview Express route definitions for video and frame analysis endpoints.
 */

import { Router } from 'express';
import { handleAnalyzeVideo } from '../controllers/analyze.controller';
import { uploadVideo } from '../middlewares/upload.middleware';

/**
 * Router handling analysis endpoints mounted under `/api/analyze`.
 */
export const analyzeRouter = Router();

/**
 * POST /api/analyze/video
 * Accepts a multipart video upload (`video` field) with an optional `fps`
 * parameter and runs the full tracking orchestration pipeline.
 */
analyzeRouter.post('/video', uploadVideo, handleAnalyzeVideo);