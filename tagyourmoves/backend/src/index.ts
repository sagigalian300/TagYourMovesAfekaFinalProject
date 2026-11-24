/**
 * @fileoverview Application entry point for the Tag Your Moves orchestrator
 * server. Configures Express middlewares, mounts API routes, and starts the
 * HTTP listener.
 */

import cors from 'cors';
import express, { Request, Response } from 'express';
import { config } from './config';
import { errorHandler } from './middlewares/error.middleware';
import { analyzeRouter } from './routes/analyze.routes';

const app = express();

app.use(cors());
app.use(express.json());

/**
 * GET /health
 * Lightweight liveness check for container orchestration and local monitoring.
 */
app.get('/health', (_req: Request, res: Response) => {
  res.status(200).json({ status: 'ok' });
});

app.use('/api/analyze', analyzeRouter);

app.use(errorHandler);

app.listen(config.port, () => {
  console.log(`[Orchestrator] Listening on port ${config.port}`);
  console.log(`[Orchestrator] Detection service: ${config.services.detectionUrl}`);
  console.log(`[Orchestrator] Signature service: ${config.services.signatureUrl}`);
  console.log(`[Orchestrator] Tracker service:   ${config.services.trackerUrl}`);
});