/**
 * @fileoverview Centralized Express error-handling middleware and lightweight
 * HTTP error class for consistent JSON error responses.
 */

import axios, { AxiosError } from 'axios';
import { NextFunction, Request, Response } from 'express';
import multer from 'multer';

/**
 * Represents an application error associated with an explicit HTTP status code.
 */
export class HttpError extends Error {
  /** HTTP status code to return to the client. */
  public readonly statusCode: number;

  /**
   * Creates an instance of HttpError.
   *
   * @param statusCode HTTP status code (e.g., 400, 404).
   * @param message Human-readable error description.
   */
  constructor(statusCode: number, message: string) {
    super(message);
    this.name = 'HttpError';
    this.statusCode = statusCode;
  }
}

/**
 * Extracts a descriptive message from a downstream FastAPI Axios error.
 *
 * @param err The Axios error instance.
 * @return A formatted error message including upstream details if available.
 */
function formatAxiosError(err: AxiosError): string {
  const responseData = err.response?.data as { detail?: unknown } | undefined;
  const detail =
    typeof responseData?.detail === 'string' ? `: ${responseData.detail}` : '';
  return `Upstream microservice error (${err.message})${detail}`;
}

/**
 * Express error-handling middleware that formats all unhandled errors into
 * a standardized JSON payload.
 *
 * @param err The error thrown or passed via `next(err)`.
 * @param _req Express request object.
 * @param res Express response object.
 * @param _next Express next middleware function.
 */
export function errorHandler(
  err: unknown,
  _req: Request,
  res: Response,
  _next: NextFunction,
): void {
  console.error('[Orchestrator Error]', err);

  if (err instanceof HttpError) {
    res.status(err.statusCode).json({ error: err.message });
    return;
  }

  if (err instanceof multer.MulterError) {
    const statusCode = err.code === 'LIMIT_FILE_SIZE' ? 413 : 400;
    res.status(statusCode).json({ error: `Upload error: ${err.message}` });
    return;
  }

  if (axios.isAxiosError(err)) {
    const statusCode = err.response ? 502 : 504;
    res.status(statusCode).json({ error: formatAxiosError(err) });
    return;
  }

  if (err instanceof Error) {
    const isClientFileError = err.message.startsWith('Invalid file type');
    const statusCode = isClientFileError ? 400 : 500;
    res.status(statusCode).json({ error: err.message });
    return;
  }

  res.status(500).json({ error: 'An unexpected internal server error occurred.' });
}