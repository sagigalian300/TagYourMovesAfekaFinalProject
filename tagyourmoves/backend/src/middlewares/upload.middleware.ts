/**
 * @fileoverview Multer middleware configuration for handling incoming video
 * file uploads to temporary disk storage.
 */

import { randomUUID } from 'crypto';
import { Request } from 'express';
import fs from 'fs';
import multer, { FileFilterCallback } from 'multer';
import path from 'path';
import { config } from '../config';

fs.mkdirSync(config.storage.tempDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, config.storage.tempDir);
  },
  filename: (_req, file, cb) => {
    const extension = path.extname(file.originalname) || '.mp4';
    cb(null, `${randomUUID()}${extension}`);
  },
});

/**
 * Validates that the uploaded file has a video MIME type.
 *
 * @param _req Express request object.
 * @param file Metadata of the uploaded file.
 * @param cb Multer file filter callback.
 */
function videoFileFilter(
  _req: Request,
  file: Express.Multer.File,
  cb: FileFilterCallback,
): void {
  if (!file.mimetype.startsWith('video/')) {
    cb(new Error('Invalid file type. Only video files are accepted.'));
    return;
  }
  cb(null, true);
}

const upload = multer({
  storage,
  fileFilter: videoFileFilter,
  limits: {
    fileSize: config.storage.maxFileSizeBytes,
  },
});

/**
 * Express middleware expecting a single video file under the `video` form field.
 */
export const uploadVideo = upload.single('video');