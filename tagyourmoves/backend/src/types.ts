/**
 * @fileoverview Domain interfaces matching the Python FastAPI contracts
 * and internal orchestrator pipeline data structures.
 */

/**
 * Base bounding box and confidence fields shared by all detected entities.
 */
export interface DetectedObject {
  /** Model detection confidence score in the range [0, 1]. */
  confidence: number;
  /** X-coordinate of the top-left corner of the bounding box. */
  x1: number;
  /** Y-coordinate of the top-left corner of the bounding box. */
  y1: number;
  /** X-coordinate of the bottom-right corner of the bounding box. */
  x2: number;
  /** Y-coordinate of the bottom-right corner of the bounding box. */
  y2: number;
}

/**
 * Represents a detected basketball in a single frame.
 */
export type Ball = DetectedObject;

/**
 * Represents a detected basketball rim in a single frame.
 */
export type Rim = DetectedObject;

/**
 * Represents a detected player in a single frame, enriched across pipeline stages.
 */
export interface Player extends DetectedObject {
  /** Persistent ID assigned by the tracker service, or null before tracking. */
  tracking_id?: number | null;
  /** Team identifier (1 for team 1, 2 for team 2), or null if unassigned/referee. */
  team?: number | null;
  /** Recognized jersey number, or null if not detected. */
  shirt_number?: number | null;
  /** Whether the player is currently holding the ball. */
  has_ball: boolean;
  /** 512-dimensional appearance embedding vector populated by player-signature. */
  signature?: number[] | null;
}

/**
 * Represents a single analyzed video frame and all objects detected within it.
 */
export interface Frame {
  /** Zero-based or sequential index of the frame in the extracted sequence. */
  frame_number: number;
  /** Timestamp of the frame in milliseconds relative to the start of the video. */
  timestamp_ms: number;
  /** Detected basketball in the frame, or null if none was found. */
  ball: Ball | null;
  /** List of detected basketball rims in the frame. */
  rims: Rim[];
  /** List of detected players in the frame. */
  players: Player[];
}

/**
 * Metadata for a single image frame extracted to disk by FFmpeg.
 */
export interface ExtractedFrame {
  /** Sequential frame number matching the pipeline order. */
  frameNumber: number;
  /** Computed timestamp in milliseconds based on frame index and target FPS. */
  timestampMs: number;
  /** Absolute filesystem path to the extracted frame image. */
  filePath: string;
}

/**
 * Payload sent sequentially to the stateful tracker microservice per frame.
 * Flattens session_id alongside all Frame fields.
 */
export interface TrackRequest extends Frame {
  /** Unique identifier isolating the current video analysis session. */
  session_id: string;
}

/**
 * Response returned by the orchestrator after analyzing a full video sequence.
 */
export interface VideoAnalysisResponse {
  /** Unique session identifier used during processing. */
  session_id: string;
  /** Frame rate at which the video was sampled and analyzed. */
  fps: number;
  /** Total number of processed frames returned. */
  total_frames: number;
  /** Ordered sequence of frames with persistent tracking IDs assigned. */
  trackedFrames: Frame[];
}