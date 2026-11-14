# Dataset Clip Naming

Keep original and mirrored clips in their separate folders:

- `successful_3_point_shooting/` contains the unaugmented successful three-point clips.
- `successful_3_point_shooting_mirrored/` contains their mirrored augmented versions.
- Newly split, not-yet-labeled clips are written directly to this directory until they are classified.

Use the same filename format in all folders:

```text
clip_<source-id>_<clip-id>_<shot-type>_<result>[_<augmentation>].mp4
```

Fields:

- `source-id`: the source video's filename without its extension, in lowercase (for example, `nba_diverse_threes`).
- `clip-id`: four-digit clip identifier, zero-padded (for example, `0001` or `0121`). Keep the same ID for a clip and its augmented versions.
- `shot-type`: shot distance, such as `2pt` or `3pt`. Use `unknown` until the clip is labeled.
- `result`: `made` or `missed`. Use `unknown` until the clip is labeled.
- `augmentation`: optional transformation suffix, such as `mirrored`. Omit it for the unaugmented clip; do not use an `original` suffix.

Examples:

```text
successful_3_point_shooting/clip_nba_diverse_threes_0001_3pt_made.mp4
successful_3_point_shooting_mirrored/clip_nba_diverse_threes_0001_3pt_made_mirrored.mp4
successful_3_point_shooting/clip_nba_diverse_threes_0121_2pt_missed.mp4
clip_nba_diverse_threes_0001_unknown_unknown.mp4
```

Use lowercase letters, digits, and underscores, and keep the field order consistent. The clip-splitting pipeline uses the source video filename as the `source-id`, numbers clips from `0001`, and sets both labels to `unknown` until they are annotated. A different source video gets a different `source-id`, so its clip IDs can start from `0001` without colliding with clips from other sources.
