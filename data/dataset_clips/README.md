# Dataset Clip Naming

Use the same filename format for every dataset clip:

```text
clip_<source-id>_<clip-id>_<shot-type>_<result>[_<augmentation>].mp4
```

Fields:

- `source-id`: short identifier for the source video, such as `nba01`.
- `clip-id`: four-digit clip identifier, zero-padded (for example, `0001` or `0121`). Keep the same ID for a clip and its augmented versions.
- `shot-type`: shot distance, such as `2pt` or `3pt`.
- `result`: `made` or `missed`.
- `augmentation`: optional transformation suffix, such as `mirrored`. Omit it for the unaugmented clip; do not use an `original` suffix.

Examples:

```text
clip_nba01_0001_3pt_made.mp4
clip_nba01_0001_3pt_made_mirrored.mp4
clip_nba01_0121_2pt_missed.mp4
```

Use lowercase letters, digits, and underscores, and keep the field order consistent. A different source video should have a different `source-id`, so its clip IDs can start from `0001` without colliding with clips from other sources.
