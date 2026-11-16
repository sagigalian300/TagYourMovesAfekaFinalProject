# Data Pipeline

## Download and split a video

[`download_and_split_clips.py`](./download_and_split_clips.py) downloads a YouTube video and splits it into clips whenever it detects a camera transition.

The script downloads the configured video to `data/raw_videos/nba_diverse_threes.mp4` at a maximum resolution of 720p. It detects scenes with PySceneDetect's content detector, then uses FFmpeg to write each unclassified scene to `data/dataset_clips/`. Once labeled, keep original clips in their class folder and augmented versions in their corresponding augmentation folder.

Newly split clips use the dataset naming format:

```text
clip_<source-id>_<4-digit-clip-id>_unknown_unknown.mp4
```

The source ID comes from the source video's filename, and clip IDs are numbered from `0001`. The shot type and result are `unknown` until the clips are labeled. See [the dataset clip naming guide](../data/dataset_clips/README.md) for the full naming convention.

Run the script from the repository root:

```powershell
python data_pipeline/download_and_split_clips.py
```

The pipeline requires the packages in the repository's `requirements.txt` and FFmpeg on the system.
