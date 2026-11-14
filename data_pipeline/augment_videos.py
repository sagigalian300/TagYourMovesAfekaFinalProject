import subprocess
from pathlib import Path

def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).parent.parent

def mirror_videos(input_dir: Path, output_dir: Path):
    """
    Horizontally mirror unaugmented MP4 clips into the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = [
        f for f in input_dir.iterdir()
        if f.suffix.lower() == ".mp4" and not f.stem.endswith("_mirrored")
    ]
    print(f"Found {len(video_files)} videos to augment.")
    
    for video_path in video_files:
        out_filename = f"{video_path.stem}_mirrored{video_path.suffix}"
        output_path = output_dir / out_filename
        
        # ffmpeg command for fast horizontal flipping (-vf hflip)
        # -c:a copy preserves the original audio without re-encoding
        command = [
            "ffmpeg", 
            "-y",                   # Overwrite output files without asking
            "-i", str(video_path),  # Input file
            "-vf", "hflip",         # Video filter: horizontal flip
            "-c:a", "copy",         # Copy audio stream without changes
            str(output_path)        # Output file
        ]
        
        print(f"Mirroring {video_path.name} -> {out_filename}...")
        # Run the ffmpeg command. Hiding standard output to keep the console clean.
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            check=True
        )
        
    print(f"\nMirrored videos saved to '{output_dir}'.")

def main():
    root_dir = get_project_root()
    
    clips_folder = root_dir / "data" / "dataset_clips"
    input_folder = clips_folder / "successful_3_point_shooting"
    output_folder = clips_folder / "successful_3_point_shooting_mirrored"
    
    mirror_videos(input_folder, output_folder)

if __name__ == "__main__":
    main()