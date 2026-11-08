import os
import subprocess
from pathlib import Path

def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).parent.parent

def mirror_videos(input_dir: Path, output_dir: Path):
    """
    Applies a horizontal flip (mirror) augmentation to all MP4 videos in the input directory.
    Uses ffmpeg for fast processing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all mp4 files in the input directory
    video_files = [f for f in input_dir.iterdir() if f.suffix.lower() == '.mp4']
    print(f"Found {len(video_files)} videos to augment.")
    
    for video_path in video_files:
        # Create the output filename, e.g., clip_001_mirrored.mp4
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
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    print(f"\nSuccess! All mirrored videos saved to '{output_dir.name}/'")

def main():
    root_dir = get_project_root()
    
    # Updated to match your exact folder name
    input_folder = root_dir / "data" / "dataset_clips" / "successful_3_point_shooting"
    
    # New folder for the augmented videos to keep them strictly separated
    output_folder = root_dir / "data" / "dataset_clips" / "successful_3_point_shooting_mirrored"
    
    mirror_videos(input_folder, output_folder)

if __name__ == "__main__":
    main()