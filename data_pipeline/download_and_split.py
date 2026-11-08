import os
from pathlib import Path
import yt_dlp
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect import split_video_ffmpeg

def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).parent.parent

def download_nba_video(url: str, output_path: Path) -> Path:
    """Downloads a video from YouTube at a maximum resolution of 720p."""
    print(f"Downloading video from: {url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]',
        'outtmpl': str(output_path),
        'quiet': False
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def split_video_to_clips(video_path: Path, output_dir: Path):
    """Analyzes the video for camera transitions and splits it into shorter clips."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    # threshold=50.0 makes it less sensitive to camera flashes
    # min_scene_len=90 ensures clips are at least 3 seconds long (assuming 30 FPS)
    scene_manager.add_detector(ContentDetector(threshold=50.0, min_scene_len=90))
    
    print(f"Detecting scenes in {video_path.name}...")
    scene_manager.detect_scenes(video, show_progress=True)
    
    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        print("No scenes found.")
        return

    print("Splitting video...")
    split_video_ffmpeg(
        input_video_path=str(video_path),
        scene_list=scene_list,
        output_dir=str(output_dir),
        output_file_template="clip_$SCENE_NUMBER.mp4",
        show_progress=True
    )

def main():
    root_dir = get_project_root()
    raw_video_path = root_dir / "data" / "raw_videos" / "nba_diverse_threes.mp4"
    clips_folder = root_dir / "data" / "dataset_clips"
    
    youtube_url = "https://www.youtube.com/watch?v=f7tobLeKN90"
    
    download_nba_video(youtube_url, raw_video_path)
    split_video_to_clips(raw_video_path, clips_folder)

if __name__ == "__main__":
    main()
