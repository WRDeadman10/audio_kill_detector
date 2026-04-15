import ffmpeg
import os

def extract_audio(video_path: str, output_path: str) -> None:
    "Extract audio from video using FFmpeg"
    try:
        # Check if output file already exists and handle overwrite control
        if os.path.exists(output_path):
            overwrite = input(f"File {output_path} already exists. Overwrite? (y/n): ")
            if overwrite.lower() != 'y':
                print("Operation cancelled.")
                return

        # Extract audio using ffmpeg-python
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec='pcm_s16le', ar=22050, ac=1)
            .run()
        )

        print(f"Audio extracted successfully to {output_path}")
    except Exception as e:
        print(f"Error extracting audio: {e}")
