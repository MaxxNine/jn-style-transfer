import subprocess
import os

def add_audio_to_video(video_path: str, output_path: str, audio_source: str):
        audio_file = "audio.aac"
        # Extract audio
        subprocess.call(["ffmpeg", "-i", audio_source, "-vn", "-acodec", "copy", audio_file])

        # Merge audio and video
        subprocess.call(["ffmpeg", "-i", video_path, "-i", audio_file, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", output_path])

        # Delete temporary audio file
        os.remove(audio_file)
