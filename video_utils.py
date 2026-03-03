"""
video_utils.py - FFmpeg merge, frame extraction, and file handling utilities.
"""

import subprocess
import wave
import tempfile
import os
import shutil
from pathlib import Path


def check_ffmpeg():
    """Verify FFmpeg is installed and accessible."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg not found. Install it with: brew install ffmpeg (macOS) "
            "or sudo apt install ffmpeg (Linux)"
        )


def pcm_to_wav(pcm_data: bytes, output_path: str, sample_rate: int = 48000, channels: int = 2) -> str:
    """Convert raw 16-bit PCM bytes to a WAV file."""
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit = 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return output_path


def merge_video_and_score(video_path: str, score_wav_path: str, output_path: str) -> str:
    """
    Layer Lyria musical score under Veo's ambient audio.

    Ambient audio from Veo is reduced to 30% volume, musical score at 70%.
    Uses amix to blend both audio tracks, keeping the shortest duration.
    """
    check_ffmpeg()

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', score_wav_path,
        '-filter_complex',
        '[0:a]volume=0.3[ambient];'
        '[1:a]volume=0.7[music];'
        '[ambient][music]amix=inputs=2:duration=shortest[out]',
        '-map', '0:v',
        '-map', '[out]',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg merge failed: {result.stderr}")

    return output_path


def merge_video_score_only(video_path: str, score_wav_path: str, output_path: str) -> str:
    """
    Replace video audio entirely with Lyria score.
    Fallback if Veo video has no ambient audio track.
    """
    check_ffmpeg()

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', score_wav_path,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg merge failed: {result.stderr}")

    return output_path


def extract_keyframes(video_path: str, output_dir: str, interval_seconds: float = 2.5) -> list[str]:
    """
    Extract key frames from a video clip at regular intervals.
    Used for multi-frame video clip support (stretch goal).

    Returns list of extracted frame file paths.
    """
    check_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f'fps=1/{interval_seconds}',
        '-q:v', '2',  # High quality JPEG (2 = near lossless)
        os.path.join(output_dir, 'frame_%04d.jpg')
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Frame extraction failed: {result.stderr}")

    frames = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith('frame_')]
    )
    return frames


def stitch_video_clips(clip_paths: list[str], output_path: str) -> str:
    """
    Concatenate multiple video clips into one continuous video.
    Used for multi-frame video clip support (stretch goal).
    """
    check_ffmpeg()

    # Create a concat file list
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for clip in clip_paths:
            f.write(f"file '{os.path.abspath(clip)}'\n")
        concat_file = f.name

    try:
        # Re-encode to ensure compatibility across clips with different codecs/resolutions
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-c:a', 'aac', '-b:a', '192k',
            '-movflags', '+faststart',
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Video stitching failed: {result.stderr}")
    finally:
        os.unlink(concat_file)

    return output_path


def get_video_duration(video_path: str) -> float:
    """Get the duration of a video file in seconds."""
    check_ffmpeg()

    cmd = [
        'ffprobe', '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    return float(result.stdout.strip())


def has_audio_stream(video_path: str) -> bool:
    """Check if a video file contains an audio stream."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'a',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return bool(result.stdout.strip())


def extract_all_frames(video_path: str, output_dir: str) -> tuple[list[str], float]:
    """
    Extract ALL frames from a video at original framerate.

    Returns (list of frame paths, fps of the original video).
    """
    check_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)

    # Get the original video FPS
    cmd_fps = [
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd_fps, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe fps failed: {result.stderr}")
    fps_str = result.stdout.strip()
    # fps_str is like "24/1" or "30000/1001"
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)

    # Extract all frames as PNG (lossless for colorization quality)
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-q:v', '1',
        os.path.join(output_dir, 'frame_%06d.png')
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Frame extraction failed: {result.stderr}")

    frames = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith('frame_')]
    )
    return frames, fps


def reassemble_frames(
    frame_dir: str, output_path: str, fps: float,
    audio_path: str = None,
) -> str:
    """
    Reassemble colorized frames back into a video at the original FPS.

    Args:
        frame_dir: Directory containing colorized_NNNNNN.png files
        output_path: Where to write the final MP4
        fps: Frames per second for the output
        audio_path: Optional audio track to mux in (WAV or AAC)
    """
    check_ffmpeg()

    input_pattern = os.path.join(frame_dir, 'colorized_%06d.png')

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', input_pattern,
    ]

    if audio_path:
        cmd.extend(['-i', audio_path])

    cmd.extend([
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p',
    ])

    if audio_path:
        cmd.extend(['-c:a', 'aac', '-b:a', '192k', '-shortest'])

    cmd.extend(['-movflags', '+faststart', output_path])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Frame reassembly failed: {result.stderr}")

    return output_path
