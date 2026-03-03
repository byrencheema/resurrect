"""
pipeline.py - Core AI pipeline: analyze → colorize → animate → score → merge.

Orchestrates Gemini 3.1 Pro (scene analysis), NanoBanana 2 (colorization),
Veo 3.1 (animation), and Lyria RealTime (musical score) into a single pipeline.

Two pipelines:
  1. resurrect_video() — Video-to-video (primary): upload B&W film → full resurrected video
  2. resurrect_image() — Photo-to-video: upload B&W photo → 8-sec resurrected clip
"""

import asyncio
import json
import os
import shutil
import tempfile
import io

from google import genai
from google.genai import types
from PIL import Image
import numpy as np

from lyria_scorer import generate_score
from video_utils import (
    pcm_to_wav,
    merge_video_and_score,
    merge_video_score_only,
    has_audio_stream,
    extract_keyframes,
    stitch_video_clips,
    get_video_duration,
)


def _parse_json_response(text: str) -> dict:
    """Parse JSON from a model response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    return json.loads(text)


# ---------------------------------------------------------------------------
# Scene Analysis — Image (single photo)
# ---------------------------------------------------------------------------

async def analyze_scene(client: genai.Client, image_bytes: bytes) -> dict:
    """
    Analyze a B&W photograph using Gemini 3.1 Pro.

    Returns structured JSON with era, setting, colors, movement, and music metadata
    that drives the rest of the pipeline.
    """
    response = await client.aio.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            """Analyze this old black and white photograph. Return valid JSON only, with no markdown formatting or code fences:
{
    "era": "estimated decade (e.g. 1920s)",
    "setting": "location description",
    "people": "description of people, clothing, expressions",
    "mood": "emotional tone",
    "colors": {
        "sky": "historically accurate color",
        "buildings": "historically accurate color",
        "clothing": "historically accurate colors",
        "skin": "natural skin tone description",
        "lighting": "warm/cool, time of day"
    },
    "movement": "what subtle motion would exist in this scene",
    "ambient_sounds": "what you would hear",
    "music": {
        "genre": "period-appropriate genre",
        "tempo": "slow/medium/fast",
        "instruments": "specific instruments",
        "mood": "musical mood description"
    }
}"""
        ],
    )

    return _parse_json_response(response.text)


# ---------------------------------------------------------------------------
# Scene Analysis — Video (full clip understanding)
# ---------------------------------------------------------------------------

async def analyze_video(client: genai.Client, video_path: str, num_scenes: int = None) -> dict:
    """
    Analyze a full B&W video clip using Gemini 3.1 Pro's video understanding.

    Uploads the entire video to Gemini and gets a comprehensive, timestamped
    scene-by-scene analysis. This is far superior to analyzing a single frame
    because it understands:
      - Scene transitions and narrative flow
      - Different moods across the clip
      - Movement and action context
      - Period-appropriate music that evolves with the content

    Returns structured JSON with overall analysis + per-scene breakdowns.
    """
    # Upload the video file to Gemini Files API
    uploaded_file = await client.aio.files.upload(file=video_path)

    prompt = """Analyze this old black and white film clip for restoration. Return valid JSON only, no markdown fences.

Return a JSON object with:
1. "overall" — global analysis of the entire clip
2. "scenes" — array of scene segments with timestamps

Format:
{
    "overall": {
        "era": "estimated decade (e.g. 1920s)",
        "title_guess": "best guess at what this film/footage is",
        "setting": "general location and context",
        "mood": "overall emotional tone",
        "narrative": "brief description of what happens across the clip",
        "music": {
            "genre": "period-appropriate genre for the overall clip",
            "tempo": "slow/medium/fast",
            "instruments": "specific instruments",
            "mood": "musical mood that fits the overall piece"
        }
    },
    "scenes": [
        {
            "timestamp": "MM:SS-MM:SS",
            "description": "what happens in this segment",
            "people": "description of people, clothing, expressions",
            "mood": "emotional tone of this specific scene",
            "colors": {
                "sky": "historically accurate color",
                "buildings": "historically accurate color",
                "clothing": "historically accurate colors",
                "skin": "natural skin tone description",
                "lighting": "warm/cool, time of day"
            },
            "movement": "what motion exists in this scene",
            "ambient_sounds": "what you would hear",
            "veo_prompt": "detailed cinematic prompt to animate this scene: describe the motion, camera angle, and atmosphere in 2-3 sentences"
        }
    ]
}

Be specific about colors — use real color names (burnt sienna, slate blue, ivory) not vague ones.
For veo_prompt, describe the MOTION that should happen, not just the static scene."""

    response = await client.aio.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=[uploaded_file, prompt],
    )

    # Clean up the uploaded file
    try:
        await client.aio.files.delete(name=uploaded_file.name)
    except Exception:
        pass  # Non-critical cleanup

    return _parse_json_response(response.text)


# ---------------------------------------------------------------------------
# NanoBanana 2 — Colorization
# ---------------------------------------------------------------------------

async def colorize_frame(
    client: genai.Client, image_bytes: bytes, scene_data: dict
) -> tuple[Image.Image, bytes]:
    """
    Colorize a B&W image using NanoBanana 2 (gemini-3.1-flash-image-preview).

    Args:
        client: GenAI client
        image_bytes: Raw JPEG bytes of the B&W frame
        scene_data: Scene analysis dict (must have "colors" and "era" keys,
                    works with both per-scene and overall analysis formats)

    Returns (PIL Image, raw image bytes) tuple.
    """
    colors = scene_data.get("colors", {})
    era = scene_data.get("era", "early 20th century")

    response = await client.aio.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            f"""Colorize this black and white photograph with historically accurate colors.
Era: {era}.
Sky: {colors.get('sky', 'natural blue')}. Buildings: {colors.get('buildings', 'natural stone/brick tones')}.
Clothing: {colors.get('clothing', 'period-appropriate colors')}. Skin tones: {colors.get('skin', 'natural skin tones')}.
Lighting: {colors.get('lighting', 'natural daylight')}.
Preserve all original detail, grain, and composition.
Photorealistic result. Do not add or remove any elements."""
        ],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            raw_bytes = part.inline_data.data
            pil_image = Image.open(io.BytesIO(raw_bytes))
            return pil_image, raw_bytes

    raise RuntimeError("NanoBanana 2 did not return an image in the response.")


# ---------------------------------------------------------------------------
# Veo 3.1 — Animation
# ---------------------------------------------------------------------------

async def animate_frame(
    client: genai.Client,
    colorized_pil: Image.Image,
    scene_data: dict,
    output_path: str,
) -> str:
    """
    Animate a colorized image using Veo 3.1, producing an 8-second video
    with native ambient audio.

    Args:
        client: GenAI client
        colorized_pil: Colorized PIL Image
        scene_data: Scene analysis dict. If it has a "veo_prompt" key (from video
                    analysis), uses that. Otherwise builds a prompt from the fields.
        output_path: Where to save the MP4 file

    Returns path to the saved MP4 file.
    """
    # Use the custom Veo prompt from video analysis if available
    if "veo_prompt" in scene_data:
        prompt = (
            f"{scene_data['veo_prompt']} "
            f"Ambient audio: {scene_data.get('ambient_sounds', 'period-appropriate ambient sounds')}. "
            f"Documentary style. Maintain photographic quality. No modern elements."
        )
    else:
        era = scene_data.get("era", "early 20th century")
        movement = scene_data.get("movement", "subtle natural motion")
        ambient = scene_data.get("ambient_sounds", "period-appropriate ambient sounds")
        prompt = (
            f"Gentle, cinematic animation of this {era} photograph coming to life. "
            f"{movement}. Ambient audio: {ambient}. "
            f"Documentary style. Subtle, natural motion only. "
            f"Maintain photographic quality. No modern elements."
        )

    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt,
        image=colorized_pil,
        config=types.GenerateVideosConfig(
            aspect_ratio="16:9",
            person_generation="allow_adult",
        ),
    )

    # Poll until generation is complete (can take 11 sec to 6+ min)
    while not operation.done:
        await asyncio.sleep(5)
        operation = client.operations.get(operation)

    # Download and save the generated video
    generated_video = operation.response.generated_videos[0]
    client.files.download(file=generated_video.video)
    generated_video.video.save(output_path)

    return output_path


# ---------------------------------------------------------------------------
# Photo → Video Pipeline
# ---------------------------------------------------------------------------

async def resurrect_image(client: genai.Client, input_image, tmp_dir: str = None):
    """
    B&W photo → colorized, animated, scored 8-sec video.

    Returns: (colorized_image_path, final_video_path, scene_analysis, status_text)
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="resurrect_")

    if isinstance(input_image, np.ndarray):
        img = Image.fromarray(input_image)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        image_bytes = buf.getvalue()
    elif isinstance(input_image, str):
        with open(input_image, "rb") as f:
            image_bytes = f.read()
    elif isinstance(input_image, bytes):
        image_bytes = input_image
    else:
        raise ValueError(f"Unsupported input type: {type(input_image)}")

    scene_analysis = await analyze_scene(client, image_bytes)

    colorized_pil, _ = await colorize_frame(client, image_bytes, scene_analysis)
    colorized_path = os.path.join(tmp_dir, "colorized.png")
    colorized_pil.save(colorized_path, format="PNG")

    raw_video_path = os.path.join(tmp_dir, "raw_video.mp4")
    animate_task = asyncio.create_task(
        animate_frame(client, colorized_pil, scene_analysis, raw_video_path)
    )
    score_task = asyncio.create_task(
        generate_score(client, scene_analysis, duration_seconds=10)
    )

    _, score_pcm = await asyncio.gather(animate_task, score_task)

    score_wav_path = os.path.join(tmp_dir, "score.wav")
    pcm_to_wav(score_pcm, score_wav_path)

    final_video_path = os.path.join(tmp_dir, "resurrected.mp4")
    if has_audio_stream(raw_video_path):
        merge_video_and_score(raw_video_path, score_wav_path, final_video_path)
    else:
        merge_video_score_only(raw_video_path, score_wav_path, final_video_path)

    return colorized_path, final_video_path, scene_analysis, "Done! Your photograph has been resurrected."


# ---------------------------------------------------------------------------
# Video → Video Pipeline (Primary Feature)
# ---------------------------------------------------------------------------

async def resurrect_video(
    client: genai.Client,
    video_path: str,
    tmp_dir: str = None,
    frame_interval_seconds: float = 3.0,
    progress_callback=None,
) -> dict:
    """
    Full Resurrect pipeline for video clips.

    Pipeline:
      1. Upload full video to Gemini 3.1 Pro → timestamped scene-by-scene analysis
      2. Extract key frames with FFmpeg
      3. Match each frame to its scene segment → per-frame color palette
      4. Colorize each frame with NanoBanana 2 (using scene-specific colors)
      5. Animate each colorized frame with Veo 3.1 (parallel, scene-specific prompts)
      6. Generate Lyria score using overall mood/music analysis
      7. Stitch all clips → merge with score

    Args:
        client: Google GenAI client (with api_version='v1alpha')
        video_path: Path to the input B&W video file
        tmp_dir: Temp directory for intermediate files
        frame_interval_seconds: Extract one frame every N seconds
        progress_callback: Optional async callable(status_str) for UI updates

    Returns dict with colorized_frames, clip_paths, final_video_path, scene_analysis, status.
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="resurrect_vid_")

    frames_dir = os.path.join(tmp_dir, "frames")
    colorized_dir = os.path.join(tmp_dir, "colorized")
    clips_dir = os.path.join(tmp_dir, "clips")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(colorized_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)

    async def status(msg):
        if progress_callback:
            await progress_callback(msg)

    # --- Step 1: Analyze the FULL video with Gemini video understanding ---
    await status("Uploading video to Gemini 3.1 Pro for full scene analysis...")
    video_analysis = await analyze_video(client, video_path)
    overall = video_analysis.get("overall", {})
    scenes = video_analysis.get("scenes", [])

    # --- Step 2: Extract key frames ---
    await status("Extracting key frames from video...")
    frame_paths = extract_keyframes(video_path, frames_dir, frame_interval_seconds)

    if not frame_paths:
        return {
            "colorized_frames": [],
            "clip_paths": [],
            "final_video_path": None,
            "scene_analysis": video_analysis,
            "status": "No frames could be extracted from the video.",
        }

    # --- Step 3: Match frames to scene segments ---
    # Calculate the timestamp (in seconds) for each extracted frame
    frame_timestamps = [i * frame_interval_seconds for i in range(len(frame_paths))]

    def _get_scene_for_timestamp(ts_seconds: float) -> dict:
        """Find the scene segment that covers a given timestamp."""
        for scene in scenes:
            ts_str = scene.get("timestamp", "")
            try:
                start_str, end_str = ts_str.split("-")
                start_parts = start_str.strip().split(":")
                end_parts = end_str.strip().split(":")
                start_sec = int(start_parts[0]) * 60 + int(start_parts[1])
                end_sec = int(end_parts[0]) * 60 + int(end_parts[1])
                if start_sec <= ts_seconds <= end_sec:
                    # Merge era from overall into scene data
                    merged = {**scene, "era": overall.get("era", "early 20th century")}
                    return merged
            except (ValueError, IndexError):
                continue
        # Fallback: use first scene or overall data
        if scenes:
            return {**scenes[0], "era": overall.get("era", "early 20th century")}
        return {
            "era": overall.get("era", "early 20th century"),
            "colors": {},
            "movement": "subtle natural motion",
            "ambient_sounds": "period-appropriate ambient sounds",
        }

    # Build per-frame scene data
    frame_scene_data = [_get_scene_for_timestamp(ts) for ts in frame_timestamps]

    # --- Step 4: Colorize all frames (with per-scene color palettes) ---
    await status(f"Colorizing {len(frame_paths)} frames with NanoBanana 2...")
    colorized_frames = []
    colorized_pils = []

    for i, (frame_path, scene_data) in enumerate(zip(frame_paths, frame_scene_data)):
        with open(frame_path, "rb") as f:
            frame_bytes = f.read()

        try:
            colorized_pil, _ = await colorize_frame(client, frame_bytes, scene_data)
        except Exception as e:
            # If colorization fails for a frame, skip it
            continue

        colorized_path = os.path.join(colorized_dir, f"colorized_{i:04d}.png")
        colorized_pil.save(colorized_path, format="PNG")
        colorized_frames.append(colorized_path)
        colorized_pils.append((i, colorized_pil))

    if not colorized_pils:
        return {
            "colorized_frames": [],
            "clip_paths": [],
            "final_video_path": None,
            "scene_analysis": video_analysis,
            "status": "All colorization attempts failed.",
        }

    # --- Step 5: Animate each colorized frame (parallel) + generate score ---
    total_duration = len(colorized_pils) * 8
    score_duration = min(total_duration + 2, 598)  # Lyria max ~10 min session

    await status(
        f"Animating {len(colorized_pils)} frames with Veo 3.1 + generating "
        f"{score_duration}s musical score with Lyria (in parallel)..."
    )

    animation_tasks = []
    for idx, (frame_idx, colorized_pil) in enumerate(colorized_pils):
        clip_path = os.path.join(clips_dir, f"clip_{idx:04d}.mp4")
        scene_data = frame_scene_data[frame_idx]
        task = asyncio.create_task(
            animate_frame(client, colorized_pil, scene_data, clip_path)
        )
        animation_tasks.append(task)

    # Build score analysis from the overall video analysis
    score_analysis = {
        "music": overall.get("music", {
            "genre": "orchestral",
            "tempo": "medium",
            "instruments": "piano, strings",
            "mood": "nostalgic, cinematic",
        })
    }
    score_task = asyncio.create_task(
        generate_score(client, score_analysis, duration_seconds=score_duration)
    )

    results = await asyncio.gather(*animation_tasks, score_task, return_exceptions=True)

    clip_results = results[:-1]
    score_result = results[-1]

    clip_paths = [r for r in clip_results if not isinstance(r, Exception)]

    if not clip_paths:
        return {
            "colorized_frames": colorized_frames,
            "clip_paths": [],
            "final_video_path": None,
            "scene_analysis": video_analysis,
            "status": "All Veo animation tasks failed. Check API access and try again.",
        }

    # --- Step 6: Stitch all clips together ---
    await status("Stitching video clips together...")
    stitched_path = os.path.join(tmp_dir, "stitched.mp4")
    if len(clip_paths) > 1:
        stitch_video_clips(clip_paths, stitched_path)
    else:
        shutil.copy2(clip_paths[0], stitched_path)

    # --- Step 7: Merge with musical score ---
    await status("Merging video with musical score...")
    final_video_path = os.path.join(tmp_dir, "resurrected.mp4")

    if isinstance(score_result, Exception):
        shutil.copy2(stitched_path, final_video_path)
        status_msg = f"Video resurrected without score (score error: {score_result})"
    else:
        score_wav_path = os.path.join(tmp_dir, "score.wav")
        pcm_to_wav(score_result, score_wav_path)

        if has_audio_stream(stitched_path):
            merge_video_and_score(stitched_path, score_wav_path, final_video_path)
        else:
            merge_video_score_only(stitched_path, score_wav_path, final_video_path)

        status_msg = (
            f"Done! Resurrected {len(clip_paths)} scenes from "
            f"{len(frame_paths)} extracted frames into a "
            f"{len(clip_paths) * 8}-second video with musical score."
        )

    return {
        "colorized_frames": colorized_frames,
        "clip_paths": clip_paths,
        "final_video_path": final_video_path,
        "scene_analysis": video_analysis,
        "status": status_msg,
    }
