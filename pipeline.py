"""
pipeline.py - Core AI pipeline: analyze → colorize → animate → score → merge.

Orchestrates Gemini 3.1 Pro (scene analysis), NanoBanana 2 (colorization),
Veo 3.1 (animation), and Lyria RealTime (musical score) into a single pipeline.
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


# ---------------------------------------------------------------------------
# Step 1: Gemini 3.1 Pro — Scene Analysis
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

    # Parse the JSON from the response, stripping any markdown fences
    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        text = "\n".join(lines[1:-1])

    return json.loads(text)


# ---------------------------------------------------------------------------
# Step 2: NanoBanana 2 — Colorization
# ---------------------------------------------------------------------------

async def colorize_frame(
    client: genai.Client, image_bytes: bytes, scene_analysis: dict
) -> tuple[Image.Image, bytes]:
    """
    Colorize a B&W image using NanoBanana 2 (gemini-3.1-flash-image-preview).

    Uses scene analysis to guide historically accurate colorization.
    Returns (PIL Image, raw image bytes) tuple.
    """
    colors = scene_analysis["colors"]
    era = scene_analysis["era"]

    response = await client.aio.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            f"""Colorize this black and white photograph with historically accurate colors.
Era: {era}.
Sky: {colors['sky']}. Buildings: {colors['buildings']}.
Clothing: {colors['clothing']}. Skin tones: {colors['skin']}.
Lighting: {colors['lighting']}.
Preserve all original detail, grain, and composition.
Photorealistic result. Do not add or remove any elements."""
        ],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )

    # Extract the colorized image from the response
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            # Get raw bytes for saving/passing to Veo
            raw_bytes = part.inline_data.data
            # Also get PIL Image for display
            pil_image = Image.open(io.BytesIO(raw_bytes))
            return pil_image, raw_bytes

    raise RuntimeError("NanoBanana 2 did not return an image in the response.")


# ---------------------------------------------------------------------------
# Step 3: Veo 3.1 — Animation
# ---------------------------------------------------------------------------

async def animate_frame(
    client: genai.Client,
    colorized_pil: Image.Image,
    scene_analysis: dict,
    output_path: str,
) -> str:
    """
    Animate a colorized image using Veo 3.1, producing an 8-second video
    with native ambient audio.

    Note: Veo generation can take 11 seconds to 6+ minutes.
    Returns path to the saved MP4 file.
    """
    # Pass PIL image directly — the SDK accepts PIL Image objects
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=f"""Gentle, cinematic animation of this {scene_analysis['era']} photograph
coming to life. {scene_analysis['movement']}.
Ambient audio: {scene_analysis['ambient_sounds']}.
Documentary style. Subtle, natural motion only.
Maintain photographic quality. No modern elements.""",
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
# Full Pipeline Orchestrator
# ---------------------------------------------------------------------------

async def resurrect_image(client: genai.Client, input_image, tmp_dir: str = None):
    """
    Full Resurrect pipeline: B&W image → colorized, animated, scored video.

    Args:
        client: Google GenAI client (with api_version='v1alpha' for Lyria support)
        input_image: numpy array (from Gradio) or file path string or bytes
        tmp_dir: optional temp directory path for output files

    Returns:
        (colorized_image_path, final_video_path, scene_analysis, status_text)
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="resurrect_")

    # --- Convert input to JPEG bytes ---
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

    # --- Step 1: Scene Analysis ---
    scene_analysis = await analyze_scene(client, image_bytes)

    # --- Step 2: Colorize ---
    colorized_pil, colorized_bytes = await colorize_frame(client, image_bytes, scene_analysis)

    # Save colorized image for display
    colorized_path = os.path.join(tmp_dir, "colorized.png")
    colorized_pil.save(colorized_path, format="PNG")

    # --- Step 3 & 4: Animate and Score in parallel ---
    raw_video_path = os.path.join(tmp_dir, "raw_video.mp4")

    animate_task = asyncio.create_task(
        animate_frame(client, colorized_pil, scene_analysis, raw_video_path)
    )
    score_task = asyncio.create_task(
        generate_score(client, scene_analysis, duration_seconds=10)
    )

    # Wait for both to complete
    _, score_pcm = await asyncio.gather(animate_task, score_task)

    # Save score as WAV
    score_wav_path = os.path.join(tmp_dir, "score.wav")
    pcm_to_wav(score_pcm, score_wav_path)

    # --- Step 5: Merge video + score ---
    final_video_path = os.path.join(tmp_dir, "resurrected.mp4")

    if has_audio_stream(raw_video_path):
        merge_video_and_score(raw_video_path, score_wav_path, final_video_path)
    else:
        merge_video_score_only(raw_video_path, score_wav_path, final_video_path)

    return colorized_path, final_video_path, scene_analysis, "Done! Your photograph has been resurrected."


# ---------------------------------------------------------------------------
# Video-to-Video Pipeline (Primary Feature)
# ---------------------------------------------------------------------------

async def resurrect_video(
    client: genai.Client,
    video_path: str,
    tmp_dir: str = None,
    frame_interval_seconds: float = 3.0,
    progress_callback=None,
) -> dict:
    """
    Full Resurrect pipeline for video clips:
    B&W video → extract frames → analyze → colorize each → animate each → score → stitch.

    This is the primary feature: take an old silent film clip (e.g., Charlie Chaplin)
    and transform it into a colorized, animated, musically scored video.

    Args:
        client: Google GenAI client (with api_version='v1alpha')
        video_path: Path to the input B&W video file
        tmp_dir: Temp directory for intermediate files
        frame_interval_seconds: Extract one frame every N seconds
        progress_callback: Optional async callback for status updates

    Returns:
        dict with keys:
            - colorized_frames: list of colorized frame paths
            - clip_paths: list of generated video clip paths
            - final_video_path: path to the final stitched + scored video
            - scene_analysis: scene analysis dict
            - status: status message
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="resurrect_vid_")

    frames_dir = os.path.join(tmp_dir, "frames")
    colorized_dir = os.path.join(tmp_dir, "colorized")
    clips_dir = os.path.join(tmp_dir, "clips")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(colorized_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)

    # --- Step 1: Extract key frames from the video ---
    frame_paths = extract_keyframes(video_path, frames_dir, frame_interval_seconds)

    if not frame_paths:
        return {
            "colorized_frames": [],
            "clip_paths": [],
            "final_video_path": None,
            "scene_analysis": {},
            "status": "No frames could be extracted from the video.",
        }

    # --- Step 2: Analyze first frame (use for all frames for consistency) ---
    with open(frame_paths[0], "rb") as f:
        first_frame_bytes = f.read()

    scene_analysis = await analyze_scene(client, first_frame_bytes)

    # --- Step 3: Colorize all frames ---
    colorized_frames = []
    colorized_pils = []

    for i, frame_path in enumerate(frame_paths):
        with open(frame_path, "rb") as f:
            frame_bytes = f.read()

        colorized_pil, colorized_bytes = await colorize_frame(
            client, frame_bytes, scene_analysis
        )

        colorized_path = os.path.join(colorized_dir, f"colorized_{i:04d}.png")
        colorized_pil.save(colorized_path, format="PNG")
        colorized_frames.append(colorized_path)
        colorized_pils.append(colorized_pil)

    # --- Step 4: Animate each colorized frame with Veo (parallel) + generate score ---
    # Calculate total score duration: 8 seconds per frame + some overlap
    total_duration = len(frame_paths) * 8
    score_duration = total_duration + 2  # slight buffer

    # Launch all Veo animations in parallel with the score generation
    animation_tasks = []
    for i, colorized_pil in enumerate(colorized_pils):
        clip_path = os.path.join(clips_dir, f"clip_{i:04d}.mp4")
        task = asyncio.create_task(
            animate_frame(client, colorized_pil, scene_analysis, clip_path)
        )
        animation_tasks.append(task)

    score_task = asyncio.create_task(
        generate_score(client, scene_analysis, duration_seconds=score_duration)
    )

    # Wait for all animations and score to complete
    results = await asyncio.gather(*animation_tasks, score_task, return_exceptions=True)

    # Separate animation results from score result
    clip_results = results[:-1]
    score_result = results[-1]

    # Collect successful clips
    clip_paths = []
    for i, result in enumerate(clip_results):
        if isinstance(result, Exception):
            # Skip failed clips
            continue
        clip_paths.append(result)

    if not clip_paths:
        return {
            "colorized_frames": colorized_frames,
            "clip_paths": [],
            "final_video_path": None,
            "scene_analysis": scene_analysis,
            "status": "All animation tasks failed. Check API access and try again.",
        }

    # --- Step 5: Stitch all clips together ---
    stitched_path = os.path.join(tmp_dir, "stitched.mp4")
    if len(clip_paths) > 1:
        stitch_video_clips(clip_paths, stitched_path)
    else:
        # Single clip, just copy/rename
        shutil.copy2(clip_paths[0], stitched_path)

    # --- Step 6: Merge with musical score ---
    final_video_path = os.path.join(tmp_dir, "resurrected.mp4")

    if isinstance(score_result, Exception):
        # Score generation failed — use stitched video without score
        shutil.copy2(stitched_path, final_video_path)
        status = f"Video resurrected (score generation failed: {score_result})"
    else:
        score_pcm = score_result
        score_wav_path = os.path.join(tmp_dir, "score.wav")
        pcm_to_wav(score_pcm, score_wav_path)

        if has_audio_stream(stitched_path):
            merge_video_and_score(stitched_path, score_wav_path, final_video_path)
        else:
            merge_video_score_only(stitched_path, score_wav_path, final_video_path)

        status = (
            f"Done! Resurrected {len(clip_paths)} clips from {len(frame_paths)} "
            f"extracted frames into a {len(clip_paths) * 8}-second video with musical score."
        )

    return {
        "colorized_frames": colorized_frames,
        "clip_paths": clip_paths,
        "final_video_path": final_video_path,
        "scene_analysis": scene_analysis,
        "status": status,
    }
