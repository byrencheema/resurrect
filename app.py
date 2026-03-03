"""
app.py - Gradio UI + main orchestration for Resurrect.

Upload an old B&W video clip or photograph → get back a colorized,
animated, musically scored video.

Modes:
  1. Resurrect Video — Reimagine mode: extract frames → colorize → animate with Veo → score
  2. Colorize Video — Preserve original motion, frame-by-frame colorization + score
  3. Resurrect Photo — Single photo → 8-sec animated + scored video
"""

import asyncio
import os
import io
import tempfile

import gradio as gr
from PIL import Image
from google import genai

from pipeline import (
    analyze_scene,
    colorize_frame,
    animate_frame,
    resurrect_video,
    colorize_video,
)
from lyria_scorer import generate_score
from video_utils import pcm_to_wav, merge_video_and_score, merge_video_score_only, has_audio_stream


# ---------------------------------------------------------------------------
# Client Setup
# ---------------------------------------------------------------------------

def get_client() -> genai.Client:
    """Create a Google GenAI client from environment variable.
    Uses v1alpha API version for Lyria RealTime compatibility."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a key at https://aistudio.google.com/apikey"
        )
    return genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1alpha"},
    )


# ---------------------------------------------------------------------------
# Video-to-Video Processing (Primary Feature)
# ---------------------------------------------------------------------------

async def process_video(video_path, frame_interval):
    """
    Full Resurrect pipeline for video clips.
    Takes a B&W video → full video analysis → colorizes → animates → scores.
    Yields intermediate results for real-time UI updates.
    """
    if video_path is None:
        yield None, None, None, {}, "Please upload a black & white video clip."
        return

    client = get_client()
    tmp_dir = tempfile.mkdtemp(prefix="resurrect_vid_")
    interval = float(frame_interval)

    # Shared state for progress updates from the pipeline
    latest_status = ["Starting..."]

    async def on_progress(msg):
        latest_status[0] = msg

    try:
        yield None, None, None, {}, "Uploading video to Gemini for full scene analysis..."

        result = await resurrect_video(
            client=client,
            video_path=video_path,
            tmp_dir=tmp_dir,
            frame_interval_seconds=interval,
            progress_callback=on_progress,
        )

        colorized_path = result["colorized_frames"][0] if result["colorized_frames"] else None
        final_video = result["final_video_path"]
        scene = result["scene_analysis"]
        status_msg = result["status"]

        yield colorized_path, final_video, video_path, scene, status_msg

    except Exception as e:
        yield None, None, video_path, {}, f"Error: {e}"


# ---------------------------------------------------------------------------
# Photo-to-Video Processing (Secondary Feature)
# ---------------------------------------------------------------------------

async def process_image(image):
    """
    Full Resurrect pipeline for single photos.
    Yields intermediate results so the UI updates at each stage.
    """
    if image is None:
        yield None, None, {}, "Please upload a black & white photograph."
        return

    client = get_client()

    # Convert numpy array to JPEG bytes
    img = Image.fromarray(image)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    image_bytes = buf.getvalue()

    # --- Step 1: Scene Analysis ---
    yield None, None, {}, "Step 1/5: Analyzing scene with Gemini 3.1 Pro..."

    try:
        scene_analysis = await analyze_scene(client, image_bytes)
    except Exception as e:
        yield None, None, {}, f"Scene analysis failed: {e}"
        return

    yield None, None, scene_analysis, "Step 2/5: Colorizing with NanoBanana 2..."

    # --- Step 2: Colorize ---
    try:
        colorized_pil, colorized_bytes = await colorize_frame(client, image_bytes, scene_analysis)
    except Exception as e:
        yield None, None, scene_analysis, f"Colorization failed: {e}"
        return

    # Save colorized image for display
    tmp_dir = tempfile.mkdtemp(prefix="resurrect_")
    colorized_path = os.path.join(tmp_dir, "colorized.png")
    colorized_pil.save(colorized_path, format="PNG")

    yield colorized_path, None, scene_analysis, "Step 3/5: Generating animation (Veo 3.1) + musical score (Lyria) in parallel... This may take a few minutes."

    # --- Step 3 & 4: Animate + Score in parallel ---
    try:
        raw_video_path = os.path.join(tmp_dir, "raw_video.mp4")
        animate_task = asyncio.create_task(
            animate_frame(client, colorized_pil, scene_analysis, raw_video_path)
        )
        score_task = asyncio.create_task(
            generate_score(client, scene_analysis, duration_seconds=10)
        )
        _, score_pcm = await asyncio.gather(animate_task, score_task)
    except Exception as e:
        yield colorized_path, None, scene_analysis, f"Animation/scoring failed: {e}"
        return

    yield colorized_path, None, scene_analysis, "Step 4/5: Merging video and musical score..."

    # --- Step 5: Merge ---
    try:
        score_wav_path = os.path.join(tmp_dir, "score.wav")
        pcm_to_wav(score_pcm, score_wav_path)

        final_video_path = os.path.join(tmp_dir, "resurrected.mp4")

        if has_audio_stream(raw_video_path):
            merge_video_and_score(raw_video_path, score_wav_path, final_video_path)
        else:
            merge_video_score_only(raw_video_path, score_wav_path, final_video_path)
    except Exception as e:
        if os.path.exists(raw_video_path):
            yield colorized_path, raw_video_path, scene_analysis, f"Merge failed ({e}), showing raw video."
            return
        yield colorized_path, None, scene_analysis, f"Video merge failed: {e}"
        return

    yield colorized_path, final_video_path, scene_analysis, "Step 5/5: Done! Your photograph has been resurrected."


# ---------------------------------------------------------------------------
# Colorize-Only Video Processing
# ---------------------------------------------------------------------------

async def process_colorize_video(video_path, colorize_every_n, add_vocals, lyrics_text):
    """
    Colorize-only mode: preserves original motion, just adds color + score.
    Yields intermediate results for real-time UI updates.
    """
    if video_path is None:
        yield None, None, {}, "Please upload a black & white video clip."
        return

    client = get_client()
    tmp_dir = tempfile.mkdtemp(prefix="resurrect_colorize_")
    every_n = int(colorize_every_n)

    async def on_progress(msg):
        pass  # Status updates come via yields below

    try:
        yield None, None, {}, "Starting colorize-only pipeline (preserves original motion)..."

        result = await colorize_video(
            client=client,
            video_path=video_path,
            tmp_dir=tmp_dir,
            colorize_every_n=every_n,
            progress_callback=on_progress,
            vocals_lyrics=lyrics_text.strip() if add_vocals and lyrics_text.strip() else None,
        )

        final_video = result["final_video_path"]
        scene = result["scene_analysis"]
        status_msg = result["status"]

        yield final_video, video_path, scene, status_msg

    except Exception as e:
        yield None, video_path, {}, f"Error: {e}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    """Build and return the Gradio Blocks interface."""

    # Material Design 3 inspired theme
    theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#fef7e0", c100="#feeab3", c200="#fedd82",
            c300="#fed04f", c400="#fec62b", c500="#febb0f",
            c600="#feb40e", c700="#feaa0b", c800="#fea109",
            c900="#fe9104", c950="#e67c00",
            name="amber",
        ),
        neutral_hue=gr.themes.Color(
            c50="#fafafa", c100="#f5f5f5", c200="#eeeeee",
            c300="#e0e0e0", c400="#bdbdbd", c500="#9e9e9e",
            c600="#757575", c700="#616161", c800="#424242",
            c900="#212121", c950="#121212",
            name="neutral",
        ),
        font=[gr.themes.GoogleFont("Google Sans Flex"), "Roboto", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Roboto Mono"), "monospace"],
        radius_size=gr.themes.sizes.radius_lg,
    ).set(
        body_background_fill="#f8f9fa",
        body_text_color="#1f1f1f",
        block_background_fill="#ffffff",
        block_border_width="0px",
        block_shadow="0 1px 3px 0 rgba(0,0,0,0.1), 0 1px 2px -1px rgba(0,0,0,0.1)",
        block_label_text_color="#444746",
        block_title_text_color="#1f1f1f",
        border_color_primary="#c4c7c5",
        input_background_fill="#ffffff",
        input_border_color="#c4c7c5",
        input_border_width="1px",
        button_primary_background_fill="#febb0f",
        button_primary_background_fill_hover="#e6a90e",
        button_primary_text_color="#1f1f1f",
        button_primary_border_color="transparent",
        button_secondary_background_fill="#e8eaed",
        button_secondary_background_fill_hover="#dadce0",
        button_secondary_text_color="#1f1f1f",
        button_large_radius="*radius_lg",
        checkbox_background_color="#ffffff",
        checkbox_border_color="#c4c7c5",
        slider_color="#febb0f",
    )

    css = """
    .main-title h1 {
        text-align: center;
        font-weight: 500;
        font-size: 2.5rem;
        color: #1f1f1f;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }
    .subtitle h3 {
        text-align: center;
        font-weight: 400;
        color: #444746;
        margin-top: 4px;
        font-size: 1.1rem;
    }
    .hero-text p {
        text-align: center;
        color: #5f6368;
        max-width: 640px;
        margin: 0 auto 24px;
        line-height: 1.6;
    }
    .status-box textarea {
        font-size: 0.85rem;
        background: #f8f9fa !important;
        border-color: #e8eaed !important;
    }
    .gradio-container {
        max-width: 1200px !important;
    }
    .tab-nav button {
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 10px 20px !important;
    }
    .tab-nav button.selected {
        border-bottom: 3px solid #febb0f !important;
        color: #1f1f1f !important;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="Resurrect — Old Films to Living Color",
        theme=theme,
        css=css,
    ) as demo:
        gr.Markdown(
            "# Resurrect",
            elem_classes="main-title",
        )
        gr.Markdown(
            "### Old films, living color, original score",
            elem_classes="subtitle",
        )
        gr.Markdown(
            "Upload an old black & white video clip or photograph. "
            "Resurrect will colorize it, bring it to life with motion, and "
            "compose a period-appropriate musical score — all powered by Google's multimodal AI.",
            elem_classes="hero-text",
        )

        # ============================================================
        # Tab 1: Video → Video (Primary Feature)
        # ============================================================
        with gr.Tab("Resurrect Video", id="video_tab"):
            gr.Markdown(
                "**Upload an old B&W video clip** (e.g., Charlie Chaplin, early newsreels, "
                "silent films). Resurrect will extract key frames, colorize them, animate "
                "each one, and compose a musical score for the full piece."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    input_video = gr.Video(
                        label="Upload B&W Video Clip",
                        sources=["upload"],
                    )
                    frame_interval = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=3.0,
                        step=0.5,
                        label="Frame Extract Interval (seconds)",
                        info="Extract one key frame every N seconds. Lower = more frames = longer processing.",
                    )
                    btn_video = gr.Button(
                        "Resurrect Video",
                        variant="primary",
                        size="lg",
                    )
                    video_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=3,
                        elem_classes="status-box",
                    )

                with gr.Column(scale=1):
                    video_scene_info = gr.JSON(label="Scene Analysis")

            with gr.Row():
                video_colorized = gr.Image(label="Colorized Frame Preview", height=400)
                video_output = gr.Video(label="Resurrected Video", autoplay=True)

            with gr.Row():
                video_original = gr.Video(label="Original (for comparison)")

            btn_video.click(
                fn=process_video,
                inputs=[input_video, frame_interval],
                outputs=[video_colorized, video_output, video_original, video_scene_info, video_status],
            )

        # ============================================================
        # Tab 2: Colorize Video (Preserve Original Motion)
        # ============================================================
        with gr.Tab("Colorize Video", id="colorize_tab"):
            gr.Markdown(
                "**Colorize-only mode**: Preserves the original motion frame-by-frame. "
                "Best for clips where you want to keep the exact performance (e.g., Charlie Chaplin's "
                "physical comedy) but add color and a musical score. No Veo re-animation — just "
                "NanoBanana 2 colorization on every frame + Lyria score."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    colorize_input_video = gr.Video(
                        label="Upload B&W Video Clip",
                        sources=["upload"],
                    )
                    colorize_every_n = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=1,
                        step=1,
                        label="Colorize Every Nth Frame",
                        info="1 = every frame (best quality, slowest). 3 = every 3rd frame (faster, slight choppiness).",
                    )
                    colorize_add_vocals = gr.Checkbox(
                        label="Add vocal textures to score",
                        value=False,
                        info="Use Lyria VOCALIZATION mode for oohs/aahs vocal textures. Text below influences mood.",
                    )
                    colorize_lyrics = gr.Textbox(
                        label="Vocal mood/theme (optional)",
                        placeholder="E.g., 'Walking through the rain, memories remain...' (influences style, not literal singing)",
                        lines=3,
                        visible=True,
                    )
                    btn_colorize_video = gr.Button(
                        "Colorize Video",
                        variant="primary",
                        size="lg",
                    )
                    colorize_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=3,
                        elem_classes="status-box",
                    )

                with gr.Column(scale=1):
                    colorize_scene_info = gr.JSON(label="Scene Analysis")

            with gr.Row():
                colorize_output = gr.Video(label="Colorized Video", autoplay=True)
                colorize_original = gr.Video(label="Original (for comparison)")

            btn_colorize_video.click(
                fn=process_colorize_video,
                inputs=[colorize_input_video, colorize_every_n, colorize_add_vocals, colorize_lyrics],
                outputs=[colorize_output, colorize_original, colorize_scene_info, colorize_status],
            )

        # ============================================================
        # Tab 3: Photo → Video (Secondary Feature)
        # ============================================================
        with gr.Tab("Resurrect Photo", id="photo_tab"):
            gr.Markdown(
                "**Upload a single B&W photograph** to generate an 8-second "
                "colorized, animated video with a musical score."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Upload B&W Photograph",
                        type="numpy",
                        height=400,
                    )
                    btn_photo = gr.Button(
                        "Resurrect Photo",
                        variant="primary",
                        size="lg",
                    )
                    photo_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=2,
                        elem_classes="status-box",
                    )

                with gr.Column(scale=1):
                    photo_scene_info = gr.JSON(label="Scene Analysis")

            with gr.Row():
                photo_colorized = gr.Image(label="Colorized", height=400)
                photo_video_out = gr.Video(label="Resurrected Video", autoplay=True)

            btn_photo.click(
                fn=process_image,
                inputs=[input_image],
                outputs=[photo_colorized, photo_video_out, photo_scene_info, photo_status],
            )

        # ============================================================
        # Examples
        # ============================================================
        example_dir = os.path.join(os.path.dirname(__file__), "samples")
        if os.path.isdir(example_dir):
            image_examples = [
                os.path.join(example_dir, f)
                for f in sorted(os.listdir(example_dir))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            video_examples = [
                os.path.join(example_dir, f)
                for f in sorted(os.listdir(example_dir))
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
            ]
            if video_examples:
                gr.Examples(
                    examples=video_examples,
                    inputs=input_video,
                    label="Sample B&W Video Clips",
                )
            if image_examples:
                gr.Examples(
                    examples=image_examples,
                    inputs=input_image,
                    label="Sample B&W Photographs",
                )

    return demo


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
