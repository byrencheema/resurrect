# Resurrect: Old Films → Living Color + Score

## Context

Building a hackathon project for YC x Google DeepMind: Multimodal Frontier Hackathon (March 7, 2026). Solo build, 6.5 hours (10:30 AM - 5:00 PM). The concept: upload an old B&W photo or silent film clip → get back a colorized, animated, musically scored video. Uses Gemini 3.1 + NanoBanana 2 + Veo 3.1 + Lyria RealTime.

## Confirmed APIs

| Model | ID | What it does | Cost |
|-------|----|-------------|------|
| Gemini 3.1 Pro | `gemini-3.1-pro-preview` | Analyzes scene, writes prompts for other models | Paid (hackathon gives high-rate limits) |
| NanoBanana 2 | `gemini-3.1-flash-image-preview` | Colorizes B&W frames | Paid |
| Veo 3.1 | `veo-3.1-generate-preview` | Image → 8-sec video with native audio | Paid ($0.40/sec = $3.20 per clip) |
| Lyria RealTime | `models/lyria-realtime-exp` | Streaming musical score via WebSocket | FREE |

**NOT available as API:** Lyria 3 vocals (Gemini App only, no developer endpoint). Not needed for this project — we use Lyria RealTime for instrumental scoring + Veo for ambient audio.

## Tech Stack

- **Python 3.11+** — all Google SDKs have first-class Python support
- **`google-genai`** — single SDK for Gemini, NanoBanana, Veo, Lyria
- **Gradio** — instant UI with file upload, image/video display, audio playback. Zero frontend code.
- **FFmpeg** — merge Veo video + Lyria score into final output
- **asyncio** — Lyria RealTime requires async WebSocket

## Project Structure

```
resurrect/
├── app.py              # Gradio UI + main orchestration
├── pipeline.py         # Core AI pipeline (analyze → colorize → animate → score)
├── lyria_scorer.py     # Lyria RealTime WebSocket session management
├── video_utils.py      # FFmpeg merge, frame extraction, file handling
├── requirements.txt    # google-genai, gradio, Pillow, numpy
└── samples/            # Pre-loaded demo B&W photos for quick demo
    ├── chaplin.jpg
    ├── 1920s_street.jpg
    └── victorian_portrait.jpg
```

## Implementation Plan

### Step 1: Project Setup (10:30 - 10:45)

1. Confirm API access with DeepMind staff — ask what model IDs are available, get API key
2. Create project directory
3. Install dependencies:
   ```
   uv init resurrect && cd resurrect
   uv add google-genai gradio Pillow numpy
   ```
4. Verify each API with a minimal test call

### Step 2: Gemini 3.1 Scene Analyzer (10:45 - 11:15)

**File: `pipeline.py`**

Gemini 3.1 takes the uploaded B&W image and returns structured JSON:

```python
async def analyze_scene(client, image_bytes):
    response = await client.aio.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            """Analyze this old black and white photograph. Return valid JSON only:
            {
                "era": "estimated decade (e.g. 1920s)",
                "setting": "location description",
                "people": "description of people, clothing, expressions",
                "mood": "emotional tone",
                "colors": {
                    "sky": "color",
                    "buildings": "color",
                    "clothing": "colors",
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
        ]
    )
    return json.loads(response.text)
```

**Test:** Upload a B&W photo, print the JSON. Verify it returns sensible era/color/music analysis.

### Step 3: NanoBanana 2 Colorizer (11:15 - 12:00)

**File: `pipeline.py`**

Takes the B&W image + Gemini's color analysis → returns a colorized version.

```python
async def colorize_frame(client, image_bytes, scene_analysis):
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
        config=types.GenerateContentConfig(response_modalities=["IMAGE"])
    )
    return response.candidates[0].content.parts[0].inline_data.data
```

**Test:** Upload B&W photo → get colorized image back. Verify colors match era.

### Step 4: Veo 3.1 Animator (12:00 - 1:00)

**File: `pipeline.py`**

Takes the colorized image → generates 8-second video with native ambient audio.

```python
async def animate_frame(client, colorized_image_bytes, scene_analysis):
    image = types.Image.from_bytes(data=colorized_image_bytes)

    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=f"""Gentle, cinematic animation of this {scene_analysis['era']}
        photograph coming to life. {scene_analysis['movement']}.
        Ambient audio: {scene_analysis['ambient_sounds']}.
        Documentary style. Subtle, natural motion only.
        Maintain photographic quality. No modern elements.""",
        image=image,
        config=types.GenerateVideosConfig(
            duration_seconds="8",
            aspect_ratio="16:9",
            person_generation="allow_all"
        )
    )

    # Poll until complete (11 sec - 6 min)
    while not operation.done:
        await asyncio.sleep(5)
        operation = client.operations.get(operation)

    video = operation.response.generated_videos[0]
    video_bytes = client.files.download(file=video.video)
    return video_bytes
```

**Test:** Feed colorized image in → get 8-sec MP4 back. Verify it has ambient audio. Check generation time.

**IMPORTANT:** Veo can take up to 6 minutes. For the demo, pre-generate 2-3 results. Do 1 live generation on stage to show it works.

### Step 5: Lyria RealTime Scorer (1:00 - 2:00)

**File: `lyria_scorer.py`**

Generates 8+ seconds of musical score matching the scene mood via WebSocket.

```python
async def generate_score(client, scene_analysis, duration_seconds=10):
    music = scene_analysis["music"]

    audio_chunks = []

    async with client.aio.live.music.connect(
        model='models/lyria-realtime-exp'
    ) as session:
        await session.set_weighted_prompts(prompts=[
            types.WeightedPrompt(text=music["genre"], weight=2.0),
            types.WeightedPrompt(text=music["instruments"], weight=1.5),
            types.WeightedPrompt(text=music["mood"], weight=1.0),
        ])

        tempo_map = {"slow": 72, "medium": 100, "fast": 130}
        await session.set_music_generation_config(
            config=types.LiveMusicGenerationConfig(
                bpm=tempo_map.get(music["tempo"], 100),
                density=0.3,
                brightness=0.4,
                music_generation_mode="QUALITY"
            )
        )

        await session.play()

        collected = 0
        async for message in session.receive():
            for chunk in message.server_content.audio_chunks:
                audio_chunks.append(chunk.data)
                collected += len(chunk.data)
                # 48kHz * 2 channels * 2 bytes * duration
                if collected >= 48000 * 2 * 2 * duration_seconds:
                    break
            else:
                continue
            break

        await session.stop()

    return b''.join(audio_chunks)
```

**Output:** Raw 16-bit PCM, 48kHz stereo. Write to WAV file before merging.

**Test:** Generate 10 seconds of score for a "1920s ragtime piano, playful" scene. Verify it sounds right.

### Step 6: FFmpeg Merge (2:00 - 2:30)

**File: `video_utils.py`**

Combine Veo video (with ambient audio) + Lyria score into final output.

```python
import subprocess
import wave

def pcm_to_wav(pcm_data, output_path, sample_rate=48000, channels=2):
    """Convert raw PCM bytes to WAV file."""
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

def merge_video_and_score(video_path, score_wav_path, output_path):
    """Layer Lyria score under Veo's ambient audio."""
    subprocess.run([
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
        output_path
    ], check=True)
```

**Test:** Merge a Veo output with a Lyria WAV. Play the result. Verify both audio layers are present.

### Step 7: Gradio UI (2:30 - 3:30)

**File: `app.py`**

```python
import gradio as gr
from pipeline import resurrect_image
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

async def process(image):
    # Returns: (original, colorized, video_path, scene_description)
    return await resurrect_image(client, image)

with gr.Blocks(title="Resurrect", theme=gr.themes.Base()) as demo:
    gr.Markdown("# Resurrect\n### Old photographs → living color with a score")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload B&W photo", type="numpy")
            btn = gr.Button("Resurrect", variant="primary", size="lg")

        with gr.Column():
            status = gr.Textbox(label="Status", interactive=False)
            scene_info = gr.JSON(label="Scene Analysis")

    with gr.Row():
        colorized = gr.Image(label="Colorized")
        video_out = gr.Video(label="Resurrected")

    btn.click(
        fn=process,
        inputs=[input_image],
        outputs=[colorized, video_out, scene_info, status]
    )

demo.launch()
```

**Layout:**
- Left: upload area + Resurrect button
- Right: scene analysis JSON
- Bottom left: colorized still image
- Bottom right: final video player (colorized + moving + scored)

### Step 8: Multi-Frame Support for Video Clips (3:30 - 4:00)

If time permits, add support for uploading a video clip (not just a photo):

1. Extract key frames every 2-3 seconds using FFmpeg
2. Analyze first frame with Gemini 3.1 (use analysis for all frames for consistency)
3. Colorize each frame with NanoBanana 2
4. Animate each frame with Veo 3.1 (8 sec each)
5. Score the full sequence with Lyria RealTime (steer prompts per scene if mood changes)
6. Stitch all Veo clips together with FFmpeg

**This is a stretch goal.** MVP is single image → single 8-sec video. Demo with that if time is tight.

### Step 9: Demo Prep (4:00 - 5:00)

1. Pre-generate 2-3 results from different eras:
   - 1920s street scene
   - Victorian portrait
   - 1940s wartime photo
2. Have one ready to generate LIVE on stage (pick one that generates fast)
3. Practice the narrative:
   - Show the original B&W photo
   - "This was taken in [year]. No color. No sound. No motion."
   - Click Resurrect
   - While Veo generates, show the colorized still + scene analysis
   - Play the final video with score
   - "Every pixel of color, every frame of motion, every note of music — generated by AI."

## Fallback Plan

If any API is unavailable or broken at the hackathon:

| Problem | Fallback |
|---------|----------|
| Veo 3.1 down/slow | Show colorized stills only + Lyria score as separate audio. Still impressive. |
| Lyria RealTime down | Use Veo's native audio only (ambient sounds). No musical score. |
| NanoBanana 2 down | Use Gemini 3.1's image generation capabilities (slower, lower quality) |
| API rate limited | Pre-generate everything. Demo from cache. Do 1 live generation. |

## Verification / Demo Checklist

- [ ] Upload B&W photo → Gemini returns valid scene JSON
- [ ] Scene JSON → NanoBanana returns colorized image
- [ ] Colorized image → Veo returns 8-sec MP4 with ambient audio
- [ ] Scene JSON → Lyria returns 10 sec of period-appropriate music
- [ ] FFmpeg merges video + score into final MP4
- [ ] Gradio UI displays before/after side by side
- [ ] Full pipeline runs end-to-end in under 3 minutes
- [ ] 3 pre-generated demos ready for presentation
- [ ] 1 live demo prepared with a reliable photo

## Pre-Hackathon Prep (before Saturday)

1. `uv init resurrect && cd resurrect`
2. `uv add google-genai gradio Pillow numpy`
3. Verify FFmpeg is installed: `brew install ffmpeg`
4. Get a Google AI Studio API key: https://aistudio.google.com/apikey
5. Test each API individually with a hello-world call
6. Collect 5-10 good public domain B&W photos (Library of Congress, Wikimedia)
7. Clone the quickstart repo for reference: `git clone https://github.com/google-gemini/veo-3-nano-banana-gemini-api-quickstart`
