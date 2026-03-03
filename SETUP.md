# Resurrect — Setup Guide

## Prerequisites

- **Python 3.11+**
- **FFmpeg** (with ffprobe)
- **Google Gemini API key** with access to:
  - Gemini 3.1 Pro (`gemini-3.1-pro-preview`) — scene analysis
  - NanoBanana 2 (`gemini-3.1-flash-image-preview`) — colorization
  - Veo 3.1 (`veo-3.1-generate-preview`) — animation
  - Lyria RealTime (`lyria-realtime-exp`) — music scoring

## Install

```bash
# Clone and enter the repo
git clone <repo-url> && cd resurrect

# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (if not already installed)
# macOS:  brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: download from https://ffmpeg.org/download.html
```

## Configure

```bash
export GEMINI_API_KEY="your-key-here"
```

Get a key at https://aistudio.google.com/apikey

## Run

```bash
python app.py
```

Opens at **http://localhost:7860**

## Verify

1. Open the UI in your browser
2. Go to the **Resurrect Photo** tab
3. Upload any B&W image (grab one from `docs/test_inputs_reference.md`)
4. Click **Resurrect Photo** — you should see:
   - Scene analysis JSON populate
   - A colorized version of the photo appear
   - A short animated video with music generate

If any step fails, the status box will show which API errored.

## Modes

| Tab | What it does |
|-----|-------------|
| **Resurrect Video** | Extract frames → colorize → Veo re-animate each → stitch → Lyria score |
| **Colorize Video** | Frame-by-frame colorization preserving original motion + score |
| **Resurrect Photo** | Single photo → colorize → animate → score |

## Sample Content

See `docs/test_inputs_reference.md` for curated public domain B&W videos and photos. Quick start:

```bash
# Download a short Chaplin clip (~57 MB)
wget "https://archive.org/download/CC_1914_02_02_MakingALiving/CC_1914_02_02_MakingALiving.mp4"

# Trim to 10 seconds for fast testing
ffmpeg -i CC_1914_02_02_MakingALiving.mp4 -t 10 -c copy chaplin_10s.mp4
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GEMINI_API_KEY not set` | `export GEMINI_API_KEY="..."` |
| `FFmpeg not found` | Install FFmpeg and ensure it's on your PATH |
| Veo times out | Normal for long clips; try shorter input or increase frame interval |
| NanoBanana returns empty | Image may have been safety-filtered; try a different image |
| Lyria no audio | Ensure API key has Lyria access; check `v1alpha` API version |
