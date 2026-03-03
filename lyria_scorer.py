"""
lyria_scorer.py - Lyria RealTime WebSocket session management for musical score generation.

Uses Google's Lyria RealTime API (models/lyria-realtime-exp) to generate
period-appropriate musical scores based on scene analysis metadata.

Two modes:
  1. generate_score() — Instrumental score (genre + instruments + mood)
  2. generate_vocal_score() — Score with vocals/lyrics (adds vocal prompts + optional lyrics)

Output: Raw 16-bit PCM audio, 48kHz stereo.

IMPORTANT: The client MUST be created with http_options={'api_version': 'v1alpha'}
for Lyria RealTime to work.
"""

import asyncio

from google import genai
from google.genai import types


# Audio format constants
SAMPLE_RATE = 48000  # 48 kHz
CHANNELS = 2         # Stereo
BYTES_PER_SAMPLE = 2  # 16-bit PCM
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE  # 192,000 bytes/sec

MODEL = "models/lyria-realtime-exp"


async def generate_score(
    client: genai.Client,
    scene_analysis: dict,
    duration_seconds: int = 10,
) -> bytes:
    """
    Generate a musical score matching the scene mood via Lyria RealTime WebSocket.

    Args:
        client: Google GenAI client (must use api_version='v1alpha')
        scene_analysis: Scene analysis dict from Gemini 3.1 Pro
        duration_seconds: How many seconds of music to generate (default 10)

    Returns:
        Raw 16-bit PCM audio bytes (48kHz, stereo).
        Convert to WAV with video_utils.pcm_to_wav() before use.
    """
    music = scene_analysis.get("music", {})
    genre = music.get("genre", "orchestral")
    instruments = music.get("instruments", "piano, strings")
    mood = music.get("mood", "nostalgic, cinematic")
    tempo = music.get("tempo", "medium")

    # Map tempo descriptions to BPM
    tempo_map = {"slow": 72, "medium": 100, "fast": 130}
    try:
        bpm = tempo_map.get(str(tempo).lower(), 100)
    except (AttributeError, TypeError):
        bpm = 100

    audio_chunks = []
    target_bytes = BYTES_PER_SECOND * duration_seconds

    async with client.aio.live.music.connect(model=MODEL) as session:
        # Set weighted prompts to guide the musical style
        await session.set_weighted_prompts(prompts=[
            types.WeightedPrompt(text=genre, weight=2.0),
            types.WeightedPrompt(text=instruments, weight=1.5),
            types.WeightedPrompt(text=mood, weight=1.0),
        ])

        # Configure generation parameters
        config = types.LiveMusicGenerationConfig(
            bpm=bpm,
            density=0.3,
            brightness=0.4,
            music_generation_mode=types.MusicGenerationMode.QUALITY,
            temperature=1.0,
            guidance=4.0,
        )
        await session.set_music_generation_config(config=config)

        # Start playback / generation
        await session.play()

        # Collect audio chunks until we have enough data
        # Timeout: allow 2x the requested duration + 15s buffer for settling
        max_wait = duration_seconds * 2 + 15
        start_time = asyncio.get_event_loop().time()
        collected = 0

        async for message in session.receive():
            if message.server_content and message.server_content.audio_chunks:
                audio_data = message.server_content.audio_chunks[0].data
                audio_chunks.append(audio_data)
                collected += len(audio_data)
                if collected >= target_bytes:
                    break
            elif hasattr(message, 'filtered_prompt') and message.filtered_prompt:
                pass  # Prompt was safety-filtered; continue with whatever plays

            # Timeout check
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait:
                break

            # Yield to event loop to prevent blocking
            await asyncio.sleep(1e-12)

        # Stop the session gracefully
        await session.stop()

    return b"".join(audio_chunks)


async def generate_vocal_score(
    client: genai.Client,
    scene_analysis: dict,
    duration_seconds: int = 10,
    lyrics: str = None,
) -> bytes:
    """
    Generate a musical score WITH vocal textures via Lyria RealTime WebSocket.

    Uses VOCALIZATION mode which produces wordless vocal textures (oohs, aahs,
    humming) as an instrument layer. This is NOT coherent singing with lyrics —
    Lyria RealTime is instrumental-only. Actual lyrics-based singing requires
    Lyria 3 which has no developer API.

    The lyrics parameter influences the musical mood/style through weighted
    prompts rather than producing literal sung words.

    Args:
        client: Google GenAI client (must use api_version='v1alpha')
        scene_analysis: Scene analysis dict from Gemini 3.1 Pro
        duration_seconds: How many seconds of music to generate
        lyrics: Optional text to influence the vocal mood/style (not literal singing)

    Returns:
        Raw 16-bit PCM audio bytes (48kHz, stereo).
    """
    music = scene_analysis.get("music", {})
    genre = music.get("genre", "folk ballad")
    instruments = music.get("instruments", "acoustic guitar, vocals")
    mood = music.get("mood", "nostalgic, storytelling")
    tempo = music.get("tempo", "medium")

    tempo_map = {"slow": 72, "medium": 100, "fast": 130}
    try:
        bpm = tempo_map.get(str(tempo).lower(), 100)
    except (AttributeError, TypeError):
        bpm = 100

    # Build vocal-focused prompts
    prompts = [
        types.WeightedPrompt(text=genre, weight=2.0),
        types.WeightedPrompt(text="singing vocals, human voice, choir", weight=2.5),
        types.WeightedPrompt(text=instruments, weight=1.0),
        types.WeightedPrompt(text=mood, weight=1.0),
    ]

    # Lyrics influence the musical style/mood, not literal singing
    if lyrics:
        prompts.append(
            types.WeightedPrompt(text=f"song about: {lyrics}", weight=3.0)
        )

    audio_chunks = []
    target_bytes = BYTES_PER_SECOND * duration_seconds

    async with client.aio.live.music.connect(model=MODEL) as session:
        await session.set_weighted_prompts(prompts=prompts)

        # Use VOCALIZATION mode to steer toward vocal-like textures
        config = types.LiveMusicGenerationConfig(
            bpm=bpm,
            density=0.5,
            brightness=0.5,
            music_generation_mode=types.MusicGenerationMode.VOCALIZATION,
            temperature=1.0,
            guidance=4.0,
        )
        await session.set_music_generation_config(config=config)

        await session.play()

        max_wait = duration_seconds * 2 + 15
        start_time = asyncio.get_event_loop().time()
        collected = 0

        async for message in session.receive():
            if message.server_content and message.server_content.audio_chunks:
                audio_data = message.server_content.audio_chunks[0].data
                audio_chunks.append(audio_data)
                collected += len(audio_data)
                if collected >= target_bytes:
                    break
            elif hasattr(message, 'filtered_prompt') and message.filtered_prompt:
                pass

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait:
                break

            await asyncio.sleep(1e-12)

        await session.stop()

    return b"".join(audio_chunks)
