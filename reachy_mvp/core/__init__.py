"""Core functionality for Reachy MVP."""

# VAD and audio capture
from .vad_capture import VADSpeechCapture, VAD

# OpenAI client
from .openai_client import OpenAIAudioClient

# Streaming base classes
from .streaming import (
    StreamChunk,
    StreamMiddleware,
    StreamPipeline,
)

# Middleware implementations
from .middlewares import (
    SentenceBufferMiddleware,
    FilterMiddleware,
    TTSStreamMiddleware,
    InterruptibleTTSMiddleware,
)

__all__ = [
    # Audio capture
    "VADSpeechCapture",
    "VAD",
    # OpenAI
    "OpenAIAudioClient",
    # Streaming base
    "StreamChunk",
    "StreamMiddleware",
    "StreamPipeline",
    # Middlewares
    "SentenceBufferMiddleware",
    "FilterMiddleware",
    "TTSStreamMiddleware",
    "InterruptibleTTSMiddleware",
]
