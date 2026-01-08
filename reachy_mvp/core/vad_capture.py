"""
Voice Activity Detection and speech capture using Silero VAD.

This module provides VAD-based speech capture with circular buffer pattern
adapted from the GLaDOS reference implementation.
"""

import asyncio
import logging
from collections import deque
from pathlib import Path
import io
import wave

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort

# Reduce ONNX runtime verbosity
ort.set_default_logger_severity(4)

logger = logging.getLogger(__name__)


class VAD:
    """
    Voice Activity Detection using Silero VAD v5 ONNX model.

    Adapted from GLaDOS implementation.
    """

    SAMPLE_RATE: int = 16000  # or 8000 only!

    def __init__(self, model_path: Path | str):
        """Initialize VAD with ONNX model.

        Args:
            model_path: Path to silero_vad_v5.onnx model
        """
        providers = ort.get_available_providers()
        # Remove problematic providers
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        if "CoreMLExecutionProvider" in providers:
            providers.remove("CoreMLExecutionProvider")

        self.ort_sess = ort.InferenceSession(
            str(model_path),
            sess_options=ort.SessionOptions(),
            providers=providers,
        )

        self._state: NDArray[np.float32]
        self._context: NDArray[np.float32]
        self._last_sr: int
        self._last_batch_size: int

        self.reset_states()

    def reset_states(self, batch_size: int = 1) -> None:
        """Reset VAD internal state."""
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(
        self,
        audio_sample: NDArray[np.float32],
        sample_rate: int = SAMPLE_RATE
    ) -> NDArray[np.float32]:
        """
        Process audio sample and return VAD confidence.

        Args:
            audio_sample: Audio with shape (batch_size, num_samples)
            sample_rate: Sample rate (8000 or 16000)

        Returns:
            VAD confidence score (0.0 to 1.0)
        """
        num_samples = 512 if sample_rate == 16000 else 256

        if audio_sample.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {audio_sample.shape[-1]} "
                f"(Supported: 256 for 8kHz, 512 for 16kHz)"
            )

        batch_size = audio_sample.shape[0]
        context_size = 64 if sample_rate == 16000 else 32

        # Reset state if parameters changed
        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sample_rate):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        # Concatenate context
        audio_sample = np.concatenate([self._context, audio_sample], axis=1)

        # Run inference
        ort_inputs = {
            "input": audio_sample.astype(np.float32),
            "state": self._state,
            "sr": np.array(sample_rate, dtype=np.int64),
        }
        ort_outs = self.ort_sess.run(None, ort_inputs)
        out, state = ort_outs
        self._state = state

        # Update context
        self._context = audio_sample[..., -context_size:]
        self._last_sr = sample_rate
        self._last_batch_size = batch_size

        return np.squeeze(out)

    def __del__(self) -> None:
        """Clean up ONNX session."""
        if hasattr(self, "ort_sess"):
            del self.ort_sess


class VADSpeechCapture:
    """
    Captures speech using VAD with circular buffer pattern from GLaDOS.

    Uses Silero VAD to detect voice activity, buffers pre-activation audio,
    and detects speech pauses to determine when user has finished speaking.
    """

    # Constants from GLaDOS (in milliseconds)
    VAD_SIZE = 32          # ms per VAD chunk
    BUFFER_SIZE = 800      # ms pre-activation buffer
    PAUSE_LIMIT = 640      # ms silence to end speech

    def __init__(
        self,
        audio_stream,
        sample_rate: int = 16000,
        vad_model_path: str = "models/silero_vad_v5.onnx",
        vad_threshold: float = 0.5,
    ):
        """
        Initialize VAD speech capture.

        Args:
            audio_stream: PyAudio stream for reading audio
            sample_rate: Audio sample rate (default 16kHz)
            vad_model_path: Path to Silero VAD model
            vad_threshold: VAD confidence threshold (0.0-1.0)
        """
        self.audio_stream = audio_stream
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold

        # Initialize VAD
        self.vad = VAD(model_path=vad_model_path)

        # Calculate buffer sizes in chunks
        # VAD processes 512 samples at 16kHz = 32ms
        self.vad_chunk_samples = 512 if sample_rate == 16000 else 256
        self.vad_chunk_duration_ms = self.VAD_SIZE

        # Circular buffer: 800ms / 32ms = 25 chunks
        self.buffer_max_chunks = self.BUFFER_SIZE // self.VAD_SIZE
        self._buffer: deque[NDArray[np.int16]] = deque(maxlen=self.buffer_max_chunks)

        # Pause detection: 640ms / 32ms = 20 chunks
        self.pause_chunks = self.PAUSE_LIMIT // self.VAD_SIZE

        # State
        self._recording_started = False
        self._samples: list[NDArray[np.int16]] = []
        self._gap_counter = 0

    async def capture_speech(self) -> bytes | None:
        """
        Capture speech segment with pause detection.

        Returns:
            WAV bytes of captured speech, or None if no speech detected
        """
        logger.debug("Listening for speech...")

        self.reset()
        timeout_chunks = (30 * 1000) // self.VAD_SIZE  # 30 second timeout
        chunks_read = 0

        while chunks_read < timeout_chunks:
            try:
                # Read audio chunk
                audio_data = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.audio_stream.read,
                    self.vad_chunk_samples,
                    False  # exception_on_overflow=False
                )

                # Convert to numpy array
                chunk = np.frombuffer(audio_data, dtype=np.int16)

                # Run VAD
                vad_confidence = await self._process_vad(chunk)

                # Handle audio based on state
                if not self._recording_started:
                    self._manage_pre_activation_buffer(chunk, vad_confidence)
                else:
                    speech_complete = self._process_activated_audio(chunk, vad_confidence)
                    if speech_complete:
                        logger.info(f"Speech captured ({len(self._samples)} chunks)")
                        return self._samples_to_wav()

                chunks_read += 1

            except Exception as e:
                logger.warning(f"Error reading audio: {e}")
                break

        logger.debug("Timeout or no speech detected")
        return None

    async def _process_vad(self, chunk: NDArray[np.int16]) -> bool:
        """
        Run VAD on audio chunk.

        Args:
            chunk: Audio chunk (int16)

        Returns:
            True if voice activity detected, False otherwise
        """
        # Normalize to float32 [-1.0, 1.0]
        audio_float = chunk.astype(np.float32) / 32768.0

        # Reshape for VAD (needs batch dimension)
        audio_float = audio_float.reshape(1, -1)

        # Run VAD in thread executor to avoid blocking
        confidence = await asyncio.get_event_loop().run_in_executor(
            None,
            self.vad,
            audio_float,
            self.sample_rate
        )

        return float(confidence) > self.vad_threshold

    def _manage_pre_activation_buffer(
        self,
        chunk: NDArray[np.int16],
        vad_confidence: bool
    ) -> None:
        """
        Manage circular buffer and detect voice activation.

        Args:
            chunk: Audio chunk
            vad_confidence: VAD result
        """
        self._buffer.append(chunk)

        if vad_confidence:
            logger.info("Voice activity detected!")
            # Transfer buffer to samples and start recording
            self._samples = list(self._buffer)
            self._recording_started = True
            self._gap_counter = 0

    def _process_activated_audio(
        self,
        chunk: NDArray[np.int16],
        vad_confidence: bool
    ) -> bool:
        """
        Process audio after activation, detect pauses.

        Args:
            chunk: Audio chunk
            vad_confidence: VAD result

        Returns:
            True if speech is complete (pause detected), False otherwise
        """
        self._samples.append(chunk)

        if not vad_confidence:
            self._gap_counter += 1
            if self._gap_counter >= self.pause_chunks:
                logger.debug(f"Pause detected ({self.PAUSE_LIMIT}ms)")
                return True  # Speech complete
        else:
            self._gap_counter = 0  # Reset on voice activity

        return False

    def _samples_to_wav(self) -> bytes:
        """
        Convert captured samples to WAV bytes.

        Returns:
            WAV file as bytes (16kHz, mono, int16)
        """
        # Concatenate all chunks
        audio_data = np.concatenate(self._samples)

        # Create WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        wav_buffer.seek(0)
        return wav_buffer.read()

    def reset(self) -> None:
        """Reset capture state for new speech segment."""
        self._recording_started = False
        self._samples.clear()
        self._gap_counter = 0
        self._buffer.clear()
        self.vad.reset_states()
