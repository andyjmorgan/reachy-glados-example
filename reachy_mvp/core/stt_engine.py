"""
Speech-to-Text Engine for Reachy MVP

Provides local speech-to-text transcription with noise detection,
using Faster Whisper for efficient inference.
"""

import logging
import io
import wave
from dataclasses import dataclass
from typing import Optional

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@dataclass
class STTResult:
    """Result of speech-to-text transcription"""

    text: str
    is_speech: bool
    confidence: float
    language: str

    # Detailed metrics for debugging
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    duration: float

    def __str__(self):
        return (
            f"STTResult(text='{self.text[:50]}...', is_speech={self.is_speech}, "
            f"confidence={self.confidence:.2f}, language={self.language})"
        )


class STTEngine:
    """
    Local speech-to-text engine with noise detection.

    Uses Faster Whisper for efficient transcription and provides
    confidence metrics to distinguish speech from background noise.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: Optional[str] = None,
        language: str = "en",
        min_speech_confidence: float = 0.6,
        max_no_speech_prob: float = 0.6,
        min_compression_ratio: float = 0.5,
        max_compression_ratio: float = 2.5,
        min_text_length: int = 3,
    ):
        """
        Initialize the STT engine.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cpu or cuda)
            compute_type: Compute type (auto-selected if None: int8 for CPU, float16 for GPU)
            language: Language code or "auto" for detection
            min_speech_confidence: Minimum avg_logprob to consider speech (higher = stricter)
            max_no_speech_prob: Maximum no_speech_prob to consider speech (lower = stricter)
            min_compression_ratio: Minimum compression ratio
            max_compression_ratio: Maximum compression ratio
            min_text_length: Minimum text length in characters
        """
        self.model_size = model_size
        self.device = device

        # Auto-select compute type based on device if not specified
        if compute_type is None:
            self.compute_type = "int8" if device == "cpu" else "float16"
        else:
            self.compute_type = compute_type

        self.language = language if language != "auto" else None

        # Noise detection thresholds
        self.min_speech_confidence = min_speech_confidence
        self.max_no_speech_prob = max_no_speech_prob
        self.min_compression_ratio = min_compression_ratio
        self.max_compression_ratio = max_compression_ratio
        self.min_text_length = min_text_length

        logger.info(
            f"Initializing Faster Whisper model: {model_size} on {device} "
            f"with {self.compute_type} compute"
        )

        # Initialize Faster Whisper model
        # Downloads model on first run (~150MB for base model)
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=self.compute_type,
        )

        logger.info("STT engine initialized successfully")

    def transcribe(self, audio_wav: bytes) -> STTResult:
        """
        Transcribe audio and detect if it contains speech or just noise.

        Args:
            audio_wav: WAV audio bytes (16kHz mono int16)

        Returns:
            STTResult with transcription and speech detection

        Raises:
            Exception: If transcription fails
        """
        logger.debug(f"Transcribing audio ({len(audio_wav)} bytes)")

        try:
            # Faster Whisper expects file-like object or path
            audio_file = io.BytesIO(audio_wav)

            # Get audio duration from WAV header
            duration = self._get_wav_duration(audio_wav)

            # Transcribe with Faster Whisper
            segments, info = self.model.transcribe(
                audio_file,
                language=self.language,
                beam_size=5,
                vad_filter=False,  # We already have VAD
                word_timestamps=False,
            )

            # Collect all segments into full transcription
            full_text = ""
            total_logprob = 0.0
            segment_count = 0

            for segment in segments:
                full_text += segment.text
                total_logprob += segment.avg_logprob
                segment_count += 1

            # Calculate average log probability
            avg_logprob = total_logprob / segment_count if segment_count > 0 else -1.0

            # Get metadata from transcription info
            language = info.language
            no_speech_prob = getattr(info, 'no_speech_prob', 0.0)

            # Calculate compression ratio (text length / audio duration)
            # This helps detect hallucinations (very high) or noise (very low)
            text_length = len(full_text.strip())
            compression_ratio = text_length / duration if duration > 0 else 0.0

            # Determine if this is actual speech vs noise
            is_speech = self._is_speech(
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                compression_ratio=compression_ratio,
                text_length=text_length,
            )

            # Calculate overall confidence (inverse of no_speech_prob)
            confidence = 1.0 - no_speech_prob

            result = STTResult(
                text=full_text.strip(),
                is_speech=is_speech,
                confidence=confidence,
                language=language,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                compression_ratio=compression_ratio,
                duration=duration,
            )

            logger.info(
                f"Transcription complete: is_speech={is_speech}, "
                f"confidence={confidence:.2f}, "
                f"text_len={text_length}, "
                f"avg_logprob={avg_logprob:.2f}, "
                f"no_speech_prob={no_speech_prob:.2f}, "
                f"compression={compression_ratio:.2f}"
            )

            if is_speech:
                logger.info(f"Detected speech: '{full_text}'")
            else:
                logger.info(f"Detected background noise (no speech). Text was: '{full_text}'")

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise

    def _is_speech(
        self,
        avg_logprob: float,
        no_speech_prob: float,
        compression_ratio: float,
        text_length: int,
    ) -> bool:
        """
        Determine if audio contains actual speech vs background noise.

        Uses multiple heuristics:
        - avg_logprob: How confident the model is (higher = more confident)
        - no_speech_prob: Direct probability of no speech (lower = more speech)
        - compression_ratio: Text length vs audio duration (too low/high = not speech)
        - text_length: Minimum characters required

        Args:
            avg_logprob: Average log probability of transcription
            no_speech_prob: Probability of no speech
            compression_ratio: Characters per second ratio
            text_length: Length of transcribed text

        Returns:
            True if speech detected, False if just noise
        """
        # Check all thresholds
        confidence_ok = avg_logprob > self.min_speech_confidence
        no_speech_ok = no_speech_prob < self.max_no_speech_prob
        compression_ok = (
            self.min_compression_ratio < compression_ratio < self.max_compression_ratio
        )
        length_ok = text_length >= self.min_text_length

        # All conditions must be met for speech
        is_speech = confidence_ok and no_speech_ok and compression_ok and length_ok

        # Debug logging
        if not is_speech:
            reasons = []
            if not confidence_ok:
                reasons.append(
                    f"low confidence (avg_logprob={avg_logprob:.2f} < {self.min_speech_confidence})"
                )
            if not no_speech_ok:
                reasons.append(
                    f"high no_speech_prob ({no_speech_prob:.2f} > {self.max_no_speech_prob})"
                )
            if not compression_ok:
                reasons.append(
                    f"bad compression_ratio ({compression_ratio:.2f} not in "
                    f"[{self.min_compression_ratio}, {self.max_compression_ratio}])"
                )
            if not length_ok:
                reasons.append(
                    f"text too short ({text_length} < {self.min_text_length})"
                )

            logger.debug(f"Noise detected: {', '.join(reasons)}")

        return is_speech

    def _get_wav_duration(self, audio_wav: bytes) -> float:
        """
        Get duration of WAV audio in seconds.

        Args:
            audio_wav: WAV audio bytes

        Returns:
            Duration in seconds
        """
        try:
            with wave.open(io.BytesIO(audio_wav), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            logger.warning(f"Failed to get WAV duration: {e}")
            return 1.0  # Default fallback