"""
Unit tests for VAD speech capture module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from collections import deque
import io
import wave

from reachy_mvp.core.vad_capture import VAD, VADSpeechCapture


class TestVAD:
    """Tests for VAD class."""

    @pytest.fixture
    def vad_model_path(self, tmp_path):
        """Create a temporary VAD model path."""
        model_file = tmp_path / "test_vad.onnx"
        model_file.touch()
        return str(model_file)

    @patch('reachy_mvp.core.vad_capture.ort.InferenceSession')
    def test_vad_initialization(self, mock_session, vad_model_path):
        """Test VAD initializes correctly."""
        vad = VAD(model_path=vad_model_path)

        assert vad.SAMPLE_RATE == 16000
        assert hasattr(vad, 'ort_sess')
        assert hasattr(vad, '_state')
        mock_session.assert_called_once()

    @patch('reachy_mvp.core.vad_capture.ort.InferenceSession')
    def test_vad_reset_states(self, mock_session, vad_model_path):
        """Test VAD state reset."""
        vad = VAD(model_path=vad_model_path)
        vad.reset_states(batch_size=2)

        assert vad._state.shape == (2, 2, 128)
        assert vad._last_sr == 0
        assert vad._last_batch_size == 0

    @patch('reachy_mvp.core.vad_capture.ort.InferenceSession')
    def test_vad_processes_audio_chunk(self, mock_session, vad_model_path):
        """Test VAD processes audio chunk correctly."""
        # Mock inference session
        mock_sess_instance = MagicMock()
        mock_sess_instance.run.return_value = (
            np.array([0.8]),  # High confidence
            np.zeros((2, 1, 128), dtype=np.float32)  # State
        )
        mock_session.return_value = mock_sess_instance

        vad = VAD(model_path=vad_model_path)

        # Create audio sample (512 samples for 16kHz)
        audio_sample = np.random.rand(1, 512).astype(np.float32)

        result = vad(audio_sample, sample_rate=16000)

        assert mock_sess_instance.run.called
        assert isinstance(result, np.ndarray)

    @patch('reachy_mvp.core.vad_capture.ort.InferenceSession')
    def test_vad_wrong_sample_count_raises_error(self, mock_session, vad_model_path):
        """Test VAD raises error for wrong sample count."""
        vad = VAD(model_path=vad_model_path)

        # Wrong number of samples
        audio_sample = np.random.rand(1, 256).astype(np.float32)

        with pytest.raises(ValueError, match="Provided number of samples"):
            vad(audio_sample, sample_rate=16000)


class TestVADSpeechCapture:
    """Tests for VADSpeechCapture class."""

    @pytest.fixture
    def mock_audio_stream(self):
        """Create mock PyAudio stream."""
        stream = Mock()
        stream.read = Mock(return_value=b'\x00' * 1024)  # Silence
        return stream

    @pytest.fixture
    def vad_model_path(self, tmp_path):
        """Create temporary VAD model path."""
        model_file = tmp_path / "test_vad.onnx"
        model_file.touch()
        return str(model_file)

    @patch('reachy_mvp.core.vad_capture.VAD')
    def test_vad_speech_capture_initialization(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test VADSpeechCapture initializes correctly."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            sample_rate=16000,
            vad_model_path=vad_model_path,
            vad_threshold=0.5
        )

        assert capture.sample_rate == 16000
        assert capture.vad_threshold == 0.5
        assert capture.vad_chunk_samples == 512
        assert capture.buffer_max_chunks == 25  # 800ms / 32ms
        assert capture.pause_chunks == 20  # 640ms / 32ms
        assert isinstance(capture._buffer, deque)
        assert capture._buffer.maxlen == 25

    @patch('reachy_mvp.core.vad_capture.VAD')
    def test_pre_activation_buffer_fills(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test circular buffer fills before activation."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path
        )

        # Add chunks to buffer (no voice activity)
        for i in range(30):  # More than buffer size
            chunk = np.zeros(512, dtype=np.int16)
            capture._manage_pre_activation_buffer(chunk, vad_confidence=False)

        # Buffer should be at max size
        assert len(capture._buffer) == 25
        assert not capture._recording_started

    @patch('reachy_mvp.core.vad_capture.VAD')
    def test_voice_activation_starts_recording(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test voice activity starts recording."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path
        )

        # Fill buffer
        for i in range(10):
            chunk = np.zeros(512, dtype=np.int16)
            capture._manage_pre_activation_buffer(chunk, vad_confidence=False)

        # Voice detected
        chunk = np.ones(512, dtype=np.int16)
        capture._manage_pre_activation_buffer(chunk, vad_confidence=True)

        assert capture._recording_started
        assert len(capture._samples) == 11  # Buffer contents + current chunk transferred
        assert capture._gap_counter == 0

    @patch('reachy_mvp.core.vad_capture.VAD')
    def test_pause_detection_completes_speech(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test pause detection marks speech as complete."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path
        )

        capture._recording_started = True
        capture._samples = [np.ones(512, dtype=np.int16) for _ in range(10)]

        # Process silence chunks (below pause limit)
        for i in range(capture.pause_chunks - 1):
            chunk = np.zeros(512, dtype=np.int16)
            result = capture._process_activated_audio(chunk, vad_confidence=False)
            assert result is False  # Not complete yet
            assert capture._gap_counter == i + 1

        # Final silence chunk should complete
        chunk = np.zeros(512, dtype=np.int16)
        result = capture._process_activated_audio(chunk, vad_confidence=False)
        assert result is True  # Speech complete!

    @patch('reachy_mvp.core.vad_capture.VAD')
    def test_voice_activity_resets_gap_counter(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test voice activity resets gap counter."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path
        )

        capture._recording_started = True
        capture._gap_counter = 10

        # Voice activity should reset counter
        chunk = np.ones(512, dtype=np.int16)
        result = capture._process_activated_audio(chunk, vad_confidence=True)

        assert result is False  # Not complete
        assert capture._gap_counter == 0  # Reset!

    @patch('reachy_mvp.core.vad_capture.VAD')
    def test_samples_to_wav_creates_valid_wav(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test WAV file creation from samples."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            sample_rate=16000,
            vad_model_path=vad_model_path
        )

        # Create test samples (1 second of audio)
        capture._samples = [
            np.ones(512, dtype=np.int16) * 1000  # Some amplitude
            for _ in range(31)  # ~1 second at 16kHz
        ]

        wav_bytes = capture._samples_to_wav()

        # Verify it's valid WAV
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 0

        # Parse WAV to verify parameters
        wav_buffer = io.BytesIO(wav_bytes)
        with wave.open(wav_buffer, 'rb') as wf:
            assert wf.getnchannels() == 1  # Mono
            assert wf.getsampwidth() == 2  # 16-bit
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 512 * 31

    @patch('reachy_mvp.core.vad_capture.VAD')
    def test_reset_clears_state(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test reset clears all internal state."""
        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path
        )

        # Set up some state
        capture._recording_started = True
        capture._samples = [np.ones(512, dtype=np.int16) for _ in range(10)]
        capture._gap_counter = 5
        capture._buffer = deque([np.ones(512, dtype=np.int16) for _ in range(10)], maxlen=25)

        # Reset
        capture.reset()

        assert capture._recording_started is False
        assert len(capture._samples) == 0
        assert capture._gap_counter == 0
        assert len(capture._buffer) == 0

    @pytest.mark.asyncio
    @patch('reachy_mvp.core.vad_capture.VAD')
    async def test_capture_speech_timeout(self, mock_vad, mock_audio_stream, vad_model_path):
        """Test capture_speech returns None on timeout without speech."""
        # Mock VAD to return low confidence (no speech)
        mock_vad_instance = MagicMock()
        mock_vad_instance.return_value = 0.1  # Below threshold
        mock_vad.return_value = mock_vad_instance

        capture = VADSpeechCapture(
            audio_stream=mock_audio_stream,
            vad_model_path=vad_model_path,
            vad_threshold=0.5
        )

        # Mock _process_vad to return False (no speech)
        capture._process_vad = AsyncMock(return_value=False)

        # Override timeout for faster test
        capture.VAD_SIZE = 100  # 100ms chunks
        timeout_chunks = 5  # 5 chunks = 500ms timeout

        result = await capture.capture_speech()

        # Should timeout without capturing speech
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
