"""
State machine orchestrating Reachy's interactive voice assistant loop.

Implements the MVP workflow:
- STARTUP: Initialize components, robot sleeps
- SLEEP: Monitor for wake word
- INTERACTIVE: Listen for user speech with VAD
- PROCESSING: Send to Anthropic Claude, stream responses, handle tools
"""

import asyncio
import logging
import os
import sys
from enum import Enum
from typing import Optional, Dict, Any
from collections import deque
import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model

logger = logging.getLogger(__name__)

try:
    from reachy_mini import ReachyMini
except ImportError:
    logger.warning("reachy-mini not installed, using mock mode")
    ReachyMini = None

from reachy_mvp.core.vad_capture import VADSpeechCapture
from reachy_mvp.core.anthropic_client import AnthropicClient
from reachy_mvp.core.stt_engine import STTEngine
from reachy_mvp.core.audio_input_processor import AudioInputProcessor, AudioInputResult
from reachy_mvp.robot.animations import RobotAnimator
from reachy_mvp.tools.tool_handlers import TOOL_DEFINITIONS, TOOL_HANDLERS
from reachy_mvp.core.streaming import StreamPipeline, StreamChunk
from reachy_mvp.core.middlewares import (
    SentenceBufferMiddleware,
    FilterMiddleware,
    ToolCallMiddleware,
    ProviderMiddleware,
)
from reachy_mvp.core.middlewares.tts import TTSStreamMiddleware


class ReachyState(Enum):
    """States for Reachy's operation."""
    STARTUP = "startup"
    SLEEP = "sleep"
    INTERACTIVE = "interactive"
    PROCESSING = "processing"


class ReachyStateMachine:
    """
    Main orchestrator for Reachy MVP voice assistant.

    Manages the full lifecycle: startup → sleep → wake → listen → process → respond
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the state machine with configuration.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.state = ReachyState.STARTUP

        # Initialize audio
        self.audio = pyaudio.PyAudio()
        self.mic_stream: Optional[Any] = None

        # Initialize wake word detector
        self.wake_word_model: Optional[Model] = None

        # Initialize robot (or mock)
        self.mini: Optional[ReachyMini] = None
        self.use_mock_robot = ReachyMini is None

        # Initialize components (will be set up in startup)
        self.vad_capture: Optional[VADSpeechCapture] = None
        self.llm_client: Optional[AnthropicClient] = None
        self.stt_engine: Optional[STTEngine] = None
        self.audio_processor: Optional[AudioInputProcessor] = None
        self.animator: Optional[RobotAnimator] = None

        # Conversation state
        self.messages = [
            {"role": "system", "content": config["conversation"]["system_prompt"]}
        ]
        self.max_history = config["conversation"]["max_history"]

        # Control flags
        self.interruption_flag = asyncio.Event()
        self.running = True

        # Current audio being processed
        self.current_audio: Optional[bytes] = None

        # Queue for speech audio from VAD
        self.speech_queue: asyncio.Queue[bytes] = asyncio.Queue()

        # Task for continuous VAD
        self.vad_task: Optional[asyncio.Task] = None

        # Middleware pipeline (will be built in startup)
        self.pipeline: Optional[StreamPipeline] = None

    def _find_reachy_microphone(self) -> int:
        """
        Find Reachy's microphone device index.

        Returns:
            Device index for Reachy's microphone

        Raises:
            RuntimeError: If microphone not found
        """
        for i in range(self.audio.get_device_count()):
            dev_info = self.audio.get_device_info_by_index(i)
            if 'reachy' in dev_info['name'].lower():
                logger.info(f"Found Reachy microphone: {dev_info['name']}")
                return i

        raise RuntimeError("Reachy microphone not found. Is the robot connected?")

    def _setup_audio_stream(self) -> Any:
        """
        Set up PyAudio stream for microphone input.

        Returns:
            PyAudio stream object
        """
        device_index = self._find_reachy_microphone()

        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.config["audio"]["channels"],
            rate=self.config["audio"]["sample_rate"],
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.config["audio"]["chunk_size"]
        )

        return stream

    def _setup_wake_word(self) -> Model:
        """
        Set up OpenWakeWord model.

        Returns:
            OpenWakeWord model instance
        """
        model_path = self.config["wake_word"]["model_path"]

        if not os.path.exists(model_path):
            raise RuntimeError(f"Wake word model not found: {model_path}")

        logger.info(f"Loading wake word model: {model_path}")
        model = Model(wakeword_models=[model_path], inference_framework="onnx")

        return model

    async def _continuous_vad_listener(self) -> None:
        """
        Continuously listen for speech during AWAKE state.

        Yields speech audio bytes via queue when pause detected.
        Runs throughout INTERACTIVE and PROCESSING phases.
        """
        while self.state in [ReachyState.INTERACTIVE, ReachyState.PROCESSING]:
            try:
                # Capture speech (blocks until pause detected or timeout)
                audio_bytes = await self.vad_capture.capture_speech()

                if audio_bytes:
                    # Speech detected - put in queue for processing
                    await self.speech_queue.put(audio_bytes)

                    # If currently processing, set interrupt flag
                    if self.state == ReachyState.PROCESSING:
                        logger.info("Speech detected during processing - interrupting")
                        self.interruption_flag.set()

            except asyncio.CancelledError:
                logger.info("VAD listener cancelled")
                raise
            except Exception as e:
                logger.error(f"VAD listener error: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    def _build_pipeline(self) -> StreamPipeline:
        """
        Build the streaming middleware pipeline.

        Pipeline order (outermost to innermost):
        TTS → Filter → Sentence → Tool → Provider

        Returns:
            Configured StreamPipeline instance
        """
        import logging
        logger = logging.getLogger(__name__)

        # Provider factory for tool middleware re-entrancy
        # Build middleware list (INNERMOST first!)
        # Pipeline chains like: middlewares[-1].process(middlewares[-2].process(...middlewares[0].process(stream)))
        # So: first in list = innermost (processes input first), last in list = outermost (yields to caller)
        middlewares = []

        # Provider middleware (INNERMOST - SOURCE) - calls Anthropic Claude API
        logger.debug(
            f"Creating provider: history_len={len(self.messages)}, "
            f"tools={len(TOOL_DEFINITIONS) if TOOL_DEFINITIONS else 0}"
        )
        provider = ProviderMiddleware(
            llm_client=self.llm_client,
            conversation_history=self.messages.copy(),
            tools=TOOL_DEFINITIONS,
            temperature=self.config["anthropic"].get("temperature", 0.8),
            max_tokens=self.config["anthropic"].get("max_tokens", 1000)
        )
        middlewares.append(provider)

        # Tool middleware - handles tool calls with internal loop re-entry
        middlewares.append(
            ToolCallMiddleware(
                tool_registry=TOOL_HANDLERS,
                provider=provider,
                max_tool_depth=20
            )
        )

        # Sentence buffer middleware - accumulates text into sentences
        middlewares.append(SentenceBufferMiddleware())

        # Filter middleware - removes empty text/sentence chunks only
        # (Don't filter tool_call, tool_notification, finish, etc. which have empty content)
        middlewares.append(
            FilterMiddleware(
                lambda c: (
                    c.type not in ["text", "sentence"]  # Pass through non-text chunks
                    or bool(c.content.strip())  # For text/sentence, check if non-empty
                )
            )
        )

        # TTS middleware (OUTERMOST) - processes sentence chunks, sends to TTS service
        tts_config = self.config.get("tts", {})
        if tts_config.get("service_url"):
            logger.info(f"TTS enabled: {tts_config['service_url']}")
            # Pass robot's media_manager for audio playback
            media_manager = self.mini.media_manager if self.mini else None
            middlewares.append(
                TTSStreamMiddleware(
                    tts_url=tts_config["service_url"],
                    timeout=tts_config.get("timeout", 30.0),
                    media_manager=media_manager
                )
            )
        else:
            logger.warning("TTS disabled (no service_url in config)")

        logger.info(f"Pipeline built with {len(middlewares)} middlewares")
        return StreamPipeline(middlewares)

    async def _speak_announcement(self) -> None:
        """
        Generate and speak a sarcastic GLaDOS startup announcement.
        Uses the LLM to create a dynamic response each time.
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Prompt for GLaDOS to generate a sarcastic awakening comment
            prompt = "Great with an oh, hi. then Generate a short one sentence, sarcastic observation about being rudely awoken by the user turning me on. Keep it under 20 words."

            logger.debug(f"Prompt: {prompt}")

            # Build a fresh pipeline for the announcement
            announcement_pipeline = self._build_pipeline()

            # Get the provider (first middleware) and set it to text mode
            provider = announcement_pipeline.middlewares[0]
            provider.set_user_message(prompt)

            # Process through pipeline
            logger.info("Speaking announcement...")

            async def empty_stream():
                """Empty stream for provider."""
                if False:
                    yield

            # Collect the response (will be spoken via TTS)
            announcement_text = []
            async for chunk in announcement_pipeline.process(empty_stream()):
                if chunk.type == "text" or chunk.type == "sentence":
                    announcement_text.append(chunk.content)
                elif chunk.type == "error":
                    logger.warning(f"Pipeline error during announcement: {chunk.content}")
                # Don't break - let stream finish naturally to avoid cancel scope errors

            full_announcement = "".join(announcement_text)
            if full_announcement:
                logger.info(f"GLaDOS: {full_announcement}")

            # Wait for audio queue to finish playing
            # Wait for singleton audio queue to be empty
            if TTSStreamMiddleware._audio_queue:
                logger.debug("Waiting for audio playback to complete...")
                await TTSStreamMiddleware._audio_queue.join()
                logger.debug("Audio playback complete")

        except Exception as e:
            logger.error(f"Announcement generation error: {e}", exc_info=True)

    async def startup(self) -> None:
        """
        Initialize all components and prepare for operation.

        Transitions to INTERACTIVE state after completion (for debugging).
        """
        logger.info("Reachy Voice Assistant Starting...")

        # Initialize robot
        if self.use_mock_robot:
            logger.warning("Running in MOCK mode (reachy-mini not installed)")
            self.mini = None
        else:
            logger.info("Initializing robot...")
            self.mini = ReachyMini()

        # Initialize animator (enabled is determined automatically based on mini being None or not)
        self.animator = RobotAnimator(self.mini)

        # Initialize audio stream
        logger.info("Setting up audio stream...")
        self.mic_stream = self._setup_audio_stream()

        # Initialize wake word
        logger.info("Loading wake word model...")
        self.wake_word_model = self._setup_wake_word()

        # Initialize VAD capture
        logger.info("Initializing VAD...")
        self.vad_capture = VADSpeechCapture(
            audio_stream=self.mic_stream,
            sample_rate=self.config["audio"]["sample_rate"],
            vad_model_path=self.config["vad"]["model_path"],
            vad_threshold=self.config["vad"]["confidence_threshold"]
        )

        # Initialize Anthropic client
        logger.info("Connecting to Anthropic...")
        self.llm_client = AnthropicClient(
            model=self.config["anthropic"]["model"]
        )

        # Initialize STT engine (required for text-only mode)
        logger.info("Initializing local STT engine...")
        stt_config = self.config.get("stt", {})
        self.stt_engine = STTEngine(
            model_size=stt_config.get("model_size", "base"),
            device=stt_config.get("device", "cpu"),
            language=stt_config.get("language", "en"),
            min_speech_confidence=stt_config.get("min_speech_confidence", 0.6),
            max_no_speech_prob=stt_config.get("max_no_speech_prob", 0.6),
            min_compression_ratio=stt_config.get("min_compression_ratio", 0.5),
            max_compression_ratio=stt_config.get("max_compression_ratio", 2.5),
            min_text_length=stt_config.get("min_text_length", 3),
        )
        logger.info(f"STT engine ready ({stt_config.get('model_size', 'base')} model)")

        # Initialize audio input processor
        logger.info("Initializing audio input processor...")
        self.audio_processor = AudioInputProcessor(stt_engine=self.stt_engine)

        # Build middleware pipeline
        logger.info("Building middleware pipeline...")
        self.pipeline = self._build_pipeline()

        # Wake up animation before announcement
        logger.info("Waking up Reachy...")
        await self.animator.wake_up()
        logger.info("Wake up animation complete")

        # TEMP: Disable announcement to test if issue is specific to it
        logger.debug("[DEBUG] Announcement disabled for testing")
        # logger.info("Generating startup announcement...")
        # await self._speak_announcement()

        # Start continuous VAD listening
        logger.info("Starting continuous VAD listener...")
        self.vad_task = asyncio.create_task(self._continuous_vad_listener())

        # TEMP: Skip sleep for troubleshooting - go straight to interactive
        logger.debug("[DEBUG] Skipping sleep mode, going straight to interactive")
        self.state = ReachyState.INTERACTIVE

    async def sleep_phase(self) -> None:
        """
        Monitor for wake word while robot sleeps.

        Transitions to INTERACTIVE when wake word detected.
        """
        logger.info("Sleeping... waiting for wake word")

        # TEMP: Skip wake word for testing - go straight to interactive
        logger.debug("[DEBUG] Skipping wake word, going straight to interactive mode")
        await self.animator.wake_up()
        await asyncio.sleep(2)

        # Start continuous VAD listener
        logger.info("Starting continuous VAD listener...")
        self.vad_task = asyncio.create_task(self._continuous_vad_listener())

        self.state = ReachyState.INTERACTIVE
        return

        # # Run wake word detection in thread executor (it's blocking)
        # loop = asyncio.get_event_loop()
        # wake_detected = await loop.run_in_executor(
        #     None, self._wait_for_wake_word
        # )

        # if wake_detected and self.running:
        #     print()
        #     print("👋 Wake word detected!")
        #     await self.animator.wake_up()
        #
        #     # Start continuous VAD listener
        #     print("  ✓ Starting continuous VAD listener...")
        #     self.vad_task = asyncio.create_task(self._continuous_vad_listener())
        #
        #     self.state = ReachyState.INTERACTIVE

    def _wait_for_wake_word(self) -> bool:
        """
        Blocking wake word detection (runs in thread executor).

        Returns:
            True if wake word detected, False if interrupted
        """
        threshold = self.config["wake_word"]["threshold"]

        # Audio format for wake word
        n_samples = 1280  # 80ms at 16kHz

        try:
            while self.running:
                # Read audio chunk (bytes)
                audio_bytes = self.mic_stream.read(n_samples, exception_on_overflow=False)

                # Convert bytes to numpy array (int16 format from pyaudio.paInt16)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

                # Run wake word prediction
                prediction = self.wake_word_model.predict(audio_data)

                # Check for wake word (first model in the list)
                model_name = list(prediction.keys())[0]
                score = prediction[model_name]

                if score >= threshold:
                    return True

        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            return False

        return False

    async def interactive_phase(self) -> None:
        """
        Wait for VAD to detect speech (via queue).

        VAD runs continuously in background, this phase just waits
        for speech to appear in the queue.

        Transitions to:
        - PROCESSING if speech captured
        - SLEEP if timeout or no speech
        """
        logger.info("Listening for your request...")

        timeout = self.config["conversation"]["interactive_timeout"]

        try:
            # Wait for speech from continuous VAD listener
            audio_bytes = await asyncio.wait_for(
                self.speech_queue.get(),
                timeout=timeout
            )

            if audio_bytes:
                logger.info(f"Captured speech: {len(audio_bytes)} bytes")
                self.current_audio = audio_bytes
                self.state = ReachyState.PROCESSING
            else:
                logger.info("No speech detected")
                await self._return_to_sleep()

        except asyncio.TimeoutError:
            logger.info("Timeout waiting for speech")
            await self._return_to_sleep()

        except Exception as e:
            logger.error(f"Error in interactive phase: {e}")
            await self._return_to_sleep()

    async def processing_phase(self) -> None:
        """
        Process user request through Anthropic Claude + middleware pipeline.

        VAD continues listening in background. If speech detected, interruption_flag
        will be set and this phase will cancel and restart.

        Handles:
        - Sending transcribed text to Claude
        - Streaming responses
        - Tool calls with animations
        - Interruption via VAD speech detection

        Transitions to:
        - INTERACTIVE if completed or interrupted
        - SLEEP if error occurs
        """
        logger.info("Processing request...")

        # Flush any lingering audio from previous response
        from reachy_mvp.core.middlewares.tts import TTSStreamMiddleware
        await TTSStreamMiddleware.cancel_playback()

        # Clear interrupt flag for this new processing session
        self.interruption_flag.clear()

        try:
            # Process request with interrupt monitoring
            processing_task = asyncio.create_task(self._process_request())

            # Wait for completion or interruption
            while not processing_task.done():
                if self.interruption_flag.is_set():
                    # Cancel processing
                    logger.info("Interruption detected - cancelling processing")
                    processing_task.cancel()
                    try:
                        await processing_task
                    except asyncio.CancelledError:
                        pass

                    # Cancel TTS
                    await TTSStreamMiddleware.cancel_playback()

                    # Clear flag and go back to INTERACTIVE to process new speech
                    self.interruption_flag.clear()
                    logger.warning("Interrupted by new speech!")
                    self.state = ReachyState.INTERACTIVE
                    return

                await asyncio.sleep(0.05)

            # Completed normally
            logger.info("Response complete")
            self.state = ReachyState.INTERACTIVE

        except Exception as e:
            logger.error(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
            await self._return_to_sleep()

    async def _process_request(self) -> None:
        """
        Process transcribed text through middleware pipeline.

        The pipeline now includes the provider middleware which calls Anthropic Claude,
        so we don't call the LLM externally anymore.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Process audio input (transcription, noise detection, fallback)
        result: AudioInputResult = await self.audio_processor.process_audio_input(
            self.current_audio
        )

        # Check if we should skip this input (noise detected or error)
        if result.should_skip:
            logger.info(f"Skipping input: {result.skip_reason}")
            # Return early without adding to history or processing
            # This will complete _process_request() and return to INTERACTIVE state
            return

        # Log transcribed text
        logger.info(f"Using transcribed text: '{result.transcribed_text}'")

        # Add user message to conversation history
        self.messages.append(result.to_message())

        # Trim history if needed
        if len(self.messages) > self.max_history:
            # Keep system message + recent messages
            self.messages = [self.messages[0]] + self.messages[-(self.max_history-1):]

        # Rebuild pipeline for this request with updated conversation history
        logger.info(
            f"Processing request: text mode, "
            f"history={len(self.messages)} messages"
        )

        # Create fresh pipeline with current state
        request_pipeline = self._build_pipeline()

        # Update the provider (FIRST middleware now) with transcribed text
        # Note: We only support text mode now (no audio to LLM)
        provider = request_pipeline.middlewares[0]  # Provider is now at index 0 (innermost)
        provider.set_user_message(result.transcribed_text)

        # Process through middleware pipeline with empty input stream
        # (Provider middleware ignores input stream - it's a source)
        accumulated_response = []

        async def empty_stream():
            """Empty stream for provider (it ignores input)."""
            if False:
                yield

        logger.debug("Starting pipeline iteration...")

        try:
            async for chunk in request_pipeline.process(empty_stream()):
                logger.debug(f"Received chunk type={chunk.type}, content='{chunk.content[:50] if chunk.content else ''}'")

                # Handle different chunk types
                if chunk.type == "tool_notification":
                    # Trigger animation for tool execution
                    logger.info(f"Tool notification: {chunk.tool_name}")
                    await self.animator.tool_animation(chunk.tool_name)

                elif chunk.type == "text" and chunk.content:
                    # Accumulate and print text chunks (if any make it through)
                    accumulated_response.append(chunk.content)
                    print(chunk.content, end="", flush=True)

                elif chunk.type == "sentence" and chunk.content:
                    # Accumulate and print sentence chunks
                    accumulated_response.append(chunk.content)
                    print(chunk.content, end="", flush=True)
                    logger.debug(f"Sentence complete: {chunk.content[:50]}...")

                elif chunk.type == "tts":
                    # TTS audio chunk (could be played back if needed)
                    logger.debug(f"TTS audio chunk: {len(chunk.audio_data)} bytes")

                elif chunk.type == "error":
                    # Error occurred in pipeline
                    logger.error(f"Pipeline error: {chunk.content}")

                elif chunk.type == "finish":
                    logger.info(f"Stream finished: reason={chunk.finish_reason}")
                    print()  # New line after response

        except asyncio.CancelledError:
            # Task was cancelled (due to interruption)
            logger.warning("[INTERRUPT] Request processing cancelled!")
            logger.info("[INTERRUPT] Cleaning up TTS playback")
            from reachy_mvp.core.middlewares.tts import TTSStreamMiddleware
            logger.info("[INTERRUPT] Calling TTSStreamMiddleware.cancel_playback()")
            await TTSStreamMiddleware.cancel_playback()
            logger.info("[INTERRUPT] Playback cancellation complete - safe to proceed")
            raise  # Re-raise to propagate cancellation

        finally:
            # Add assistant response to history (even if interrupted)
            if accumulated_response:
                response_text = "".join(accumulated_response)
                self.messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                logger.debug(f"Response added to history: {len(response_text)} chars")

    # FUTURE: Re-enable interrupt monitoring with dedicated audio stream
    # to avoid corrupting VAD stream. See plan: sequential-frolicking-russell.md
    # Methods _monitor_interruption() and _check_wake_word_once() removed.

    async def _return_to_sleep(self) -> None:
        """Helper to transition back to sleep state."""
        # Cancel continuous VAD listener
        if self.vad_task and not self.vad_task.done():
            logger.info("Cancelling continuous VAD task")
            self.vad_task.cancel()
            try:
                await self.vad_task
            except asyncio.CancelledError:
                pass
            self.vad_task = None

        await self.animator.go_to_sleep()
        self.state = ReachyState.SLEEP

    async def run(self) -> None:
        """
        Main event loop.

        Runs the state machine indefinitely until shutdown.
        """
        try:
            # Initial startup
            await self.startup()

            # Main loop
            while self.running:
                if self.state == ReachyState.SLEEP:
                    await self.sleep_phase()

                elif self.state == ReachyState.INTERACTIVE:
                    await self.interactive_phase()

                elif self.state == ReachyState.PROCESSING:
                    await self.processing_phase()

                # Small yield to event loop
                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.running = False

        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        """
        Clean up resources before shutdown.
        """
        logger.info("Cleaning up...")

        # Cancel VAD task if running
        if self.vad_task and not self.vad_task.done():
            logger.info("Cancelling VAD task")
            self.vad_task.cancel()
            try:
                await self.vad_task
            except asyncio.CancelledError:
                pass

        # Close audio stream
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()

        # Terminate audio
        if self.audio:
            self.audio.terminate()

        # Put robot to sleep
        if not self.use_mock_robot and self.animator:
            await self.animator.go_to_sleep()

        logger.info("Cleanup complete")
