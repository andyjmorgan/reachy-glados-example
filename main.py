"""
Main entry point for Reachy MVP Voice Assistant.

Usage:
    python main.py

Requirements:
    - OPENAI_API_KEY environment variable must be set
    - GLaDOS TTS service should be running (optional for MVP)
    - Reachy robot should be connected (or run in mock mode)
"""

import asyncio
import os
import sys
import warnings
import io
import contextlib
import logging
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging with the required format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class StderrFilter:
    """
    Filter stderr output to suppress noisy async cleanup errors.

    Filters out "Exception ignored" messages from httpx/openai async generator cleanup.
    These are benign cleanup warnings that don't affect functionality.
    """
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.buffer = ""
        self.suppressing = False  # Track if we're currently suppressing a traceback

    def write(self, text):
        # Buffer the text
        self.buffer += text

        # If we have a complete line (ends with newline)
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            # Keep the last incomplete line in buffer
            self.buffer = lines[-1]

            # Process complete lines
            for line in lines[:-1]:
                # Check if this starts a traceback we want to suppress
                if "Exception ignored" in line:
                    self.suppressing = True
                    continue

                # If we're suppressing, check if this line is part of the traceback
                if self.suppressing:
                    # Traceback lines to suppress
                    if (
                        line.strip().startswith("Traceback") or
                        line.strip().startswith("File ") or
                        "AsyncLibraryNotFoundError" in line or
                        "RuntimeError" in line or
                        "async generator" in line or
                        "httpx" in line or
                        "openai" in line or
                        "sniffio" in line or
                        "httpcore" in line or
                        line.strip() == ""  # Empty lines in traceback
                    ):
                        # Continue suppressing
                        continue
                    else:
                        # End of traceback, stop suppressing
                        self.suppressing = False

                # Output non-suppressed lines
                if not self.suppressing:
                    self.original_stderr.write(line + '\n')

    def flush(self):
        # Flush any buffered content if not suppressing
        if self.buffer and not self.suppressing:
            self.original_stderr.write(self.buffer)
            self.buffer = ""
        self.original_stderr.flush()

    def fileno(self):
        return self.original_stderr.fileno()


# Install stderr filter
sys.stderr = StderrFilter(sys.__stderr__)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from reachy_mvp.core.state_machine import ReachyStateMachine


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create config.yaml in the current directory."
        )

    with open(config_file) as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid config file: {e}")


def check_environment() -> None:
    """
    Check required environment variables and dependencies.

    Raises:
        SystemExit: If critical requirements are missing
    """
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        print()
        print("Please set your OpenAI API key in .env file:")
        print("  OPENAI_API_KEY=sk-...")
        print()
        print("Or export as environment variable:")
        print("  export OPENAI_API_KEY='sk-...'")
        print()
        sys.exit(1)

    # Check model files
    required_models = [
        "models/silero_vad_v5.onnx",
        "models/reachy.onnx"
    ]

    missing_models = []
    for model_path in required_models:
        if not Path(model_path).exists():
            missing_models.append(model_path)

    if missing_models:
        print("❌ Error: Required model files not found:")
        for model in missing_models:
            print(f"  - {model}")
        print()
        print("Please download required models:")
        print("  - Silero VAD: https://github.com/snakers4/silero-vad/raw/master/files/silero_vad_v5.onnx")
        print("  - Reachy wake word: (should be in models/ directory)")
        print()
        sys.exit(1)

    print("✅ Environment check passed")
    print()


def print_banner() -> None:
    """Print startup banner."""
    print()
    print("=" * 60)
    print("  Reachy Voice Assistant MVP")
    print("=" * 60)
    print()
    print("  🤖 Interactive voice assistant for Reachy Mini")
    print("  🎤 Wake word: 'Hey Reachy'")
    print("  ⚡ Powered by OpenAI GPT-4o Audio")
    print()
    print("=" * 60)
    print()


def print_instructions() -> None:
    """Print usage instructions."""
    print()
    print("📋 Instructions:")
    print("  1. Say 'Hey Reachy' to wake the robot")
    print("  2. Speak your request when the robot wakes up")
    print("  3. Wait for the response")
    print("  4. Say 'Hey Reachy' again to interrupt")
    print("  5. Press Ctrl+C to exit")
    print()
    print("=" * 60)
    print()


async def main() -> None:
    """
    Main entry point.

    Loads configuration, checks environment, and starts the state machine.
    """
    try:
        # Print banner
        print_banner()

        # Check environment
        check_environment()

        # Load configuration
        print("📄 Loading configuration...")
        config = load_config()
        print("  ✓ Configuration loaded")
        print()

        # Print instructions
        print_instructions()

        # Create and run state machine
        state_machine = ReachyStateMachine(config)
        await state_machine.run()

    except KeyboardInterrupt:
        print()
        print("  👋 Goodbye!")

    except Exception as e:
        print()
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
