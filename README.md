# Reachy GLaDOS Voice Assistant

A complete voice assistant implementation for the Reachy Mini robot with GLaDOS personality from Portal.

## Features

- 🎤 **Continuous VAD-based speech detection** using Silero VAD
- 🤖 **Anthropic Claude integration** for AI responses with tool calling
- 🗣️ **Local STT** using faster-whisper (no cloud dependency)
- 🔊 **TTS integration** with interruptible playback
- ⚡ **Real-time interruption** - speak anytime to interrupt current response
- 🎭 **Robot animations** synchronized with speech and tool execution
- 🔧 **Tool calling** with animation feedback

## Architecture

### State Machine
- **SLEEP**: Wake word monitoring (OpenWakeWord)
- **AWAKE**: Continuous VAD listening
- **INTERACTIVE**: Wait for speech via queue
- **PROCESSING**: Process through middleware with interrupt support

### Middleware Pipeline
```
User Speech → VAD → STT → Anthropic Claude → Tool Middleware → Sentence Buffer → Filter → TTS → Robot
```

## Setup

### Prerequisites

1. Python 3.10+
2. Reachy Mini robot (or run in mock mode)
3. Anthropic API key

### Installation

```bash
# Clone the repository
git clone git@github.com:andyjmorgan/reachy-glados-example.git
cd reachy-glados-example

# Install dependencies
pip install -r requirements.txt

# Download required models
mkdir -p models

# Download Silero VAD model
wget https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx -O models/silero_vad_v5.onnx

# Download wake word model (or train your own)
# Place your wake word model at: models/reachy.onnx
```

### Configuration

1. Create `.env` file:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

2. Update `config.yaml` with your settings:
   - TTS service URL
   - Model paths
   - Thresholds and timeouts

### Running

```bash
python main.py
```

## Key Components

### VAD Capture (`reachy_mvp/core/vad_capture.py`)
- Continuous voice activity detection
- Circular buffer pattern for pre-activation audio
- Pause detection for natural turn-taking

### State Machine (`reachy_mvp/core/state_machine.py`)
- Orchestrates entire lifecycle
- Manages VAD task, interrupt handling, and state transitions
- Continuous VAD listening throughout AWAKE state

### Anthropic Client (`reachy_mvp/core/anthropic_client.py`)
- Streaming responses from Claude
- Tool calling support
- Message format conversion (OpenAI ↔ Anthropic)

### Middleware Pipeline
- **Provider**: Calls Anthropic Claude API
- **Tool**: Executes function calls with re-entrancy
- **Sentence Buffer**: Accumulates text into complete sentences
- **Filter**: Removes empty chunks
- **TTS**: Sends sentences to TTS service with interruptible playback

## Configuration

### VAD Settings (`config.yaml`)
```yaml
vad:
  vad_size: 32          # ms per chunk
  buffer_size: 800      # ms pre-activation buffer
  pause_limit: 640      # ms silence to end speech
  confidence_threshold: 0.5
```

### STT Noise Detection
Adjust thresholds in `config.yaml` to tune sensitivity:
```yaml
stt:
  min_speech_confidence: -1.0    # Lower = more permissive
  max_no_speech_prob: 0.8        # Higher = more permissive
```

## Architecture Decisions

### Why Continuous VAD?
Previously, VAD stopped after capturing speech, and wake word monitoring ran during processing. This caused **stream corruption** due to concurrent access with different chunk sizes:
- VAD: 512 samples (32ms)
- Wake word: 1280 samples (80ms)

**Solution**: VAD runs continuously as the single owner of `mic_stream` during AWAKE state. Wake word monitoring only during SLEEP.

### Why Queue-Based Communication?
The continuous VAD task puts captured speech into an asyncio queue. The state machine pulls from this queue, enabling clean separation and preventing blocking issues.

### Why Interrupt via VAD?
User speech (detected by VAD) is the natural interrupt signal during conversations. Wake word is for waking up from sleep only.

## Troubleshooting

### "No speech detected" after first request
- **Fixed**: This was caused by concurrent stream access. Now resolved with continuous VAD.

### Core Audio error `-50`
- Intermittent PyAudio issue on macOS
- Non-fatal - system continues working
- Related to audio device buffer management

### TTS not playing
- Check TTS service is running at configured URL
- Verify `config.yaml` has correct `tts.service_url`

## Development

### Testing
```bash
pytest tests/
```

### Logging
All logging uses structured format:
```
[time]-[severity]-[codepath:line]-[message]
```

Levels:
- **DEBUG**: Detailed diagnostics
- **INFO**: State transitions, major events
- **WARNING**: Non-fatal issues
- **ERROR**: Failures

## Credits

Built with [Claude Code](https://claude.com/claude-code)

Inspired by GLaDOS from Portal and the original [GLaDOS implementation](https://github.com/dnhkng/GlaDOS).

## License

MIT
