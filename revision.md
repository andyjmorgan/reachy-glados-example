
## Startup
when the robot starts up, we make an introduction message, we send a message through the middleware and TTS the output. We then enter the main awake state.

## Awake state:

During this period we're continuously listening for voice commands using Silero VAD (Voice Activity Detection). VAD runs as a continuous background task throughout the AWAKE state, monitoring the microphone stream for speech.

Once VAD detects speech (and a pause), the captured audio is placed in a queue. The voice command is transcribed to text using faster-whisper STT, then sent to the middleware for processing.

The middleware processes the request through Anthropic Claude and returns a response text which we then TTS and play back to the user.

**Interruption**: VAD continues listening even during middleware processing and TTS playback. If new speech is detected during processing, VAD sets an interruption flag, causing the current middleware execution to cancel immediately (including TTS playback). The new speech is then processed.


## Middleware

The middleware is well established, but we need to ensure when we're looping tools we keep appending to the original request, including the tool request and tool responses.
The tts middleware will start the talking animation when it starts playing audio, and stop the talking animation when audio playback is complete.
The cancellation of the middleware, should trigger the tts middleware to stop audio playback and stop the talking animation.
The tts middleware is responsible for this solely.

## Sleep

During sleep, we monitor for the wake word using OpenWakeWord, entering the AWAKE state when detected.

**Wake word monitoring is ONLY active during SLEEP** - this prevents stream corruption. During AWAKE state, only VAD accesses the microphone stream.

We do not play the introduction message again until the robot is restarted.
Sleep should occur after 5 minutes of idle time (30 seconds currently for testing).



