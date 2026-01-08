
General behavior

# Startup:

On startup, the application should initialize all necessary components and services. 
We then start the wakeword monitor and put the robot to sleep.

# Sleep
We then wait for the wake word before we start the loop.
On issue of the wakeword, we start the interactivity stage. 
The sleep phase is no longer monitoring for wake word.

# Interactive stage
The microphone input is monitored for user requests.
When we detect speech we begin monitoring for the end of the speech.
Once we reach a natural pause in the speech, we process the request.
We need to be prepared for ambient noise and false starts.

# Request processing

request processing will create a middleware chain to handle the request.
the middleware pattern is described below.
request processing will send the audio clip to https://platform.openai.com/docs/models/gpt-audio using chat completions in wav format.
We'll stream the results back.
Tool calls will be intercepted by the tool calling middleware and handled as described below.
Text streamed messages will be collated to the next Period, carriage return, or end of message.
Once collated, they are sent to the TTS engine to be spoken back to the user.

## Handling tool calls
tool calls will get collated and sent to the appropriate tool handler,
We'll emit a tool call message to trigger a downstream animation.
Once we have the tool results back we'll re-call the downstream middleware again.

# Middleware pattern
the middleware pattern takes chat completions requests and passes them through a series of middleware handlers.
the middleware responds with a middlewareMessage, which is a union of Tool call, text response, or audio response.
Tool calls will result in a custom animation.
Text results will be sent to console logging.
Audio results will be queued for playback.

# Handling Interruption

While we're in request processing, if the user begins speaking again, we should interrupt the current processing.
Cancelling the current request and starting a new request processing cycle.
Cancelling the current request should trigger an antenna wiggle.

review the project here: for reference on interuption handling, listening, etc:

/Users/andrewmorgan/Personal/source/reachy/glados-all-in-one/GLaDOS



