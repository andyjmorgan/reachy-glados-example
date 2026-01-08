"""
OpenAI Realtime Audio API integration.

This module handles communication with OpenAI's audio API, sending audio
and receiving streaming text/tool call responses.
"""

import os
import base64
from typing import AsyncIterator, Optional
import json

from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


from reachy_mvp.core.streaming import StreamChunk


class OpenAIAudioClient:
    """
    Handles OpenAI gpt-4o-audio API calls with streaming.

    Sends audio input to OpenAI and streams back text responses
    and tool calls compatible with existing middleware framework.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-audio-preview",
        text_model: Optional[str] = None
    ):
        """
        Initialize OpenAI audio client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use for audio requests (default: gpt-4o-audio-preview)
            text_model: Model to use for text-only requests (defaults to gpt-4o-search-preview)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model  # Audio model
        self.text_model = text_model or "gpt-4o-search-preview"  # Text model with built-in web search

    async def send_audio_stream(
        self,
        audio_bytes: bytes,
        conversation_history: list,
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        web_search_options: Optional[dict] = None
    ) -> AsyncIterator[StreamChunk]:
        """
        Send audio to OpenAI and stream responses.

        Args:
            audio_bytes: WAV audio bytes (16kHz, mono, int16)
            conversation_history: List of previous messages
            tools: Optional list of tool definitions
            temperature: Response randomness (0-2)
            max_tokens: Maximum tokens in response
            web_search_options: Optional web search configuration (location, etc.)

        Yields:
            StreamChunk objects:
                - type="text", content=<str>
                - type="tool_call", tool_name=<str>, arguments=<dict>, id=<str>
                - type="finish"
        """
        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Build messages array
        messages = conversation_history.copy()
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": "wav"
                    }
                }
            ]
        })

        # Prepare request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            request_params["tools"] = tools

        # Note: web_search_options not supported in Chat Completions API
        # Only available in Responses API
        # For location context, add it to system prompt instead

        try:
            # Create streaming completion
            response = await self.client.chat.completions.create(**request_params)

            # Process streaming chunks
            async for chunk in response:
                # Extract delta from chunk
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta
                finish_reason = choice.finish_reason

                # Handle text content
                if hasattr(delta, 'content') and delta.content:
                    yield StreamChunk(
                        type="text",
                        content=delta.content
                    )

                # Handle tool calls
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        # Tool calls come in chunks, accumulate them
                        if hasattr(tool_call, 'function'):
                            func = tool_call.function
                            if hasattr(func, 'name') and func.name:
                                # Parse arguments if available
                                args = {}
                                if hasattr(func, 'arguments') and func.arguments:
                                    try:
                                        args = json.loads(func.arguments)
                                    except json.JSONDecodeError:
                                        # Partial arguments, skip for now
                                        continue

                                yield StreamChunk(
                                    type="tool_call",
                                    content="",
                                    tool_name=func.name,
                                    arguments=args,
                                    tool_call_id=tool_call.id if hasattr(tool_call, 'id') else None
                                )

                # Handle finish
                if finish_reason:
                    yield StreamChunk(
                        type="finish",
                        content="",
                        finish_reason=finish_reason
                    )
                    break

        except Exception as e:
            # Yield error chunk
            yield StreamChunk(
                type="error",
                content=f"OpenAI API error: {str(e)}"
            )

    async def send_text_stream(
        self,
        text: str,
        conversation_history: list,
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        web_search_options: Optional[dict] = None
    ) -> AsyncIterator[StreamChunk]:
        """
        Send text (not audio) to OpenAI and stream responses.

        Useful for testing or tool call responses.

        Args:
            text: Text message to send
            conversation_history: List of previous messages
            tools: Optional list of tool definitions
            temperature: Response randomness (0-2)
            max_tokens: Maximum tokens in response
            web_search_options: Optional web search configuration (location, etc.)

        Yields:
            StreamChunk objects (same format as send_audio_stream)
        """
        messages = conversation_history.copy()
        messages.append({
            "role": "user",
            "content": text
        })

        request_params = {
            "model": self.text_model,  # Use text model for text-only requests
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            request_params["tools"] = tools

        # Note: web_search_options not supported in Chat Completions API
        # Only available in Responses API
        # For location context, add it to system prompt instead

        try:
            response = await self.client.chat.completions.create(**request_params)

            async for chunk in response:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta
                finish_reason = choice.finish_reason

                if hasattr(delta, 'content') and delta.content:
                    yield StreamChunk(
                        type="text",
                        content=delta.content
                    )

                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if hasattr(tool_call, 'function'):
                            func = tool_call.function
                            if hasattr(func, 'name') and func.name:
                                args = {}
                                if hasattr(func, 'arguments') and func.arguments:
                                    try:
                                        args = json.loads(func.arguments)
                                    except json.JSONDecodeError:
                                        continue

                                yield StreamChunk(
                                    type="tool_call",
                                    content="",
                                    tool_name=func.name,
                                    arguments=args,
                                    tool_call_id=tool_call.id if hasattr(tool_call, 'id') else None
                                )

                if finish_reason:
                    yield StreamChunk(
                        type="finish",
                        content="",
                        finish_reason=finish_reason
                    )
                    break

        except Exception as e:
            yield StreamChunk(
                type="error",
                content=f"OpenAI API error: {str(e)}"
            )
