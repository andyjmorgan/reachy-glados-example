"""
Anthropic Claude API integration.

This module handles communication with Anthropic's Claude API, sending text
and receiving streaming text/tool call responses.
"""

import os
from typing import AsyncIterator, Optional, List, Dict, Any
import json

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from reachy_mvp.core.streaming import StreamChunk


class AnthropicClient:
    """
    Handles Anthropic Claude API calls with streaming.

    Sends text input to Claude and streams back text responses
    and tool calls compatible with existing middleware framework.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514"
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-sonnet-4-20250514)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )

        self.client = AsyncAnthropic(api_key=self.api_key)
        self.model = model

    async def send_text_stream(
        self,
        text: str,
        conversation_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> AsyncIterator[StreamChunk]:
        """
        Send text to Claude and stream responses.

        Args:
            text: Text message to send (can be empty for tool result continuations)
            conversation_history: List of previous messages (OpenAI format)
            tools: Optional list of tool definitions (Anthropic format)
            temperature: Response randomness (0-1)
            max_tokens: Maximum tokens in response

        Yields:
            StreamChunk objects:
                - type="text", content=<str>
                - type="tool_call", tool_name=<str>, arguments=<dict>, tool_call_id=<str>
                - type="finish"
        """
        # Convert OpenAI format to Anthropic format
        system_message, anthropic_messages = self._convert_messages(
            conversation_history, text
        )

        request_params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }

        if system_message:
            request_params["system"] = system_message

        if tools:
            request_params["tools"] = tools

        try:
            async with self.client.messages.stream(**request_params) as stream:
                async for event in stream:
                    # Handle different event types
                    if event.type == "content_block_start":
                        # New content block starting
                        if hasattr(event, 'content_block'):
                            block = event.content_block
                            if block.type == "tool_use":
                                # Tool call starting - we'll yield it when we have all the data
                                pass

                    elif event.type == "content_block_delta":
                        # Content chunk
                        delta = event.delta
                        if delta.type == "text_delta":
                            # Text content
                            yield StreamChunk(
                                type="text",
                                content=delta.text
                            )
                        elif delta.type == "input_json_delta":
                            # Tool call arguments being streamed
                            # We'll accumulate and yield when complete
                            pass

                    elif event.type == "content_block_stop":
                        # Content block complete
                        # Check if this was a tool use block
                        if hasattr(stream, 'current_message_snapshot'):
                            message = stream.current_message_snapshot
                            if message and message.content:
                                for block in message.content:
                                    if block.type == "tool_use":
                                        # Yield tool call
                                        yield StreamChunk(
                                            type="tool_call",
                                            tool_name=block.name,
                                            arguments=block.input,
                                            tool_call_id=block.id
                                        )

                    elif event.type == "message_stop":
                        # Message complete
                        if hasattr(stream, 'current_message_snapshot'):
                            message = stream.current_message_snapshot
                            yield StreamChunk(
                                type="finish",
                                finish_reason=message.stop_reason if message else "stop"
                            )

        except Exception as e:
            error_msg = f"Anthropic API error: {str(e)}"
            yield StreamChunk(
                type="error",
                content=error_msg,
                metadata={"error": str(e)}
            )

    def _convert_messages(
        self,
        openai_messages: List[Dict[str, Any]],
        new_user_text: str
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert OpenAI message format to Anthropic format.

        OpenAI: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        Anthropic: system="...", messages=[{"role": "user", "content": "..."}, ...]

        Also handles tool messages in Anthropic format.

        Args:
            openai_messages: Messages in OpenAI format
            new_user_text: New user message to append (can be empty)

        Returns:
            (system_message, anthropic_messages)
        """
        system_message = None
        anthropic_messages = []

        for msg in openai_messages:
            role = msg.get("role")

            if role == "system":
                # Extract system message (Anthropic uses separate parameter)
                system_message = msg.get("content")

            elif role == "user":
                # User message - pass through
                anthropic_messages.append({
                    "role": "user",
                    "content": msg.get("content", "")
                })

            elif role == "assistant":
                # Assistant message
                content = msg.get("content")
                tool_calls = msg.get("tool_calls")

                if tool_calls:
                    # Assistant made tool calls - convert to Anthropic format
                    content_blocks = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})

                    for tc in tool_calls:
                        func = tc.get("function", {})
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id"),
                            "name": func.get("name"),
                            "input": json.loads(func.get("arguments", "{}"))
                        })

                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content_blocks
                    })
                else:
                    # Regular assistant message
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content or ""
                    })

            elif role == "tool":
                # Tool result - convert to Anthropic format
                # Anthropic expects tool results in user messages
                tool_call_id = msg.get("tool_call_id")
                result_content = msg.get("content", "")

                # Check if last message is user with tool_result content
                # If so, append to it; otherwise create new user message
                if (anthropic_messages and
                    anthropic_messages[-1]["role"] == "user" and
                    isinstance(anthropic_messages[-1]["content"], list)):
                    # Append to existing tool result message
                    anthropic_messages[-1]["content"].append({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": result_content
                    })
                else:
                    # Create new user message with tool result
                    anthropic_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": result_content
                        }]
                    })

        # Add new user message if provided
        if new_user_text:
            anthropic_messages.append({
                "role": "user",
                "content": new_user_text
            })

        return system_message, anthropic_messages
