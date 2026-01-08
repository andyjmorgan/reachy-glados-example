"""
Provider middleware that wraps Anthropic Claude client for re-entrant tool calling.

This middleware acts as the source of streaming responses by calling Anthropic's API.
Unlike other middlewares that transform streams, this middleware creates streams.

Text-only (no audio support).
"""

import asyncio
import logging
from typing import AsyncIterator, Optional, List, Dict, Any

from reachy_mvp.core.streaming import StreamMiddleware, StreamChunk
from reachy_mvp.core.anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)


class ProviderMiddleware(StreamMiddleware):
    """
    Middleware that calls Anthropic Claude API to generate streaming responses.

    This middleware wraps the Anthropic client and can be re-invoked by tool middleware
    to continue conversations after tool execution.

    Key features:
    - Creates streams (doesn't transform them)
    - Stores conversation history
    - Supports text input only (no audio)
    - Can be re-invoked with updated history for tool calls
    - Ignores input stream (it's a source, not a transformer)
    """

    def __init__(
        self,
        llm_client: AnthropicClient,
        user_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.8,
        max_tokens: int = 1000
    ):
        """
        Initialize provider middleware.

        Args:
            llm_client: Anthropic client instance
            user_message: Text message for text input
            conversation_history: List of message dicts (system, user, assistant, tool)
            tools: List of tool definitions for function calling (Anthropic format)
            temperature: Claude temperature parameter
            max_tokens: Claude max_tokens parameter
        """
        self.llm_client = llm_client
        self.user_message = user_message
        self.conversation_history = conversation_history or []
        self.tools = tools
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.debug(
            f"ProviderMiddleware initialized: "
            f"user_message={bool(user_message)}, "
            f"history_len={len(self.conversation_history)}, "
            f"tools={len(tools) if tools else 0}"
        )

    def set_user_message(self, message: str) -> None:
        """
        Set text message for the request.

        Args:
            message: User text message
        """
        self.user_message = message
        logger.debug(f"Provider user message set: {message[:50]}...")

    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """
        Add messages to conversation history.

        Used by tool middleware to add tool results before re-calling provider.

        Args:
            messages: List of message dicts to append to history
        """
        self.conversation_history.extend(messages)
        logger.debug(
            f"Provider messages added: {len(messages)} messages, "
            f"total history={len(self.conversation_history)}"
        )

    async def process(self, stream: AsyncIterator[StreamChunk]) -> AsyncIterator[StreamChunk]:
        """
        Process stream by calling Anthropic API and yielding response chunks.

        Note: This middleware IGNORES the input stream - it's a source, not a transformer.
        The input stream parameter is required by the StreamMiddleware interface but unused.

        Args:
            stream: Input stream (ignored)

        Yields:
            StreamChunk objects from Anthropic response
        """
        logger.debug(
            f"Provider request params: "
            f"history_len={len(self.conversation_history)}, "
            f"user_msg={bool(self.user_message)}, "
            f"tools={len(self.tools) if self.tools else 0}"
        )

        llm_stream = None
        try:
            # Call Anthropic with text input
            logger.debug("Provider calling send_text_stream()")
            llm_stream = self.llm_client.send_text_stream(
                text=self.user_message or "",  # Empty string for continuation
                conversation_history=self.conversation_history,
                tools=self.tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Yield all chunks from Anthropic stream
            chunk_count = 0
            tool_call_count = 0
            text_chars = 0

            async for chunk in llm_stream:
                chunk_count += 1

                if chunk.type == "tool_call":
                    tool_call_count += 1
                    logger.info(
                        f"Provider yielding tool_call: "
                        f"name={chunk.tool_name}, "
                        f"args={chunk.arguments}, "
                        f"id={chunk.tool_call_id}"
                    )
                elif chunk.type == "text" and chunk.content:
                    text_chars += len(chunk.content)
                    logger.debug(f"Provider yielding text: {chunk.content[:30]}...")
                elif chunk.type == "finish":
                    logger.info(
                        f"Provider stream complete: "
                        f"chunks={chunk_count}, "
                        f"tool_calls={tool_call_count}, "
                        f"text_chars={text_chars}, "
                        f"finish_reason={chunk.finish_reason}"
                    )

                yield chunk

            logger.info("Provider middleware completed successfully")

        except asyncio.CancelledError:
            # Stream was cancelled (e.g., due to interruption)
            logger.info("Provider middleware cancelled, closing LLM stream")
            # Explicitly close the stream while we have async context
            if llm_stream is not None:
                try:
                    # Close the async generator properly
                    await llm_stream.aclose()
                    logger.debug("LLM stream closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing LLM stream: {e}")
            raise  # Re-raise to propagate cancellation

        except Exception as e:
            logger.error(f"Provider middleware error: {e}", exc_info=True)
            # Yield error chunk
            yield StreamChunk(
                type="error",
                content=f"Provider error: {str(e)}",
                metadata={"error": str(e), "error_type": type(e).__name__}
            )
