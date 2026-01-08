"""
Tool call middleware for intercepting and executing tool calls from OpenAI.

Integrates with the provider middleware pattern to enable re-entrant tool calling:
- Detects tool calls from OpenAI
- Yields tool_notification chunks for animations
- Executes tools
- Creates new provider with tool results
- Re-enters pipeline to get OpenAI's continuation
"""

import logging
from typing import AsyncIterator, Callable, Dict, Any, List
import json

from reachy_mvp.core.streaming import StreamMiddleware, StreamChunk, StreamPipeline
from reachy_mvp.tools.tool_handlers import get_tool_handler

logger = logging.getLogger(__name__)


class ToolCallMiddleware(StreamMiddleware):
    """
    Middleware that intercepts tool calls and handles re-entry via internal loop.

    This middleware wraps the provider middleware, using an internal do-while loop to:
    1. Iterate through provider's stream, detecting tool_call chunks
    2. Yield tool_notification chunks (for animations) but accumulate tool_calls
    3. Execute tool handlers and accumulate results
    4. After stream ends: if tool calls detected, add them + results to provider history
    5. Call provider.process() again with updated history (loop continues)
    6. Exit loop when stream has no tool calls

    This eliminates recursive pipeline creation - tool middleware loops internally,
    calling the same provider instance multiple times with updated history.
    """

    def __init__(
        self,
        tool_registry: Dict[str, Callable],
        provider: Any,
        max_tool_depth: int = 10
    ):
        """
        Initialize tool call middleware.

        Args:
            tool_registry: Dictionary mapping tool names to handler functions
            provider: Reference to ProviderMiddleware instance to call for re-entry
            max_tool_depth: Maximum iteration depth for nested tool calls (safety limit)
        """
        self.tool_registry = tool_registry
        self.provider = provider
        self.max_tool_depth = max_tool_depth

        logger.debug(
            f"ToolCallMiddleware initialized: "
            f"tools={list(tool_registry.keys())}, "
            f"max_depth={max_tool_depth}"
        )

    async def process(
        self,
        stream: AsyncIterator[StreamChunk]
    ) -> AsyncIterator[StreamChunk]:
        """
        Process stream, intercepting tool calls and handling re-entry with internal loop.

        Args:
            stream: Input stream from downstream (provider)

        Yields:
            StreamChunks including tool_notification, text, and other types
        """
        logger.info("ToolCallMiddleware processing")
        logger.debug("ToolCallMiddleware.process() starting")

        # Get reference to provider (downstream middleware)
        # Since this is a pipeline, stream is coming from the provider
        provider_stream = stream

        try:
            # Do-while loop: keep calling provider until no more tool calls
            continue_loop = True
            iteration = 0

            while continue_loop:
                iteration += 1
                logger.info(f"[TOOL] Loop iteration {iteration}")

                # Track tool calls in this iteration
                tool_calls_accumulated = []
                tool_results = []

                # Iterate through provider's stream
                async for chunk in provider_stream:
                    logger.debug(f"ToolCallMiddleware received chunk: {chunk.type}")

                    if chunk.is_tool_call():
                        # Tool call detected - accumulate it, don't yield
                        tool_name = chunk.tool_name
                        arguments = chunk.arguments or {}
                        tool_call_id = chunk.tool_call_id

                        logger.info(
                            f"[TOOL] Tool call detected: name={tool_name}, "
                            f"args={arguments}, id={tool_call_id}"
                        )

                        # Accumulate tool call info
                        tool_calls_accumulated.append({
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments) if arguments else "{}"
                            }
                        })

                        # Yield tool_notification (for animations) but NOT the tool_call itself
                        notification = StreamChunk.tool_notification(
                            tool_name=tool_name,
                            arguments=arguments
                        )
                        logger.debug(f"Yielding tool_notification: {tool_name}")
                        yield notification

                        # Execute tool and accumulate result
                        logger.info(f"Executing tool: {tool_name}")
                        result = await self._execute_tool(tool_name, arguments)
                        logger.info(f"Tool execution complete: {tool_name}, result_len={len(result)}")

                        tool_results.append({
                            "id": tool_call_id,
                            "name": tool_name,
                            "result": result
                        })

                    else:
                        # Pass through all non-tool-call chunks
                        logger.debug(f"Passing through {chunk.type} chunk")
                        yield chunk

                # Stream finished - check if we accumulated any tool calls
                if tool_calls_accumulated:
                    logger.info(f"[TOOL] Iteration {iteration} had {len(tool_calls_accumulated)} tool calls, looping again")

                    # Check recursion depth
                    if iteration >= self.max_tool_depth:
                        logger.warning(
                            f"Max tool depth ({self.max_tool_depth}) reached, "
                            f"stopping iteration"
                        )
                        yield StreamChunk.error(
                            f"Maximum tool call depth ({self.max_tool_depth}) exceeded",
                            error_type="MAX_DEPTH_EXCEEDED"
                        )
                        continue_loop = False
                        break

                    # Build messages to add to provider's conversation history
                    messages_to_add = self._build_tool_messages(
                        tool_calls_accumulated,
                        tool_results
                    )

                    logger.debug(f"Adding {len(messages_to_add)} messages to provider history")

                    # Add messages to the provider's conversation history
                    # This modifies the provider's state for next iteration
                    self.provider.add_messages(messages_to_add)

                    # The old stream should be exhausted (we iterated through it)
                    # But explicitly close it to ensure cleanup happens in async context
                    try:
                        await provider_stream.aclose()
                    except Exception as e:
                        logger.debug(f"Error closing old provider stream: {e}")

                    # Get new stream from provider for next iteration
                    provider_stream = self.provider.process(self._empty_stream())

                else:
                    # No tool calls - we're done
                    logger.info(f"[TOOL] Iteration {iteration} had no tool calls, exiting loop")
                    continue_loop = False

            logger.info("ToolCallMiddleware loop complete")

            # Ensure final stream is closed
            try:
                if provider_stream is not None:
                    await provider_stream.aclose()
                    logger.debug("Final provider stream closed")
            except Exception as e:
                logger.debug(f"Error closing final provider stream: {e}")

        except Exception as e:
            logger.error(f"ToolCallMiddleware error: {e}", exc_info=True)
            yield StreamChunk.error(
                f"Tool middleware error: {str(e)}",
                error_type=type(e).__name__
            )

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> str:
        """
        Execute tool handler with arguments.

        Args:
            tool_name: Name of tool to execute
            arguments: Dict of arguments for tool

        Returns:
            Tool result as string
        """
        logger.debug(f"Looking up handler for tool: {tool_name}")
        try:
            # Get handler
            handler = get_tool_handler(tool_name)

            if handler is None:
                error_msg = f"Unknown tool: '{tool_name}'"
                logger.error(error_msg)
                return f"Error: {error_msg}"

            # Execute handler with arguments
            logger.debug(f"Calling handler for {tool_name} with args: {arguments}")
            result = await handler(**arguments)
            logger.debug(f"Handler returned: {result[:100]}...")

            return result

        except TypeError as e:
            # Wrong arguments provided
            error_msg = f"Invalid arguments for tool '{tool_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}"

        except Exception as e:
            # Other execution error
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}"

    def _build_tool_messages(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Build messages to add to conversation history for tool results.

        OpenAI requires:
        1. Assistant message with tool_calls array
        2. Tool message(s) with matching tool_call_id and result

        Args:
            tool_calls: List of tool call objects from OpenAI
            tool_results: List of dicts with 'id', 'name', and 'result' keys

        Returns:
            List of message dicts to add to conversation history
        """
        messages = []

        # 1. Add assistant message with tool_calls
        # (OpenAI needs to see what it requested)
        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": None,  # No text content when making tool calls
                "tool_calls": tool_calls
            })
            logger.debug(f"Built assistant message with {len(tool_calls)} tool_calls")

        # 2. Add tool result messages (one for each tool call)
        for tool_result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_result["id"],
                "name": tool_result["name"],
                "content": tool_result["result"]
            })
            logger.debug(f"Built tool message: id={tool_result['id']}, result_len={len(tool_result['result'])}")

        return messages

    async def _empty_stream(self) -> AsyncIterator[StreamChunk]:
        """
        Create an empty async generator for provider middleware.

        Provider middleware ignores its input stream (it's a source, not transformer),
        so we pass an empty stream.

        Yields:
            Nothing - this is an empty generator
        """
        # Empty generator - yields nothing
        if False:
            yield
