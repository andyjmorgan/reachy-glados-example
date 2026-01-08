"""
Tool implementations for Reachy MVP.

Simplified to just datetime functionality as requested.
"""

from datetime import datetime
from typing import Dict, Callable, Awaitable, Optional


# Tool handler functions

async def get_time() -> str:
    """
    Get the current time.

    Returns:
        Current time formatted as string
    """
    now = datetime.now()
    return f"The current time is {now.strftime('%I:%M %p')}"


async def get_date() -> str:
    """
    Get the current date.

    Returns:
        Current date formatted as string
    """
    now = datetime.now()
    return f"Today is {now.strftime('%A, %B %d, %Y')}"


# Anthropic Claude function definitions
# Note: Anthropic doesn't have built-in web search like OpenAI
# But supports web fetching and search via custom tools or extended thinking
# For now, using only time/date tools

TOOL_DEFINITIONS = [
    {
        "name": "get_time",
        "description": "Get the current time",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_date",
        "description": "Get the current date",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# TODO: Add Anthropic extended thinking or web search capabilities
# Anthropic supports: computer_use, bash, text_editor as server-side tools
# For web search, we'd need to implement a custom tool with an external API


# Tool handler registry

TOOL_HANDLERS: Dict[str, Callable[..., Awaitable[str]]] = {
    "get_time": get_time,
    "get_date": get_date,
}


def get_tool_handler(tool_name: str) -> Optional[Callable[..., Awaitable[str]]]:
    """
    Get a tool handler by name.

    Args:
        tool_name: Name of the tool

    Returns:
        Tool handler function or None if not found
    """
    return TOOL_HANDLERS.get(tool_name)
