"""Safe, typed, statically registered M3A video tools."""

from cutagent.tools.factory import create_m3a_registry, create_media_tool_registry
from cutagent.tools.registry import ToolRegistry

__all__ = ["ToolRegistry", "create_m3a_registry", "create_media_tool_registry"]
