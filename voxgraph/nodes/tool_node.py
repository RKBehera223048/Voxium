"""
Voxium — Tool Execution Node
================================
LangGraph node that executes tool calls requested by the agent LLM.

When the agent_node produces actions (tool calls), this node executes
them and returns results. The graph can loop back to agent_node for
multi-step tool chains.

Reads: actions
Writes: tool_results, messages
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List

from langchain_core.messages import ToolMessage

from voxgraph.state import VoxiumState, AgentAction

logger = logging.getLogger(__name__)


async def tool_node(state: VoxiumState) -> Dict[str, Any]:
    """
    LangGraph node: Execute queued tool calls.

    Iterates over state.actions and dispatches each to the appropriate
    tool handler. Results are stored in state.tool_results and added
    as ToolMessages for the next LLM invocation.
    """
    actions = state.get("actions", [])
    if not actions:
        return {"tool_results": []}

    results: List[Dict[str, Any]] = []
    messages = []

    for action in actions:
        intent = action.get("intent", "")
        params = action.get("parameters", {})

        logger.info("Executing tool: %s (params=%s)", intent, params)

        try:
            result = await _dispatch_tool(intent, params)
            results.append({
                "intent": intent,
                "success": True,
                "result": result,
            })
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=intent,
            ))
        except Exception as e:
            logger.error("Tool execution failed: %s: %s", intent, e)
            results.append({
                "intent": intent,
                "success": False,
                "error": str(e),
            })
            messages.append(ToolMessage(
                content=f"Error: {e}",
                tool_call_id=intent,
            ))

    return {
        "tool_results": results,
        "actions": [],  # Clear actions after execution
        "messages": messages,
    }


async def _dispatch_tool(intent: str, params: Dict[str, Any]) -> Any:
    """
    Dispatch a tool call to the appropriate handler.
    
    Uses LangChain @tool functions from llm.tools.
    """
    from llm.tools import (
        tool_read_file, tool_write_file, tool_list_directory,
        tool_read_clipboard, tool_write_clipboard, tool_search_memory
    )
    
    tool_map = {
        "file.read": tool_read_file,
        "file.write": tool_write_file,
        "file.list": tool_list_directory,
        "clipboard.read": tool_read_clipboard,
        "clipboard.write": tool_write_clipboard,
        "search.memory": tool_search_memory,
    }
    
    if intent in tool_map:
        return await tool_map[intent].ainvoke(params)
        
    raise ValueError(f"Unknown tool intent: {intent}")
