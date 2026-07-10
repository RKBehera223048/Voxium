from __future__ import annotations

import logging
from typing import List

from langchain_core.tools import tool

from tools.file_manager import read_file, write_file, list_directory
from tools.clipboard import read_clipboard, write_clipboard
from tools.search import search_memory

logger = logging.getLogger(__name__)

@tool
async def tool_read_file(path: str) -> str:
    """
    Read the contents of a file.
    
    Use this tool to read the text content of any file on the system.
    Provide the exact path to the file.
    
    Args:
        path: The absolute or relative path to the file.
        
    Returns:
        The content of the file.
    """
    logger.debug(f"Calling tool_read_file with path={path}")
    return await read_file(path)

@tool
async def tool_write_file(path: str, content: str) -> str:
    """
    Write content to a file.
    
    Use this tool to create or overwrite a file with new text content.
    Provide the exact path and the content to write.
    
    Args:
        path: The absolute or relative path to the file.
        content: The text content to write.
        
    Returns:
        A success message indicating the file was written.
    """
    logger.debug(f"Calling tool_write_file with path={path}")
    return await write_file(path, content)

@tool
async def tool_list_directory(path: str) -> List[str]:
    """
    List the contents of a directory.
    
    Use this tool to explore the filesystem by listing files and folders
    within a specific directory.
    
    Args:
        path: The absolute or relative path to the directory.
        
    Returns:
        A list of file and directory names.
    """
    logger.debug(f"Calling tool_list_directory with path={path}")
    return await list_directory(path)

@tool
async def tool_read_clipboard() -> str:
    """
    Read text content from the system clipboard.
    
    Use this tool when you need to access text that the user has recently copied.
    
    Returns:
        The text currently in the clipboard.
    """
    logger.debug("Calling tool_read_clipboard")
    return await read_clipboard()

@tool
async def tool_write_clipboard(text: str) -> str:
    """
    Write text content to the system clipboard.
    
    Use this tool to copy text so the user can easily paste it elsewhere.
    
    Args:
        text: The text to copy to the clipboard.
        
    Returns:
        A success message.
    """
    logger.debug("Calling tool_write_clipboard")
    return await write_clipboard(text)

@tool
async def tool_search_memory(query: str) -> str:
    """
    Search the agent's memory for previously stored information.
    
    Use this tool to recall facts, previous conversation context, or learned data.
    
    Args:
        query: The search query string.
        
    Returns:
        The search results containing relevant memory snippets.
    """
    logger.debug(f"Calling tool_search_memory with query={query}")
    return await search_memory(query)
