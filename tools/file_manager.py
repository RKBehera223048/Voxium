from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

async def read_file(path: str) -> str:
    """
    Read the contents of a file asynchronously.
    
    Args:
        path: The absolute or relative path to the file.
        
    Returns:
        The content of the file as a string.
    """
    logger.info(f"Reading file: {path}")
    # Stub implementation
    return f"Dummy content for {path}"

async def write_file(path: str, content: str) -> str:
    """
    Write content to a file asynchronously.
    
    Args:
        path: The absolute or relative path to the file.
        content: The text content to write.
        
    Returns:
        A success message.
    """
    logger.info(f"Writing to file: {path}")
    # Stub implementation
    return f"Successfully wrote to {path}"

async def list_directory(path: str) -> List[str]:
    """
    List the contents of a directory asynchronously.
    
    Args:
        path: The absolute or relative path to the directory.
        
    Returns:
        A list of file and directory names.
    """
    logger.info(f"Listing directory: {path}")
    # Stub implementation
    return ["dummy_file1.txt", "dummy_file2.txt", "dummy_dir"]
