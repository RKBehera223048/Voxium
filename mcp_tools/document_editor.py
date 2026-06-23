"""
Voxium — Document Editor Tool (Stub)
=======================================
MCP action handler for document formatting, highlighting,
and text manipulation in the active editor.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def handle_document_action(
    action: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Handle document editing agent commands."""
    logger.info("Document action: %s params=%s", action, parameters)

    return {
        "success": False,
        "error": "Document editor not yet configured",
        "action": action,
    }
