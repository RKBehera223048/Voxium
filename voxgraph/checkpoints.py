"""
Voxium — LangGraph Checkpoint Storage
========================================
SQLite-backed checkpoint persistence for LangGraph sessions.

Enables:
    - Session resume after server restart
    - Conversation history across multiple interactions
    - Checkpoint-based state rollback
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default checkpoint database path
DEFAULT_CHECKPOINT_DB = os.getenv("CHECKPOINT_DB_PATH", "data/db/checkpoints.sqlite")


def get_checkpointer():
    """
    Get or create the SQLite checkpointer for LangGraph.

    Returns a SqliteSaver instance configured with the checkpoint database.
    Falls back to MemorySaver if sqlite dependency is unavailable.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = Path(DEFAULT_CHECKPOINT_DB)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        checkpointer = SqliteSaver.from_conn_string(str(db_path))
        logger.info("LangGraph checkpointer: SQLite at %s", db_path)
        return checkpointer

    except ImportError:
        logger.warning(
            "langgraph-checkpoint-sqlite not installed. "
            "Using in-memory checkpointer (state won't persist across restarts). "
            "Install with: pip install langgraph-checkpoint-sqlite"
        )
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    except Exception as e:
        logger.error("Failed to create SQLite checkpointer: %s", e)
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
